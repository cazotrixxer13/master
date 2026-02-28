import os
import jax
import jax.numpy as jnp
from flax import linen as nn
import functools
from model import PRIOR_MEAN, PRIOR_COV, SIGMA_X, LatentVariableModel
from vi import elbo as vi_elbo, vi_fit as vi_fit_diag
from utils import load_data, plot_loss
from tqdm import trange
from vi_numpyro import vi_fit_full_cov, trace_elbo_full_cov
from numpyro.infer import Trace_ELBO


# -----------------------------
# Encoder network (Flax MLP)
# -----------------------------
class Encoder(nn.Module):
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, x):
        """
        Parameters
        ----------
        x : jnp.ndarray
            Input with shape (batch, 3).

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            - mu: (batch, 2)
            - log_std: (batch, 2)
        """
        h = nn.tanh(nn.Dense(self.hidden_dim)(x))
        h = nn.tanh(nn.Dense(self.hidden_dim)(h))
        out = nn.Dense(4)(h)  # 2 for mu, 2 for log_std
        mu, log_std = jnp.split(out, 2, axis=-1)
        return mu, log_std


def amortized_elbo_single(
    x_obs: jnp.ndarray,
    key: jax.Array,
    params,
    encoder: Encoder,
    gen_model: LatentVariableModel,
    num_samples: int = 1,
) -> jnp.ndarray:
    """
    ELBO for a single observation using amortized variational parameters.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observation with shape (3,).
    key : jax.Array
        PRNG key.
    params : PyTree
        Parameters of the encoder network.
    encoder : Encoder
        Encoder network module.
    gen_model : LatentVariableModel
        Generative model p(x, z).
    num_samples : int, optional
        Number of reparameterization samples M (eps ~ N(0, I)), by default 1.

    Returns
    -------
    jnp.ndarray
        Scalar ELBO.
    """
    # Hint: Use encoder.apply(params, x_obs[None, :]) to get the mean and log_std, which you may need to squeeze again
    #       Draw fresh eps_samples and use vi_elbo to compute the ELBO
    mu, log_std = encoder.apply(params, x_obs[None, :])
    mu = jnp.squeeze(mu, axis=0)
    log_std = jnp.squeeze(log_std, axis=0)
    key, subkey = jax.random.split(key)
    eps_samples = jax.random.normal(subkey, shape=(num_samples, 2))
    return vi_elbo(gen_model, mu, log_std, x_obs, eps_samples)


def amortized_elbo_batch(
    params,
    encoder: Encoder,
    gen_model: LatentVariableModel,
    x_batch: jnp.ndarray,
    key: jax.Array,
    num_samples: int = 1,
) -> jnp.ndarray:
    """
    Vectorized ELBO over a batch by vmapping the single-x ELBO.

    Parameters
    ----------
    params : PyTree
        Encoder parameters.
    encoder : Encoder
        Encoder network module.
    gen_model : LatentVariableModel
        Generative model p(x, z).
    x_batch : jnp.ndarray
        Batch with shape (B, 3).
    key : jax.Array
        PRNG key.
    num_samples : int, optional
        Number of reparameterization samples M per data point, by default 1.

    Returns
    -------
    jnp.ndarray
        Scalar mean ELBO over the batch.
    """
    batch_size = x_batch.shape[0]
    keys = jax.random.split(key, batch_size)
    # vmap over x_obs and key; params/encoder/gen_model shared
    vmapped_elbo = jax.vmap(
        amortized_elbo_single, in_axes=(0, 0, None, None, None, None)
    )
    # Call the vmapped ELBO and return the mean
    return jnp.mean(vmapped_elbo(x_batch, keys, params, encoder, gen_model, num_samples))


def loss_fn(
    params,
    encoder: Encoder,
    gen_model: LatentVariableModel,
    x_batch,
    key,
    num_samples=1,
):
    """
    Compute training loss as negative ELBO.

    Parameters
    ----------
    params : PyTree
        Encoder parameters.
    encoder : Encoder
        Encoder network module.
    gen_model : LatentVariableModel
        Generative model p(x, z).
    x_batch : jnp.ndarray
        Batch with shape (B, 3).
    key : jax.Array
        PRNG key.
    num_samples : int, optional
        Number of reparameterization samples M per data point, by default 1.

    Returns
    -------
    jnp.ndarray
        Scalar loss (-ELBO).
    """
    return -amortized_elbo_batch(
        params, encoder, gen_model, x_batch, key, num_samples=num_samples
    )


# -----------------------------
# Training loop (simple SGD)
# -----------------------------


@functools.partial(
    jax.jit,
    static_argnames=("encoder", "gen_model", "lr", "num_samples"),
)
def train_step(
    params,
    encoder: Encoder,
    gen_model: LatentVariableModel,
    x_batch,
    key,
    lr=1e-3,
    num_samples=1,
):
    """
    Single SGD step on the amortized VI objective.

    Parameters
    ----------
    params : PyTree
        Current encoder parameters.
    encoder : Encoder
        Encoder network module.
    gen_model : LatentVariableModel
        Generative model p(x, z).
    x_batch : jnp.ndarray
        Batch with shape (B, 3).
    key : jax.Array
        PRNG key.
    lr : float, optional
        Learning rate, by default 1e-3.
    num_samples : int, optional
        Number of reparameterization samples M per data point, by default 1.

    Returns
    -------
    tuple
        - new_params: PyTree of updated parameters
        - loss: scalar loss value
    """
    # compute grad of loss wrt params
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params, encoder, gen_model, x_batch, key, num_samples)

    # SGD update
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


def train_encoder(
    encoder: Encoder,
    x_train: jnp.ndarray,
    gen_model: LatentVariableModel,
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_samples: int = 1,
    key: jax.Array = jax.random.PRNGKey(0),
):
    """
    Train the amortized encoder on a dataset.

    Parameters
    ----------
    encoder : Encoder
        Encoder network module.
    x_train : jnp.ndarray
        Training data with shape (N, 3).
    gen_model : LatentVariableModel
        Generative model p(x, z).
    num_epochs : int, optional
        Number of epochs, by default 50.
    batch_size : int, optional
        Batch size, by default 64.
    lr : float, optional
        Learning rate, by default 1e-3.
    num_samples : int, optional
        Number of reparameterization samples M per data point, by default 1.
    key : jax.Array, optional
        PRNG key, by default jax.random.PRNGKey(0).

    Returns
    -------
    tuple
        - params: PyTree of trained encoder parameters
        - losses: (steps,) array of training losses
    """
    N = x_train.shape[0]

    # create model and initialize
    key_init, key_train = jax.random.split(key)

    # dummy input for init
    dummy_x = jnp.zeros((1, 3))
    params = encoder.init(key_init, dummy_x)

    losses = []
    key_loop = key_train
    with trange(num_epochs, desc="Training Epochs", leave=True) as pbar:
        for epoch in pbar:
            epoch_losses = []
            for i in range(0, N, batch_size):
                # no shuffling for simplicity
                x_batch = x_train[i : i + batch_size]

                key_loop, key_step = jax.random.split(key_loop)
                params, loss = train_step(
                    params,
                    encoder,
                    gen_model,
                    x_batch,
                    key_step,
                    lr=lr,
                    num_samples=num_samples,
                )

                losses.append(loss)
                epoch_losses.append(loss)

            avg_loss = float(jnp.mean(jnp.array(epoch_losses)))
            pbar.set_postfix(loss=avg_loss)

    return params, jnp.array(losses)


def main():
    os.makedirs("figures/avi", exist_ok=True)

    gen_model = LatentVariableModel(
        prior_mean=PRIOR_MEAN, prior_cov=PRIOR_COV, sigma_x=SIGMA_X
    )

    x_train, x_test = load_data()
    key = jax.random.PRNGKey(42)
    encoder = Encoder(hidden_dim=32)
    num_samples_elbo_eval = 100_000

    # Train the encoder and get the parameters and losses. Plot the loss curve and include in your report.
    params, losses = train_encoder(
        encoder=encoder,
        x_train=x_train,
        gen_model=gen_model,
        num_epochs=20,
        batch_size=128,
        lr=1e-3,
        num_samples=1,
        key=key,
    )

    plot_loss(
        losses,
        save_path="figures/avi/avi_training_loss.png",
    )

    amort_gaps_diag, amort_gaps_full = [], []

    for idx, x_obs in enumerate(x_test):
        key, subkey = jax.random.split(key)
        # Get mean and log_std predicted by encoder and compute the ELBO using num_samples_elbo_eval samples
        mu_pred, log_std_pred = encoder.apply(params, x_obs[None, :])
        mu_q_pred = jnp.squeeze(mu_pred, axis=0)
        log_std_pred = jnp.squeeze(log_std_pred, axis=0)
        cov_q_pred = jnp.diag(jnp.exp(2.0 * log_std_pred))
        elbo_amortized = vi_elbo(gen_model, mu_q_pred, log_std_pred, x_obs, jax.random.normal(subkey, shape=(num_samples_elbo_eval, 2)))

        # Fit the diagonal VI and compute the ELBO with num_samples_elbo_eval samples
        mu_q_diag, cov_q_diag, losses_diag = vi_fit_diag(gen_model, x_obs)
        log_std_diag = jnp.log(jnp.sqrt(jnp.diag(cov_q_diag)))
        elbo_vi_diag = vi_elbo(gen_model, mu_q_diag, log_std_diag, x_obs, jax.random.normal(subkey, shape=(num_samples_elbo_eval, 2)))

        # Same for full-covariance VI
        mu_q_full_cov, cov_q_full_cov, losses_full_cov, param_map_full_cov = vi_fit_full_cov(x_obs)
        elbo_vi_full_cov = trace_elbo_full_cov(x_obs, param_map_full_cov, num_samples=num_samples_elbo_eval, key=subkey)

        if idx < 5:
            # Plot posterior and variational distributions for x_obs
            gen_model.plot_posterior(
                x_obs,
                q_mean=[mu_q_pred, mu_q_diag, mu_q_full_cov],
                q_cov=[cov_q_pred, cov_q_diag, cov_q_full_cov],
                titles=[
                    f"Amortized VI, ELBO={elbo_amortized:.2f}",
                    f"Diagonal VI, ELBO={elbo_vi_diag:.2f}, Gap={elbo_vi_diag - elbo_amortized:.2f}",
                    f"Full Covariance VI, ELBO={elbo_vi_full_cov:.2f}, Gap={elbo_vi_full_cov - elbo_amortized:.2f}",
                ],
                save_path=f"figures/avi/vi_comparison_x{idx}.png",
                suptitle=f"Target: $p(z \mid x_{idx})$",
            )

        # Compute amortization gap for both diagonal and full-covariance VI. In the end, report the statistics (see assignment sheet).
        amort_gaps_diag.append(float(elbo_vi_diag - elbo_amortized))
        amort_gaps_full.append(float(elbo_vi_full_cov - elbo_amortized))
    
    amort_gaps_diag = jnp.array(amort_gaps_diag)
    amort_gaps_full = jnp.array(amort_gaps_full)
    
    mean_diag = float(jnp.mean(amort_gaps_diag))
    std_diag = float(jnp.std(amort_gaps_diag))
    min_diag = float(jnp.min(amort_gaps_diag))
    max_diag = float(jnp.max(amort_gaps_diag))
    mean_full = float(jnp.mean(amort_gaps_full))
    std_full = float(jnp.std(amort_gaps_full))
    min_full = float(jnp.min(amort_gaps_full))
    max_full = float(jnp.max(amort_gaps_full))

    print("Amortization gap (Diagonal VI):")
    print(f"  mean = {mean_diag:.4f}")
    print(f"  std  = {std_diag:.4f}")
    print(f"  min  = {min_diag:.4f}")
    print(f"  max  = {max_diag:.4f}")

    print("Amortization gap (Full-cov VI):")
    print(f"  mean = {mean_full:.4f}")
    print(f"  std  = {std_full:.4f}")
    print(f"  min  = {min_full:.4f}")
    print(f"  max  = {max_full:.4f}")


if __name__ == "__main__":
    main()
