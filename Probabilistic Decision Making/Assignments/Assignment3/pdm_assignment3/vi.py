import jax
import jax.numpy as jnp
from model import PRIOR_MEAN, PRIOR_COV, SIGMA_X, LatentVariableModel
from jax.scipy.stats import multivariate_normal
from utils import load_data, plot_loss
from gamma import estimate_gamma_with_vi_posterior
import os


def elbo(
    model: LatentVariableModel,
    mu: jnp.ndarray,
    log_std: jnp.ndarray,
    x_obs: jnp.ndarray,
    eps_samples: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute a Monte Carlo estimate of the ELBO for a diagonal-Gaussian q.

    Parameters
    ----------
    model : LatentVariableModel
        Generative model p(x, z).
    mu : jnp.ndarray
        Mean of q(z) with shape (2,).
    log_std : jnp.ndarray
        Log standard deviations of q(z) with shape (2,).
    x_obs : jnp.ndarray
        Observed data with shape (3,).
    eps_samples : jnp.ndarray
        Standard Normal samples with shape (M, 2).

    Returns
    -------
    jnp.ndarray
        Scalar ELBO estimate.
    """
    z_samples = mu + jnp.exp(log_std) * eps_samples
    log_p = model.log_joint(x_obs, z_samples)

    q_cov = jnp.diag(jnp.exp(log_std)**2)
    log_q = multivariate_normal.logpdf(z_samples, mean=mu, cov=q_cov)

    return jnp.mean(log_p - log_q)




def vi_fit(
    model: LatentVariableModel,
    x_obs: jnp.ndarray,
    num_steps=2000,
    lr=1e-3,
    num_samples=50,
    key=jax.random.PRNGKey(0),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run vanilla VI with a diagonal-Gaussian q for a single observation.

    Parameters
    ----------
    model : LatentVariableModel
        Generative model p(x, z).
    x_obs : jnp.ndarray
        Observed data with shape (3,).
    num_steps : int, optional
        Number of optimization steps, by default 2000.
    lr : float, optional
        Learning rate, by default 1e-3.
    num_samples : int, optional
        Number of MC samples per ELBO estimate, by default 50.
    key : jax.Array, optional
        PRNG key, by default jax.random.PRNGKey(0).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        - mu_q: (2,) variational mean
        - cov_q: (2, 2) variational covariance
        - losses: (num_steps,) negative ELBO trajectory
    """
    mu_q, cov_q, losses = None, None, None
    
    
    mu = model.prior_mean
    log_std = jnp.zeros_like(mu)
    losses = jnp.zeros((num_steps,))
    
    def loss_fn(mu, log_std, x_obs, eps_samples):
        return -elbo(model=model, mu=mu, log_std=log_std, x_obs=x_obs, eps_samples=eps_samples)


    grad_fn = jax.grad(loss_fn, argnums=(0, 1))

    for t in range(num_steps):
        key, subkey = jax.random.split(key)
        eps_samples = jax.random.normal(subkey, shape=(num_samples, 2))

        loss_val = loss_fn(mu, log_std, x_obs, eps_samples)
        g_mu, g_log_std = grad_fn(mu, log_std, x_obs, eps_samples)

        mu = mu - lr * g_mu
        log_std = log_std - lr * g_log_std

        losses = losses.at[t].set(loss_val)

    std = jnp.exp(log_std)
    cov_q = jnp.diag(std**2)
    mu_q = mu
    
    return mu_q, cov_q, losses


def main():
    os.makedirs("figures/vi", exist_ok=True)

    model = LatentVariableModel(
        prior_mean=PRIOR_MEAN, prior_cov=PRIOR_COV, sigma_x=SIGMA_X
    )

    num_samples_elbo_eval = 100_000
    key = jax.random.PRNGKey(42)

    _, x_test = load_data()
    for idx, x_obs in enumerate(x_test[:5]):
        key_vi, key = jax.random.split(key)

        # Run vi_fit and plot_loss
        mu_q, cov_q, losses = vi_fit(
            model=model,
            x_obs=x_obs,
            num_steps=1000,
            lr=1e-3,
            num_samples=50,
            key=key_vi,
        )
        plot_loss(losses, f"figures/vi/loss_x{idx}.png")

        # Compute ELBO for final VI params with `num_samples_elbo_eval` samples
        std_q = jnp.sqrt(jnp.diag(cov_q))
        log_std_q = jnp.log(std_q)

        key_eval, key = jax.random.split(key)
        eps_eval = jax.random.normal(key_eval, shape=(num_samples_elbo_eval, 2))

        elbo_vi = elbo(
            model=model,
            mu=mu_q,
            log_std=log_std_q,
            x_obs=x_obs,
            eps_samples=eps_eval,
        )

        model.plot_posterior(
            x_obs,
            q_mean=[mu_q],
            q_cov=[cov_q],
            titles=[
                f"VI on $p(z \mid x_{idx})$ , ELBO={elbo_vi:.3f}",
            ],
            save_path=f"figures/vi/posterior_x{idx}.png",
        )

        # Compute gamma_q using `estimate_gamma_with_vi_posterior` and add it to your table
        gamma_q = estimate_gamma_with_vi_posterior(mu_q, cov_q)
        print(f"[{idx}] gamma_p: {gamma_q}")


if __name__ == "__main__":
    main()
