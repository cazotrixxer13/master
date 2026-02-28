import os
import jax
import jax.numpy as jnp
import jax.random as random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from model import LatentVariableModel, PRIOR_MEAN, PRIOR_COV, SIGMA_X
from utils import load_data, plot_loss
from matplotlib import pyplot as plt
from gamma import estimate_gamma_with_vi_posterior


def numpyro_model(x_obs: jnp.ndarray):
    """
    Create the NumPyro model for the latent variable model p(z) p(x | z).

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observed data with shape (3,).

    Notes
    -----
    - Use ``dist.Normal(...).to_event(1)`` to create an isotropic multivariate
      Normal distribution.
    - Use ``dist.MultivariateNormal(...)`` to create a full-covariance
      multivariate Normal distribution.
    """
    # Hint: Use dist.Normal(...).to_event(1) to create an isotropic multivariate Normal distribution.
    #       Use dist.MultivariateNormal(...) to create a full covariance multivariate Normal distribution.
    z = numpyro.sample(
        "z",
        dist.MultivariateNormal(loc=PRIOR_MEAN, covariance_matrix=PRIOR_COV),
    )

    mu_x = LatentVariableModel.f(z)
    numpyro.sample(
        "x",
        dist.Normal(loc=mu_x, scale=SIGMA_X).to_event(1),
        obs=x_obs,
    )


# Note: we won't use x_obs here, but it's required by the SVI interface
def numpyro_guide_full_cov(x_obs):
    """
    Create a full-covariance Gaussian guide q(z) for SVI in NumPyro.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observed data with shape (3,).

    Notes
    -----
    - Define parameters via ``numpyro.param`` for the mean and covariance.
    - Use ``dist.MultivariateNormal(...)`` for a full-covariance Gaussian.
    """
    # Hint: Use numpyro.param(...) to create the parameters for the mean and covariance.
    #       Use dist.Normal(...).to_event(1) to create an isotropic multivariate Normal distribution.
    #       Use dist.MultivariateNormal(...) to create a full covariance multivariate Normal distribution.
    mu_q = numpyro.param("mu_q", PRIOR_MEAN)

    cov_q = numpyro.param(
        "cov_q",
        jnp.eye(2),
        constraint=constraints.positive_definite,
    )

    numpyro.sample(
        "z",
        dist.MultivariateNormal(loc=mu_q, covariance_matrix=cov_q),
    )


def vi_fit_full_cov(
    x_obs: jnp.ndarray,
    num_steps: int = 3000,
    lr: float = 3e-3,
    num_samples: int = 64,
    key: jax.Array = random.PRNGKey(0),
):
    """
    Fit a full-covariance Gaussian guide with SVI for a single observation.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observed data with shape (3,).
    num_steps : int, optional
        Number of optimization steps, by default 3000.
    lr : float, optional
        Learning rate, by default 3e-3.
    num_samples : int, optional
        Number of particles for the ELBO estimator, by default 64.
    key : jax.Array, optional
        PRNG key, by default ``random.PRNGKey(0)``.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict]
        - mu_q: (2,) variational mean
        - cov_q: (2, 2) variational covariance
        - losses: (num_steps,) negative ELBO trajectory
        - param_map: parameter dictionary usable for ELBO recomputation
    """
    optimizer = numpyro.optim.Adam(step_size=lr)
    svi = SVI(
        numpyro_model,
        numpyro_guide_full_cov,
        optimizer,
        loss=Trace_ELBO(num_particles=num_samples),
    )
    svi_result = svi.run(key, num_steps=num_steps, x_obs=x_obs)
    params = svi_result.params
    # extract parameters of the model
    mu_q, cov_q = params["mu_q"], params["cov_q"]

    # This is used to recompute the ELBO in the end easily:
    param_map = svi.get_params(svi_result.state)
    return mu_q, cov_q, svi_result.losses, param_map


def trace_elbo_full_cov(
    x_obs: jnp.ndarray,
    param_map,
    num_samples: int = 64,
    key: jax.Array = random.PRNGKey(0),
):
    """
    Compute the ELBO for the fitted full-covariance Gaussian guide.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observed data with shape (3,).
    param_map : dict
        Parameter dictionary returned from ``vi_fit_full_cov``.
    num_samples : int, optional
        Number of particles for the ELBO estimator, by default 64.
    key : jax.Array, optional
        PRNG key, by default ``random.PRNGKey(0)``.

    Returns
    -------
    jnp.ndarray
        Scalar ELBO value (note: we negate since loss is negative ELBO).
    """
    elbo = Trace_ELBO(num_particles=num_samples)
    # We negate since the loss is the negative ELBO
    return -elbo.loss(
        rng_key=key,
        param_map=param_map,
        model=numpyro_model,
        guide=numpyro_guide_full_cov,
        x_obs=x_obs,
    )


def main():
    os.makedirs("figures/vi_numpyro", exist_ok=True)

    model = LatentVariableModel(
        prior_mean=PRIOR_MEAN, prior_cov=PRIOR_COV, sigma_x=SIGMA_X
    )

    num_samples_elbo_eval = 100_000
    key = jax.random.PRNGKey(42)

    _, x_test = load_data()
    for idx, x_obs in enumerate(x_test[:5]):
        # Run vi_fit_full_cov and plot_loss
        key_fit, key = jax.random.split(key)
        mean_full_cov, cov_full_cov, losses_full_cov, param_map_full_cov = vi_fit_full_cov(
            x_obs=x_obs,
            num_steps=1000,
            lr=3e-3,
            num_samples=50,
            key=key_fit,
        )
        plot_loss(losses_full_cov, f"figures/vi_numpyro/loss_{idx}.png")

        # Recompute the ELBO for the final VI params with `num_samples_elbo_eval` samples
        key_eval, key = jax.random.split(key)
        elbo_full_cov = trace_elbo_full_cov(
            x_obs=x_obs,
            param_map=param_map_full_cov,
            num_samples=num_samples_elbo_eval,
            key=key_eval,
        )

        model.plot_posterior(
            x_obs,
            q_mean=[mean_full_cov],
            q_cov=[cov_full_cov],
            titles=[
                f"Full Covariance VI, ELBO={elbo_full_cov:.3f}",
            ],
            save_path=f"figures/vi_numpyro/posterior_{idx}.png",
        )

        # Compute gamma_q using `estimate_gamma_with_vi_posterior` and add it to your table
        gamma_q = estimate_gamma_with_vi_posterior(mean_full_cov, cov_full_cov)
        print(f"[{idx}] gamma_p: {gamma_q}")


if __name__ == "__main__":
    main()
