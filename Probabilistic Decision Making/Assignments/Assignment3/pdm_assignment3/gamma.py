import jax
import jax.numpy as jnp
from model import LatentVariableModel, PRIOR_MEAN, PRIOR_COV, SIGMA_X
from jax.scipy.special import logsumexp
from utils import load_data


def g(z: jnp.array) -> jnp.array:
    """
    Compute the function g(z) = z1^2 + z2^2 + z1*z2 + z1 + z2.

    Parameters
    ----------
    z : jnp.ndarray
        Input array with shape (..., 2).

    Returns
    -------
    jnp.ndarray
        Array of shape (...,) with g evaluated elementwise.
    """
    z1 = z[..., 0]
    z2 = z[..., 1]
    
    return z1**2 + z2**2 + z1 * z2 + z1 + z2


def estimate_gamma_with_vi_posterior(
    mu_q: jnp.ndarray, cov_q: jnp.ndarray
) -> jnp.ndarray:
    """
    Estimate gamma_q = E_q[g(z)] under a Gaussian variational posterior q.

    Parameters
    ----------
    mu_q : jnp.ndarray
        Mean of q(z) with shape (2,).
    cov_q : jnp.ndarray
        Covariance of q(z) with shape (2, 2).

    Returns
    -------
    jnp.ndarray
        Scalar estimate of gamma_q.
    """
    mu1 = mu_q[0] 
    mu2 = mu_q[1]
    s11 = cov_q[0, 0]
    s22 = cov_q[1, 1]
    s12 = cov_q[0, 1]

    return (s11 + mu1**2) + (s22 + mu2**2) + (s12 + mu1 * mu2) + mu1 + mu2


def estimate_gamma_with_mcmc_samples(samples: jnp.ndarray) -> jnp.ndarray:
    """
    Estimate gamma_p = E_p[g(z)] using MCMC samples.

    Parameters
    ----------
    samples : jnp.ndarray
        Array of posterior samples with shape (N, 2).

    Returns
    -------
    jnp.ndarray
        Scalar Monte Carlo estimate of gamma_p.
    """
    if samples.ndim == 3: samples = samples.reshape(-1, samples.shape[-1])
    
    return jnp.mean(g(z=samples))


def compute_gamma_numerically(
    x_obs: jnp.ndarray, num_grid_points: int = 2000, k_std: float = 5.0
) -> jnp.ndarray:
    """
    Numerically compute E_p[g(z) | x_obs] on a dense 2D grid.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observed data with shape (3,).
    num_grid_points : int, optional
        Number of grid points per dimension, by default 2000.
    k_std : float, optional
        Grid extends ±k_std standard deviations around the prior mean, by default 5.0.

    Returns
    -------
    jnp.ndarray
        Scalar numerical estimate of E_p[g(z) | x_obs].

    Notes
    -----
    Uses log-sum-exp normalization for numerical stability. The Riemann sum is
    over a uniform grid so the cell area cancels between numerator and denominator.
    """
    gen_model = LatentVariableModel(
        prior_mean=PRIOR_MEAN, prior_cov=PRIOR_COV, sigma_x=SIGMA_X
    )

    z1_std = jnp.sqrt(PRIOR_COV[0, 0])
    z2_std = jnp.sqrt(PRIOR_COV[1, 1])

    z1_grid = jnp.linspace(
        PRIOR_MEAN[0] - k_std * z1_std,
        PRIOR_MEAN[0] + k_std * z1_std,
        num_grid_points,
    )
    z2_grid = jnp.linspace(
        PRIOR_MEAN[1] - k_std * z2_std,
        PRIOR_MEAN[1] + k_std * z2_std,
        num_grid_points,
    )

    z1_mesh, z2_mesh = jnp.meshgrid(z1_grid, z2_grid)
    z_grid = jnp.stack(
        [z1_mesh.ravel(), z2_mesh.ravel()], axis=-1
    )  # (num_grid_points^2, 2)

    # Evaluate log unnormalized posterior at grid points
    log_joint = jax.jit(lambda z: gen_model.log_joint(x_obs, z))
    log_p = jax.vmap(log_joint)(z_grid)  # (N^2,)
    # Normalize with log-sum-exp (cell area cancels)
    log_Z = logsumexp(log_p)
    log_w = log_p - log_Z
    w = jnp.exp(log_w)  # normalized weights over the grid points
    # Evaluate g(z) on grid
    g_vals = g(z_grid)
    # Expected value
    return jnp.sum(w * g_vals)


if __name__ == "__main__":
    x_train, x_test = load_data()
    for idx, x_obs in enumerate(x_test[:5]):
        gamma_p = compute_gamma_numerically(x_obs)
        print(f"[{idx}] gamma_p: {gamma_p}")
