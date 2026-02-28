import jax
import jax.numpy as jnp
from dataclasses import dataclass

Array = jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GPParams:
    """
    Gaussian process hyperparameters \lambda.
    All parameters are stored in log-space to enforce positivity:
      amplitude = exp(log_amp)
      lengthscale = exp(log_ell)
      noise_std = exp(log_noise)
      period = exp(log_period)
    """

    log_amp: Array  # scalar
    log_ell: Array  # scalar
    log_noise: Array  # scalar
    log_period: Array  # scalar (for periodic kernel)

    # Make this dataclass a JAX PyTree so Optax/JAX can traverse it
    def tree_flatten(self):
        children = (self.log_amp, self.log_ell, self.log_noise, self.log_period)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        log_amp, log_ell, log_noise, log_period = children
        return cls(
            log_amp=log_amp, log_ell=log_ell, log_noise=log_noise, log_period=log_period
        )


def pairwise_squared_euclidean_dist(x1: Array, x2: Array) -> Array:
    diff = x1[:, None, :] - x2[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def rbf_kernel(x1: Array, x2: Array, params: GPParams) -> Array:
    """
    RBF kernel.

    Parameters
    ----------
    x1 : Array
        Inputs with shape (N, 1).
    x2 : Array
        Inputs with shape (M, 1).
    params : GPParams
        Hyperparameters (log-space).

    Returns
    -------
    Array
        Kernel matrix with shape (N, M).
    """
    
    sqrt_dist = pairwise_squared_euclidean_dist(x1, x2)
    return (jnp.exp(params.log_amp)**2) * jnp.exp(-sqrt_dist / (2.0 * jnp.exp(params.log_ell)**2))


def matern32_kernel(x1: Array, x2: Array, params: GPParams) -> Array:
    """
    Matérn ν=3/2 kernel (closed form).

    Parameters
    ----------
    x1 : Array
        Inputs with shape (N, 1).
    x2 : Array
        Inputs with shape (M, 1).
    params : GPParams
        Hyperparameters (log-space).

    Returns
    -------
    Array
        Kernel matrix with shape (N, M).
    """

    v = 3/2
    
    sqrt_dist = pairwise_squared_euclidean_dist(x1, x2)
    z = jnp.sqrt(2*v) * jnp.sqrt(sqrt_dist) / jnp.exp(params.log_ell)
    return (jnp.exp(params.log_amp)**2) * (1.0 + z) * jnp.exp(-z)


def periodic_kernel(x1: Array, x2: Array, params: GPParams) -> Array:
    """
    Periodic (sinusoidal) kernel.

    Parameters
    ----------
    x1 : Array
        Inputs with shape (N, 1)
    x2 : Array
        Inputs with shape (M, 1).
    params : GPParams
        Hyperparameters (log-space).

    Returns
    -------
    Array
        Kernel matrix with shape (N, M).
    """

    sqrt_dist = pairwise_squared_euclidean_dist(x1, x2)
    sin_term = jnp.sin(jnp.pi * jnp.sqrt(sqrt_dist) / jnp.exp(params.log_period))
    return (jnp.exp(params.log_amp)**2) * jnp.exp(-2.0 * ((sin_term**2) / (jnp.exp(params.log_ell)**2)))
