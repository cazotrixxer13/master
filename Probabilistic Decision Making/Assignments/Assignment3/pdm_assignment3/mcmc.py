import os
from model import LatentVariableModel, PRIOR_MEAN, PRIOR_COV, SIGMA_X
import jax.numpy as jnp
import jax.random as random
import jax
from jax import lax
from jax.scipy.stats import multivariate_normal
from vi_numpyro import numpyro_model
from utils import load_data
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import gelman_rubin, effective_sample_size, autocorrelation
import numpy as np
from matplotlib import pyplot as plt
from gamma import estimate_gamma_with_mcmc_samples

def metropolis_hastings_random_walk(
    x_obs: jnp.ndarray,
    gen_model: LatentVariableModel,
    num_samples: int = 10_000,
    sigma_proposal: float = 0.5,
    key: jax.Array = random.PRNGKey(0),
    init: jnp.ndarray | None = None,
    burn_in: int = 1_000,
    num_chains: int = 1,
):
    """
    Metropolis–Hastings with Gaussian random-walk proposal targeting gen_model.log_joint.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observation with shape (3,).
    gen_model : LatentVariableModel
        Generative model providing log_joint(x, z).
    num_samples : int, optional
        Total MH steps (including burn-in), by default 10_000.
    sigma_proposal : float, optional
        Proposal std for isotropic Gaussian random walk, by default 0.5.
    key : jax.Array, optional
        PRNG key, by default random.PRNGKey(0).
    init : jnp.ndarray | None, optional
        Initial state z0 with shape (2,) or (num_chains, 2). Defaults to the prior mean.
    burn_in : int, optional
        Number of initial samples to discard, by default 1_000.
    num_chains : int, optional
        Number of parallel chains to run, by default 1.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - samples: (num_chains, N_kept, 2) array of post burn-in samples, where N_kept = num_samples - burn_in
        - accept_rate: (num_chains,) average acceptance rate per chain
    """
    if init is None:
        # Use prior mean as a reasonable default initializer
        init = gen_model.prior_mean
        init = jnp.broadcast_to(init, (num_chains, 2))
    
    N_kept = num_samples - burn_in

    z_list = []
    alpha_list = []

    logp_init = gen_model.log_joint(x_obs, init) 

    for index in range(num_samples):
        key, key_prop, key_u = random.split(key, 3)

        # z' = z + sigma_proposal * epsilon, epsilon ~ N(0, I)
        epsilon = random.normal(key_prop, shape=init.shape)
        next = init + sigma_proposal * epsilon

        # log_alpha = log p(x, z') - log p(x, z)
        log_alpha = gen_model.log_joint(x_obs, next) - logp_init

        # α = min(1, exp(log_alpha))
        alpha = jnp.minimum(1.0, jnp.exp(log_alpha))

        u = random.uniform(key_u, shape=alpha.shape)
        accept = u <= alpha

        init = jnp.where(accept[:, None], next, init)
        logp_init = jnp.where(accept, gen_model.log_joint(x_obs, next), logp_init)

        z_list.append(init)
        alpha_list.append(alpha)

    samples, accept_rate = jnp.swapaxes(jnp.stack(z_list, axis=0)[-N_kept:, :, :], 0, 1), jnp.stack(alpha_list, axis=0)[-N_kept:, :].mean(axis=0)
    return samples, accept_rate


def mala(
    x_obs: jnp.ndarray,
    gen_model: LatentVariableModel,
    num_samples: int = 10_000,
    step_size: float = 0.1,
    key: jax.Array = random.PRNGKey(0),
    init: jnp.ndarray | None = None,
    burn_in: int = 1_000,
    num_chains: int = 1,
):
    """
    Metropolis-Adjusted Langevin Algorithm (MALA) with isotropic Gaussian proposal.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observation with shape (3,).
    gen_model : LatentVariableModel
        Generative model providing log_joint(x, z).
    num_samples : int, optional
        Total MALA steps (including burn-in), by default 10_000.
    step_size : float, optional
        Proposal variance (eta) for MALA; covariance is eta * I, by default 0.1.
    key : jax.Array, optional
        PRNG key, by default random.PRNGKey(0).
    init : jnp.ndarray | None, optional
        Initial state z0 with shape (2,) or (num_chains, 2). Defaults to the prior mean.
    burn_in : int, optional
        Number of initial samples to discard, by default 1_000.
    num_chains : int, optional
        Number of parallel chains to run, by default 1.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - samples: (num_chains, N_kept, 2) array of post burn-in samples, where N_kept = num_samples - burn_in
        - accept_rate: (num_chains,) average acceptance rate per chain
    """
    if init is None:
        init = gen_model.prior_mean
        init = jnp.broadcast_to(init, (num_chains, 2))

    N_kept = num_samples - burn_in
    
    logp_init = gen_model.log_joint(x_obs, init)
    
    grad_log_joint = jax.vmap(jax.grad(lambda z: gen_model.log_joint(x_obs, z)))

    z_list = []
    alpha_list = []

    for index in range(num_samples):

        # g = ∇_z log p(x, z)
        grad_init = grad_log_joint(init)

        # mean: m_init = z + (step_size / 2) * g
        mean_init = init + 0.5 * step_size * grad_init

        key, key_eps, key_u = random.split(key, 3)
        epsilon = random.normal(key_eps, shape=init.shape)

        # z' = m_init + sqrt(step_size) * epsilon
        z_dash = mean_init + jnp.sqrt(step_size) * epsilon

        # Log p(x, z')
        logp_given_z_dash = gen_model.log_joint(x_obs, z_dash)

        # g' = ∇_z log p(x, z')
        grad_logp_given_z_dash = grad_log_joint(z_dash)

        # log q(z' | z): Gaussian with mean_init and covariance step_size * I
        diff_forward = z_dash - mean_init
        sq_norm_forward = jnp.sum(diff_forward**2, axis=-1)
        logq_forward = -0.5 * (
            2 * jnp.log(2.0 * jnp.pi * step_size) + sq_norm_forward / step_size
        )

        # log q(z | z'): mean for reverse proposal uses z_prop and grad_prop
        mean_prop = z_dash + 0.5 * step_size * grad_logp_given_z_dash
        diff_reverse = init - mean_prop
        sq_norm_reverse = jnp.sum(diff_reverse**2, axis=-1)
        logq_backwards = -0.5 * ( 2 * jnp.log(2.0 * jnp.pi * step_size) + sq_norm_reverse / step_size )

        # log r = [log p(x,z') + log q(z | z')] - [log p(x,z) + log q(z' | z)]
        log_r = (logp_given_z_dash + logq_backwards) - (logp_init + logq_forward)

        # α = min(1, exp(log_r))
        alpha = jnp.minimum(1.0, jnp.exp(log_r))

        u = random.uniform(key_u, shape=alpha.shape)
        accept = u <= alpha 

        init = jnp.where(accept[:, None], z_dash, init)
        logp_init = jnp.where(accept, gen_model.log_joint(x_obs, z_dash), logp_init)

        z_list.append(init)
        alpha_list.append(alpha)

    samples, accept_rate = jnp.swapaxes(jnp.stack(z_list, axis=0)[-N_kept:, :, :], 0, 1), jnp.stack(alpha_list, axis=0)[-N_kept:, :].mean(axis=0)
    return samples, accept_rate


def nuts(
    x_obs: jnp.ndarray,
    num_samples: int = 2_000,
    num_warmup: int = 1_000,
    num_chains: int = 1,
    key: jax.Array = random.PRNGKey(0),
):
    """
    Run NUTS using NumPyro for the model `numpyro_model`.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observation with shape (3,).
    num_samples : int, optional
        Number of post-warmup samples per chain, by default 2_000.
    num_warmup : int, optional
        Number of warmup (adaptation) steps per chain, by default 1_000.
    num_chains : int, optional
        Number of chains to run in parallel, by default 1.
    key : jax.Array, optional
        PRNG key, by default random.PRNGKey(0).

    Returns
    -------
    jnp.ndarray
        Samples of z with shape (num_chains, num_samples, 2).
    """
    kernel = NUTS(numpyro_model)
    # pass appropriate arguments to MCMC()
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(key, x_obs=x_obs)
    samples_dict = mcmc.get_samples(group_by_chain=True)
    samples = samples_dict["z"]
    return samples


def compute_diagnostics(samples: jnp.ndarray):
    """
    Compute R-hat and Effective Sample Size (ESS) per dimension.

    Parameters
    ----------
    samples : jnp.ndarray
        Samples with shape (num_chains, num_samples, 2) or (num_samples, 2).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        - rhat: (2,) array of R-hat values
        - ess: (2,) array of effective sample sizes
    """
    samples_chains = samples[None, ...] if samples.ndim == 2 else samples # (num_chains, num_samples, dim)

    rhat = gelman_rubin(samples_chains)
    ess = effective_sample_size(samples_chains)
    return rhat, ess


def plot_acfs(
    samples_list,
    max_lag: int,
    save_path: str,
    titles,
):
    """
    Plot ACFs for multiple sample arrays side-by-side.

    Parameters
    ----------
    samples_list : list
        Iterable of arrays with shape (draws, 2) or (chains, draws, 2).
    max_lag : int
        Maximum lag to compute the ACF.
    save_path : str
        Output path where the figure will be saved.
    titles : list
        Iterable of strings (same length as samples_list) to title each subplot.
    """
    samples_list = list(samples_list)
    titles = list(titles)
    assert len(samples_list) == len(
        titles
    ), "titles must match the number of sample arrays"

    def to_series(samples: np.ndarray) -> np.ndarray:
        arr = np.asarray(samples)
        if arr.ndim == 3:
            c, n, d = arr.shape
            return arr.reshape(c * n, d)
        if arr.ndim == 2:
            return arr
        raise ValueError(
            "Each samples array must be (draws, dim) or (chains, draws, dim)"
        )

    def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
        x = np.asarray(x) - np.mean(x)
        var = np.dot(x, x)
        acf_vals = np.empty(max_lag + 1, dtype=float)
        for lag in range(max_lag + 1):
            if lag == 0:
                acf_vals[lag] = 1.0
            else:
                acf_vals[lag] = np.dot(x[:-lag], x[lag:]) / var
        return acf_vals

    k = len(samples_list)
    lags = np.arange(max_lag + 1)
    fig, axs = plt.subplots(1, k, figsize=(6 * k, 4), squeeze=False)
    fig.set_facecolor("white")
    for i, (samples, title) in enumerate(zip(samples_list, titles)):
        ax = axs[0, i]
        series = to_series(samples)
        num_dims = series.shape[-1]
        for z_idx in range(num_dims):
            acf_vals = acf_1d(series[:, z_idx], max_lag)
            ax.plot(lags, acf_vals, label=f"z{z_idx+1}")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(title)
        ax.legend()

    plt.suptitle("Autocorrelation Function (ACF)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_mcmc(
    x_obs: jnp.ndarray,
    method: str,
    num_samples: int = 100_000,
    burn_in: int = 1_000,
    num_chains: int = 4,
):
    """
    Run a specified MCMC method and return samples and diagnostics.

    Parameters
    ----------
    x_obs : jnp.ndarray
        Observation with shape (3,).
    method : str
        One of {"mh", "mala", "nuts"}.
    num_samples : int, optional
        Number of samples (or post-warmup samples for NUTS), by default 100_000.
    burn_in : int, optional
        Burn-in length for MH/MALA, by default 1_000.
    num_chains : int, optional
        Number of chains, by default 4.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        - samples: array of samples with chain dimension
        - rhat: (2,) R-hat values
        - ess: (2,) effective sample sizes
    """
    num_iters = num_samples + burn_in
    gen_model = LatentVariableModel(
        prior_mean=PRIOR_MEAN, prior_cov=PRIOR_COV, sigma_x=SIGMA_X
    )

    if method == "mh":
        # Call metropolis_hastings_random_walk with appropriate arguments and assign to samples and accept_rate
        samples, accept_rate = metropolis_hastings_random_walk(x_obs, gen_model, num_samples=num_iters, burn_in=burn_in, num_chains=num_chains, sigma_proposal=5.0)
    elif method == "mala":
        # Call mala with appropriate arguments and assign to samples and accept_rate
        samples, accept_rate = mala(x_obs, gen_model, num_samples=num_iters, burn_in=burn_in, num_chains=num_chains, step_size=1.0)
    elif method == "nuts":
        # Call nuts with appropriate arguments and assign to samples
        samples = nuts(x_obs, num_samples=num_samples, num_chains=num_chains)
        accept_rate = jnp.nan
    else:
        raise ValueError(f"Invalid method: {method}")

    # Diagnostics: R-hat and ESS
    rhat, ess = compute_diagnostics(samples)
    return samples, rhat, ess


def main():
    """
    Run MH, MALA, and NUTS across a subset of test observations and
    generate ACF plots and posterior overlays.
    """
    os.makedirs("figures/mcmc_custom_sig_5_step_1", exist_ok=True)

    gen_model = LatentVariableModel(
        prior_mean=PRIOR_MEAN, prior_cov=PRIOR_COV, sigma_x=SIGMA_X
    )

    x_train, x_test = load_data()

    print("Running MCMC methods...")

    num_samples = 10_000
    for idx, x_obs in enumerate(x_test[:5]):
        print("Metropolis-Hastings...")
        samples_mh, rhat_mh, ess_mh = run_mcmc(x_obs, "mh", num_samples=num_samples)

        print("MALA...")
        samples_mala, rhat_mala, ess_mala = run_mcmc(
            x_obs, "mala", num_samples=num_samples
        )

        print("NUTS...")
        samples_nuts, rhat_nuts, ess_nuts = run_mcmc(
            x_obs, "nuts", num_samples=num_samples
        )

        def format_title(method, rhat, ess):
            rhat_str = ", ".join([f"{v:.2f}" for v in rhat])
            ess_str = ", ".join([f"{v:.0f}" for v in ess])
            return f"ACF ({method.upper()}), $\\hat{{R}}$ = [{rhat_str}], ESS = [{ess_str}]"

        # Plot ACFs
        plot_acfs(
            [samples_mh, samples_mala, samples_nuts],
            max_lag=200,
            save_path=f"figures/mcmc_custom_sig_5_step_1/acf_x{idx}.png",
            titles=[
                format_title("mh", rhat_mh, ess_mh),
                format_title("mala", rhat_mala, ess_mala),
                format_title("nuts", rhat_nuts, ess_nuts),
            ],
        )

        # Plot posterior with samples
        gen_model.plot_posterior(
            x_obs,
            samples=[
                (
                    samples_mh.reshape(-1, samples_mh.shape[-1])
                    if samples_mh.ndim == 3
                    else samples_mh
                ),
                (
                    samples_mala.reshape(-1, samples_mala.shape[-1])
                    if samples_mala.ndim == 3
                    else samples_mala
                ),
                (
                    samples_nuts.reshape(-1, samples_nuts.shape[-1])
                    if samples_nuts.ndim == 3
                    else samples_nuts
                ),
            ],
            titles=["MH", "MALA", "NUTS"],
            save_path=f"figures/mcmc_custom_sig_5_step_1/posterior_x{idx}.png",
        )

        # Estimate gamma_p using the three MCMC methods and add to your table
        gamma_p_mh = estimate_gamma_with_mcmc_samples(samples_mh)
        gamma_p_mala = estimate_gamma_with_mcmc_samples(samples_mala)
        gamma_p_nuts = estimate_gamma_with_mcmc_samples(samples_nuts)

        print(f"Results for x_test[{idx}]:")
        print("Gamma_P MH:", gamma_p_mh)
        print("Gamma_P MALA:", gamma_p_mala)
        print("Gamma_P NUTS:", gamma_p_nuts)

if __name__ == "__main__":
    main()
