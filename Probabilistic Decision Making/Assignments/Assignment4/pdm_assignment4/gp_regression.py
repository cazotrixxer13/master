import numpy as np
import functools
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from matplotlib import pyplot as plt
from pathlib import Path
from gp_kernels import (
    GPParams,
    rbf_kernel,
    matern32_kernel,
    periodic_kernel,
)
from plotting_utils import (
    KernelPlotData,
    DatasetPlotData,
    render_dataset_summary,
    FullyBayesianKernelDraws,
    FullyBayesianDatasetPlotData,
    render_fully_bayesian_draws,
)
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

Array = jnp.ndarray


def gp_prior_sample(
    key: jax.Array,
    x: Array,
    kernel_fn: Callable[[Array, Array, GPParams], Array],
    params: GPParams,
    jitter: float = 1e-4,
    num_functions: int = 3,
) -> Array:
    """
    Draw function values f(x) ~ N(0, k(x,x)).

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    x : Array
        Inputs with shape (N, 1).
    kernel_fn : callable
        Kernel function returning k(x1,x2,params).
    params : GPParams
        Hyperparameters.
    jitter : float, optional
        Diagonal jitter for numerical stability, by default 1e-4.
    num_functions : int, optional
        Number of independent function draws, by default 3.

    Returns
    -------
    Array
        Samples with shape (num_functions, N).
    """
    N = x.shape[0]

    K = kernel_fn(x, x, params)
    K = K + jitter * jnp.eye(N)

    mean = jnp.zeros((N,))
    samples = random.multivariate_normal(key, mean=mean, cov=K, shape=(num_functions,))

    return samples


def gp_posterior(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    kernel_fn: Callable[[Array, Array, GPParams], Array],
    params: GPParams,
    jitter: float = 1e-4,
) -> Tuple[Array, Array]:
    """
    Compute GP posterior p(f_* | x_*, x, y, lambda) for standard GP regression:
      y = f(x) + ε, ε ~ N(0, σ_n^2 I)

    Parameters
    ----------
    x_train : Array
        Train inputs with shape (N, 1).
    y_train : Array
        Train targets with shape (N,).
    x_test : Array
        Test inputs with shape (M, 1).
    kernel_fn : callable
        Kernel function.
    params : GPParams
        Hyperparameters.
    jitter : float, optional
        Small diagonal jitter added to training covariance, by default 1e-4.

    Returns
    -------
    tuple[Array, Array]
        mean : (M,) posterior mean
        cov  : (M,M) posterior covariance
    """
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train).reshape(-1)
    x_test = jnp.asarray(x_test)
    N = x_train.shape[0]

    K_tt = kernel_fn(x_train, x_train, params)
    A = K_tt + (jnp.exp(params.log_noise)**2 + jitter) * jnp.eye(N)

    K_star_t = kernel_fn(x_test, x_train, params)
    K_star_star = kernel_fn(x_test, x_test, params)

    mean = K_star_t @ jnp.linalg.solve(A, y_train)
    cov = K_star_star - K_star_t @ jnp.linalg.solve(A, K_star_t.T)

    return mean, cov


def gp_log_marginal_likelihood(
    x_train: Array,
    y_train: Array,
    kernel_fn: Callable[[Array, Array, GPParams], Array],
    params: GPParams,
    jitter: float = 1e-4,
) -> Array:
    """
    Compute log p(y | x, lambda) for GP regression.

    Parameters
    ----------
    x_train : Array
        Train inputs with shape (N, D).
    y_train : Array
        Train targets with shape (N,).
    kernel_fn : callable
        Kernel function.
    params : GPParams
        Hyperparameters.
    jitter : float, optional
        Jitter added to covariance diagonal, by default 1e-4.

    Returns
    -------
    Array
        Scalar log marginal likelihood.
    """
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train).reshape(-1)
    N = x_train.shape[0]

    K_tt = kernel_fn(x_train, x_train, params)
    Sigma = K_tt + (jnp.exp(params.log_noise)**2 + jitter) * jnp.eye(N)

    quad = y_train @ jnp.linalg.solve(Sigma, y_train)
    sign, logdet = jnp.linalg.slogdet(Sigma)

    return -0.5 * (quad + logdet + N * jnp.log(2.0 * jnp.pi))


def fit_hyperparams_ml2(
    init_params: GPParams,
    x_train: Array,
    y_train: Array,
    kernel_fn: Callable[[Array, Array, GPParams], Array],
    num_steps: int = 500,
    lr: float = 5e-2,
) -> Tuple[GPParams, Array]:
    """
    Fit GP hyperparameters by ML-II (type-II maximum likelihood).

    Parameters
    ----------
    init_params : GPParams
        Initial hyperparameters.
    x_train : Array
        Train inputs, shape (N, 1).
    y_train : Array
        Targets, shape (N,).
    kernel_fn : callable
        Kernel function.
    num_steps : int, optional
        Optimization steps, by default 500.
    lr : float, optional
        Learning rate, by default 5e-2.

    Returns
    -------
    tuple[GPParams, Array]
        params : fitted hyperparameters
        losses : (num_steps,) negative log marginal likelihood
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_params)

    @functools.partial(jax.jit, static_argnames=("kernel_fn",))
    def step_ml2(
        params: GPParams,
        opt_state,
        x_train: Array,
        y_train: Array,
        kernel_fn: Callable[[Array, Array, GPParams], Array],
    ) -> Tuple[GPParams, optax.OptState, Array]:
        def loss_fn(p: GPParams) -> Array:
            return -gp_log_marginal_likelihood(x_train, y_train, kernel_fn, p)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Run optimization loop
    params = init_params
    losses = []
    for _ in range(num_steps):
        params, opt_state, loss = step_ml2(
            params, opt_state, x_train, y_train, kernel_fn
        )
        losses.append(loss)
    return params, jnp.array(losses)


def load_dataset(name: str) -> Tuple[Array, Array]:
    """
    Load a toy regression dataset by name.

    Parameters
    ----------
    name : str
        Name of dataset to load. Options: {"smooth", "rough", "periodic"}

    Returns
    -------
    tuple[Array, Array, Array]
        x_train : (N, 1) array of input features
        y_train : (N,) array of targets
        x_test  : (M, 1) evenly spaced test inputs for plotting
    """
    if name == "smooth":
        path = "data/smooth.npz"
    elif name == "rough":
        path = "data/rough.npz"
    elif name == "periodic":
        path = "data/periodic.npz"
    else:
        raise ValueError(f"Invalid dataset name: {name}")

    data = np.load(path)
    x_min, x_max = data["x"].min(), data["x"].max()
    x_test = jnp.linspace(
        x_min - 0.25 * (x_max - x_min), x_max + 0.25 * (x_max - x_min), 250
    )[:, None]
    return jnp.asarray(data["x"]), jnp.asarray(data["y"]), x_test


def gp_regression_ml2(kernels, key: jax.Array, fig_root: Path):
    """
    Run Type-II maximum likelihood (ML-II) GP regression for each dataset and kernel.
    For each dataset in {"smooth","rough","periodic"}:
      - draw prior functions at test inputs
      - compute initial posterior with shared initial hyperparameters
      - optimize hyperparameters via ML-II
      - compute posterior after ML-II
      - render a multi-row summary figure saved to fig_root/dataset_name_ml2.pdf

    Parameters
    ----------
    kernels : list[tuple[str, callable]]
        Pairs of (kernel_name, kernel_fn(x1, x2, params) -> (N,M) kernel matrix).
    key : jax.Array
        PRNG key used for prior draws.
    fig_root : pathlib.Path
        Output directory for saving figures.
    """
    for dataset_name in ["smooth", "rough", "periodic"]:
        x_train, y_train, x_test = load_dataset(dataset_name)

        # Init GP hyperparameters
        hyperparams_init = GPParams(
            log_amp=jnp.log(jnp.array(1.0)),
            log_ell=jnp.log(jnp.array(1.0)),
            log_noise=jnp.log(jnp.array(0.1)),
            log_period=jnp.log(jnp.array(1.0)),
        )

        kernel_plot_rows: list[KernelPlotData] = []
        for ki, (kernel_name, kernel_fn) in enumerate(kernels):
            # Prior draws
            key, sub = random.split(key)
            prior_draws = gp_prior_sample(sub, x_test, kernel_fn, hyperparams_init, jitter=1e-4, num_functions=5)

            # Initial posterior (before type-II maximum likelihood)
            mean_init, cov_init = gp_posterior(x_train, y_train, x_test, kernel_fn, hyperparams_init, jitter=1e-4)

            # Fit hyperparams with ML-II
            hyperparams_optimized, ml2_losses = fit_hyperparams_ml2(hyperparams_init, x_train, y_train, kernel_fn, num_steps=500, lr=5e-2)

            # Posterior after ML-II
            mean_optimized, cov_optimized = gp_posterior(x_train, y_train, x_test, kernel_fn, hyperparams_optimized, jitter=1e-4)

            lml_init = gp_log_marginal_likelihood(x_train, y_train, kernel_fn, hyperparams_init, jitter=1e-4)
            lml_optimized = gp_log_marginal_likelihood(x_train, y_train, kernel_fn, hyperparams_optimized, jitter=1e-4)

            kernel_plot_rows.append(
                KernelPlotData(
                    name=f"{kernel_name}",
                    prior_draws=prior_draws,
                    mean_init=mean_init,
                    cov_init=cov_init,
                    lml_init=lml_init,
                    lml_optimized=lml_optimized,
                    ml2_losses=ml2_losses,
                    mean_optimized=mean_optimized,
                    cov_optimized=cov_optimized,
                    hyperparams_init=hyperparams_init,
                    hyperparams_optimized=hyperparams_optimized,
                )
            )

        # Assemble plot data and render once
        dataset_plot = DatasetPlotData(
            dataset_name=dataset_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            kernels=kernel_plot_rows,
            save_path=str(fig_root / f"{dataset_name}_ml2.pdf"),
        )
        render_dataset_summary(dataset_plot)


def numpyro_lambda_model(
    x_train=None, y_train=None, kernel_fn=None, kernel_name=None, use_likelihood=True
):
    """
    NumPyro model defining a prior over GP hyperparameters and (optionally)
    a likelihood contribution via the GP log marginal likelihood.

    When use_likelihood=True, the model adds a factor equal to
      log p(y | x, lambda) = GP log marginal likelihood
    so that NUTS targets p(lambda | D) ∝ p(y | x, lambda) p(lambda).

    Parameters
    ----------
    x_train : Array, optional
        Training inputs (N,1). Required when use_likelihood=True.
    y_train : Array, optional
        Training targets (N,). Required when use_likelihood=True.
    kernel_fn : callable, optional
        Kernel function k(x1, x2, params). Required when use_likelihood=True.
    kernel_name : str, optional
        Kernel name; used to decide whether to sample log_period.
    use_likelihood : bool, optional
        If True, add GP log marginal likelihood as a factor.
    """
    # Specify priors for these values directly in log-space. 
    # Use numpyro.sample with numpyro distributions (imported as `dist`, e.g. `dist.Normal`)
    log_amp = numpyro.sample("log_amp", dist.Normal(0.0, 1.0))
    log_ell = numpyro.sample("log_ell", dist.Normal(0.0, 1.0))
    log_noise = numpyro.sample("log_noise", dist.Normal(-1.0, 0.5))
    log_period = numpyro.sample("log_period", dist.Normal(0.0, 1.0))

    if use_likelihood:
        params = GPParams(
            log_amp=log_amp,
            log_ell=log_ell,
            log_noise=log_noise,
            log_period=log_period,
        )
        # Multiply log marginal likelihood
        lml = gp_log_marginal_likelihood(x_train, y_train, kernel_fn, params)
        numpyro.factor("gp_log_marginal_likelihood", lml)


def sample_prior_lambda(
    key: jax.Array, kernel_name: str, num_draws: int
) -> Dict[str, Array]:
    """
    Draw samples of GP hyperparameters from the prior p(lambda) defined
    in numpyro_lambda_model (without likelihood).

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    kernel_name : str
        Kernel name; controls whether period is included.
    num_draws : int
        Number of prior draws of lambda.

    Returns
    -------
    dict[str, Array]
        Dictionary of arrays keyed by hyperparameter names
        (e.g., "log_amp", "log_ell", "log_noise", optionally "log_period"),
        each of shape (num_draws,).
    """
    hyperparameter_draws = Predictive(
        lambda: numpyro_lambda_model(kernel_name=kernel_name, use_likelihood=False),
        num_samples=num_draws,
    )(key)
    return hyperparameter_draws


def gp_regression_fully_bayesian(kernels, key: jax.Array, fig_root: Path):
    """
    Fully Bayesian GP regression over hyperparameters using NUTS (NumPyro).
    For each dataset and kernel:
      - sample from hyperparameter-marginalized GP prior and plot
      - run NUTS to sample p(lambda | D) ∝ p(y | x, lambda) p(lambda)
      - for each hyperparameter sample, draw a function from p(f_* | x_*, D, lambda)
      - render a 2-row figure with prior draws (top) and posterior draws + mean (bottom)
        saved to fig_root/dataset_name_fully_bayesian.pdf

    Parameters
    ----------
    kernels : list[tuple[str, callable]]
        Pairs of (kernel_name, kernel_fn).
    key : jax.Array
        PRNG key used for prior/mcmc and posterior draws.
    fig_root : pathlib.Path
        Output directory for saving figures.
    """
    for dataset_name in ["smooth", "rough", "periodic"]:
        x_train, y_train, x_test = load_dataset(dataset_name)

        fb_kernel_draws = []
        for kernel_name, kernel_fn in kernels:
            # Draw 5 times from hyperparameter-marginalized prior
            prior_lambda = sample_prior_lambda(key, kernel_name=kernel_name, num_draws=5)

            prior_draw_list = []
            for i in range(5):
                params_i = GPParams(
                    log_amp=prior_lambda["log_amp"][i],
                    log_ell=prior_lambda["log_ell"][i],
                    log_noise=prior_lambda["log_noise"][i],
                    log_period=prior_lambda["log_period"][i]
                )
                prior_draw_list.append(gp_prior_sample(key, x_test, kernel_fn, params_i, jitter=1e-4, num_functions=1)[0])
            
            gp_prior_draws = jnp.stack(prior_draw_list, axis=0)

            # Use NUTS to draw samples from hyperparam posterior p(lambda | D)
            num_warmup = 500
            nuts = NUTS(numpyro_lambda_model)
            mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=100, num_chains=1)
            mcmc.run(
                key,
                kernel_fn=kernel_fn,
                kernel_name=kernel_name,
                x_train=x_train,
                y_train=y_train,
            )
            print(
                f"[{dataset_name}] {kernel_name} — NUTS posterior over hyperparameters"
            )
            mcmc.print_summary()

            # Build posterior predictive draws f_* using sampled hyperparameters
            samples = mcmc.get_samples()
            posterior_draw_list = []
            for i in range(samples["log_amp"].shape[0]):
                params_i = GPParams(
                    log_amp=samples["log_amp"][i],
                    log_ell=samples["log_ell"][i],
                    log_noise=samples["log_noise"][i],
                    log_period=samples["log_period"][i]
                )

                mean_i, cov_i = gp_posterior(x_train, y_train, x_test, kernel_fn, params_i, jitter=1e-4)
                posterior_draw_list.append(random.multivariate_normal(key, mean=mean_i, cov=cov_i))
            draws_arr = jnp.stack(posterior_draw_list, axis=0)

            mean_curve = jnp.mean(draws_arr, axis=0)
            fb_kernel_draws.append(
                FullyBayesianKernelDraws(
                    name=kernel_name,
                    prior_draws=gp_prior_draws,
                    draws=draws_arr,
                    mean_curve=mean_curve,
                )
            )

        # Render and save plot with draws per kernel
        fb_plot = FullyBayesianDatasetPlotData(
            dataset_name=dataset_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            kernels=fb_kernel_draws,
            save_path=str(fig_root / f"{dataset_name}_fully_bayesian.png"),
            alpha_draws=0.15,
        )
        render_fully_bayesian_draws(fb_plot)


def main():
    """
    Entry point for Task 1 GP experiments.
    Produces summary figures for ML-II and fully Bayesian treatments
    for all datasets and kernels under figures_task1/.
    """
    FIG_ROOT = Path("figures_task1")
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    # Kernel functions to evaluate for each dataset
    kernels = [
        ("RBF Kernel", rbf_kernel),
        ("Matern Kernel, ν=3/2", matern32_kernel),
        ("Periodic Kernel", periodic_kernel),
    ]

    # Pick an initial seed (integer)
    seed = 365365365
    key = random.PRNGKey(seed)
    gp_regression_ml2(kernels, key, FIG_ROOT)
    gp_regression_fully_bayesian(kernels, key, FIG_ROOT)


if __name__ == "__main__":
    main()
