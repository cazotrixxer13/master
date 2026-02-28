from dataclasses import dataclass
from typing import List, Optional
from gp_kernels import GPParams
import jax.numpy as jnp
from matplotlib import pyplot as plt


Array = jnp.ndarray


@dataclass
class KernelPlotData:
    name: str
    prior_draws: Array  # (S, N)
    mean_init: Array  # (M,)
    cov_init: Array  # (M,M)
    lml_init: Array  # scalar
    lml_optimized: Array  # scalar
    ml2_losses: Array  # (T,)
    mean_optimized: Array  # (M,)
    cov_optimized: Array  # (M,M)
    hyperparams_init: Optional[GPParams] = None
    hyperparams_optimized: Optional[GPParams] = None


@dataclass
class DatasetPlotData:
    dataset_name: str
    x_train: Array  # (N,1)
    y_train: Array  # (N,)
    x_test: Array  # (M,1)
    kernels: List[KernelPlotData]
    save_path: str


@dataclass
class FullyBayesianKernelDraws:
    name: str
    prior_draws: Array  # (S0, M)
    draws: Array  # (S, M)
    mean_curve: Array  # (M,)


@dataclass
class FullyBayesianDatasetPlotData:
    dataset_name: str
    x_train: Array  # (N,1)
    y_train: Array  # (N,)
    x_test: Array  # (M,1)
    kernels: List[FullyBayesianKernelDraws]
    save_path: str
    alpha_draws: float = 0.15


def plot_gp_fit_ax(
    ax,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    mean: Array,
    cov: Array,
    title: str,
    ylabel: str,
):
    std = jnp.sqrt(jnp.clip(jnp.diag(cov), a_min=0.0))
    ax.scatter(x_train[:, 0], y_train, s=20, label="Data")
    ax.plot(x_test[:, 0], mean, label="Posterior Mean")
    ax.fill_between(
        x_test[:, 0],
        mean - 1.96 * std,
        mean + 1.96 * std,
        alpha=0.2,
        label="95% Band",
    )
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(ylabel)
    ax.set_ylim([-3.5, 3.5])
    ax.legend(loc="best", fontsize=8)


def plot_fb_draws_ax(
    ax,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    draws: Array,  # (S, M)
    mean_curve: Array,  # (M,)
    title: str,
    ylabel: str,
    alpha_draws: float = 0.15,
):
    ax.scatter(x_train[:, 0], y_train, s=20, label="Data")

    # Individual function draws with small alpha
    for i in range(draws.shape[0]):
        ax.plot(x_test[:, 0], draws[i], color="C1", alpha=alpha_draws, linewidth=1.0)

    # Average curve with alpha=1
    ax.plot(
        x_test[:, 0], mean_curve, color="C0", alpha=1.0, linewidth=1.5, label="Average"
    )
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(ylabel)
    ax.set_ylim([-3.5, 3.5])
    ax.legend(loc="best", fontsize=8)


def build_kernel_title(name: str, params: Optional[GPParams]) -> str:
    if params is None:
        return name

    alpha = float(jnp.exp(params.log_amp))
    ell = float(jnp.exp(params.log_ell))
    sigma_n = float(jnp.exp(params.log_noise))
    parts = [
        rf"$\alpha={alpha:.2f}$",
        rf"$\ell={ell:.2f}$",
        rf"$\sigma_n={sigma_n:.2f}$",
    ]

    if "period" in name.lower():
        period = float(jnp.exp(params.log_period))
        parts.append(rf"$p={period:.2f}$")

    return ", ".join(parts)


def render_dataset_summary(plot_data: DatasetPlotData) -> None:
    # rows: [prior_draws, ml2_loss, posterior_initial, posterior_after_ml2]
    # cols : one per kernel (assumes exactly 3)
    fig, axes = plt.subplots(4, len(plot_data.kernels), figsize=(12, 12), dpi=120)
    # Share y-axis within each non-loss row
    for r in (0, 1, 2):
        axes[r, 1].sharey(axes[r, 0])
        axes[r, 2].sharey(axes[r, 0])

    # Row 0: prior draws
    for ki, kdata in enumerate(plot_data.kernels):
        if kdata.prior_draws is not None:
            ax = axes[0, ki]
            for i in range(kdata.prior_draws.shape[0]):
                ax.plot(plot_data.x_test[:, 0], kdata.prior_draws[i])
            ax.set_title(kdata.name, fontsize=16, pad=15)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(
                r"Draws from GP Prior $p(f_* \mid x_*, \lambda_{\text{init}})$"
            )

    for ki, kdata in enumerate(plot_data.kernels):
        # Initial posterior (row 1)
        if kdata.mean_init is not None and kdata.cov_init is not None:
            plot_gp_fit_ax(
                axes[1, ki],
                plot_data.x_train,
                plot_data.y_train,
                plot_data.x_test,
                kdata.mean_init,
                kdata.cov_init,
                title=build_kernel_title(kdata.name, kdata.hyperparams_init),
                ylabel=r"Initial Posterior $p(f_* \mid x_*, \mathcal{D}, \lambda_{\text{init}})$",
            )

        # Posterior after ML-II (row 2)
        if kdata.mean_optimized is not None and kdata.cov_optimized is not None:
            plot_gp_fit_ax(
                axes[2, ki],
                plot_data.x_train,
                plot_data.y_train,
                plot_data.x_test,
                kdata.mean_optimized,
                kdata.cov_optimized,
                title=build_kernel_title(kdata.name, kdata.hyperparams_optimized),
                ylabel=r"Posterior after ML-II $p(f_* \mid x_*, \mathcal{D}, \hat{\lambda}_{\text{ML2}})$",
            )

        if kdata.ml2_losses is not None:
            # ML-II loss (row 3)
            ax_l = axes[3, ki]
            # plot steps in logspace on the x-axis
            steps = jnp.arange(1, len(kdata.ml2_losses) + 1)
            ax_l.semilogx(steps, -kdata.ml2_losses)
            ax_l.set_title(
                f"Init LML: {float(kdata.lml_init):.2f}, Final LML: {float(kdata.lml_optimized):.2f}"
            )
            ax_l.set_xlabel("Step (log scale)")
            ax_l.set_ylabel(r"$\log p(y \mid x, \lambda)$")

    fig.tight_layout()
    fig.savefig(plot_data.save_path)
    plt.close(fig)


def render_fully_bayesian_draws(plot_data: FullyBayesianDatasetPlotData) -> None:
    # 2 rows (top: GP prior draws, bottom: posterior predictive draws), len(kernels) columns
    n_kernels = len(plot_data.kernels)
    fig, axes = plt.subplots(2, n_kernels, figsize=(4 * n_kernels, 6.5), dpi=120)
    # Share y-axis within each row
    if n_kernels > 1:
        for c in range(1, n_kernels):
            axes[0, c].sharey(axes[0, 0])
            axes[1, c].sharey(axes[1, 0])

    # Row 0: prior draws (match ML-II styling)
    for ki, kdraws in enumerate(plot_data.kernels):
        ax0 = axes[0, ki] if n_kernels > 1 else axes[0]
        for i in range(kdraws.prior_draws.shape[0]):
            ax0.plot(plot_data.x_test[:, 0], kdraws.prior_draws[i])
        ax0.set_title(kdraws.name, fontsize=16, pad=15)
        ax0.set_xlabel(r"$x$")
        ax0.set_ylabel(r"Draws from GP Prior $p(f_* \mid x_*)$")

    # Row 1: posterior predictive draws + average
    for ki, kdraws in enumerate(plot_data.kernels):
        ax1 = axes[1, ki] if n_kernels > 1 else axes[1]
        plot_fb_draws_ax(
            ax1,
            plot_data.x_train,
            plot_data.y_train,
            plot_data.x_test,
            kdraws.draws,
            kdraws.mean_curve,
            title=kdraws.name,
            ylabel=r"Posterior Draws $p(f_* \mid x_*, \mathcal{D})$",
            alpha_draws=plot_data.alpha_draws,
        )
    fig.tight_layout()
    fig.savefig(plot_data.save_path)
    plt.close(fig)
