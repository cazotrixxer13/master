import os
import jax
import matplotlib.pyplot as plt
from typing import Tuple, Union, List, Optional, Callable
from tqdm import tqdm
from jax import numpy as jnp
from jax.nn import sigmoid, log_sigmoid
from jax.random import PRNGKey, multivariate_normal
from jax.scipy.stats.multivariate_normal import logpdf as logpdf_mvnorm


def load_data(dataset_name: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if dataset_name == "moons":
        data = jnp.load("data/moons.npz")
    elif dataset_name == "linear":
        data = jnp.load("data/linear.npz")
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return jnp.array(data["X"]), jnp.array(data["y"])


def create_2d_grid(
    X: jnp.ndarray, num_grid: int = 100
) -> Tuple[jnp.ndarray, float, float, float, float]:
    """
    Create a 2D grid of points in the input space.
    Returns the grid points, the minimum and maximum values of the first and second feature, respectively.
    """
    x1_min, x1_max = jnp.min(X[:, 0]) - 1.0, jnp.max(X[:, 0]) + 1.0
    x2_min, x2_max = jnp.min(X[:, 1]) - 1.0, jnp.max(X[:, 1]) + 1.0
    xx1 = jnp.linspace(x1_min, x1_max, num_grid)
    xx2 = jnp.linspace(x2_min, x2_max, num_grid)
    xx1_mesh, xx2_mesh = jnp.meshgrid(xx1, xx2, indexing="xy")
    X_grid = jnp.stack([xx1_mesh.ravel(), xx2_mesh.ravel()], axis=1)

    return X_grid, x1_min, x1_max, x2_min, x2_max


def plot_prob_fns(
    prob_fns: Union[
        Callable[[jnp.ndarray], jnp.ndarray], List[Callable[[jnp.ndarray], jnp.ndarray]]
    ],
    X: jnp.ndarray,
    y: jnp.ndarray,
    save_path: Optional[str] = None,
    titles: Optional[List[str]] = None,
    num_grid: int = 100,
    suptitle: Optional[str] = None,
) -> None:
    """
    Plots the probability functions over a 2D grid.
    Args:
        prob_fns: A list of probability functions.
                  Each function should take X_grid (a grid of points in the input space) and return a probability for each row in X_grid.
        X: The input data (without feature transformation).
        y: The target data (binary labels).
        titles: A list of titles for the plots.
        num_grid: The number of grid points.
        suptitle: The title of the figure.
    """
    if not isinstance(prob_fns, list):
        prob_fns = [prob_fns]

    n_plots = len(prob_fns)
    n_cols = min(n_plots, 3)
    n_rows = int(jnp.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = (
        axes.reshape(-1) if n_plots > 1 else [axes]
    )  # Ensure axes is 1D array for indexing

    for idx, prob_fn in enumerate(prob_fns):
        X_grid, x1_min, x1_max, x2_min, x2_max = create_2d_grid(X, num_grid)
        probs = prob_fn(X_grid).reshape(num_grid, num_grid)

        ax = axes[idx]
        scatter1 = ax.scatter(
            X[y == -1, 0],
            X[y == -1, 1],
            c="w",
            edgecolor="black",
            label="y=-1",
            s=50,
            alpha=0.9,
            linewidth=1,
        )
        scatter2 = ax.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            c="orange",
            edgecolor="black",
            label="y=1",
            s=50,
            alpha=0.9,
            linewidth=1,
        )
        contour = ax.imshow(
            probs,
            origin="lower",
            extent=[x1_min, x1_max, x2_min, x2_max],
            aspect="auto",
            cmap="viridis",
            alpha=0.7,
            vmin=0,
            vmax=1,
        )
        fig.colorbar(contour, ax=ax)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if titles is not None:
            ax.set_title(titles[idx])

        ax.legend()

    # Hide unused subplots if any
    for i in range(len(prob_fns), len(axes)):
        axes[i].axis("off")

    plt.suptitle(suptitle)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def log_prior(
    theta: jnp.ndarray, prior_mean: jnp.ndarray, prior_cov: jnp.ndarray
) -> float:
    """
    Returns log(p(theta)) where p(theta) is the Gaussian prior with mean prior_mean and covariance prior_cov.
    """
    return logpdf_mvnorm(theta, prior_mean, prior_cov)


def log_likelihood(theta: jnp.ndarray, Phi_X: jnp.ndarray, y: jnp.ndarray) -> float:
    """
    Returns the scalar log(p(y | X, theta)). Note that the input is already phi(X) and y is in {-1, +1}.
    For numerical stability, use the log_sigmoid function.
    """
    return jnp.sum(log_sigmoid(y * jnp.dot(Phi_X, theta)))


def log_unnormalized_posterior(
    theta: jnp.ndarray,
    Phi_X: jnp.ndarray,
    y: jnp.ndarray,
    prior_mean: jnp.ndarray,
    prior_cov: jnp.ndarray,
) -> float:
    """
    Returns the log of the unnormalized posterior \\tilde{p}(theta | y, X).
    """
    return log_likelihood(theta, Phi_X, y) + log_prior(theta, prior_mean, prior_cov)


def get_likelihood_per_x_fn(
    theta: jnp.ndarray, phi_fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a function that takes a jnp.array X_grid and returns a jnp.array of probabilities
    [p(y=1 | x^{(1)}, theta), ..., p(y=1 | x^{(N)}, theta)]
    where x^{(i)} is the i-th row of X_grid.
    """
    return lambda X_grid: sigmoid(phi_fn(X_grid) @ theta)


def bayesian_logistic_regression(
    X: jnp.ndarray,
    y: jnp.ndarray,
    phi_fn: Callable[[jnp.ndarray], jnp.ndarray],
    mu_prior: jnp.ndarray,
    cov_prior: jnp.ndarray,
    prng_key: PRNGKey,
    dataset_name: str,
    num_samples_plots: int = 3,
    num_samples_mc: int = 1000,
):
    """
    Task 1 - Bayesian Logistic Regression.
    Args:
        X: The input data (without feature transformation).
        y: The target data (binary labels).
        phi_fn: The feature transformation function.
        dataset_name: The name of the dataset, must be either "linear" or "moons".
        num_samples_plots: The number of samples to plot from the prior.
        num_samples_mc: The number of samples to use for the Monte Carlo approximation.
    """
    key_prior, key_map, key_laplace, key_mc, key_brute_force_mc = jax.random.split(
        prng_key, 5
    )
    Phi_X = phi_fn(X)
    n_features = Phi_X.shape[1]

    # TODO: Sample `num_samples_plots` times out of the Gaussian prior using `multivariate_normal`.
    #       Use the PRNGKey `key_prior` for this.
    #       For each sample theta, use get_likelihood_per_x_fn to create a function that takes a jnp.array X_grid and returns a jnp.array of probabilities
    samples = multivariate_normal(key_prior, mean=mu_prior, cov=cov_prior, shape=(num_samples_plots,))
    p_y_given_x_theta = list(map(lambda theta: get_likelihood_per_x_fn(theta, phi_fn), samples))
    plot_prob_fns(
        p_y_given_x_theta,
        X,
        y,
        suptitle="Likelihood $p(y=1 \\mid x, \\theta)$ for different $\\theta$ drawn from prior",
        save_path=f"plots/{dataset_name}/likelihood_theta_from_prior.png",
    )

    neg_log_unnorm_posterior = jax.jit(
        lambda theta: -log_unnormalized_posterior(theta, Phi_X, y, mu_prior, cov_prior)
    )
    # TODO: Pick a suitable initial theta for `get_map_estimate`.
    theta_init = jnp.zeros(n_features)
    theta_map = get_map_estimate(
        neg_log_unnorm_posterior,
        theta_init=theta_init,
        save_path=f"plots/{dataset_name}/loss_grad_descent.png",
    )

    # TODO: Visualize the likelihood $p(y=1 | x, theta_map)$ using `plot_prob_fns`.
    p_y_given_x_theta_map = [get_likelihood_per_x_fn(theta_map, phi_fn)]
    plot_prob_fns(
        p_y_given_x_theta_map,
        X,
        y,
        suptitle="Likelihood $p(y=1 \\mid x, \\theta_{\\text{MAP}})$",
        save_path=f"plots/{dataset_name}/likelihood_theta_map.png",
    )

    mu_laplace, cov_laplace = laplace_approximation(theta_map, neg_log_unnorm_posterior)

    # TODO: Sample `num_samples_plots` times out of the Laplace approximation of the posterior.
    #       Use the PRNGKey `key_laplace` for this.
    #       Again, use `get_likelihood_per_x_fn` to create a function for each sample theta, that takes a jnp.array X_grid and returns a jnp.array of probabilities p(y=1 | x, theta).
    samples = multivariate_normal(key_laplace, mean=mu_laplace, cov=cov_laplace, shape=(num_samples_plots,))
    p_y_given_x_theta_laplace = list(map(lambda theta: get_likelihood_per_x_fn(theta, phi_fn), samples))
    plot_prob_fns(
        p_y_given_x_theta_laplace,
        X,
        y,
        suptitle="Likelihood $p(y=1 \\mid x, \\theta)$ for different $\\theta$ drawn from Laplace approximation",
        save_path=f"plots/{dataset_name}/likelihood_theta_from_laplace_posterior.png",
    )

    # Posterior Predictive with Monte Carlo (Laplace Approximation)

    # TODO: Sample `num_samples_mc` times out of the Laplace approximation of the posterior.
    #       Use the PRNGKey `key_mc` for this.
    #       Create a function `posterior_pred_mc` that takes a jnp.array X_grid and returns a jnp.array of probabilities, which are the estimated posterior predictive probabilities for each row in X_grid. Use `theta_samples` for your MC estimate.
    theta_samples = multivariate_normal(key_mc, mean=mu_laplace, cov=cov_laplace, shape=(num_samples_mc,))
    def posterior_pred_mc(X_grid):
        return sigmoid(phi_fn(X_grid) @ theta_samples.T).mean(axis=1)

    plot_prob_fns(
        posterior_pred_mc,
        X,
        y,
        suptitle="Monte Carlo Estimate of Posterior Predictive (with Laplace)",
        save_path=f"plots/{dataset_name}/posterior_predictive_laplace_mc.png",
    )

    # Posterior Predictive with MacKay's analytic approximation

    # TODO: Implement MacKay's analytic approximation of the posterior predictive.
    def posterior_pred_mackay(X_grid):
        phi_x = phi_fn(X_grid)
        m = phi_x @ mu_laplace
        s_2 = jnp.sum((phi_x @ cov_laplace) * phi_x, axis=1)
        return sigmoid( m / ( jnp.sqrt( 1.0 + ((jnp.pi * s_2)/8.0) ) ) )

    plot_prob_fns(
        posterior_pred_mackay,
        X,
        y,
        suptitle="MacKay's analytic approximation of Posterior Predictive",
        save_path=f"plots/{dataset_name}/posterior_predictive_mackay.png",
    )

    if n_features <= 3:
        theta_samples = brute_force_posterior_sampling(
            neg_log_unnorm_posterior,
            n_features,
            theta_map,
            cov_laplace,
            key_brute_force_mc,
            num_samples_mc,
        )

        # TODO: Create a function `posterior_pred_brute_force_mc` that takes a jnp.array X_grid and returns a jnp.array of probabilities, which are the estimated posterior predictive probabilities for each row in X_grid. Use `theta_samples` for your MC estimate.
        def posterior_pred_brute_force_mc(X_grid):
            return sigmoid(phi_fn(X_grid) @ theta_samples.T).mean(axis=1)

        plot_prob_fns(
            posterior_pred_brute_force_mc,
            X,
            y,
            suptitle='Brute Force MC Estimate of Posterior Predictive ("true" posterior)',
            save_path=f"plots/{dataset_name}/posterior_predictive_brute_force_mc.png",
        )


def brute_force_posterior_sampling(
    neg_log_posterior: Callable[[jnp.ndarray], float],
    n_features: int,
    theta_map: jnp.ndarray,
    cov_laplace: jnp.ndarray,
    key: PRNGKey,
    num_samples_mc: int = 1000,
    num_stds_per_axis: int = 5,
    num_grid_points_per_axis: int = 50,
) -> jnp.ndarray:
    """
    Brute force posterior sampling.
    """
    # Direct Monte Carlo, without Laplace Approximation

    # TODO: Create a linspace for each dimension, see assignment sheet for details.
    theta_ranges = [ 
        jnp.linspace(
            theta_map[i] - num_stds_per_axis * jnp.sqrt(cov_laplace[i, i]),
            theta_map[i] + num_stds_per_axis * jnp.sqrt(cov_laplace[i, i]),
            num_grid_points_per_axis) 
            for i in range(n_features) 
    ]
    # Create a meshgrid and stack to array of shape (n_points, d)
    mesh = jnp.meshgrid(*theta_ranges, indexing="ij")
    theta_grid = jnp.stack([m.ravel() for m in mesh], axis=-1)
    unnormalized_log_posterior_grid = -jax.vmap(neg_log_posterior)(theta_grid)
    log_posterior_grid = unnormalized_log_posterior_grid - jax.scipy.special.logsumexp(
        unnormalized_log_posterior_grid
    )

    probs = jnp.exp(log_posterior_grid)
    indices = jax.random.choice(key, len(probs), shape=(num_samples_mc,), p=probs)
    theta_samples = theta_grid[indices]

    return theta_samples


def get_map_estimate(
    neg_log_unnorm_posterior: Callable[[jnp.ndarray], float],
    theta_init: jnp.ndarray,
    num_iters: int = 2000,
    step_size: float = 5e-3,
    save_path: Optional[str] = None,
) -> jnp.ndarray:
    """
    Get the MAP estimate of the posterior using gradient descent.
    Args:
        neg_log_unnorm_posterior: The negative log of the unnormalized posterior function.
        theta_init: The initial theta value.
        num_iters: The number of iterations.
        step_size: The step size.
    Returns:
        The MAP estimate of the posterior.
    """
    theta_map = theta_init
    losses = []

    num_iters, step_size = 30, 1e-1

    for i in tqdm(range(num_iters)):
        # TODO: Implement gradient descent. 
        #       Use jax to compute the gradient and track the losses in the `losses` list.
        loss, grad = jax.value_and_grad(neg_log_unnorm_posterior)(theta_map)
        losses.append(loss)
        theta_map = theta_map - step_size * grad

    # Create a plot of the loss function
    plt.figure()
    plt.plot(losses)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

    return theta_map


def laplace_approximation(
    theta_map: jnp.ndarray, neg_log_unnorm_posterior: Callable[[jnp.ndarray], float]
) -> jnp.ndarray:
    """
    Get the Laplace approximation of the posterior.
    Args:
        theta_map: The MAP estimate of the posterior.
        neg_log_posterior: The negative log-posterior function (up to normalization).
    Returns:
        The mean and covariance of the Laplace approximation of the posterior.
    """
    # TODO: Compute the parameters of the Laplace approximation of the posterior. 
    #       You can use jax to compute the Hessian.
    mu_laplace = theta_map
    cov_laplace = jnp.linalg.inv(jax.hessian(neg_log_unnorm_posterior)(theta_map))

    return mu_laplace, cov_laplace


def classify_linear_data() -> None:
    X, y = load_data("linear")
    # TODO: Define appropriate feature transformation function phi_fn for linearly separable data.
    def phi_fn(X): return jnp.concatenate([jnp.ones((X.shape[0], 1)), X], axis=1)

    # TODO: Define \mu, \Sigma for the prior p(\theta).
    mu_prior = jnp.zeros(phi_fn(X).shape[1])
    cov_prior = jnp.eye(phi_fn(X).shape[1])

    # print(phi_fn(X).shape[1])
    # print(mu_prior, cov_prior)

    key = PRNGKey(0)  # Source of pseudo-randomness
    bayesian_logistic_regression(
        X, y, phi_fn, mu_prior, cov_prior, key, dataset_name="linear"
    )


def classify_moons_data() -> None:
    X, y = load_data("moons")
    key = PRNGKey(0)
    key_features, key = jax.random.split(key)

    n_features = 500
    lengthscale = 0.4

    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    center = 0.5 * jnp.array([xmin + xmax, ymin + ymax])
    scale = jnp.array([xmax - xmin, ymax - ymin])
    centers = (
        jax.random.uniform(key_features, shape=(n_features, X.shape[1])) - 0.5
    ) * 2 * scale + center

    def phi_fn(X):
        """
        Radial Basis Function (RBF) feature transformation.
        """
        # Compute squared Euclidean distances to each center
        dists = jnp.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        features = jnp.exp(-dists / (2 * lengthscale**2))
        # Add bias (constant one)
        design_matrix = jnp.concatenate([jnp.ones((X.shape[0], 1)), features], axis=1)
        return design_matrix

    # TODO: Define \mu, \Sigma for the prior p(\theta).
    mu_prior = jnp.zeros(phi_fn(X).shape[1])
    cov_prior = jnp.eye(phi_fn(X).shape[1])

    bayesian_logistic_regression(X, y, phi_fn, mu_prior, cov_prior, key, dataset_name="moons")


def main():
    # if not os.path.exists("plots/linear"):
    #     os.makedirs("plots/linear")
    # classify_linear_data()

    if not os.path.exists("plots/moons"):
        os.makedirs("plots/moons")
    classify_moons_data()


if __name__ == "__main__":
    main()
