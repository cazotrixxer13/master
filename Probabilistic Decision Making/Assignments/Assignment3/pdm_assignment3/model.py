import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

PRIOR_MEAN = jnp.array([0.0, 0.0])
PRIOR_COV = jnp.array([[0.5, 0.35], [0.35, 1.0]])
SIGMA_X = 0.7


class LatentVariableModel:
    def __init__(self, prior_mean: jnp.ndarray, prior_cov: jnp.ndarray, sigma_x: float):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.sigma_x = sigma_x

    @staticmethod
    def f(z: jnp.ndarray) -> jnp.ndarray:
        """
        Implements f(z) as in the assignment sheet.

        Parameters
        ----------
        z : jnp.ndarray
            Latent variable with shape (..., 2).

        Returns
        -------
        jnp.ndarray
            Transformed output with shape (..., 3).
        """
        z1 = z[..., 0] 
        z2 = z[..., 1]
        
        return jnp.stack([z1**2 - 4.0, z1 * z2, z2**2], axis=-1)

    def log_p_z(self, z: jnp.ndarray):
        """
        Compute log p(z) under the Gaussian prior.

        Parameters
        ----------
        z : jnp.ndarray
            Latent variable with shape (..., 2).

        Returns
        -------
        jnp.ndarray
            Log-density values with shape (...) corresponding to each z.
        """

        return multivariate_normal.logpdf(z, mean=self.prior_mean, cov=self.prior_cov)

    def log_p_x_given_z(self, x: jnp.ndarray, z: jnp.ndarray):
        """
        Compute log p(x | z) under an isotropic Gaussian likelihood.

        The likelihood is x | z ~ N(f(z), sigma_x^2 I_3).

        Parameters
        ----------
        x : jnp.ndarray
            Observation with shape (3,) or (..., 3).
        z : jnp.ndarray
            Latent variable with shape (2,) or (..., 2).

        Returns
        -------
        jnp.ndarray
            Log-likelihood values with shape (...) corresponding to each (x, z).
        """
        cov = (self.sigma_x**2) * jnp.eye(3)
        
        return multivariate_normal.logpdf(x, mean=self.f(z), cov=cov)

    def log_joint(self, x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the joint log-density log p(x, z).

        Parameters
        ----------
        x : jnp.ndarray
            Observation with shape (3,) or (..., 3).
        z : jnp.ndarray
            Latent variable with shape (2,) or (..., 2).

        Returns
        -------
        jnp.ndarray
            Log joint values with shape (...) for each pair (x, z).
        """
        return self.log_p_z(z) + self.log_p_x_given_z(x, z)

    def plot_posterior(
        self,
        x_obs,
        q_mean=None,
        q_cov=None,
        num_grid_points=1000,
        save_path=None,
        titles=None,
        samples=None,
        suptitle=None,
    ):
        """
        Plot the posterior density p(z | x_obs) and optional overlays.

        Parameters
        ----------
        x_obs : jnp.ndarray
            Observation with shape (3,).
        q_mean : jnp.ndarray | None, optional
            Mean(s) of variational posterior(s). Either shape (2,) for a single
            distribution or (K, 2) for K distributions, by default None.
        q_cov : jnp.ndarray | None, optional
            Covariance(s) of variational posterior(s). Either shape (2, 2) for a single
            distribution or (K, 2, 2) for K distributions, by default None.
        num_grid_points : int, optional
            Number of grid points per axis to approximate the posterior density,
            by default 1000.
        save_path : str | None, optional
            If provided, path to save the resulting figure, by default None (shows interactively).
        titles : list[str] | None, optional
            Optional list of subplot titles of length K, by default None.
        samples : list[jnp.ndarray] | jnp.ndarray | None, optional
            Optional samples to overlay. If multiple panels are plotted, pass a list
            of arrays; each array should have shape (N, 2), by default None.
        suptitle : str | None, optional
            Optional figure-level title, by default None.

        Returns
        -------
        None
        """
        # Generate grid for posterior (same for all)
        z1_std = jnp.sqrt(self.prior_cov[0, 0])
        z2_std = jnp.sqrt(self.prior_cov[1, 1])

        z1_grid = jnp.linspace(
            self.prior_mean[0] - 3 * z1_std,
            self.prior_mean[0] + 3 * z1_std,
            num_grid_points,
        )
        z2_grid = jnp.linspace(
            self.prior_mean[1] - 3 * z2_std,
            self.prior_mean[1] + 3 * z2_std,
            num_grid_points,
        )

        z1_mesh, z2_mesh = jnp.meshgrid(z1_grid, z2_grid)
        z_grid = jnp.stack(
            [z1_mesh.ravel(), z2_mesh.ravel()], axis=-1
        )  # (num_grid_points^2, 2)

        # Compute unnormalized log posterior for every grid point
        log_post = jax.vmap(lambda z: self.log_joint(x_obs, z))(z_grid)
        log_post_grid = log_post.reshape(z1_mesh.shape)
        post_grid = jnp.exp(log_post_grid)
        post_grid = post_grid / jnp.sum(post_grid)

        # Allow for multiple q_mean/q_cov or samples
        if q_mean is None or q_cov is None:
            if samples is not None:
                num_q = len(samples)
            else:
                num_q = 0
        else:
            q_mean = jnp.atleast_2d(q_mean)
            q_cov = jnp.atleast_3d(q_cov)
            num_q = q_mean.shape[0]
            # If both q_mean and q_cov are shape (1,...) and no titles, treat as one plot

        if num_q == 0:
            # Single plot: just posterior
            fig, ax = plt.subplots(figsize=(6, 5))
            contour = ax.contourf(
                z1_grid, z2_grid, post_grid, levels=50, cmap="viridis"
            )
            plt.colorbar(contour, label="Posterior density", ax=ax)
            ax.set_xlabel("$z_1$")
            ax.set_ylabel("$z_2$")
            ax.set_title("Posterior density $p(z | x)$")
            if samples is not None:
                ax.scatter(
                    samples[..., 0], samples[..., 1], color="red", alpha=0.05, s=1
                )
            if save_path is not None:
                plt.savefig(save_path)
            else:
                plt.show()
            return

        # Make subplots side by side for each q or samples array
        fig, axs = plt.subplots(1, num_q, figsize=(6 * num_q, 5), squeeze=False)
        for i in range(num_q):
            ax = axs[0, i]
            # Posterior density (same for all)
            contour = ax.contourf(
                z1_grid, z2_grid, post_grid, levels=50, cmap="viridis"
            )

            if i == num_q - 1:
                plt.colorbar(contour, label="Posterior density", ax=ax)

            if q_mean is not None and q_cov is not None:
                ax.set_xlabel("$z_1$")
                ax.set_ylabel("$z_2$")

                # Plot q(z) for this mean/cov
                z_points = jnp.stack([z1_mesh.ravel(), z2_mesh.ravel()], axis=-1)
                q_logpdf = multivariate_normal.logpdf(z_points, q_mean[i], q_cov[i])
                q_density = jnp.exp(q_logpdf).reshape(z1_mesh.shape)

                cs = ax.contour(
                    z1_grid,
                    z2_grid,
                    q_density,
                    levels=10,
                    colors="red",
                    linewidths=1.5,
                    alpha=0.25,
                )

                # Add proxy for legend
                ax.plot(
                    [],
                    [],
                    color="red",
                    lw=1.5,
                    alpha=0.25,
                    label=r"Variational $q(z)$",
                )
                ax.legend(loc="upper right")

            if samples is not None:
                samples_i = samples[i]
                ax.scatter(
                    samples_i[..., 0], samples_i[..., 1], color="red", alpha=0.05, s=1
                )

            # Set title
            if titles is not None and i < len(titles) and titles[i] is not None:
                ax.set_title(str(titles[i]))
            else:
                ax.set_title("Posterior and $q(z)$")

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
