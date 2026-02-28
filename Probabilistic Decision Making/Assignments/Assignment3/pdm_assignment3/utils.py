import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt


def load_data(file_path="data/x.npz") -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load the data from the file.
    """
    with np.load(file_path) as data:
        x_train = jnp.array(data["x_train"])
        x_test = jnp.array(data["x_test"])
        return x_train, x_test


def plot_loss(
    losses: jnp.ndarray,
    save_path: str,
    title: str = "Convergence of fitting VI parameters",
):
    """
    Plot the loss curve.
    """
    plt.figure(figsize=(10, 4), dpi=100).set_facecolor("white")
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(save_path)
