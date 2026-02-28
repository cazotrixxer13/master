import numpy as np
from matplotlib import pyplot as plt

# known from Task 3.2 (5.)
TRUE_E = 4.0 / 3.0

def F_inv(u: np.ndarray) -> np.ndarray:
    # TODO: Implement the inverse of the cumulative distribution function F
    u = np.clip(u, 0.0, 1.0)
    x = 2.0 * np.sqrt(u)
    return x


def sample_F(N: int) -> np.ndarray:
    # TODO: use numpy and `F_inv` to draw N i.i.d. samples from F
    u = np.random.rand(N)
    return F_inv(u)


def estimate_mean_monte_carlo():
    N = np.arange(100, 10001, 100)
    means = np.zeros_like(N, dtype=float)

    for i, n in enumerate(N):
        x = sample_F(n)
        means[i] = x.mean()

    plt.figure(figsize=(7, 4.5))
    plt.scatter(N, means, label='MC Estimates', alpha=0.7,  color='red')
    plt.axhline(TRUE_E, linestyle='--', label='True $E_X[X] = 4/3$')
    plt.xscale('log')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('MC Estimate')
    plt.title('Monte Carlo Estimation of $E_X[X]$')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def main():
    # TODO: Set numpy random seed for reproducibility, 
    # i.e., call np.random.seed(N) with an integer N of your choice
    N = 365365365
    np.random.seed(N)
    estimate_mean_monte_carlo()


if __name__ == "__main__":
    main()