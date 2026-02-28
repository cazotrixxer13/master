from scipy.integrate import nquad
import numpy as np


def f(x, y):
    # TODO: Implement f as given in the assignment sheet.
    return np.sin(np.sqrt(x**2 + y**2) + x + y)


def monte_carlo_estimate(N, rng_key):
    rng = np.random.default_rng(seed=rng_key)
    
    x_i = rng.uniform(low=0, high=2*np.pi, size=N)
    y_i = rng.uniform(low=0, high=2*np.pi, size=N)

    return 4 * np.pi**2 * np.mean(f(x_i, y_i))


def main():
    seed = 365365365
    N = 50_000_000
    estimate = monte_carlo_estimate(N, seed)

    # TODO: Use `nquad` to numerically compute the integral and compare the result with the Monte Carlo estimate.
    result, error, infodict = nquad(f, [[0, 2*np.pi],[0, 2*np.pi]], full_output=True)
    
    print(f"Number of Samples:                  {N}")
    print(f"Monte Carlo estimate result:        {estimate:.8f}")
    print(f"nquad result:                       {result:.8f}")
    print(f"nquad estimated error:              {error:.8e}")
    print(f"Number of function evaluations:     {infodict['neval']}")
    
    diff = abs(result - estimate)
    print(f"\nDifference:                         {diff:.8f}")


if __name__ == "__main__":
    main()
