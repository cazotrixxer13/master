import numpy as np
from tqdm import tqdm
import itertools
from matplotlib import pyplot as plt
import time


class BayesianNetwork:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                "x1": [0.8],
                "x2": [0.1, 0.5],
                "x3": [0.25, 0.5, 0.8, 0.7],
                "x4": [0.1, 0.5],
                "x5": [0.1, 0.5, 0.8, 0.3],
                "x6": [0.95, 0.6],
                "x7": [0.1, 0.5],
                "x8": [0.02],
            }
        else:
            self.params = params

    def x1_prob(self):
        arr = self.params["x1"]
        return arr[0]

    def x2_prob(self, x1):
        arr = self.params["x2"]
        if x1 == 0:
            return arr[0]
        else:
            return arr[1]

    def x3_prob(self, x1, x6):
        arr = self.params["x3"]
        if x1 == 0 and x6 == 0:
            return arr[0]
        elif x1 == 0 and x6 == 1:
            return arr[1]
        elif x1 == 1 and x6 == 0:
            return arr[2]
        else:
            return arr[3]

    def x4_prob(self, x3):
        arr = self.params["x4"]
        if x3 == 0:
            return arr[0]
        else:
            return arr[1]

    def x5_prob(self, x3, x7):
        arr = self.params["x5"]
        if x3 == 0 and x7 == 0:
            return arr[0]
        elif x3 == 0 and x7 == 1:
            return arr[1]
        elif x3 == 1 and x7 == 0:
            return arr[2]
        else:
            return arr[3]

    def x6_prob(self, x8):
        arr = self.params["x6"]
        if x8 == 0:
            return arr[0]
        else:
            return arr[1]

    def x7_prob(self, x8):
        arr = self.params["x7"]
        if x8 == 0:
            return arr[0]
        else:
            return arr[1]

    def x8_prob(self):
        arr = self.params["x8"]
        return arr[0]

    def sample_joint_distribution(self, num_samples=1000, seed=0):
        np.random.seed(seed)
        
        rng = np.random.default_rng()
        samples = np.zeros((num_samples, 8), dtype=int)

        for i in range(num_samples):
            x1 = rng.binomial(1, self.x1_prob())
            x8 = rng.binomial(1, self.x8_prob())
            x6 = rng.binomial(1, self.x6_prob(x8))
            x7 = rng.binomial(1, self.x7_prob(x8))
            x2 = rng.binomial(1, self.x2_prob(x1))
            x3 = rng.binomial(1, self.x3_prob(x1, x6))
            x4 = rng.binomial(1, self.x4_prob(x3))
            x5 = rng.binomial(1, self.x5_prob(x3, x7))

            samples[i] = [x1, x2, x3, x4, x5, x6, x7, x8]

        return samples


def maximum_likelihood_estimation(samples):
    # TODO: Compute the MLE parameters for each variable, in the same format as the params dictionary above.
    mle = {
        "x1": [None],
        "x2": [None],
        "x3": [None],
        "x4": [None],
        "x5": [None],
        "x6": [None],
        "x7": [None],
        "x8": [None],
    }
    
    
    x1_vals = samples[:, 0]
    x2_vals = samples[:, 1]
    x3_vals = samples[:, 2]
    x4_vals = samples[:, 3]
    x5_vals = samples[:, 4]
    x6_vals = samples[:, 5]
    x7_vals = samples[:, 6]
    x8_vals = samples[:, 7]

    mle["x1"] = [x1_vals.mean()]

    mle["x2"] = []
    for x1_parent in [0, 1]:
        mask = (x1_vals == x1_parent)
        if mask.sum() > 0:
            mle["x2"].append(x2_vals[mask].mean())
        else:
            mle["x2"].append(0.5)

    mle["x3"] = []
    for x1_parent in [0, 1]:
        for x6_parent in [0, 1]:
            mask = (x1_vals == x1_parent) & (x6_vals == x6_parent)
            if mask.sum() > 0:
                mle["x3"].append(x3_vals[mask].mean())
            else:
                mle["x3"].append(0.5)

    mle["x4"] = []
    for x3_parent in [0, 1]:
        mask = (x3_vals == x3_parent)
        if mask.sum() > 0:
            mle["x4"].append(x4_vals[mask].mean())
        else:
            mle["x4"].append(0.5)
            
    mle["x5"] = []
    for x3_parent in [0, 1]:
        for x7_parent in [0, 1]:
            mask = (x3_vals == x3_parent) & (x7_vals == x7_parent)
            if mask.sum() > 0:
                mle["x5"].append(x5_vals[mask].mean())
            else:
                mle["x5"].append(0.5)

    mle["x6"] = []
    for x8_parent in [0, 1]:
        mask = (x8_vals == x8_parent)
        if mask.sum() > 0:
            mle["x6"].append(x6_vals[mask].mean())
        else:
            mle["x6"].append(0.5)

    mle["x7"] = []
    for x8_parent in [0, 1]:
        mask = (x8_vals == x8_parent)
        if mask.sum() > 0:
            mle["x7"].append(x7_vals[mask].mean())
        else:
            mle["x7"].append(0.5)

    mle["x8"] = [x8_vals.mean()]
    
    return mle


class MarkovChainBayesianNetwork:
    def __init__(self, length, params=None):
        self.length = length
        if params is None:
            # Randomly generate numpy array of length length with two values between 0 or 1
            self.params = np.random.rand(length, 2)
            self.params[0, 1] = (
                np.nan
            )  # Distribution p(x1) does not depend on any other variable
        else:
            assert params.shape == (length, 2), "Shape of params must be (length, 2)"
            self.params = params
            
    def compute_joint_distribution(self):
        joint_distribution = []
        # TODO: Compute full joint distribution by iterating over all possible binary states.
        #       You can use `tqdm` to show a progress bar.
        
        joint_distribution = np.empty(2 ** self.length, dtype=float)

        for idx in tqdm(range(2 ** self.length)):
            bits = np.empty(self.length, dtype=int)
            for i in range(self.length):
                bits[i] = ((idx >> i) & 1)
            p_x1_1 = self.params[0, 0]
            p = p_x1_1
            if bits[0] != 1:
                p = (1.0 - p_x1_1)

            for i in range(1, self.length):
                parent = bits[i - 1]
                child = bits[i]

                p_child1 = self.params[i, parent]
                if child == 1:
                    p *= p_child1
                else:
                    p *= (1.0 - p_child1)

            joint_distribution[idx] = p

        return np.array(joint_distribution)

    def compute_marginal_variable_elimination(self, variable_index):
        assert (
            variable_index > 0 and variable_index <= self.length
        ), "Variable index must be between 1 and length"

        p_prev_1 = self.params[0, 0]
        if variable_index == 1:
            return p_prev_1

        for i in tqdm(range(2, variable_index + 1)):
            p_prev_0 = 1.0 - p_prev_1
            p_1_given_0 = self.params[i - 1, 0]
            p_1_given_1 = self.params[i - 1, 1]
            p_prev_1 = p_prev_0 * p_1_given_0 + p_prev_1 * p_1_given_1

        return p_prev_1


def plot_params(params, mle_params, N):
    params = [item for i in range(1, len(params) + 1) for item in params[f"x{i}"]]
    mle_params = [
        item for i in range(1, len(mle_params) + 1) for item in mle_params[f"x{i}"]
    ]

    plt.figure(figsize=(10, 5))
    width = 0.4
    x = np.arange(len(params))
    plt.bar(x - width / 2, params, width=width, label="True Parameters")
    plt.bar(x + width / 2, mle_params, width=width, label="MLE Parameters")
    plt.legend()
    plt.xlabel("Variable Index")
    plt.ylabel("Parameter Value")
    plt.title(f"Parameters vs. MLE Parameters (N = {N})")
    plt.show()


def main():
    # ------------------------------------------------------------
    # Task 3.4

    bn = BayesianNetwork()
    # TODO: For different N, sample from the joint distribution of `bn` and compute the MLE parameters.
    
    N = [10, 100, 10_000, 1_000_000]

    for n in N:
        samples = bn.sample_joint_distribution(n, 365365365)
        mle_params = maximum_likelihood_estimation(samples)
        plot_params(bn.params, mle_params, n)

    # ------------------------------------------------------------
    # Task 3.5

    n = 22  # You can change this during testing
    mc = MarkovChainBayesianNetwork(length=n)
    # Compute the full joint distribution.
    t1 = time.perf_counter()
    joint_distribution = mc.compute_joint_distribution()

    # TODO: Compute the marginal distribution of xn by only using `joint_distribution`.
    xn_marginal = joint_distribution[2**(n-1):].sum()
    t2 = time.perf_counter()
    # Compute the marginal distribution of xn by using variable elimination.
    xn_marginal_var_el = mc.compute_marginal_variable_elimination(variable_index=n)
    t3 = time.perf_counter()

    print(f"\nFull Joint distribution: {xn_marginal}")
    print(f"Joint distribution with VE: {xn_marginal_var_el}\n")
    
    print(f"Time consumed for full joint computation: {t2 - t1}")
    print(f"Time consumed for joint computation with VE: {t3 - t2}")

    # Assert that xn_marginal and xn_marginal_var_el are close
    assert np.isclose(
        xn_marginal, xn_marginal_var_el, atol=1e-8
    ), f"Marginals do not match: {xn_marginal} vs {xn_marginal_var_el}"


if __name__ == "__main__":
    main()
