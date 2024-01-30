import numpy as np
import time


class BiotAnalyticalSolution:
    """
    Analytical solution according to https://www.icevirtuallibrary.com/doi/10.1680/jgeot.16.P.268
    Load function: a * sin(pi * t / 2)
    """
    def __init__(self, length, t_end, alpha, beta, a):
        # Domain Ends
        self.length = length
        self.t_end = t_end

        # Material parameter
        self.alpha = alpha
        self.beta = beta
        self.c_v = alpha * beta

        # Loading Function Parameter
        self.a = a

    def load_func(self, t):
        return np.sin(np.pi * t / 2) * self.a

    def int_term(self, n_n, t):
        term_1 = -2 * np.exp(np.outer(t, -n_n)) * n_n
        term_2 = 2 * np.outer(np.cos(np.pi * t / 2), n_n)
        term_3 = np.tile(np.pi * np.sin(np.pi * t / 2), len(n_n))

        nom = np.pi * self.a * (np.add(np.add(term_1, term_2), term_3))
        denom = (4 * n_n ** 2 + np.pi ** 2)

        result = nom / denom

        return result

    def func(self, x, n_series=10000):
        z, t = np.split(x, 2, axis=1)

        print("calculating analytical solution...")
        start = time.time()

        n_vec = np.arange(n_series)
        n_n = self.c_v * np.square((1 + 2 * n_vec) * np.pi / (2 * self.length))
        p_n = (np.add(np.outer(t, -n_n) * self.load_func(0), self.int_term(n_n, t))) * 4 / (np.pi * (2 * n_vec + 1))

        trig_val = np.outer(z, (1 + 2 * n_vec) * np.pi) / (2 * self.length)
        p = p_n * np.sin(trig_val)

        u_sum = p_n * np.cos(trig_val) * ((2 * self.length) / ((1 + 2 * n_vec) * np.pi))

        # Reduce over summation
        u_sum = np.sum(u_sum, axis=1)
        p = np.sum(p, axis=1)

        # Combine u_sum and base term (note [:,0] necessary for shape stuff in numpy)
        u = np.multiply(self.load_func(t), (self.length - z))[:, 0] - u_sum
        u = u / self.alpha
        dur = time.time() - start
        print(f"done after {dur} seconds...")

        return np.vstack([u, p]).T
