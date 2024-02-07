import numpy as np


def generate_data(args, key):
    eqn = args.equation
    if eqn == 'biot1d':
        train_data = generator_biot1d(args, key, train=True)
        test_data = generator_biot1d(args, key, train=False)
    else:
        raise NotImplementedError
    return train_data, test_data


def generator_biot1d(args, key, train=True):
    # TODO: To be adapted to input/output structure of DeepONets
    # Note: At the moment only approximating u
    hyperparams = args.problem_hyperparam
    # Unpack hyperparams
    alpha_range = hyperparams['alpha']
    beta_range = hyperparams['beta']
    a_range = hyperparams['a']
    length = hyperparams['length']
    t_end = hyperparams['t_end']

    if train:
        n_samples = args.train_samples
    else:
        n_samples = args.test_samples

    # Split key
    keys = jax.random.split(key, 5)

    raise NotImplementedError


def biotanalyticalsolution(alpha, beta, a, z, t, l_max, n_series=10000):
    """
    Analytical solution according to https://www.icevirtuallibrary.com/doi/10.1680/jgeot.16.P.268
    Load function: a * sin(pi * t / 2)
    Legacy code for this publication: https://doi.org/10.1038/s41598-023-42141-x
    Heavy optimization possible doing vectorization by accepting lists as inputs
    """
    c_v = alpha * beta

    load_vals = np.sin(np.pi * t / 2) * a

    n_vec = np.arange(n_series)

    n_n = c_v * np.square((1 + 2 * n_vec) * np.pi / (2 * l_max))

    term_1 = -2 * np.exp(np.outer(t, -n_n)) * n_n
    term_2 = 2 * np.outer(np.cos(np.pi * t / 2), n_n)
    term_3 = np.tile(np.pi * np.sin(np.pi * t / 2), len(n_n))

    nom = np.pi * a * (np.add(np.add(term_1, term_2), term_3))
    denom = (4 * n_n ** 2 + np.pi ** 2)

    int_term = nom / denom

    p_n = int_term * 4 / (np.pi * (2 * n_vec + 1))

    trig_val = np.outer(z, (1 + 2 * n_vec) * np.pi) / (2 * l_max)
    p = p_n * np.sin(trig_val)

    u_sum = p_n * np.cos(trig_val) * ((2 * l_max) / ((1 + 2 * n_vec) * np.pi))

    # Reduce over summation
    u_sum = np.sum(u_sum)
    p = np.sum(p)

    # Combine u_sum and base term
    u = np.multiply(load_vals, (l_max - z)) - u_sum
    u = u / alpha

    return np.array([u, p])
