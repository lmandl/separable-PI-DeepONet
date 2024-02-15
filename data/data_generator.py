import numpy as np


def generate_data(args, key):
    # Generate data for the given problem
    # data should be a tuple of (train_data, test_data)
    # train_data and test_data should be a tuple of ((branch_input, trunk_input), output)
    # trunk_input should be a tuple of coordinates where to evaluate and parameters, i.e. (x, t, alpha, beta, a, length)
    # branch_input should be a vector of function values
    # if the input function has two values, e.g. (u, p), then the respective branch_input should be
    # flattened and concatenated along the first axis
    eqn = args.equation
    raise NotImplementedError

    return train_data, test_data
