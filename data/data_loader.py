import os
import numpy as np


def load_data(data_dir, eqn):
    """
    load data for the given problem
    data should be a tuple of (train_data, test_data)
    train_data and test_data should be a tuple of ((branch_input, trunk_input), output)
    trunk_input should be a tuple of coordinates where to evaluate and parameters, i.e. (x, t, alpha, beta, a, length)
    branch_input should be a vector of function values
    if the input function has two values, e.g. (u, p), then the respective branch_input should be
    flattened and concatenated along the first axis
    """
    # Check if equation is implemented
    # Then load data
    if eqn == 'biot1d':
        train_data, test_data = load_biot1d(data_dir)
    elif eqn == 'antiderivative_unaligned':
        train_data, test_data = load_antiderivative_unaligned(data_dir)
    else:
        raise NotImplementedError
    return train_data, test_data


def load_biot1d(data_dir):
    # TODO: implement data load after data generation and save is completed
    """
    Load data for Biot equation in 1D
    Expects data to be saved as train.npz and test.npz in data_dir
    """
    raise NotImplementedError


def load_antiderivative_unaligned(data_dir):
    """
    Load data for antiderivative problem aligned in 1D
    see: https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html
    Expects data to be saved as train.npz and test.npz in data_dir
    """
    d = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
    x_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"].squeeze()

    d = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
    x_test = (d["X_test0"], d["X_test1"])
    y_test = d["y_test"].squeeze()
    return [x_train, y_train], [x_test, y_test]
