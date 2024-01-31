import os
import numpy as np


def load_data(data_dir, eqn):
    """
    Load data for a given equation
    """
    # Check if equation is implemented
    # Then load data
    if eqn == 'biot1d':
        train_data, test_data = load_biot1d(data_dir)
    elif eqn == 'antiderivative_aligned':
        train_data, test_data = load_antiderivative_aligned(data_dir)
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


def load_antiderivative_aligned(data_dir):
    """
    Load data for antiderivative problem aligned in 1D
    see: https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html
    Expects data to be saved as train.npz and test.npz in data_dir
    """
    d = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
    x_train, y_train = (d["X"][0], d["X"][1]), d["y"]

    d = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
    x_test = (d["X"][0], d["X"][1])
    y_test = d["y"]
    return [x_train, y_train], [x_test, y_test]


def load_antiderivative_unaligned(data_dir):
    """
    Load data for antiderivative problem aligned in 1D
    see: https://deepxde.readthedocs.io/en/latest/demos/operator/antiderivative_aligned.html
    Expects data to be saved as train.npz and test.npz in data_dir
    """
    d = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
    x_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]

    d = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
    x_test = (d["X_test0"], d["X_test1"])
    y_test = d["y_test"]
    return [x_train, y_train], [x_test, y_test]
