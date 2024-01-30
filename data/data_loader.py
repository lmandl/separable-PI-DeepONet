import jax.numpy as jnp
import os


def load_data(data_dir, eqn):
    """
    Load data for a given equation
    """
    # Check if equation is implemented
    # Then load data
    if eqn == 'biot1d':
        train_data, test_data = load_biot1d(data_dir)
    else:
        raise NotImplementedError
    return train_data, test_data


def load_biot1d(data_dir):
    """
    Load data for Biot equation in 1D
    Expects data to be saved as train.npz and test.npz in data_dir
    """
    train_data = jnp.load(os.path.join(data_dir, "train.npz"))
    test_data = jnp.load(os.path.join(data_dir, "test.npz"))
    return train_data, test_data
