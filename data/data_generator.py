import jax.numpy as jnp


def generate_data(args):
    eqn = args.equation
    if eqn == 'biot1d':
        train_data = train_generator_biot1d(args)
        test_data = test_generator_biot1d(args)
    else:
        raise NotImplementedError
    return train_data, test_data


def train_generator_biot1d(args):
    # TODO
    return NotImplementedError


def test_generator_biot1d(args):
    # TODO
    return NotImplementedError
