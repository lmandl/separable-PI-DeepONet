import jax.numpy as jnp

def generate_train_data(args):
    eqn = args.equation
    if eqn == 'biot2d':
        data = train_generator_biot1d(args)
    else:
        raise NotImplementedError
    return data

def train_generator_biot1d(args):
    #TBD
    return NotImplementedError

def generate_test_data(args):
        eqn = args.equation
        if eqn == 'biot2d':
            data = test_generator_biot1d(args)
        else:
            raise NotImplementedError
        return data

def test_generator_biot1d(args):
    # TBD
    return NotImplementedError