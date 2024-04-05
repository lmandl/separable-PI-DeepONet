import jax
import jax.numpy as jnp
from torch.utils import data
from functools import partial
import tqdm
import optax
import scipy.io
import os
import argparse

from models import setup_deeponet
from models import relative_l2, mse, train_error
from models import apply_net, step, update_model


# Data Generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, batch_size, gen_key):
        self.u = u
        self.y = y
        self.s = s
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = gen_key

    def __getitem__(self, index):
        """Generate one batch of data"""
        self.key, subkey = jax.random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jax.jit, static_argnums=(0,))
    def __data_generation(self, key_i):
        """Generates data containing batch_size samples"""
        idx = jax.random.choice(key_i, self.N, (self.batch_size,), replace=False)
        s = self.s[idx, :]
        y = self.y[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs


# Generate ics training data corresponding to one input sample
def generate_one_ics_training_data(u0, p=101):
    t_0 = jnp.zeros((p, 1))
    x_0 = jnp.linspace(0, 1, p)[:, None]

    y = jnp.hstack([t_0, x_0])
    u = jnp.tile(u0, (p, 1))
    s = u0

    return u, y, s


# Generate bcs training data corresponding to one input sample
def generate_one_bcs_training_data(key, u0, p=100):
    t_bc = jax.random.uniform(key, (p, 1))
    x_bc1 = jnp.zeros((p, 1))
    x_bc2 = jnp.ones((p, 1))

    y1 = jnp.hstack([t_bc, x_bc1])  # shape = (P, 2)
    y2 = jnp.hstack([t_bc, x_bc2])  # shape = (P, 2)

    u = jnp.tile(u0, (p, 1))
    y = jnp.hstack([y1, y2])  # shape = (P, 4)
    s = jnp.zeros((p, 1))

    return u, y, s


# Generate res training data corresponding to one input sample
def generate_one_res_training_data(key, u0, p=1000):
    subkeys = jax.random.split(key, 2)

    t_res = jax.random.uniform(subkeys[0], (p, 1))
    x_res = jax.random.uniform(subkeys[1], (p, 1))

    u = jnp.tile(u0, (p, 1))
    y = jnp.hstack([t_res, x_res])
    s = jnp.zeros((p, 1))

    return u, y, s


# Generate test data corresponding to one input sample
def generate_one_test_data(idx, usol, p=101):
    u = usol[idx]
    u0 = u[0, :]

    t = jnp.linspace(0, 1, p)
    x = jnp.linspace(0, 1, p)
    T, X = jnp.meshgrid(t, x)

    s = u.T.flatten()
    u = jnp.tile(u0, (p ** 2, 1))
    y = jnp.hstack([T.flatten()[:, None], X.flatten()[:, None]])

    return u, y, s


# Define ds/dx
def s_x_net(model_fn, params, u, t, x):
    v_x = jnp.ones(x.shape)
    s_x = jax.vjp(lambda x: apply_net(model_fn, params, u, t, x), x)[1](v_x)[0]
    return s_x


def loss_ics(model_fn, params, ics_batch):
        inputs, outputs = ics_batch
        u, y = inputs

        # Compute forward pass
        t = y[:, 0]
        x = y[:, 1]
        s_pred = apply_net(model_fn, params, u, t, x)

        # Compute loss
        loss_ic = mse(outputs.flatten(), s_pred)
        return loss_ic


def loss_bcs(model_fn, params, ics_batch):
    # Fetch data
    inputs, outputs = ics_batch
    u, y = inputs

    # Compute forward pass
    s_bc1_pred = apply_net(model_fn, params, u, y[:, 0], y[:, 1])
    s_bc2_pred = apply_net(model_fn, params, u, y[:, 2], y[:, 3])

    s_x_bc1_pred = s_x_net(model_fn, params, u, y[:, 0], y[:, 1])
    s_x_bc2_pred = s_x_net(model_fn, params, u, y[:, 2], y[:, 3])

    # Compute loss
    loss_s_bc = mse(s_bc1_pred, s_bc2_pred)
    loss_s_x_bc = mse(s_x_bc1_pred, s_x_bc2_pred)

    return loss_s_bc + loss_s_x_bc


# Define residual loss
def loss_res(model_fn, params, batch):
    # Fetch data
    inputs, outputs = batch
    u, y = inputs
    # Compute forward pass
    t = y[:,0]
    x = y[:,1]

    # Residual PDE
    s = apply_net(model_fn, params, u, t, x)
    v_x = jnp.ones(x.shape)
    v_t = jnp.ones(t.shape)

    s_t = jax.vjp(lambda t: apply_net(model_fn, params, u, t, x), t)[1](v_t)[0]
    s_x = jax.vjp(lambda x: apply_net(model_fn, params, u, t, x), x)[1](v_x)[0]
    s_xx = jax.jvp(lambda x: jax.vjp(lambda x: apply_net(model_fn, params, u, t, x), x)[1](v_x)[0], (x,), (v_x,))[1]

    pred = s_t + s * s_x - 0.01 * s_xx
    # Compute loss
    loss = mse(outputs.flatten(), pred)
    return loss


def loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch):
    loss_ics_i = loss_ics(model_fn, params, ics_batch)
    loss_bcs_i = loss_bcs(model_fn, params, bcs_batch)
    loss_res_i = loss_res(model_fn, params, res_batch)
    loss_value = loss_ics_i + loss_bcs_i + loss_res_i
    return loss_value


def main_routine(args):
    # Prepare the training data
    # Load data
    path = os.path.join(os.getcwd(), 'data/burgers/Burger.mat')  # Please use the matlab script to generate data

    data_ref = scipy.io.loadmat(path)
    u_sol = jnp.array(data_ref['output'])

    n_in = u_sol.shape[0]  # number of total input samples
    args.n_test = n_in - args.n_train  # number of input samples used for test
    if args.n_test < 0:
        raise ValueError('Number of test samples is negative')

    u0_train = u_sol[:args.n_train, 0, :]  # input samples

    # Split key for IC, BC, Residual data, and model init
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)

    # ICs data
    u_ics_train, y_ics_train, s_ics_train = (jax.vmap(generate_one_ics_training_data,
                                                      in_axes=(0, None))
                                             (u0_train, args.p_ics_train))
    u_ics_train = u_ics_train.reshape(args.n_train * args.p_ics_train, -1)
    y_ics_train = y_ics_train.reshape(args.n_train * args.p_ics_train, -1)
    s_ics_train = s_ics_train.reshape(args.n_train * args.p_ics_train, -1)

    # BCs data
    bc_keys = jax.random.split(keys[0], args.n_train)
    u_bcs_train, y_bcs_train, s_bcs_train = (jax.vmap(generate_one_bcs_training_data,
                                                      in_axes=(0, 0, None))
                                             (bc_keys, u0_train, args.p_bcs_train))

    u_bcs_train = u_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)
    y_bcs_train = y_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)
    s_bcs_train = s_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)

    # Residual data
    res_keys = jax.random.split(keys[1], args.n_train)
    u_res_train, y_res_train, s_res_train = (jax.vmap(generate_one_res_training_data,
                                                      in_axes=(0, 0, None))
                                             (res_keys, u0_train, args.p_res_train))

    u_res_train = u_res_train.reshape(args.n_train * args.p_res_train, -1)
    y_res_train = y_res_train.reshape(args.n_train * args.p_res_train, -1)
    s_res_train = s_res_train.reshape(args.n_train * args.p_res_train, -1)

    # Create data generators
    ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, args.batch_size, keys[2])
    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, args.batch_size, keys[3])
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, args.batch_size, keys[4])

    # Create model
    args, model, model_fn, params = setup_deeponet(args, keys[5])

    # Define optimizer with optax (ADAM)
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    # Data
    ics_data = iter(ics_dataset)
    bcs_data = iter(bcs_dataset)
    res_data = iter(res_dataset)
    pbar = tqdm.trange(args.epochs)

    # Training loop
    for it in pbar:
        # Fetch data
        ics_batch = next(ics_data)
        bcs_batch = next(bcs_data)
        res_batch = next(res_data)

        # Do Step
        loss, params_step, opt_state = step(optimizer, loss_fn, model_fn, opt_state,
                                            params, ics_batch, bcs_batch, res_batch)

        if it % args.log_iter == 0:
            # Compute losses
            loss_ics_value = loss_ics(model_fn, params, ics_batch)
            loss_bcs_value = loss_bcs(model_fn, params, bcs_batch)
            loss_res_value = loss_res(model_fn, params, res_batch)

            # TODO: Compute error

            # Print losses
            pbar.set_postfix({'Loss': loss,
                              'loss_ics': loss_ics_value,
                              'loss_bcs': loss_bcs_value,
                              'loss_physics': loss_res_value})


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--num_outputs', type=int, default=1, help='number of outputs')
    parser.add_argument('--hidden_dim', type=int, default=40,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')
    parser.add_argument('--separable', dest='separable', default=False, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=128, help='hidden tensor dimension in separable DeepONets')

    # Branch settings
    parser.add_argument('--branch_layers', type=int, nargs="+", default=128, help='hidden branch layer sizes')
    parser.add_argument('--n_sensors', type=int, default=101,
                        help='number of sensors for branch network, also called >>m<<')
    parser.add_argument('--branch_input_features', type=int, default=1,
                        help = 'number of input features per sensor to branch network')
    parser.add_argument('--split_branch', dest='split_branch', default=False, action='store_false',
                        help='split branch outputs into n groups for n outputs')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=128, help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=2, help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=False, action='store_false',
                        help='split trunk outputs into j groups for j outputs')

    # Training settings
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='training epochs')

    # result directory
    parser.add_argument('--result_dir', type=str, default='./result',
                        help='a directory to save results, relative to cwd')

    # log settings
    parser.add_argument('--log_iter', type=int, default=100, help='iteration to save loss and error')

    # Problem / Data Settings
    parser.add_argument('--n_train', type=int, default=1000, help='number of input samples used for training')
    parser.add_argument('--p_ics_train', type=int, default=101,
                        help='number of locations for evaluating the initial condition')
    parser.add_argument('--p_bcs_train', type=int, default=100,
                        help='number of locations for evaluating the boundary condition')
    parser.add_argument('--p_res_train', type=int, default=2500,
                        help='number of locations for evaluating the PDE residual')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')

    args_in = parser.parse_args()

    print('Project in Development')

    main_routine(args_in)
