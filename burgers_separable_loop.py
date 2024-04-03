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
from models import step, update_model, hvp_fwdfwd
from models import apply_net_sep as apply_net


# Data Generator
class DataGeneratorIC(data.Dataset):
    # IC has same t for all samples in y
    def __init__(self, u, x, t, s, batch_size, gen_key):
        self.u = u
        self.x = x
        self.t = t
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
        x = self.x[idx, :]
        t = self.t
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, (t, x))
        outputs = s
        return inputs, outputs

class DataGeneratorBC(data.Dataset):
    # IC has same t for all samples in y
    def __init__(self, u, x, t, s, batch_size, gen_key):
        self.u = u
        self.x = x
        self.t = t
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
        x = self.x
        t = self.t[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, (t, x))
        outputs = s
        return inputs, outputs

# Data Generator
class DataGeneratorRes(data.Dataset):

    def __init__(self, u, x, t, s, batch_size, gen_key):
        self.u = u
        self.x = x
        self.t = t
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
        x = self.x[idx, :]
        t = self.t[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, (t, x))
        outputs = s
        return inputs, outputs

# Define ds/dx
def s_x_net(model_fn, params, u, t, x):
    v_x = jnp.ones(x.shape)
    # Verify whether to use [0] or [1] for jvp
    s_x = jax.jvp(lambda x: apply_net(model_fn, params, u, t, x), (x,), (v_x,))[1]
    return s_x

def loss_ics(model_fn, params, ics_batch):
        inputs, outputs = ics_batch
        u, y = inputs

        # Compute forward pass
        t = y[0]
        x = y[1]
        s_pred = apply_net(model_fn, params, u, t, x)

        # Compute loss
        loss_ic = mse(outputs.flatten(), s_pred)
        return loss_ic


def loss_bcs(model_fn, params, bcs_batch):
    # Fetch data
    inputs, outputs = bcs_batch
    u, y = inputs
    x, t_bc = y
    x_bc_1, x_bc_2 = x

    # Assert shapes here for easier gradient computation
    x_bc_1 = jnp.reshape(x_bc_1, (-1, 1))
    x_bc_2 = jnp.reshape(x_bc_2, (-1, 1))
    t_bc = jnp.reshape(t_bc, (-1, 1))

    # Compute forward pass
    s_bc1_pred = apply_net(model_fn, params, u, x_bc_1, t_bc)
    s_bc2_pred = apply_net(model_fn, params, u, x_bc_2, t_bc)

    s_x_bc1_pred = s_x_net(model_fn, params, u, x_bc_1, t_bc)
    s_x_bc2_pred = s_x_net(model_fn, params, u, x_bc_2, t_bc)

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
    x, t = y

    # Residual PDE
    s = apply_net(model_fn, params, u, t, x)
    v_x = jnp.atleast_2d(jnp.ones(x.shape)).T
    v_t = jnp.atleast_2d(jnp.ones(t.shape)).T

    #TODO: Forward mode AD

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

    # Note: Data generation procedure would be quicker using vmap
    # However, comparison / adaptation to separable approach is easier with loop
    # ICs data
    # Init empty arrays for storage
    u_ics_train = []
    x_ics_train = []
    s_ics_train = []
    # Loop over initial functions
    for u_0 in u0_train:
        u = jnp.tile(u_0, (args.p_ics_train, 1))
        u_ics_train.append(u)

        x_0 = jnp.linspace(0, 1, args.p_ics_train)[:, None]
        x_ics_train.append(x_0)

        s = u_0
        s_ics_train.append(s)

    t_ics_train = jnp.zeros((1, 1))

    # Make array
    u_ics_train = jnp.array(u_ics_train)
    x_ics_train = jnp.array(x_ics_train)
    s_ics_train = jnp.array(s_ics_train)

    u_ics_train = u_ics_train.reshape(args.n_train * args.p_ics_train, -1)
    x_ics_train = x_ics_train.reshape(args.n_train * args.p_ics_train, -1)
    s_ics_train = s_ics_train.reshape(args.n_train * args.p_ics_train, -1)

    # Create data generator
    ics_dataset = DataGeneratorIC(u_ics_train, x_ics_train, t_ics_train, s_ics_train, args.batch_size, keys[2])

    # BCs data
    # Init empty arrays for storage
    u_bcs_train = []
    t_bcs_train = []
    s_bcs_train = []
    # generate keys for BCs
    bc_keys = jax.random.split(keys[0], args.n_train)
    # Loop over BCs
    for key_i, u0_i in zip(bc_keys, u0_train):
        t_bc = jax.random.uniform(key_i, (args.p_bcs_train, 1))
        t_bcs_train.append(t_bc)

        u = jnp.tile(u0_i, (args.p_bcs_train, 1))
        u_bcs_train.append(u)

        s = jnp.zeros((args.p_bcs_train, 1))
        s_bcs_train.append(s)

    x_bc1_train = jnp.zeros((1, 1))
    x_bc2_train = jnp.ones((1, 1))
    x_bc_train = (x_bc1_train, x_bc2_train)

    # Make array
    u_bcs_train = jnp.array(u_bcs_train)
    t_bcs_train = jnp.array(t_bcs_train)
    s_bcs_train = jnp.array(s_bcs_train)

    u_bcs_train = u_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)
    t_bcs_train = t_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)
    s_bcs_train = s_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)

    # Create data generator
    # Normally, this would require differentiating between BC1 and BC2
    # However, as s=0 for both, we can use the same data generator
    bcs_dataset = DataGeneratorBC(u_bcs_train, x_bc_train, t_bcs_train, s_bcs_train, args.batch_size, keys[3])
    # Note: p_bcs_train can be halved as one sample from t is used for both BCs

    # Residual data
    # Init empty arrays for storage
    u_res_train = []
    x_res_train = []
    t_res_train = []
    s_res_train = []

    # generate keys for Residuals
    res_keys = jax.random.split(keys[1], args.n_train)
    # Loop over Residuals
    for key_i, u0_i in zip(res_keys, u0_train):
        subkeys = jax.random.split(key_i, 2)
        t_res = jax.random.uniform(subkeys[0], (args.p_res_train, 1))
        #t_res = jnp.linspace(0, 1, args.p_res_train).reshape(-1,1)
        x_res = jax.random.uniform(subkeys[1], (args.p_res_train, 1))
        #x_res = jnp.linspace(0, 1, args.p_res_train).reshape(-1,1)

        x_res_train.append(x_res)
        t_res_train.append(t_res)

        u = jnp.tile(u0_i, (args.p_res_train, 1))
        u_res_train.append(u)

        s = jnp.zeros((args.p_res_train, args.p_res_train))
        s_res_train.append(s)

    # Make array
    u_res_train = jnp.array(u_res_train)
    x_res_train = jnp.array(x_res_train)
    t_res_train = jnp.array(t_res_train)
    s_res_train = jnp.array(s_res_train)

    u_res_train = u_res_train.reshape(args.n_train * args.p_res_train, -1)
    x_res_train = x_res_train.reshape(args.n_train * args.p_res_train, -1)
    t_res_train = t_res_train.reshape(args.n_train * args.p_res_train, -1)
    s_res_train = s_res_train.reshape(args.n_train * args.p_res_train, -1)

    # Create data generators
    res_dataset = DataGeneratorRes(u_res_train, x_res_train, s_res_train, args.batch_size, keys[4])

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
    parser.add_argument('--separable', dest='separable', default=True, action='store_true',
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
    parser.add_argument('--p_bcs_train', type=int, default=50,
                        help='number of locations for evaluating the boundary condition')
    parser.add_argument('--p_res_train', type=int, default=50,
                        help='number of locations for evaluating the PDE residual')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')

    args_in = parser.parse_args()

    print('Project in Development')

    main_routine(args_in)
