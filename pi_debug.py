import jax
import jax.numpy as jnp
from torch.utils import data
from utils import mse_loss
from functools import partial
from tqdm import trange
import optax
import scipy.io
import os
import argparse
from models import setup_deeponet

# TODO: Check partial jits and static_argnums for use in class methods

def apply_net(model_fn, params, branch_input, *trunk_in):
    trunk_input = jnp.stack(trunk_in, axis=-1)
    return model_fn(params, branch_input, trunk_input)

# Define ds/dx
def s_x_net(model_fn, params, u, t, x):
    v_x = jnp.ones(x.shape)
    s_x = jax.vjp(lambda x: apply_net(model_fn, params, u, t, x), x)[1](v_x)[0]
    return s_x

def residual_net(model_fn, params, u, t, x):
    s = apply_net(model_fn, params, u, t, x)
    v_x = jnp.ones(x.shape)
    v_t = jnp.ones(t.shape)

    s_t = jax.vjp(lambda t: apply_net(model_fn, params, u, t, x), t)[1](v_t)[0]
    s_x = jax.vjp(lambda x: apply_net(model_fn, params, u, t, x), x)[1](v_x)[0]
    s_xx = jax.jvp(lambda x: jax.vjp(lambda x: apply_net(model_fn, params, u, t, x), x)[1](v_x)[0], (x,), (v_x,))[1]

    res = s_t + s * s_x - 0.01 * s_xx
    return res

def loss_ics(model_fn, params, batch):
    inputs, outputs = batch
    u, y = inputs

    # Compute forward pass
    t = y[:, 0]
    x = y[:, 1]
    s_pred = apply_net(model_fn, params, u, t, x)

    # Compute loss
    loss = mse_loss(outputs.flatten(), s_pred)
    return loss

# Define boundary loss
def loss_bcs(model_fn, params, batch):
    # Fetch data
    inputs, outputs = batch
    u, y = inputs

    # Compute forward pass
    s_bc1_pred = apply_net(model_fn, params, u, y[:, 0], y[:, 1])
    s_bc2_pred = apply_net(model_fn, params, u, y[:, 2], y[:, 3])

    s_x_bc1_pred = s_x_net(model_fn, params, u, y[:, 0], y[:, 1])
    s_x_bc2_pred = s_x_net(model_fn, params, u, y[:, 2], y[:, 3])

    # Compute loss
    loss_s_bc = mse_loss(s_bc1_pred, s_bc2_pred)
    loss_s_x_bc = mse_loss(s_x_bc1_pred, s_x_bc2_pred)

    return loss_s_bc + loss_s_x_bc

# Define residual loss
def loss_res(model_fn, params, batch):
    # Fetch data
    inputs, outputs = batch
    u, y = inputs
    # Compute forward pass
    pred = residual_net(model_fn, params, u, y[:, 0], y[:, 1])

    # Compute loss
    loss = mse_loss(outputs.flatten(), pred)
    return loss

# Define total loss
def loss_and_grad(model_fn, params, ics_batch, bcs_batch, res_batch):
    def loss_fn(params_loss):
        loss_ics_i = loss_ics(model_fn, params_loss, ics_batch)
        loss_bcs_i = loss_bcs(model_fn, params_loss, bcs_batch)
        loss_res_i = loss_res(model_fn, params_loss, res_batch)
        loss = loss_ics_i + loss_bcs_i + loss_res_i
        return loss
    return jax.value_and_grad(loss_fn)(params)

def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state

# update step
@partial(jax.jit, static_argnums=(0, 1))
def step(model_fn, optimizer, params, opt_state, ics_batch, bcs_batch, res_batch):
    # Train model
    loss, gradient = loss_and_grad(model_fn, params, ics_batch, bcs_batch, res_batch)

    updates, state = optimizer.update(gradient, opt_state)
    params = optax.apply_updates(params, updates)

    return loss, params, state


# Optimize parameters in a loop
def train(model_fn, params, opt_state, ics_dataset, bcs_dataset, res_dataset, epochs=10000):
    ics_data = iter(ics_dataset)
    bcs_data = iter(bcs_dataset)
    res_data = iter(res_dataset)
    pbar = trange(epochs)
    # Main training loop
    for it in pbar:
        # Fetch data
        ics_batch = next(ics_data)
        bcs_batch = next(bcs_data)
        res_batch = next(res_data)

        loss, params, opt_state = step(model_fn, optimizer, params, opt_state, ics_batch, bcs_batch, res_batch)

        if it % 100 == 0:
            # Compute losses
            loss_ics_value = loss_ics(model_fn, params, ics_batch)
            loss_bcs_value = loss_bcs(model_fn, params, bcs_batch)
            loss_res_value = loss_res(model_fn, params, res_batch)

            #TODO: Compute error

            # Print losses
            pbar.set_postfix({'Loss': loss,
                              'loss_ics': loss_ics_value,
                              'loss_bcs': loss_bcs_value,
                              'loss_physics': loss_res_value})


    return params, opt_state


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, batch_size, key):
        self.u = u
        self.y = y
        self.s = s
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = key

    def __getitem__(self, index):
        """Generate one batch of data"""
        self.key, subkey = jax.random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    # @partial(jax.jit, static_argnums=(0,))
    def __data_generation(self, key):
        """Generates data containing batch_size samples"""
        idx = jax.random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx, :]
        y = self.y[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs


# Generate ics training data corresponding to one input sample
def generate_one_ics_training_data(key, u0, m=101, P=101):
    t_0 = jnp.zeros((P, 1))
    x_0 = jnp.linspace(0, 1, P)[:, None]

    y = jnp.hstack([t_0, x_0])
    u = jnp.tile(u0, (P, 1))
    s = u0

    return u, y, s


# Generate bcs training data corresponding to one input sample
def generate_one_bcs_training_data(key, u0, m=101, P=100):
    t_bc = jax.random.uniform(key, (P, 1))
    x_bc1 = jnp.zeros((P, 1))
    x_bc2 = jnp.ones((P, 1))

    y1 = jnp.hstack([t_bc, x_bc1])  # shape = (P, 2)
    y2 = jnp.hstack([t_bc, x_bc2])  # shape = (P, 2)

    u = jnp.tile(u0, (P, 1))
    y = jnp.hstack([y1, y2])  # shape = (P, 4)
    s = jnp.zeros((P, 1))

    return u, y, s


# Generate res training data corresponding to one input sample
def generate_one_res_training_data(key, u0, m=101, P=1000):
    subkeys = jax.random.split(key, 2)

    t_res = jax.random.uniform(subkeys[0], (P, 1))
    x_res = jax.random.uniform(subkeys[1], (P, 1))

    u = jnp.tile(u0, (P, 1))
    y = jnp.hstack([t_res, x_res])
    s = jnp.zeros((P, 1))

    return u, y, s


# Generate test data corresponding to one input sample
def generate_one_test_data(idx, usol, m=101, P=101):
    u = usol[idx]
    u0 = u[0, :]

    t = jnp.linspace(0, 1, P)
    x = jnp.linspace(0, 1, P)
    T, X = jnp.meshgrid(t, x)

    s = u.T.flatten()
    u = jnp.tile(u0, (P ** 2, 1))
    y = jnp.hstack([T.flatten()[:, None], X.flatten()[:, None]])

    return u, y, s


if __name__ == '__main__':
    # Prepare the training data
    # Load data
    path = os.path.join(os.getcwd(), 'data/burgers/Burger.mat')  # Please use the matlab script to generate data

    data = scipy.io.loadmat(path)
    usol = jnp.array(data['output'])

    N = usol.shape[0]  # number of total input samples
    N_train = 1000  # number of input samples used for training
    N_test = N - N_train  # number of input samples used for test
    m = 101  # number of sensors for input samples
    P_ics_train = 101  # number of locations for evaluating the initial condition
    P_bcs_train = 100  # number of locations for evaluating the boundary condition
    P_res_train = 2500  # number of locations for evaluating the PDE residual
    P_test = 101  # resolution of uniform grid for the test data

    # parse command line arguments
    args_in = argparse.Namespace(num_outputs=1,
                                 hidden_dim=100,
                                 stacked_do=False,
                                 branch_layers=128,
                                 n_sensors=m,
                                 branch_input_features=1,
                                 split_branch=False,
                                 trunk_layers=128,
                                 trunk_input_features=2,
                                 split_trunk=False,
                                 seed=1337,
                                 lr=1e-3,
                                 epochs=10000,
                                 result_dir='./result/burgers',
                                 log_iter=100
                                 )


    u0_train = usol[:N_train, 0, :]  # input samples

    key1 = jax.random.split(jax.random.PRNGKey(1337), 3)

    key = jax.random.PRNGKey(0)  # use different key for generating test data
    keys = jax.random.split(key, N_train)

    # Generate training data for initial condition
    u_ics_train, y_ics_train, s_ics_train = jax.vmap(generate_one_ics_training_data, in_axes=(0, 0, None, None))(keys,
                                                                                                                 u0_train,
                                                                                                                 m,
                                                                                                                 P_ics_train)

    u_ics_train = u_ics_train.reshape(N_train * P_ics_train, -1)
    y_ics_train = y_ics_train.reshape(N_train * P_ics_train, -1)
    s_ics_train = s_ics_train.reshape(N_train * P_ics_train, -1)

    # Generate training data for boundary condition
    u_bcs_train, y_bcs_train, s_bcs_train = jax.vmap(generate_one_bcs_training_data, in_axes=(0, 0, None, None))(keys,
                                                                                                                 u0_train,
                                                                                                                 m,
                                                                                                                 P_bcs_train)

    u_bcs_train = u_bcs_train.reshape(N_train * P_bcs_train, -1)
    y_bcs_train = y_bcs_train.reshape(N_train * P_bcs_train, -1)
    s_bcs_train = s_bcs_train.reshape(N_train * P_bcs_train, -1)

    # Generate training data for PDE residual
    u_res_train, y_res_train, s_res_train = jax.vmap(generate_one_res_training_data, in_axes=(0, 0, None, None))(keys,
                                                                                                                 u0_train,
                                                                                                                 m,
                                                                                                                 P_res_train)

    u_res_train = u_res_train.reshape(N_train * P_res_train, -1)
    y_res_train = y_res_train.reshape(N_train * P_res_train, -1)
    s_res_train = s_res_train.reshape(N_train * P_res_train, -1)

    # Create data set
    batch_size = 100
    ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size, key1[0])

    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size, key1[1])
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size, key1[2])

    args_in, model, model_fn, params = setup_deeponet(args_in, key)

    # Define optimizer with optax (ADAM)
    optimizer = optax.adam(learning_rate=args_in.lr)
    opt_state = optimizer.init(params)

    params, opt_state = train(model_fn, params, opt_state, ics_dataset, bcs_dataset, res_dataset, epochs=20000)