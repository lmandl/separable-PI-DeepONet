import jax
import jax.numpy as jnp
from torch.utils import data
from functools import partial
import tqdm
import time
import optax
import scipy.io
import os
import argparse
import matplotlib.pyplot as plt
import shutil
import orbax.checkpoint as ocp

from models import setup_deeponet, mse, mse_single, step, hvp_fwdfwd
from models import apply_net_sep as apply_net


# Data Generator
class DataGenerator(data.Dataset):
    def __init__(self, u, c, t, x, y, s, batch_size, gen_key):
        self.u = u
        self.c = c
        self.t = t
        self.x = x
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
        c = self.c
        t = self.t
        x = self.x
        y = self.y
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, (t, x, y, c))
        outputs = s
        return inputs, outputs


def loss_bcic(model_fn, params, data_batch):
    inputs, outputs = data_batch
    u, y_vec = inputs

    # Compute forward pass
    t = y_vec[0]
    x = y_vec[1]
    y = y_vec[2]
    c = y_vec[3]
    s_pred = apply_net(model_fn, params, u, t, x, y, c)

    # Compute loss
    loss_val = mse(outputs.flatten(), s_pred.flatten())
    return loss_val


# Define residual loss
def loss_res(model_fn, params, batch):
    # Fetch data
    inputs, _ = batch
    u, y_vec = inputs
    t, x, y, c = y_vec

    v_t = jnp.ones(t.shape)
    v_x = jnp.ones(x.shape)
    v_y = jnp.ones(y.shape)

    s_t = jax.jvp(lambda t_i: apply_net(model_fn, params, u, t_i, x, y, c), (t,), (v_t,))[1]
    _, s_xx = hvp_fwdfwd(lambda x_i: apply_net(model_fn, params, u, t, x_i, y, c), (x,), (v_x,), True)
    _, s_yy = hvp_fwdfwd(lambda y_i: apply_net(model_fn, params, u, t, x, y_i, c), (y,), (v_y,), True)

    pred = s_t - jnp.square(c) * (s_xx + s_yy)
    # Compute loss
    loss = mse_single(pred.flatten())

    return loss


def loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch):
    loss_ics_i = loss_bcic(model_fn, params, ics_batch)
    loss_bcs_i = loss_bcic(model_fn, params, bcs_batch)
    loss_res_i = loss_res(model_fn, params, res_batch)
    loss_value = 20.0 * loss_ics_i + loss_bcs_i + loss_res_i
    return loss_value


def generate_one_test_data(u_sol, c_ref, idx, p_err=101):

    u = u_sol[idx]
    u0 = u[0, :, :]

    c = c_ref[idx]

    t = jnp.linspace(0, 1, p_err).reshape(p_err, 1)
    x = jnp.linspace(0, 1, p_err).reshape(p_err, 1)
    y = jnp.linspace(0, 1, p_err).reshape(p_err, 1)

    y_vec = jnp.hstack([t, x, y, c])

    s = u.T.flatten()
    u = u0.reshape(1, -1)

    return u, y_vec, s


def get_error(model_fn, params, u_sol, c_ref, idx, p_err=101, return_data=False):
    u_test, y_test_vec, s_test, c_ref = generate_one_test_data(u_sol, c_ref, idx, p_err)

    t_test = y_test_vec[:, 0].reshape(-1, 1)
    x_test = y_test_vec[:, 1].reshape(-1, 1)
    y_test = y_test_vec[:, 2].reshape(-1, 1)

    s_pred = apply_net(model_fn, params, u_test, t_test, x_test, y_test, c)

    s_pred = s_pred[0]  # first example as only one branch input

    s_pred = s_pred.reshape((-1,), order='F')
    error = jnp.linalg.norm(s_test - s_pred) / jnp.linalg.norm(s_test)
    if return_data:
        return error, s_pred
    else:
        return error


def visualize(args, model_fn, params, result_dir, epoch, usol, c_ref, idx, test=False):

    # Generate data, and obtain error
    error_s, s_pred = get_error(model_fn, params, usol, idx, args.p_test, return_data=True)

    u = usol[idx]

    t = jnp.linspace(0, 1, args.p_test)
    x = jnp.linspace(0, 1, args.p_test)
    y = jnp.linspace(0, 1, args.p_test)

    # Reshape s_pred
    s_pred = s_pred.reshape(t.shape[0], x.shape[0], y.shape[0])

    fig = plt.figure(figsize=(18, 18))

    # Plt row: central over x
    # plt row: central over y

    # t slice start, middle, end