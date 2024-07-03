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
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil
import orbax.checkpoint as ocp

from models import setup_deeponet, step, mse
from models import apply_net_sep as apply_net


def convert_to_jax_arrays_res(res_batches):
    branch_in = []
    x_coords = []
    y_coords = []
    app_disp = []

    for data in res_batches:
        branch, (x, y, disp) = data
        branch_in.append(branch)
        x_coords.append(x)
        y_coords.append(y)
        app_disp.append(disp)

    return (
        jnp.array(branch_in),
        jnp.array(x_coords),
        jnp.array(y_coords),
        jnp.array(app_disp),
    )

def convert_to_jax_arrays_data(data_batches):
    branch_in = []
    x_coords = []
    y_coords = []
    app_disp = []
    u_ref = []
    v_ref = []
    phi_ref = []
    hist_ref = []
    f_phi = []

    for data in data_batches:
        (branch, (x, y, disp)), (u, v, phi, hist), f = data
        branch_in.append(branch)
        x_coords.append(x)
        y_coords.append(y)
        app_disp.append(disp)
        u_ref.append(u)
        v_ref.append(v)
        phi_ref.append(phi)
        hist_ref.append(hist if hist is not None else jnp.zeros_like(u))  # Handle None with zeros
        f_phi.append(f)

    return (
        jnp.array(branch_in),
        jnp.array(x_coords),
        jnp.array(y_coords),
        jnp.array(app_disp),
        jnp.array(u_ref),
        jnp.array(v_ref),
        jnp.array(phi_ref),
        jnp.array(hist_ref),
        jnp.array(f_phi),
    )


# Data Generator
class ResBatchGenerator(data.Dataset):
    def __init__(self, res_batches, batch_size, key):
        self.res_batches = res_batches
        self.N = len(res_batches)
        self.batch_size = batch_size
        self.key = key

        # Convert res_batches components to JAX arrays
        self.branch_in, self.x_coords, self.y_coords, self.app_disp = convert_to_jax_arrays_res(res_batches)

    def __getitem__(self, idx):
        self.key, subkey = jax.random.split(self.key)
        idx = jax.random.choice(subkey, self.N, (self.batch_size,), replace=False)

        batch = self.__data_generation(idx)

        return batch

    def __data_generation(self, idx):
        batch = (
            self.branch_in[idx],
            self.x_coords[idx], self.y_coords[idx], self.app_disp[idx]
        )
        return batch

class DataBatchGenerator(data.Dataset):
    def __init__(self, data_batches, batch_size, key):
        self.data_batches = data_batches
        self.N = len(data_batches)
        self.batch_size = batch_size
        self.key = key

        # Convert data_batches components to JAX arrays
        (self.branch_in, self.x_coords, self.y_coords, self.app_disp, self.u_ref, self.v_ref,
         self.phi_ref, self.hist_ref, self.f_phi) = convert_to_jax_arrays_data(data_batches)

    def __getitem__(self, idx):
        self.key, subkey = jax.random.split(self.key)
        idx = jax.random.choice(subkey, self.N, (self.batch_size,), replace=False)

        batch = self.__data_generation(idx)

        return batch

    def __data_generation(self, idx):
        batch = (
            self.branch_in[idx],
            self.x_coords[idx], self.y_coords[idx], self.app_disp[idx],
            self.u_ref[idx], self.v_ref[idx], self.phi_ref[idx], self.hist_ref[idx],
            self.f_phi[idx]
        )
        return batch


@partial(jax.jit, static_argnums=(0,))
def fraction_apply_net(model_fn, params, v_in, *x_in):

    if v_in.shape[0] != 1:
        raise ValueError('Needs single batch for branch and last trunk input')

    y_out = apply_net(model_fn, params, v_in, *x_in)

    # Since the input batch to the branch is 1 as is the trunk input for delta u, we need to reshape the output
    # i.e. [162, 162, 1, 3] -> [162, 162, 3]
    y_out = jnp.reshape(y_out, (y_out.shape[1], y_out.shape[2], y_out.shape[4]))

    u_lift = x_in[0] * y_out[:, :, 0]
    v_lift = x_in[1] * (x_in[1] - 1) * y_out[:, :, 1] + x_in[1] * x_in[2]
    phi_lift = y_out[:, :, 2]

    y_final = jnp.stack([u_lift, v_lift, phi_lift], axis=-1)

    return y_final

def visualize(damage_pred_print, damage_true_print, xDisp_pred_print, xDisp_true_print, yDisp_pred_print,
              yDisp_true_print, hist_pred, hist_ref, f_phi, branch_in,
              epoch, folder, sample_i, step_i, test=False, errors=None):

    fig = plt.figure(constrained_layout=False, figsize=(15, 9))
    gs = fig.add_gridspec(3, 5)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)
    cbformat = matplotlib.ticker.ScalarFormatter()  # create the formatter
    cbformat.set_powerlimits((-1, 1))  # set the limits for sci. not.

    if errors is not None:
        if test:
            plt.suptitle(f'test, sample {sample_i}, step: {step_i}, '
                         f'$\mathcal{{L}}_2$: {jnp.mean(errors):.3e}\n$\mathcal{{L}}_2(u)$: '
                         f'{errors[0]:.3e}, $\mathcal{{L}}_2(v)$: {errors[1]:.3e}, '
                         f'$\mathcal{{L}}_2(\phi)$: {errors[2]:.3e}')
        else:
            plt.suptitle(f'train, sample {sample_i}, step: {step_i}, '
                         f'$\mathcal{{L}}_2$: {jnp.mean(errors):.3e}\n$\mathcal{{L}}_2(u)$: '
                         f'{errors[0]:.3e}, $\mathcal{{L}}_2(v)$: {errors[1]:.3e}, '
                         f'$\mathcal{{L}}_2(\phi)$: {errors[2]:.3e}')
    else:
        if test:
            plt.suptitle(f'test, sample {sample_i}, step: {step_i}')
        else:
            plt.suptitle(f'train, sample {sample_i}, step: {step_i}')

    ax = fig.add_subplot(gs[0, 0])
    h = ax.imshow(xDisp_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(xDisp_true_print), vmax=jnp.amax(xDisp_true_print))
    ax.set_title('Pred $u$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[1, 0])
    h = ax.imshow(xDisp_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(xDisp_true_print), vmax=jnp.amax(xDisp_true_print))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('True $u$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[2, 0])
    h = ax.imshow(xDisp_pred_print - xDisp_true_print, origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(xDisp_pred_print - xDisp_true_print).max(),
                  vmax=-abs(xDisp_pred_print - xDisp_true_print).max())
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Error in $u$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[0, 1])
    h = ax.imshow(yDisp_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(yDisp_true_print), vmax=jnp.amax(yDisp_true_print))
    ax.set_title('Pred $v$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[1, 1])
    h = ax.imshow(yDisp_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(yDisp_true_print), vmax=jnp.amax(yDisp_true_print))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('True $v$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[2, 1])
    h = ax.imshow(yDisp_pred_print - yDisp_true_print, origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(yDisp_pred_print - yDisp_true_print).max(),
                  vmax=-abs(yDisp_pred_print - yDisp_true_print).max())
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Error in $v$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[0, 2])
    h = ax.imshow(damage_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(damage_true_print), vmax=jnp.amax(damage_true_print))
    ax.set_title('Pred $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[1, 2])
    h = ax.imshow(damage_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(damage_true_print), vmax=jnp.amax(damage_true_print))
    ax.set_title('True $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[2, 2])
    h = ax.imshow(damage_pred_print - damage_true_print, origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(damage_pred_print - damage_true_print).max(),
                  vmax=-abs(damage_pred_print - damage_true_print).max())
    ax.set_title('Error in $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)
    if hist_ref is not None:
        ax = fig.add_subplot(gs[0, 3])
        h = ax.imshow(hist_pred, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                      vmin=jnp.amin(hist_ref), vmax=jnp.amax(hist_ref))
        ax.set_title('Pred history')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

        ax = fig.add_subplot(gs[1, 3])
        h = ax.imshow(hist_ref, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                      vmin=jnp.amin(hist_ref), vmax=jnp.amax(hist_ref))
        ax.set_title('True history')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

        ax = fig.add_subplot(gs[2, 3])
        h = ax.imshow(hist_pred - hist_ref, origin='lower', interpolation='nearest', cmap='jet',
                      aspect=1, vmin=abs(hist_pred - hist_ref).max(),
                      vmax=-abs(hist_pred - hist_ref).max())
        ax.set_title('Error in history')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax, format=cbformat)
    else:
        ax = fig.add_subplot(gs[0, 3])
        h = ax.imshow(hist_pred, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                      vmin=jnp.amin(hist_pred), vmax=jnp.amax(hist_pred))
        ax.set_title('Pred history')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[0, 4])
    h = ax.imshow(f_phi, origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=jnp.amin(f_phi), vmax=jnp.amax(f_phi))
    ax.set_title('$f_{\phi}$')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[1, 4])
    h = ax.imshow(branch_in, origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=jnp.amin(branch_in), vmax=jnp.amax(branch_in))
    ax.set_title('branch input')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    plot_dir = os.path.join(folder, f'vis/{epoch:06d}/')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if test:
        plt.savefig(os.path.join(os.path.join(folder, plot_dir), f'test_sample_{sample_i}_step_{step_i}.png'))
    else:
        plt.savefig(os.path.join(os.path.join(folder, plot_dir), f'train_sample_{sample_i}_step_{step_i}.png'))

    plt.close(fig)


def viz_loop(model_fn, params, data_batches, epoch, folder, num_samples, num_steps, test_data=False):

    errors, pred_array, hist_array = get_errors(model_fn, params, data_batches, return_data=True, per_dim=True)

    for i in range(num_samples):
        for j in range(num_steps):
            # Get position in data_batches
            idx = i*num_steps + j

            # Get data
            inputs, ref_data, f_phi = data_batches[idx]
            branch_in, trunk_ins = inputs
            branch_in = branch_in[0, :, :, 0]

            u_ref, v_ref, phi_ref, history_ref = ref_data

            # Get the predictions
            s = pred_array[idx]
            u_out, v_out, phi_out = s[0, :, :, 0], s[0, :, :, 1], s[0, :, :, 2]

            # Get the history field
            hist = hist_array[idx][:, :, 0]

            # Prepare data for visualization
            damage_pred_print = phi_out
            damage_true_print = phi_ref
            xDisp_pred_print = u_out
            xDisp_true_print = u_ref
            yDisp_pred_print = v_out
            yDisp_true_print = v_ref

            if history_ref is not None:
                hist_ref = history_ref
            else:
                hist_ref = None

            visualize(damage_pred_print, damage_true_print, xDisp_pred_print, xDisp_true_print, yDisp_pred_print,
                      yDisp_true_print, hist, hist_ref, f_phi, branch_in, epoch, folder, i, j, test=test_data,
                      errors=errors[i*num_steps + j])


@partial(jax.jit, static_argnums=(0,))
def get_hist(model_fn, params, u_in, x, y, delta_u, hist, f_phi, mu, lmd):

    # Calculate hist from u, v, phi and current history field

    # obtain net predictions / gradients
    t_x = jnp.ones(x.shape)
    t_y = jnp.ones(y.shape)
    u_x = jax.jvp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 0], (x,), (t_x,))[1]
    u_y = jax.jvp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 0], (y,), (t_y,))[1]
    v_x = jax.jvp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 1], (x,), (t_x,))[1]
    v_y = jax.jvp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 1], (y,), (t_y,))[1]

    u_xy = u_y + v_x

    # Computing the tensile strain energy

    u_xy = 0.5 * u_xy
    M = jnp.sqrt((u_x-v_y)**2 + 4 * (u_xy**2))
    lambda1 = 0.5 * (u_x + v_y) + 0.5 * M
    lambda2 = 0.5 * (u_x + v_y) - 0.5 * M

    eigSum = (lambda1 + lambda2)
    sEnergy_pos = (0.125 * lmd * (eigSum + jnp.abs(eigSum))**2 +
                   0.25 * mu * ((lambda1 + jnp.abs(lambda1))**2 + (lambda2 + jnp.abs(lambda2))**2))

    hist_temp = jnp.maximum(f_phi, sEnergy_pos)
    hist = jnp.maximum(hist[0, :, :, 0], hist_temp)

    return hist


@partial(jax.jit, static_argnums=(0,))
def auto_regression(model_fn, params, inputs, f_phi, E=210.0*1e3, nu=0.3):

    # Calculate constants
    mu = 0.5 * E / (1 + nu)
    lmd = nu * E / ((1 - 2 * nu) * (1 + nu))

    # Use data from inputs
    branch_in, trunk_ins = inputs

    # Get the number of rows and columns
    num_rows = trunk_ins[0].shape[0]
    num_cols = trunk_ins[1].shape[0]

    # Get the number of load increases based on the number of delta u
    step_num = trunk_ins[2].shape[0]

    # Array to store the history fields
    hist_array = jnp.zeros((num_rows, num_cols, step_num))

    # Set the initial history field
    hist_array = hist_array.at[:, :, 0].set(branch_in[0, :, :, 0])

    # Array to store predictions
    pred_array = jnp.zeros((step_num, num_rows, num_cols, 3))

    # Get the trunk inputs
    x = trunk_ins[0]
    y = trunk_ins[1]

    # Loop over the number of load increases
    for i in range(step_num):
        branch_in = hist_array[:, :, i][None, :, :, None]
        delta_u = trunk_ins[2][i].reshape(-1, 1)

        # Get the predictions
        pred = fraction_apply_net(model_fn, params, branch_in, x, y, delta_u)

        # Store the predictions
        pred_array = pred_array.at[i, :, :, :].set(pred)

        # Calculate the history field
        hist = get_hist(model_fn, params, branch_in, x, y, delta_u, branch_in, f_phi, mu, lmd)

        # Store the history field
        hist_array = hist_array.at[:, :, i+1].set(hist)

    return pred_array, hist_array


@partial(jax.jit, static_argnums=(0,))
def loss_data(model_fn, params, branch_in, x_coords, y_coords, app_disp, u_ref, v_ref, phi_ref, history_ref, f_phi):

    inputs = (branch_in, (x_coords, y_coords, app_disp))

    # Calculate history field and predictions
    s, history_pred = auto_regression(model_fn, params, inputs, f_phi)

    u_out, v_out, phi_out = s[0, :, :, 0], s[0, :, :, 1], s[0, :, :, 2]

    # Calculate loss
    loss_u = mse(u_out, u_ref)
    loss_v = mse(v_out, v_ref)
    loss_phi = mse(phi_out, phi_ref)

    # Calculate the history field loss only if the reference history field is available
    if history_ref is not None:
        loss_hist = mse(history_pred, history_ref)
    else:
        loss_hist = 0.0

    loss_data_val = 1e5 * loss_u + 1e4 * loss_v + loss_phi #+ 1e-3 * loss_hist

    return loss_data_val


@partial(jax.jit, static_argnums=(0,))
def loss_res(model_fn, params, u_in, x, y, delta_u):

    # Define constants
    E = 210.0 * 1e3
    nu = 0.3
    mu = 0.5 * E / (1 + nu)
    lmd = nu * E / ((1 - 2 * nu) * (1 + nu))
    gc = 2.7
    l = 0.0625

    c11 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    c22 = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    c12 = E * nu / ((1 + nu) * (1 - 2 * nu))
    c21 = E * nu / ((1 + nu) * (1 - 2 * nu))
    c33 = 0.5 * E / (1 + nu)

    s = fraction_apply_net(model_fn, params, u_in, x, y, delta_u)
    u, v, phi = s[:, :, 0], s[:, :, 1], s[:, :, 2]

    # Define gradients
    t_x = jnp.ones(x.shape)
    t_y = jnp.ones(y.shape)
    u_x = jax.jvp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 0], (x,), (t_x,))[1]
    u_y = jax.jvp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 0], (y,), (t_y,))[1]
    v_x = jax.jvp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 1], (x,), (t_x,))[1]
    v_y = jax.jvp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 1], (y,), (t_y,))[1]
    phi_x = jax.jvp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 2], (x,), (t_x,))[1]
    phi_y = jax.jvp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 2], (y,), (t_y,))[1]

    u_xy = u_y + v_x
    nabla = phi_x ** 2 + phi_y ** 2

    M = jnp.sqrt((u_x - v_y) ** 2 + (u_xy ** 2))
    lambda1 = 0.5 * (u_x + v_y) + 0.5 * M
    lambda2 = 0.5 * (u_x + v_y) - 0.5 * M

    eigSum = (lambda1 + lambda2)
    psi_pos = 0.125 * lmd * (eigSum + jnp.abs(eigSum)) ** 2 + \
              0.25 * mu * ((lambda1 + jnp.abs(lambda1)) ** 2 + (lambda2 + jnp.abs(lambda2)) ** 2)

    hist_prev = u_in[0, :, :, 0]
    hist = jnp.maximum(hist_prev, psi_pos)
    sigmaX = c11 * u_x + c12 * v_y
    sigmaY = c21 * u_x + c22 * v_y
    tauXY = c33 * u_xy

    E_e = 0.5 * (1 - phi) ** 2 * (sigmaX * u_x + sigmaY * v_y + tauXY * u_xy)
    E_c = 0.5 * gc * (phi ** 2 / l + l * nabla) + (1 - phi) ** 2 * hist

    ## Total energy = elastic energy + fracture energy
    E_e_mean = jnp.mean(E_e)
    E_c_mean = jnp.mean(E_c)

    return E_c_mean + 10.0 * E_e_mean

@partial(jax.jit, static_argnums=(0,))
def loss_data_loop(model_fn, params, data_batches):
    branch_in, x_coords, y_coords, app_disp, u_ref, v_ref, phi_ref, hist_ref, f_phi = data_batches

    data_l = jax.vmap(loss_data, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(model_fn, params, branch_in,
                                                                                  x_coords, y_coords, app_disp, u_ref,
                                                                                  v_ref, phi_ref, hist_ref, f_phi)

    return jnp.mean(data_l)


@partial(jax.jit, static_argnums=(0,))
def loss_res_loop(model_fn, params, res_batches):
    branch_in, x_coords, y_coords, app_disp = res_batches

    res_l = jax.vmap(loss_res, in_axes=(None, None, 0, 0, 0, 0))(model_fn, params, branch_in,
                                                                    x_coords, y_coords, app_disp)

    return jnp.mean(res_l)


@partial(jax.jit, static_argnums=(0,))
def loss_fn(model_fn, params, ics_batch, data_batch, res_batch):
    loss_data_i = loss_data_loop(model_fn, params, data_batch)
    loss_res_i = loss_res_loop(model_fn, params, res_batch)
    loss_value = 1e4 * loss_data_i + loss_res_i
    return loss_value


def get_errors(model_fn, params, data_batches, return_data=False, per_dim=False):
    errors = []
    pred_array = []
    hist_array = []

    for data_batch in data_batches:
        # Get data
        inputs, ref_data, f_phi = data_batch
        u_ref, v_ref, phi_ref, history_ref = ref_data

        # Calculate history field and predictions
        s, history_pred = auto_regression(model_fn, params, inputs, f_phi)
        u_out, v_out, phi_out = s[0, :, :, 0], s[0, :, :, 1], s[0, :, :, 2]

        # Store predictions and history fields
        pred_array.append(s)
        hist_array.append(history_pred)

        # Calculate errors
        if per_dim:
            error_u = jnp.linalg.norm(u_ref - u_out) / jnp.linalg.norm(u_ref)
            error_v = jnp.linalg.norm(v_ref - v_out) / jnp.linalg.norm(v_ref)
            error_phi = jnp.linalg.norm(phi_ref - phi_out) / jnp.linalg.norm(phi_ref)
            errors.append(jnp.array([error_u, error_v, error_phi]))
        else:
            stack_pred = jnp.stack([u_out, v_out, phi_out], axis=-1)
            stack_ref = jnp.stack([u_ref, v_ref, phi_ref], axis=-1)
            error = jnp.linalg.norm(stack_ref - stack_pred) / jnp.linalg.norm(stack_ref)
            errors.append(error)
    if return_data:
        return errors, pred_array, hist_array
    else:
        return errors



def main_routine(args):
    # Check if separable network is used
    if not args.separable:
        raise ValueError('Needs separable DeepONet for separable example')

    # Trunk Networks: x, y, \delta u
    # Branch network H_(t-1)
    # During training, these history fields will be obtained from the labelled datasets
    # During testing these will be auto-regressively updated
    # The DeepONet will be trained to predict u, v, \phi
    # H in testing will be calculated from the predicted u, v, \phi

    # Problem settings
    num_cols = 162
    num_rows = 162
    # test_samples defines number of samples used for testing
    test_samples = 3

    # Load training data
    path_dataset = os.path.join(os.getcwd(), 'data/fracture/TensileFailure.mat')
    dataset_all = scipy.io.loadmat(path_dataset)
    coords = dataset_all['coordinates']

    f_phi = dataset_all['f_phi']  # initial condition
    disp_x = dataset_all['dispX']  # displacement in x, network will predict this
    disp_y = dataset_all['dispY']  # displacement in y, network will predict this
    phi = dataset_all['phi']  # damage field, network will predict this
    app_disp = dataset_all['app_disp'][0]
    history = dataset_all['history']

    x_coord = coords[:num_cols, 0].reshape(-1, 1)
    y_coord = coords[::num_cols, 1].reshape(-1, 1)

    step_num = len(app_disp)
    samp_num = f_phi.shape[0]
    train_samples = samp_num - test_samples

    num_batches = train_samples

    u_all = jnp.zeros((samp_num, num_rows, num_cols, step_num))
    v_all = jnp.zeros((samp_num, num_rows, num_cols, step_num))
    phi_all = jnp.zeros((samp_num, num_rows, num_cols, step_num))
    f_phi_all = jnp.zeros((samp_num, num_rows, num_cols))
    history_all = jnp.zeros((samp_num, num_rows, num_cols, step_num))

    for i in range(samp_num):
        for j in range(step_num):
            u_all = u_all.at[i, :, :, j].set(disp_x[i, :, j].reshape(num_rows, num_cols))
            v_all = v_all.at[i, :, :, j].set(disp_y[i, :, j].reshape(num_rows, num_cols))
            phi_all = phi_all.at[i, :, :, j].set(phi[i, :, j].reshape(num_rows, num_cols))
            history_all = history_all.at[i, :, :, j].set(history[i, :, j].reshape(num_rows, num_cols))
        f_phi_all = f_phi_all.at[i, :, :].set(f_phi[i, :].reshape(num_rows, num_cols))

    # Split key for data, residual, and model init
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)

    # Create and shuffle indices
    indices = jnp.arange(samp_num)
    indices = jax.random.permutation(keys[0], indices)

    # Split indices into training and testing
    train_indices = indices[:train_samples]
    test_indices = indices[train_samples:]

    # Prepare batches
    data_batches = []  # (branch_in, (x, y, delta_u), (u_ref, v_ref, phi_ref, hist_ref), f_phi)) ref for single step
    res_batches = []  # (branch_in, (x, y, delta_u))
    test_batches = []  # ((branch_in, (x, y, delta_u)), (u_ref, v_ref, phi_ref, hist_ref), f_phi)) ref for all steps

    # Prepare training data
    for i in train_indices:
        for j in range(step_num):
            branch_in = history_all[i, :, :, j][None, :, :, None]
            res_batches.append((branch_in, (x_coord, y_coord, app_disp[j].reshape(-1,1))))
            # Last time step has no reference history data and will bet set to None
            if j != step_num - 1:
                data_batches.append(((branch_in, (x_coord, y_coord, app_disp[j].reshape(-1,1))),
                                     (u_all[i, :, :, j], v_all[i, :, :, j],
                                      phi_all[i, :, :, j], history_all[i, :, :, j+1]),
                                     f_phi_all[i, :, :]))
            else:
                data_batches.append(((branch_in, (x_coord, y_coord, app_disp[j].reshape(-1,1))),
                                     (u_all[i, :, :, j], v_all[i, :, :, j],
                                      phi_all[i, :, :, j], None),
                                     f_phi_all[i, :, :]))

    # Prepare testing data
    for i in test_indices:
        branch_in = history_all[i, :, :, 0][None, :, :, None]
        for j in range(step_num):
            if j != step_num - 1:
                test_batches.append(((branch_in, (x_coord, y_coord, app_disp[j].reshape(-1, 1))),
                                     (u_all[i, :, :, j], v_all[i, :, :, j],
                                      phi_all[i, :, :, j], history_all[i, :, :, j+1]),
                                     f_phi_all[i, :, :]))
            else:
                test_batches.append(((branch_in, (x_coord, y_coord, app_disp[j].reshape(-1,1))),
                                     (u_all[i, :, :, j], v_all[i, :, :, j],
                                      phi_all[i, :, :, j], None),
                                     f_phi_all[i, :, :]))

    # Create data generators
    data_dataset = DataBatchGenerator(data_batches, batch_size=num_batches, key=keys[1])
    res_dataset = ResBatchGenerator(res_batches, batch_size=num_batches, key=keys[2])

    # Data
    data_data = iter(data_dataset)
    res_data = iter(res_dataset)

    # Create model
    args, model, model_fn, params = setup_deeponet(args, keys[3])

    # Define optimizer with optax (ADAM)
    # optimizer
    if args.lr_scheduler == 'exponential_decay':
        lr_scheduler = optax.exponential_decay(args.lr, args.lr_schedule_steps, args.lr_decay_rate)
    elif args.lr_scheduler == 'constant':
        lr_scheduler = optax.constant_schedule(args.lr)
    else:
        raise ValueError(f"learning rate scheduler {args.lr_scheduler} not implemented.")
    optimizer = optax.adam(learning_rate=lr_scheduler)
    opt_state = optimizer.init(params)

    # create dir for saving results
    result_dir = os.path.join(os.getcwd(), os.path.join(args.result_dir, f'{time.strftime("%Y%m%d-%H%M%S")}'))
    log_file = os.path.join(result_dir, 'log.csv')
    # Create directory
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if os.path.exists(os.path.join(result_dir, 'vis')):
        shutil.rmtree(os.path.join(result_dir, 'vis'))
    if os.path.exists(log_file):
        os.remove(log_file)

    # Set up checkpointing
    if args.checkpoint_iter > 0:
        options = ocp.CheckpointManagerOptions(max_to_keep=args.checkpoints_to_keep,
                                               save_interval_steps=args.checkpoint_iter,
                                               save_on_steps=[args.epochs]  # save on last iteration
                                               )
        mngr = ocp.CheckpointManager(
            os.path.join(result_dir, 'ckpt'), options=options, item_names=('metadata', 'params')
        )

    # Restore checkpoint if available
    if args.checkpoint_path is not None:
        # Restore checkpoint
        restore_mngr = ocp.CheckpointManager(args.checkpoint_path, item_names=('metadata', 'params'))
        ckpt = restore_mngr.restore(
            restore_mngr.latest_step(),
            args=ocp.args.Composite(
                metadata=ocp.args.JsonRestore(),
                params=ocp.args.StandardRestore(params),
            ),
        )
        # Extract restored params
        params = ckpt.params
        offset_epoch = ckpt.metadata['total_iterations']  # define offset for logging
    else:
        offset_epoch = 0

    # Save arguments
    with open(os.path.join(result_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    with open(log_file, 'a') as f:
        f.write('epoch,loss,loss_data_value,loss_res_value,err_val,runtime\n')

    # First visualization
    if args.vis_iter > 0:
        # Visualize train example
        viz_loop(model_fn, params, data_batches, offset_epoch, result_dir, train_samples, step_num, test_data=False)
        # Visualize test example
        viz_loop(model_fn, params, test_batches, offset_epoch, result_dir, test_samples, step_num, test_data=True)

    # Iterations
    pbar = tqdm.trange(args.epochs)

    # Training loop
    for it in pbar:

        if it == 1:
            # start timer and exclude first iteration (compile time)
            start = time.time()

        # Fetch data
        data_batch = next(data_data)
        res_batch = next(res_data)

        # Do Step
        loss, params, opt_state = step(optimizer, loss_fn, model_fn, opt_state,
                                       params, None, data_batch, res_batch)

        if it % args.log_iter == 0:
            # Compute losses (Note that this is for the current batch)
            loss_data_value = loss_data_loop(model_fn, params, data_batch)
            loss_res_value = loss_res_loop(model_fn, params, res_batch)

            # compute error over test data
            errors = get_errors(model_fn, params, test_batches, return_data=False, per_dim=False)

            err_val = jnp.mean(jnp.array(errors))

            # Print losses
            pbar.set_postfix({'l': f'{loss:.2e}',
                              'l_data': f'{loss_data_value:.2e}',
                              'l_r': f'{loss_res_value:.2e}',
                              'e': f'{err_val:.2e}'})

            # get runtime
            if it == 0:
                runtime = 0
            else:
                runtime = time.time() - start

            # Save results
            with open(log_file, 'a') as f:
                f.write(f'{it + 1 + offset_epoch}, {loss}, {loss_data_value}, '
                        f'{loss_res_value}, {err_val}, {runtime}\n')

        # Visualize result
        if (it + 1) % args.vis_iter == 0 and args.vis_iter > 0:
            # Visualize train example
            viz_loop(model_fn, params, data_batches, offset_epoch + it + 1, result_dir,
                     train_samples, step_num, test_data=False)
            # Visualize test example
            viz_loop(model_fn, params, test_batches, offset_epoch + it + 1, result_dir,
                     test_samples, step_num, test_data=True)

        # Save checkpoint
        mngr.save(
            it + 1 + offset_epoch,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(params),
                metadata=ocp.args.JsonSave({'total_iterations': it + 1}),
            ),
        )

    mngr.wait_until_finished()

    # Save results
    runtime = time.time() - start
    with open(log_file, 'a') as f:
        f.write(f'{it + 1 + offset_epoch}, {loss}, {loss_data_value}, '
                f'{loss_res_value}, {err_val}, {runtime}\n')


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--num_outputs', type=int, default=3, help='number of outputs')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')
    parser.add_argument('--separable', dest='separable', default=True, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=128, help='hidden tensor dimension in separable DeepONets')

    # Branch settings
    parser.add_argument('--branch_cnn', dest='branch_cnn', default=True, action='store_true',)
    parser.add_argument('--branch_cnn_blocks', nargs="+", action='append',
                        default=[[16, 3, 3, "relu"], ["max_pool", 2, 2, 2, 2], [32, 3, 3, "relu"],
                                 ["max_pool", 2, 2, 2, 2], [256, "relu"], [100, "tanh"]],
                        help='branch cnn blocks, list of length 4 are Conv2D blocks (features, kernel_size_1, '
                             'kernel_size 2, ac_fun); list of length 5 are Pool2D blocks (pool_type, kernel_size_1, '
                             'kernel_size_2, stride_1, stride_2); list of length 2 are Dense blocks (features, ac_fun);'
                             ' flatten will be performed before first dense layer; '
                             'no conv/pool block after first Dense allowed')
    parser.add_argument('--branch_cnn_input_channels', type=int, default=1,
                        help='hidden branch layer sizes')
    parser.add_argument('--branch_cnn_input_size', type=int, nargs="+", default=[162, 162],
                        help='number of sensors for branch network, also called >>m<<')
    parser.add_argument('--split_branch', dest='split_branch', default=False, action='store_true',
                        help='split branch outputs into n groups for n outputs')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=[50, 50, 50, 50, 50, 50],
                        help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=3,
                        help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=True, action='store_true',
                        help='split trunk outputs into j groups for j outputs')

    # Training settings
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100000, help='training epochs')
    parser.add_argument('--lr_scheduler', type=str,
                        default='exponential_decay', choices=['constant', 'exponential_decay'],
                        help='learning rate scheduler')
    parser.add_argument('--lr_schedule_steps', type=int, default=5000, help='decay steps for lr scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='decay rate for lr scheduler')

    # result directory
    parser.add_argument('--result_dir', type=str, default='results/fracture/separable/',
                        help='a directory to save results, relative to cwd')

    # log settings
    parser.add_argument('--log_iter', type=int, default=100, help='iteration to save loss and error')
    parser.add_argument('--vis_iter', type=int, default=2500, help='iteration to save visualization')

    # Checkpoint settings
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint file for restoring, uses latest checkpoint')
    parser.add_argument('--checkpoint_iter', type=int, default=2500,
                        help='iteration of checkpoint file')
    parser.add_argument('--checkpoints_to_keep', type=int, default=1,
                        help='number of checkpoints to keep')

    args_in = parser.parse_args()

    main_routine(args_in)