import jax
import jax.numpy as jnp
from torch.utils import data
from functools import partial
import itertools
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


# Data Generator
class DataGenerator(data.Dataset):
    def __init__(self, data):
        self.data = data
        self.N = len(data)

    def __getitem__(self, index):
        """Generate one example of data"""
        example = self.__data_generation(index)
        return example

    def __data_generation(self, index):
        """Generates data containing one sample"""
        example = self.data[index]
        return example


def fraction_apply_net(model_fn, params, v_in, *x_in):

    if v_in.shape[0] != 1:
        raise ValueError('Needs single batch for branch and last trunk input')

    y_out = apply_net(model_fn, params, v_in, *x_in)

    # Since the input batch to the branch is 1 as is the trunk input for delta u, we need to reshape the output
    # i.e. [162, 162, 1, 3] -> [162, 162, 3]
    y_out = jnp.reshape(y_out, (y_out.shape[1], y_out.shape[2], y_out.shape[4]))

    u_lift = x_in[0] * y_out[:, :, 0]
    v_lift = x_in[1] * (x_in[1] - 1) * y_out[:, :, 1] + x_in[1] * x_in[2] # Check if this is correct
    phi_lift = y_out[:, :, 2]

    y_final = jnp.stack([u_lift, v_lift, phi_lift], axis=-1)

    return y_final


def loss_data(model_fn, params, data_batch):
    inputs, outputs, f_phi = data_batch
    u_in, y = inputs

    u_ref, v_ref, phi_ref, history_ref = outputs

    # Calculate history field and predictions
    s, history_pred = auto_regression(model_fn, params, inputs, f_phi)

    u_out, v_out, phi_out = s[0, :, :, 0], s[0, :, :, 1], s[0, :, :, 2]

    # Calculate loss
    loss_u = mse(u_out, u_ref)
    loss_v = mse(v_out, v_ref)
    loss_phi = mse(phi_out, phi_ref)

    loss_hist = mse(history_pred, history_ref)

    loss_data_val = 1e5 * loss_u + 1e4 * loss_v + loss_phi + 1e-2 * loss_hist

    return loss_data_val


# Define residual loss
def loss_res(model_fn, params, res_batch):
    # Note: vDelta is u in reference code
    # Note: x in reference code contains x & y & t coordinates
    # here: trunk_in
    # Note: Similar as in get_hist

    # Fetch data
    inputs, _, _ = res_batch
    u_in, trunk_in = inputs

    x = trunk_in[0]
    y = trunk_in[1]
    delta_u = trunk_in[2]

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

    return E_c_mean + 10.0*E_e_mean


def loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch):
    loss_data_i = loss_data(model_fn, params, bcs_batch)
    loss_res_i = loss_res(model_fn, params, res_batch)
    loss_value = 1e4*loss_data_i + loss_res_i
    return loss_value


def visualize(damage_pred_print, damage_true_print, xDisp_pred_print, xDisp_true_print, yDisp_pred_print,
              yDisp_true_print, epoch, folder, sample_i, step_i, test=False, errors=None):

    fig = plt.figure(constrained_layout=False, figsize=(9,9))
    gs = fig.add_gridspec(3, 3)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)
    cbformat = matplotlib.ticker.ScalarFormatter()  # create the formatter
    cbformat.set_powerlimits((-1, 1))  # set the limits for sci. not.

    if errors is not None:
        if test:
            plt.suptitle(f'test, sample {sample_i}, step: {step_i}, L2: {jnp.mean(errors):.3e}\nL2_u: '
                         f'{errors[0]:.3e}, L2_v: {errors[1]:.3e}, L2_phi: {errors[2]:.3e}')
        else:
            plt.suptitle(f'train, sample {sample_i}, step: {step_i}, L2: {jnp.mean(errors):.3e}\nL2_u: '
                         f'{errors[0]:.3e}, L2_v: {errors[1]:.3e}, L2_phi: {errors[2]:.3e}')
    else:
        if test:
            plt.suptitle(f'test, sample {sample_i}, step: {step_i}')
        else:
            plt.suptitle(f'train, sample {sample_i}, step: {step_i}')

    ax = fig.add_subplot(gs[0, 2])
    h = ax.imshow(damage_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  )#vmin=jnp.amin(damage_true_print), vmax=jnp.amax(damage_true_print))
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
    h = ax.imshow(abs(damage_pred_print - damage_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(damage_pred_print - damage_true_print).max(),
                  vmax=-abs(damage_pred_print - damage_true_print).max())
    ax.set_title('Error in $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[0, 1])
    h = ax.imshow(yDisp_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  )#vmin=jnp.amin(yDisp_true_print), vmax=jnp.amax(yDisp_true_print))
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
    h = ax.imshow(abs(yDisp_pred_print - yDisp_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(yDisp_pred_print - yDisp_true_print).max(),
                  vmax=-abs(yDisp_pred_print - yDisp_true_print).max())
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Error in $v$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax, format=cbformat)

    ax = fig.add_subplot(gs[0, 0])
    h = ax.imshow(xDisp_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  )#vmin=jnp.amin(xDisp_true_print), vmax=jnp.amax(xDisp_true_print))
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
    h = ax.imshow(abs(xDisp_pred_print - xDisp_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(xDisp_pred_print - xDisp_true_print).max(),
                  vmax=-abs(xDisp_pred_print - xDisp_true_print).max())
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Error in $u$(x)')
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


def viz_loop(model_fn, params, data_batch, ref_sol, test_flag, epoch, result_dir):

    step_num = ref_sol[0].shape[-1]
    samp_num = ref_sol[0].shape[0]

    if test_flag:

        errors, pred_array, history = get_errors(model_fn, params, data_batch, ref_sol,
                                                 train=False, return_data=True, per_dim=True)

        for samp_i in range(samp_num):
            for step_i in range(step_num):
                i = samp_i * step_num + step_i
                u_pred, v_pred, phi_pred = (pred_array[samp_i, :, :, step_i, 0],
                                            pred_array[samp_i, :, :, step_i, 1],
                                            pred_array[samp_i, :, :, step_i, 2])
                u_ref_i, v_ref_i, phi_ref_i = (ref_sol[0][samp_i, :, :, step_i],
                                               ref_sol[1][samp_i, :, :, step_i],
                                               ref_sol[2][samp_i, :, :, step_i])
                visualize(phi_pred, phi_ref_i, u_pred, u_ref_i, v_pred, v_ref_i, epoch, result_dir, samp_i, step_i,
                          test_flag, errors[i, :])

    else:

        errors, pred_array, history = get_errors(model_fn, params, data_batch, ref_sol,
                                                 train=True, return_data=True, per_dim=True)
        for samp_i in range(samp_num):
            for step_i in range(step_num):
                i = samp_i * step_num + step_i
                u_pred, v_pred, phi_pred = (pred_array[i, :, :, 0],
                                            pred_array[i, :, :, 1],
                                            pred_array[i, :, :, 2])
                u_ref_i, v_ref_i, phi_ref_i = (ref_sol[0][samp_i, :, :, step_i],
                                               ref_sol[1][samp_i, :, :, step_i],
                                               ref_sol[2][samp_i, :, :, step_i])
                visualize(phi_pred, phi_ref_i, u_pred, u_ref_i, v_pred, v_ref_i, epoch, result_dir, samp_i, step_i,
                          test_flag, errors[i, :])


def get_hist(model_fn, params, u_in, x, y, delta_u, hist, f_phi, mu, lmd, gc, l, B):

    # Calculate hist from u, v, phi and current history field

    # obtain net predictions / gradients
    t_x = jnp.ones(x.shape)
    t_y = jnp.ones(y.shape)
    u_x = jax.jvp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 0], (x,), (t_x,))[1]
    u_y = jax.jvp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 0], (y,), (t_y,))[1]
    v_x = jax.jvp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 1], (x,), (t_x,))[1]
    v_y = jax.jvp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, delta_u)[:, :, 1], (y,), (t_y,))[1]

    u_xy = u_y + v_x

    #Computing the tensile strain energy
    u_xy = 0.5*u_xy
    M = jnp.sqrt((u_x-v_y)**2 + 4*(u_xy**2))
    lambda1 = 0.5*(u_x + v_y) + 0.5*M
    lambda2 = 0.5*(u_x + v_y) - 0.5*M

    eigSum = (lambda1 + lambda2)
    sEnergy_pos = (0.125*lmd * (eigSum + jnp.abs(eigSum))**2 +
                   0.25*mu*((lambda1 + jnp.abs(lambda1))**2 + (lambda2 + jnp.abs(lambda2))**2))

    hist_temp = jnp.maximum(f_phi, sEnergy_pos)
    hist = jnp.maximum(hist, hist_temp)

    return hist

# Check Jit possibility
def auto_regression(model_fn, params, data_batch, f_phi):

    # Define constants
    E = 210.0 * 1e3
    nu = 0.3
    mu = 0.5 * E / (1 + nu)
    lmd = nu * E / ((1 - 2 * nu) * (1 + nu))
    gc = 2.7
    l = 0.0625
    B = 92

    branch_in, trunk_ins = data_batch

    # Get the number of rows and columns
    num_rows = trunk_ins[0].shape[0]
    num_cols = trunk_ins[1].shape[0]

    # Get the number of load increases based on the number of delta u
    n_t = trunk_ins[2].shape[0]

    # Array to store the history fields
    hist_array = jnp.zeros((num_rows, num_cols, n_t))

    # Set the initial history field
    hist_array = hist_array.at[:, :, 0].set(branch_in[0, :, :, 0])

    # Array to store predictions
    pred_array = jnp.zeros((n_t, num_rows, num_cols, 3))

    # Get the trunk inputs
    x = trunk_ins[0]
    y = trunk_ins[1]

    for i in range(0, n_t):
        u_in = jnp.zeros((1, num_rows, num_cols, 1))  # 1 at the front for batch size needed for branch
        u_in = u_in.at[0, :, :, 0].set(hist_array[:, :, i])
        delta_u = trunk_ins[2][i].reshape(-1, 1)

        # Get the prediction
        y_pred = fraction_apply_net(model_fn, params, u_in, x, y, delta_u)

        # Store the prediction
        pred_array = pred_array.at[i, :, :, :].set(y_pred)

        # Calculate the history field
        current_h = u_in[0, :, :, 0]
        hist = get_hist(model_fn, params, u_in, x, y, delta_u, current_h, f_phi, mu, lmd, gc, l, B)

        hist_array = hist_array.at[:, :, i+1].set(hist)

    return pred_array, hist_array


def get_errors(model_fn, params, data_batch, ref_sol, train=False, return_data=False, per_dim=False):

    # Get number of samples and number of load increases from the reference solution
    samp_num = ref_sol[0].shape[0]
    step_num = ref_sol[0].shape[-1]

    u_ref, v_ref, phi_ref = ref_sol

    errors = []
    x_shape = u_ref.shape[1]
    y_shape = u_ref.shape[2]

    if not train:
        # Test Error
        pred_array_all = jnp.zeros((samp_num, x_shape, y_shape, step_num, 3))
        hist_array_all = jnp.zeros((samp_num, x_shape, y_shape, step_num))

        for samp_i in range(samp_num):

            # use the current sample
            data_i, f_phi_i = data_batch[samp_i]

            # Auto-regression
            pred_array, hist_array = auto_regression(model_fn, params, data_i, f_phi_i)
            hist_array_all.at[samp_i, :, :, :].set(hist_array)


            for step_i in range(step_num):
                # Store the predictions
                pred_array_all = pred_array_all.at[samp_i, :, :, step_i, 0].set(pred_array[step_i, :, :, 0])
                pred_array_all = pred_array_all.at[samp_i, :, :, step_i, 1].set(pred_array[step_i, :, :, 1])
                pred_array_all = pred_array_all.at[samp_i, :, :, step_i, 2].set(pred_array[step_i, :, :, 2])

                u_pred, v_pred, phi_pred = (pred_array[step_i, :, :, 0],
                                            pred_array[step_i, :, :, 1],
                                            pred_array[step_i, :, :, 2])
                u_ref_i, v_ref_i, phi_ref_i = (u_ref[samp_i, :, :, step_i],
                                               v_ref[samp_i, :, :, step_i],
                                               phi_ref[samp_i, :, :, step_i])
                if per_dim:
                    error_u = jnp.linalg.norm(u_ref_i - u_pred) / jnp.linalg.norm(u_ref_i)
                    error_v = jnp.linalg.norm(v_ref_i - v_pred) / jnp.linalg.norm(v_ref_i)
                    error_phi = jnp.linalg.norm(phi_ref_i - phi_pred) / jnp.linalg.norm(phi_ref_i)
                    errors.append([error_u, error_v, error_phi])
                else:
                    stack_pred = jnp.stack([u_pred, v_pred, phi_pred], axis=-1)
                    stack_ref = jnp.stack([u_ref_i, v_ref_i, phi_ref_i], axis=-1)
                    error = jnp.linalg.norm(stack_ref - stack_pred) / jnp.linalg.norm(stack_ref)
                    errors.append(error)
    else:
        # Train Error
        pred_array_all = jnp.zeros((len(data_batch), x_shape, y_shape, 3))
        hist_array_all = []
        for samp_i in range(samp_num):
            for step_i in range(step_num):
                i = samp_i * step_num + step_i
                inputs, _, _ = data_batch[i]
                u_in, trunk_in = inputs
                x, y, delta_u = trunk_in
                s = fraction_apply_net(model_fn, params, u_in, x, y, delta_u)
                pred_array_all = pred_array_all.at[i, :, :, :].set(s)
                u, v, phi = s[:, :, 0], s[:, :, 1], s[:, :, 2]
                u_ref_i, v_ref_i, phi_ref_i = (u_ref[samp_i, :, :, step_i],
                                               v_ref[samp_i, :, :, step_i],
                                               phi_ref[samp_i, :, :, step_i])
                if per_dim:
                    error_u = jnp.linalg.norm(u_ref_i - u) / jnp.linalg.norm(u_ref_i)
                    error_v = jnp.linalg.norm(v_ref_i - v) / jnp.linalg.norm(v_ref_i)
                    error_phi = jnp.linalg.norm(phi_ref_i - phi) / jnp.linalg.norm(phi_ref_i)
                    errors.append([error_u, error_v, error_phi])
                else:
                    stack_pred = jnp.stack([u, v, phi], axis=-1)
                    stack_ref = jnp.stack([u_ref_i, v_ref_i, phi_ref_i], axis=-1)
                    error = jnp.linalg.norm(stack_ref - stack_pred) / jnp.linalg.norm(stack_ref)
                    errors.append(error)

    if return_data:
        return jnp.array(errors), pred_array_all, hist_array_all
    else:
        return jnp.array(errors)


def main_routine(args):
    # Check if separable network is used
    if not args.separable:
        raise ValueError('Needs separable DeepONet for separable example')

    # Trunk Networks: x, y, \delta u
    # Branch network [H_(t-1), H_(t-2), H_(t-3)]
    # For the first \delta u, the branch network will take [H_0, 0, 0]
    # For the second \delta u, the branch network will take [H_1, H_0, 0]
    # For the third \delta u, the branch network will take [H_2, H_1, H_0]
    # For the t-th \delta u, the branch network will take [H_(t-1), H_(t-2), H_(t-3)]
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

    # Full separable approach is not possible for this as history fields and stress have to be combined
    # For this example, we rectify this by using batches over stress in the trunk and history in the branch
    # This is not a full separable approach, but shows the variability for this example

    # Prepare training batches
    data_batches = []
    res_batches = []
    test_batches = []

    # Prepare training and test data
    for i in range(samp_num):
        if i < samp_num - test_samples:
            for j in range(step_num):
                branch_input = jnp.zeros((1, num_rows, num_cols, 1))  # 1 at the front for batch size needed for branch
                branch_input = branch_input.at[0, :, :, 0].set(history[i, :, j].reshape(num_rows, num_cols))

                res_batches.append(((branch_input, [x_coord, y_coord, app_disp[j].reshape(-1, 1)]), [], []))
                data_batches.append(((branch_input, [x_coord, y_coord, app_disp[j].reshape(-1, 1)]),
                                     (u_all[i, :, :, j], v_all[i, :, :, j],
                                      phi_all[i, :, :, j], history_all[i, :, :, j]),
                                     (f_phi_all[i, :, :])))
        else:
            branch_input = jnp.zeros((1, num_rows, num_cols, 1))  # 1 at the front for batch size needed for branch
            branch_input = branch_input.at[0, :, :, 0].set(history[i, :, 0].reshape(num_rows, num_cols))
            test_batches.append(((branch_input, [x_coord, y_coord, app_disp.reshape(-1, 1)]), f_phi_all[i, :, :]))

    # Reference data for training and testing
    train_ref = [u_all[:samp_num - test_samples, :, :, :],
                 v_all[:samp_num - test_samples, :, :, :],
                 phi_all[:samp_num - test_samples, :, :]]
    test_ref = [u_all[samp_num - test_samples:, :, :, :],
                v_all[samp_num - test_samples:, :, :, :],
                phi_all[samp_num - test_samples:, :, :]]

    # Create data generators
    data_dataset = DataGenerator(data_batches)
    res_dataset = DataGenerator(res_batches)

    # Data
    data_data = itertools.cycle(data_dataset)
    res_data = itertools.cycle(res_dataset)

    # Split key for IC, BC, Residual data, and model init
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 7)

    # Create model
    args, model, model_fn, params = setup_deeponet(args, keys[6])

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
        viz_loop(model_fn, params, res_batches, train_ref, False, offset_epoch, result_dir)
        # Visualize test example
        viz_loop(model_fn, params, test_batches, test_ref, True, offset_epoch, result_dir)

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
            loss_data_value = loss_data(model_fn, params, data_batch)
            loss_res_value = loss_res(model_fn, params, res_batch)

            # compute error over test data
            errors = get_errors(model_fn, params, test_batches, test_ref, train=False,
                                return_data=False, per_dim=False)

            err_val = jnp.mean(errors)

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
                f.write(f'{it+1+offset_epoch}, {loss}, {loss_data_value}, '
                        f'{loss_res_value}, {err_val}, {runtime}\n')

        # Visualize result
        if (it+1) % args.vis_iter == 0 and args.vis_iter > 0:
            # Visualize train example
            viz_loop(model_fn, params, res_batches, train_ref, False,  offset_epoch+it+1, result_dir)
            # Visualize test example
            viz_loop(model_fn, params, test_batches, test_ref, True, offset_epoch+it+1, result_dir)

        # Save checkpoint
        mngr.save(
            it+1+offset_epoch,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(params),
                metadata=ocp.args.JsonSave({'total_iterations': it+1}),
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
                        default=[[32, 3, 3, "relu"], ["max_pool", 2, 2, 2, 2], [64, 3, 3, "relu"],
                                 ["avg_pool", 2, 2, 2, 2], [256, "relu"], [32, "tanh"]],
                        help='branch cnn blocks, list of length 4 are Conv2D blocks (features, kernel_size_1, '
                             'kernel_size 2, ac_fun); list of length 5 are Pool2D blocks (pool_type, kernel_size_1, '
                             'kernel_size_2, stride_1, stride_2); list of length 2 are Dense blocks (features, ac_fun);'
                             ' flatten will be performed before first dense layer; '
                             'no conv/pool block after first Dense allowed')
    parser.add_argument('--branch_cnn_input_channels', type=int, default=1,
                        help='hidden branch layer sizes')
    parser.add_argument('--branch_cnn_input_size', type=int, nargs="+", default=[162, 162],
                        help='number of sensors for branch network, also called >>m<<')
    parser.add_argument('--split_branch', dest='split_branch', default=True, action='store_true',
                        help='split branch outputs into n groups for n outputs')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=[50, 50, 50, 50, 50],
                        help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=3,
                        help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=True, action='store_true',
                        help='split trunk outputs into j groups for j outputs')

    # Training settings
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200000, help='training epochs')
    parser.add_argument('--lr_scheduler', type=str, default='exponential_decay', choices=['constant', 'exponential_decay'],
                        help='learning rate scheduler')
    parser.add_argument('--lr_schedule_steps', type=int, default=2000, help='decay steps for lr scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for lr scheduler')

    # result directory
    parser.add_argument('--result_dir', type=str, default='results/fracture/separable/',
                        help='a directory to save results, relative to cwd')

    # log settings
    parser.add_argument('--log_iter', type=int, default=1000, help='iteration to save loss and error')
    parser.add_argument('--vis_iter', type=int, default=10000, help='iteration to save visualization')

    # Checkpoint settings
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint file for restoring, uses latest checkpoint')
    parser.add_argument('--checkpoint_iter', type=int, default=5000,
                        help='iteration of checkpoint file')
    parser.add_argument('--checkpoints_to_keep', type=int, default=5,
                        help='number of checkpoints to keep')

    args_in = parser.parse_args()

    main_routine(args_in)