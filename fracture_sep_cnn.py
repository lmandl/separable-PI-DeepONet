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
import matplotlib
import matplotlib.pyplot as plt
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
        inputs, outputs = self.__data_generation(index)
        return inputs, outputs

    #@partial(jax.jit, static_argnums=(0,))
    def __data_generation(self, index):
        """Generates data containing one sample"""
        example = self.data[index]
        outputs = example[1]
        inputs = example[0]
        return inputs, outputs


def fraction_apply_net(model_fn, params, v_in, *x_in):
    # TODO: Maybe some input scaling is needed

    if v_in.shape[0] != 1:
        raise ValueError('Needs single batch for branch and last trunk input')

    y_out = apply_net(model_fn, params, v_in, *x_in)

    # Since the input batch to the branch is 1 as is the trunk input for delta u, we need to reshape the output
    # i.e. [162, 162, 1, 3] -> [162, 162, 3]
    y_out = jnp.reshape(y_out, (y_out.shape[1], y_out.shape[2], y_out.shape[4]))

    # Transpose each output
    #y_out = jnp.stack([y_out[:, :, 0].T, y_out[:, :, 1].T, y_out[:, :, 2].T], axis=-1)

    u_lift = x_in[0] * y_out[:, :, 0]
    v_lift = x_in[1] * (x_in[1] - 1) * y_out[:, :, 1] + x_in[1] * x_in[2] # Check if this is correct
    phi_lift = y_out[:, :, 2]

    y_final = jnp.stack([u_lift, v_lift, phi_lift], axis=-1)

    return y_final


def loss_data(model_fn, params, ics_batch):
    inputs, outputs = ics_batch
    u_in, y = inputs

    u_ref, v_ref, phi_ref = outputs

    s = fraction_apply_net(model_fn, params, u_in, *y)
    u_out, v_out, phi_out = s[:, :, 0], s[:, :, 1], s[:, :, 2]

    loss_u = mse(u_out, u_ref)
    loss_v = mse(v_out, v_ref)
    loss_phi = mse(phi_out, phi_ref)

    loss_data_val = 1e5 * loss_u + 1e4 * loss_v + loss_phi

    return loss_data_val


# Define residual loss
def loss_res(model_fn, params, res_batch):
    # Note: vDelta is u in reference code
    # Note: x in reference code contains x & y & t coordinates
    # here: trunk_in
    # Note: Similar as in get_hist

    # Fetch data
    inputs, _ = res_batch
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

    hist_prev = u_in[0, :, :, 0] # TODO: Check if this is correct
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
              yDisp_true_print, epoch, folder, idx, test=False, errors=None):

    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(3, 3)

    if errors is not None:
        if test:
            plt.suptitle(f'test_{idx}, L2: {jnp.mean(errors):.3e}, L2_u: '
                         f'{errors[0]:.3e}, L2_v: {errors[1]:.3e}, L2_phi: {errors[2]:.3e}')
        else:
            plt.suptitle(f'train_{idx}, L2: {jnp.mean(errors):.3e}, L2_u: '
                         f'{errors[0]:.3e}, L2_v: {errors[1]:.3e}, L2_phi: {errors[2]:.3e}')
    else:
        if test:
            plt.suptitle(f'test_{idx}')
        else:
            plt.suptitle(f'train_{idx}')

    ax = fig.add_subplot(gs[0, 2])
    h = ax.imshow(damage_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(damage_true_print), vmax=jnp.amax(damage_true_print))
    ax.set_title('Pred $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[1, 2])
    h = ax.imshow(damage_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(damage_true_print), vmax=jnp.amax(damage_true_print))
    ax.set_title('True $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[2, 2])
    h = ax.imshow(abs(damage_pred_print - damage_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(damage_pred_print - damage_true_print).max(),
                  vmax=-abs(damage_pred_print - damage_true_print).max())
    ax.set_title('Error in $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[0, 1])
    h = ax.imshow(yDisp_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(yDisp_true_print), vmax=jnp.amax(yDisp_true_print))
    ax.set_title('Pred $v$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[1, 1])
    h = ax.imshow(yDisp_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(yDisp_true_print), vmax=jnp.amax(yDisp_true_print))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('True $v$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[2, 1])
    h = ax.imshow(abs(yDisp_pred_print - yDisp_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(yDisp_pred_print - yDisp_true_print).max(),
                  vmax=-abs(yDisp_pred_print - yDisp_true_print).max())
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Error in $v$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[0, 0])
    h = ax.imshow(xDisp_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(xDisp_true_print), vmax=jnp.amax(xDisp_true_print))
    ax.set_title('Pred $u$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[1, 0])
    h = ax.imshow(xDisp_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1,
                  vmin=jnp.amin(xDisp_true_print), vmax=jnp.amax(xDisp_true_print))
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('True $u$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[2, 0])
    h = ax.imshow(abs(xDisp_pred_print - xDisp_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1, vmin=abs(xDisp_pred_print - xDisp_true_print).max(),
                  vmax=-abs(xDisp_pred_print - xDisp_true_print).max())
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Error in $u$(x)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    if test:
        plot_dir = os.path.join(folder, f'vis/{epoch:06d}/test_{idx}/')
    else:
        plot_dir = os.path.join(folder, f'vis/{epoch:06d}/train_{idx}/')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig.tight_layout()

    plt.savefig(os.path.join(os.path.join(folder, plot_dir), 'pred.png'))
    plt.close(fig)


def viz_loop(model_fn, params, data_batch, u_ref, v_ref, phi_ref, n_t, test_flag, crackTip, epoch, result_dir):

    if test_flag:

        errors, pred_array, history = get_errors(model_fn, params, data_batch, u_ref, v_ref, phi_ref,
                                                 train=False, return_data=True, per_dim=True, crackTip=crackTip)

        for i in range(0, n_t):
            u_pred, v_pred, phi_pred = pred_array[i][:, :, 0], pred_array[i][:, :, 1], pred_array[i][:, :, 2]
            u_ref_i, v_ref_i, phi_ref_i = u_ref[:, :, i], v_ref[:, :, i], phi_ref[:, :, i]
            visualize(phi_pred, phi_ref_i, u_pred, u_ref_i, v_pred, v_ref_i, epoch, result_dir, i,
                      test_flag, errors[i,:])

    else:

        errors, pred_array, history = get_errors(model_fn, params, data_batch, u_ref, v_ref, phi_ref, train=True,
                                                 return_data=True, per_dim=True, crackTip=None)
        for i in range(0, n_t):
            inputs, _ = data_batch[i]
            u_in, trunk_in = inputs
            x, y, delta_u = trunk_in

            u_pred, v_pred, phi_pred = pred_array[i][:, :, 0], pred_array[i][:, :, 1], pred_array[i][:, :, 2]
            u_ref_i, v_ref_i, phi_ref_i = u_ref[:, :, i], v_ref[:, :, i], phi_ref[:, :, i]
            visualize(phi_pred, phi_ref_i, u_pred, u_ref_i, v_pred, v_ref_i, epoch, result_dir, i,
                      test_flag, errors[i,:])


def get_hist(model_fn, params, u_in, x, y, delta_u, hist, crackTip, mu, lmd, gc, l, B):

    # Calculate hist from u, v, phi and current history field
    init_hist = get_init_hist(x, y, crackTip, l, B, gc)

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

    hist_temp = jnp.maximum(init_hist, sEnergy_pos)
    hist = jnp.maximum(hist, hist_temp)

    return hist

def get_init_hist(x, y, crackTip, l, B, cEnerg):

    xx, yy = jnp.meshgrid(x.flatten(), y.flatten())
    init_hist = jnp.zeros((xx.shape[0], xx.shape[1]))
    dist = jnp.where(xx > crackTip, jnp.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2), jnp.abs(yy - 0.5))
    init_hist = jnp.where(dist < 0.5 * l, B * cEnerg * 0.5 * (1 - (2 * dist / l)) / l, init_hist)

    return init_hist


def auto_regression(model_fn, params, trunk_ins, crackTip):

    # Define constants
    E = 210.0 * 1e3
    nu = 0.3
    mu = 0.5 * E / (1 + nu)
    lmd = nu * E / ((1 - 2 * nu) * (1 + nu))
    gc = 2.7
    l = 0.0625
    B = 92

    # Get the number of rows and columns
    num_rows = trunk_ins[0][0].shape[0]
    num_cols = trunk_ins[0][1].shape[0]

    # Get the number of load increases based on the number of delta u
    n_t = trunk_ins[0][2].shape[0]

    # Array to store the history fields
    hist_array = jnp.zeros((num_rows, num_cols, n_t))

    # Set the initial history field
    H_0 = get_init_hist(trunk_ins[0][0], trunk_ins[0][1], crackTip, l, B, gc)
    hist_array = hist_array.at[:, :, 0].set(H_0)

    # Array to store predictions
    pred_array = jnp.zeros((n_t, num_rows, num_cols, 3))

    for i in range(0, n_t-1):
        u_in = jnp.zeros((1, num_rows, num_cols, 3))  # 1 at the front for batch size needed for branch
        u_in = u_in.at[0, :, :, 0].set(hist_array[:, :, i])
        if i - 1 >= 0:
            u_in = u_in.at[0, :, :, 1].set(hist_array[:, :, i-1])
        if i - 2 >= 0:
            u_in = u_in.at[0, :, :, 2].set(hist_array[:, :, i-2])

        # Get the trunk inputs
        x = trunk_ins[i][0]
        y = trunk_ins[i][1]
        delta_u = trunk_ins[i][2]

        # Get the prediction
        y_pred = fraction_apply_net(model_fn, params, u_in, x, y, delta_u)

        # Store the prediction
        pred_array = pred_array.at[i, :, :, :].set(y_pred)

        # Calculate the history field
        current_h = u_in[0, :, :, 0]
        hist = get_hist(model_fn, params, u_in, x, y, delta_u, current_h, crackTip, mu, lmd, gc, l, B)

        hist_array = hist_array.at[:, :, i+1].set(hist)

    return pred_array, hist_array


def get_errors(model_fn, params, data_batch, u_ref, v_ref, phi_ref,
               train = False, return_data = False, per_dim = False, crackTip=0.5):

    if not train:

        # Get the number of load increases based on the number of delta u
        n_t = len(data_batch)

        # Auto-regression
        pred_array, hist_array = auto_regression(model_fn, params, data_batch, crackTip)

        errors = []

        for i in range(0, n_t):
            u_pred, v_pred, phi_pred = pred_array[i][:, :, 0], pred_array[i][:, :, 1], pred_array[i][:, :, 2]
            u_ref_i, v_ref_i, phi_ref_i = u_ref[:, :, i], v_ref[:, :, i], phi_ref[:, :, i]
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

        errors = []
        x_shape = u_ref.shape[0]
        y_shape = u_ref.shape[1]
        pred_array = jnp.zeros((len(data_batch), x_shape, y_shape, 3))
        hist_array = []
        for i, data in enumerate(data_batch):
            inputs, _ = data
            u_in, trunk_in = inputs
            x, y, delta_u = trunk_in
            s = fraction_apply_net(model_fn, params, u_in, x, y, delta_u)
            pred_array = pred_array.at[i, :, :, :].set(s)
            u, v, phi = s[:, :, 0], s[:, :, 1], s[:, :, 2]
            u_ref_i, v_ref_i, phi_ref_i = u_ref[:, :, i], v_ref[:, :, i], phi_ref[:, :, i]
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
        return jnp.array(errors), pred_array, hist_array
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
    num_load = 7
    num_cols = 162
    num_rows = 162

    # Load training data
    path_dataset = os.path.join(os.getcwd(), 'data/fracture/dataset1.mat')
    dataset_all = scipy.io.loadmat(path_dataset)
    x = dataset_all['cordinates']
    phi = dataset_all['phi']
    u = dataset_all['u']
    v = dataset_all['v']
    stress_train = dataset_all['stress']

    path_dataset_hist = os.path.join(os.getcwd(), 'data/fracture/history1.mat')
    dataset_hist = scipy.io.loadmat(path_dataset_hist)
    hist = dataset_hist['hist']

    x_cord = x[:num_cols, 0].reshape(-1,1)
    y_cord = x[::num_cols, 1].reshape(-1,1)

    u_train = jnp.zeros((num_rows, num_cols, num_load))
    v_train = jnp.zeros((num_rows, num_cols, num_load))
    phi_train = jnp.zeros((num_rows, num_cols, num_load))

    for i in range(num_load):
        u_train = u_train.at[:, :, i].set(u[:, i].reshape(num_rows, num_cols))
        v_train = v_train.at[:, :, i].set(v[:, i].reshape(num_rows, num_cols))
        phi_train = phi_train.at[:, :, i].set(phi[:, i].reshape(num_rows, num_cols))

    # Full separable approach is not possible for this as history fields and stress have to be combined
    # For this example, we rectify this by using batches over stress in the trunk and history in the branch
    # This is not a full separable approach, but shows the variability for this example

    # Prepare training batches
    data_batches = []
    res_batches = []

    for i in range(num_load):
        branch_input = jnp.zeros((1, num_rows, num_cols, 3)) #1 at the front for batch size needed for branch
        branch_input = branch_input.at[0, :, :, 0].set(hist[:, i].reshape(num_rows, num_cols))
        if i-1 >= 0:
            branch_input = branch_input.at[0, :, :, 1].set(hist[:, i-1].reshape(num_rows, num_cols))
        if i-2 >= 0:
            branch_input = branch_input.at[0, :, :, 2].set(hist[:, i-2].reshape(num_rows, num_cols))
        res_batches.append(((branch_input, [x_cord, y_cord, stress_train[i].reshape(-1,1)]), []))
        data_batches.append(((branch_input, [x_cord, y_cord, stress_train[i].reshape(-1,1)]),
                             (u_train[:, :, i], v_train[:, :, i], phi_train[:, :, i])))

    # Prepare test data
    path_dataset_test = os.path.join(os.getcwd(), 'data/fracture/dataset5.mat')
    dataset_test = scipy.io.loadmat(path_dataset_test)
    x_test = dataset_test['cordinates']
    phi = dataset_test['phi']
    u = dataset_test['u']
    v = dataset_test['v']
    stress_test = dataset_test['stress']
    x_cord_test = x_test[:num_cols, 0].reshape(-1,1)
    y_cord_test = x_test[::num_cols, 1].reshape(-1,1)
    num_load_test = 7

    crackTip = 0.5 # Crack tip location TODO: Check if this is correct / where do i get the location from?

    u_test = jnp.zeros((num_rows, num_cols, num_load_test))
    v_test = jnp.zeros((num_rows, num_cols, num_load_test))
    phi_test = jnp.zeros((num_rows, num_cols, num_load_test))

    for i in range(num_load_test):
        u_test = u_train.at[:, :, i].set(u[:, i].reshape(num_rows, num_cols))
        v_test = v_train.at[:, :, i].set(v[:, i].reshape(num_rows, num_cols))
        phi_test = phi_train.at[:, :, i].set(phi[:, i].reshape(num_rows, num_cols))

    #path_dataset_hist_test = os.path.join(os.getcwd(), 'data/fracture/history5.mat')

    test_batches = []
    for i in range(num_load):
        test_batches.append([x_cord_test, y_cord_test, stress_test[i].reshape(-1, 1)])

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
        viz_loop(model_fn, params, res_batches, u_train, v_train, phi_train, num_load, False, 0,
                 offset_epoch, result_dir)
        # Visualize test example
        viz_loop(model_fn, params, test_batches, u_test, v_test, phi_test, num_load_test, True, crackTip,
                 offset_epoch, result_dir)

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
            # Compute losses
            loss_data_value = loss_data(model_fn, params, data_batch)
            loss_res_value = loss_res(model_fn, params, res_batch)

            # compute error over test data
            errors = get_errors(model_fn, params, test_batches, u_test, v_test, phi_test, train=False,
                                return_data=False, per_dim=False, crackTip=crackTip)
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
            viz_loop(model_fn, params, res_batches, u_train, v_train, phi_train, num_load, False, 0,
                     offset_epoch+it+1, result_dir)
            # Visualize test example
            viz_loop(model_fn, params, test_batches, u_test, v_test, phi_test, num_load_test, True, crackTip,
                     offset_epoch+it+1, result_dir)

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
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')
    parser.add_argument('--separable', dest='separable', default=True, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=32, help='hidden tensor dimension in separable DeepONets')

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
    parser.add_argument('--branch_cnn_input_channels', type=int, default=3,
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
    parser.add_argument('--lr_scheduler', type=str, default='constant', choices=['constant', 'exponential_decay'],
                        help='learning rate scheduler')
    parser.add_argument('--lr_schedule_steps', type=int, default=1000, help='decay steps for lr scheduler')
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