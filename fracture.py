import jax
import jax.numpy as jnp
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

from models import setup_deeponet, apply_net, step, mse

matplotlib.use('Agg')


def fraction_apply_net(model_fn, params, v_in, *x_in):
    if len(x_in) == 1:
        x_in = x_in[0]
    else:
        x_in = jnp.stack(x_in, axis=-1)

    """"
    lb_elem_br = jnp.min(v_in, axis=0)
    ub_elem_br = jnp.max(v_in, axis=0)

    # Note: doesn't work with ub_elem_br == lb_elem_br
    v_in = 2.0 * (v_in - lb_elem_br) / (ub_elem_br - lb_elem_br) - 1.0

    lb_elem_tr = jnp.min(x_in, axis=0)
    ub_elem_tr = jnp.max(x_in, axis=0)

    x_in = 2.0 * (x_in - lb_elem_tr) / (ub_elem_tr - lb_elem_tr) - 1.0
    """

    y_out = apply_net(model_fn, params, v_in, x_in)

    u_lift = x_in[:, 0] * y_out[:, 0]
    v_lift = x_in[:, 1] * (x_in[:, 1] - 1) * y_out[:, 1] + x_in[:, 1] * v_in[:, 0]
    phi_lift = y_out[:, 2]

    y_final = jnp.concatenate([u_lift[:, None], v_lift[:, None], phi_lift[:, None]], axis=1)

    return y_final

def loss_data(model_fn, params, ics_batch):
    inputs, outputs = ics_batch
    u_in, y = inputs

    u_ref, v_ref, phi_ref = outputs

    s = fraction_apply_net(model_fn, params, u_in, y)
    u_out, v_out, phi_out = s[:, 0], s[:, 1], s[:, 2]

    loss_u = mse(u_out, u_ref[:, 0])
    loss_v = mse(v_out, v_ref[:, 0])
    loss_phi = mse(phi_out, phi_ref[:, 0])

    loss_data_val = 1e5 * loss_u + 1e4 * loss_v + loss_phi

    return loss_data_val


# Define residual loss
def loss_res(model_fn, params, res_batch):
    # Note: vDelta is u in reference code
    # Note: x in reference code contains x & y & t coordinates
    # here: trunk_in
    # Fetch data
    inputs, _ = res_batch
    u_in, trunk_in = inputs

    x = trunk_in[:, 0]
    y = trunk_in[:, 1]
    t = trunk_in[:, 2]

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

    s = fraction_apply_net(model_fn, params, u_in, x, y, t)
    u, v, phi = s[:, 0], s[:, 1], s[:, 2]

    # Define gradients
    v_x = jnp.ones(x.shape)
    v_y = jnp.ones(y.shape)
    u_x = jax.vjp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, t)[:, 0], x)[1](v_x)[0]
    u_y = jax.vjp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, t)[:, 0], x)[1](v_y)[0]
    v_x = jax.vjp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, t)[:, 1], x)[1](v_x)[0]
    v_y = jax.vjp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, t)[:, 1], x)[1](v_y)[0]
    phi_x = jax.vjp(lambda x: fraction_apply_net(model_fn, params, u_in, x, y, t)[:, 2], x)[1](v_x)[0]
    phi_y = jax.vjp(lambda y: fraction_apply_net(model_fn, params, u_in, x, y, t)[:, 2], x)[1](v_y)[0]

    u_xy = u_y + v_x
    nabla = phi_x ** 2 + phi_y ** 2

    M = jnp.sqrt((u_x - v_y) ** 2 + (u_xy ** 2))
    lambda1 = 0.5 * (u_x + v_y) + 0.5 * M
    lambda2 = 0.5 * (u_x + v_y) - 0.5 * M

    eigSum = (lambda1 + lambda2)
    psi_pos = 0.125 * lmd * (eigSum + jnp.abs(eigSum)) ** 2 + \
              0.25 * mu * ((lambda1 + jnp.abs(lambda1)) ** 2 + (lambda2 + jnp.abs(lambda2)) ** 2)

    hist_prev = trunk_in[:, 2]
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
              yDisp_true_print, epoch, folder, idx, test=False):
    fig = plt.figure(constrained_layout=False, figsize=(6, 7))
    gs = fig.add_gridspec(3, 3)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.3)

    ax = fig.add_subplot(gs[0, 2])
    h = ax.imshow(damage_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1)
    ax.set_title('Pred $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    h.set_clim(vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[1, 2])
    h = ax.imshow(damage_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1)
    ax.set_title('True $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    h.set_clim(vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[2, 2])
    h = ax.imshow(abs(damage_pred_print - damage_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1)
    ax.set_title('Error in $\phi$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    # h.set_clim(vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[0, 1])
    h = ax.imshow(yDisp_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1)
    ax.set_title('Pred $v$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    h.set_clim(vmin=-0.001, vmax=0.006)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[1, 1])
    h = ax.imshow(yDisp_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('True $v$(x)')
    h.set_clim(vmin=-0.001, vmax=0.006)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[2, 1])
    h = ax.imshow(abs(yDisp_pred_print - yDisp_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Error in $v$(x)')
    h.set_clim(vmin=0, vmax=0.002)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[0, 0])
    h = ax.imshow(xDisp_pred_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1)
    ax.set_title('Pred $u$(x)')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    h.set_clim(vmin=-0.001, vmax=0.006)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[1, 0])
    h = ax.imshow(xDisp_true_print, origin='lower', interpolation='nearest', cmap='jet', aspect=1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('True $u$(x)')
    h.set_clim(vmin=-0.001, vmax=0.006)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    ax = fig.add_subplot(gs[2, 0])
    h = ax.imshow(abs(xDisp_pred_print - xDisp_true_print), origin='lower', interpolation='nearest', cmap='jet',
                  aspect=1)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_title('Error in $u$(x)')
    h.set_clim(vmin=0, vmax=0.002)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, ax=ax, cax=cax)

    if test:
        plot_dir = os.path.join(folder, f'vis/{epoch:06d}/test_{idx}/')
    else:
        plot_dir = os.path.join(folder, f'vis/{epoch:06d}/train_{idx}/')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.savefig(os.path.join(os.path.join(folder, plot_dir), 'pred.png'))
    plt.close(fig)


def viz_loop(model_fn, params, zipped, num_rows, num_cols, offset_epoch, result_dir):

    for (v_data, x_data, u_ref, v_ref, phi_ref, flag, n_t) in zipped:
        y_pred = model_fn(params, v_data, x_data)
        u_pred, v_pred, phi_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
        # flag == True for test data
        if not flag:
            u_true = u_ref.reshape(n_t, -1)
            v_true = v_ref.reshape(n_t, -1)
            phi_true = phi_ref.reshape(n_t, -1)

        u_pred = u_pred.reshape(n_t, -1)
        v_pred = v_pred.reshape(n_t, -1)
        phi_pred = phi_pred.reshape(n_t, -1)

        for i in range(0, n_t):
            damage_pred = phi_pred[i, :].reshape(num_rows, num_cols)
            damage_true = phi_true[i, :].reshape(num_rows, num_cols)
            xDisp_pred = u_pred[i, :].reshape(num_rows, num_cols)
            xDisp_true = u_true[i, :].reshape(num_rows, num_cols)
            yDisp_pred = v_pred[i, :].reshape(num_rows, num_cols)
            yDisp_true = v_true[i, :].reshape(num_rows, num_cols)

            visualize(damage_pred, damage_true, xDisp_pred, xDisp_true, yDisp_pred,
                      yDisp_true, offset_epoch, result_dir, i, flag)

def main_routine(args):
    if args.separable:
        raise ValueError('Needs normal DeepONet, not separable DeepONet')

    # Load and Prepare the training data
    # Load data
    num_cases = 7  # number of cases
    num_train = num_cases
    num_test = num_cases
    num_cols = 162
    num_rows = 162
    path_dataset = os.path.join(os.getcwd(), 'data/fracture/dataset1.mat')

    dataset_all = scipy.io.loadmat(path_dataset)
    x = dataset_all['cordinates']
    phi = dataset_all['phi']
    u = dataset_all['u']
    v = dataset_all['v']
    stress_train = dataset_all['stress'].T

    path_dataset_hist = os.path.join(os.getcwd(), 'data/fracture/history1.mat')
    dataset_hist = scipy.io.loadmat(path_dataset_hist)
    hist = dataset_hist['hist']
    tmp_length = x.shape[0]
    num_gap = 1
    num_sample = int(tmp_length/num_gap)
    id_sample = jnp.arange(0, num_sample)

    x_new = jnp.zeros((num_sample*num_cases, 6))
    appDisp_new = jnp.zeros((num_sample*num_cases, 1))

    for i in range(0, num_cases):
        x_new = x_new.at[i*num_sample:(i+1)*num_sample,0:2].set(x[id_sample, 0:2])
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 2:3].set(hist[id_sample, i:i+1])
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 3:4].set(u[id_sample, i:i+1])
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 4:5].set(v[id_sample, i:i+1])
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 5:6].set(phi[id_sample, i:i+1])

        appDisp_new = appDisp_new.at[i*num_sample:(i+1)*num_sample, :].set(stress_train[:, i])

    x_train = x_new[:, 0:3]
    u_train = x_new[:, 3:4]
    v_train = x_new[:, 4:5]
    phi_train = x_new[:, 5:6]

    vDelta_train = appDisp_new[:, 0:1]

    # Fetch data
    data_batch = ((vDelta_train, x_train), (u_train, v_train, phi_train))
    res_batch = ((vDelta_train, x_train), [])

    # Load and Prepare the test data
    path_dataset_test = os.path.join(os.getcwd(), 'data/fracture/dataset5.mat')
    dataset = scipy.io.loadmat(path_dataset_test)
    x = dataset['cordinates']
    phi_test = dataset['phi']
    u_test = dataset['u']
    v_test = dataset['v']
    stress_test = dataset['stress'].T

    path_dataset_hist_test = os.path.join(os.getcwd(), 'data/fracture/history5.mat')
    dataset_hist = scipy.io.loadmat(path_dataset_hist_test)
    hist = dataset_hist['hist']

    x_new = jnp.zeros((num_sample * num_cases, 6))
    appDisp_new = jnp.zeros((num_sample * num_cases, 1))

    for i in range(0, num_cases):
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 0:2].set(x[id_sample, 0:2])
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 2:3].set(hist[id_sample, i:i + 1])
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 3:4].set(u_test[id_sample, i:i+1])
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 4:5].set(v_test[id_sample, i:i+1])
        x_new = x_new.at[i * num_sample:(i + 1) * num_sample, 5:6].set(phi_test[id_sample, i:i+1])

        appDisp_new = appDisp_new.at[i * num_sample:(i + 1) * num_sample, :].set(stress_test[:, i])

    x_test = x_new[:, 0:3]
    u_test = x_new[:, 3:4]
    v_test = x_new[:, 4:5]
    phi_test = x_new[:, 5:6]
    vDelta_test = appDisp_new

    err_ref = jnp.concat([u_test, v_test, phi_test], axis=1)

    # Split key for IC, BC, Residual data, and model init
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 7)

    # Create model
    args, model, model_fn, params = setup_deeponet(args, keys[6])

    # Define optimizer with optax (ADAM)
    optimizer = optax.adam(learning_rate=args.lr)
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

        # Visualize train and test example
        zipped = zip([vDelta_train, vDelta_test], [x_train, x_test], [u_train, u_test],
                     [v_train, v_test], [phi_train, phi_test], [False, True], [num_train, num_test])

        viz_loop(model_fn, params, zipped, num_rows, num_cols, offset_epoch, result_dir)

    # Iterations
    pbar = tqdm.trange(args.epochs)

    # Training loop
    for it in pbar:

        if it == 1:
            # start timer and exclude first iteration (compile time)
            start = time.time()

        # Do Step
        loss, params, opt_state = step(optimizer, loss_fn, model_fn, opt_state,
                                       params, None, data_batch, res_batch)

        if it % args.log_iter == 0:
            # Compute losses
            loss_data_value = loss_data(model_fn, params, data_batch)
            loss_res_value = loss_res(model_fn, params, res_batch)

            # compute error over test data
            err_pred = model_fn(params, vDelta_test, x_test)
            err_val = jnp.linalg.norm(err_ref - err_pred) / jnp.linalg.norm(err_ref)

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

            # Visualize train and test example
            zipped = zip([vDelta_train, vDelta_test], [x_train, x_test], [u_train, u_test],
                         [v_train, v_test], [phi_train, phi_test], [False, True], [num_train, num_test])

            viz_loop(model_fn, params, zipped, num_rows, num_cols, it+1+offset_epoch, result_dir)

        # Save checkpoint
        mngr.save(
            it+1+offset_epoch,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(params),
                metadata=ocp.args.JsonSave({'total_iterations': it+1}),
            ),
        )

    mngr.wait_until_finished()

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--num_outputs', type=int, default=3, help='number of outputs')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')
    parser.add_argument('--separable', dest='separable', default=False, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=0, help='hidden tensor dimension in separable DeepONets')

    # Branch settings
    parser.add_argument('--branch_layers', type=int, nargs="+", default=[50, 50, 50, 50, 50],
                        help='hidden branch layer sizes')
    parser.add_argument('--n_sensors', type=int, default=1,
                        help='number of sensors for branch network, also called >>m<<')
    parser.add_argument('--branch_input_features', type=int, default=1,
                        help='number of input features per sensor to branch network')
    parser.add_argument('--split_branch', dest='split_branch', default=False, action='store_true',
                        help='split branch outputs into n groups for n outputs')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=[50, 50, 50, 50, 50],
                        help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=3,
                        help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=False, action='store_true',
                        help='split trunk outputs into j groups for j outputs')

    # Training settings
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200000, help='training epochs')

    # result directory
    parser.add_argument('--result_dir', type=str, default='results/fracture/normal/',
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