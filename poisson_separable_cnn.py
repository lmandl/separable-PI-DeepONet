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
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models import setup_deeponet, mse, mse_single, step, hvp_fwdfwd
from models import apply_net_sep as apply_net


# Data Generator
class DataGenerator(data.Dataset):
    def __init__(self, u, x, y, diff, batch_size, gen_key):
        self.u = u
        self.x = x
        self.y = y
        self.diff = diff
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
        u = self.u[idx]
        x = self.x
        y = self.y
        diff = self.diff
        # Construct batch
        inputs = (u, (x, y, diff))
        outputs = []
        return inputs, outputs


def loss_fn(model_fn, params, ic_batch, bc_batch, full_batch, return_individual_losses=False):
    # IC loss not applicable, we use the full prediction to calculate both res and bc loss
    # We use the structure of the linspaced inputs
    inputs, _ = full_batch
    branch_in, trunk_in = inputs
    x, y, k = trunk_in

    # BC Loss
    s = apply_net(model_fn, params, branch_in, x, y, k)
    # BC loss: u(0, y) = 0, u(1, y) = 0
    bc_x_0 = s[:, 0, :, :]
    bc_x_1 = s[:, -1, :, :]
    bc_loss_x = mse_single(bc_x_0) + mse_single(bc_x_1)
    # BC loss: u(x, 0) = 0, u(x, 1) = 0
    bc_y_0 = s[:, :, 0, :]
    bc_y_1 = s[:, :, -1, :]
    bc_loss_y = mse_single(bc_y_0) + mse_single(bc_y_1)
    # Arrays have same size -> mean of both mse over x and y is possible
    bc_loss = (bc_loss_x + bc_loss_y) * 0.5

    # Residual Loss
    v_x = jnp.ones(x.shape)
    v_y = jnp.ones(y.shape)

    s_xx = hvp_fwdfwd(lambda x_i: apply_net(model_fn, params, branch_in, x_i, y, k), (x,), (v_x,), False)
    s_yy = hvp_fwdfwd(lambda y_i: apply_net(model_fn, params, branch_in, x, y_i, k), (y,), (v_y,), False)

    # Reshape k and branch_in to match the shape of s_xx and s_yy
    k = k[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]
    branch_in = branch_in[:, :, :, jnp.newaxis, :]

    pde = k * s_xx + k * s_yy + branch_in

    loss_res = mse_single(pde)

    loss_value = bc_loss + loss_res

    if return_individual_losses:
        return loss_value, bc_loss, loss_res
    else:
        return loss_value


def get_error(model_fn, params, u_in, s_ref, x_test, y_test, diff_test, idx,
              return_data=False, per_diff=False):
    u_idx_i = u_in[idx]
    s_pred = apply_net(model_fn, params, u_idx_i[jnp.newaxis, :, :, :], x_test, y_test, diff_test)

    s_test = s_ref[idx, :]
    # Reshape to match the shape of s_pred
    s_test = s_test[jnp.newaxis, :, :, :, jnp.newaxis]
    diff = s_test - s_pred

    # error per diffusivity
    if per_diff:
        l2_err_diff = jnp.linalg.norm(diff, axis=(1, 2))[0, :, 0] / jnp.linalg.norm(s_test, axis=(1, 2))[0, :, 0]

    error = jnp.linalg.norm(diff) / jnp.linalg.norm(s_test)
    if return_data:
        if per_diff:
            return error, l2_err_diff, s_pred, s_test
        else:
            return error, s_pred, s_test
    else:
        if per_diff:
            return error, l2_err_diff
        else:
            return error


def visualize(model_fn, params, u_in, s_ref, x_test, y_test, diff_test, idx, epoch, result_dir):
    # Get error
    error_s, err_diff, s_pred, s_test = get_error(model_fn, params, u_in, s_ref, x_test, y_test,
                                                  diff_test, idx, return_data=True, per_diff=True)

    diff = s_test - s_pred

    n_rows = 3
    n_cols = len(diff_test) + 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for ax in axs.flatten():
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    s_min = jnp.min(s_test)
    s_max = jnp.max(s_test)
    diff_max = jnp.max(jnp.abs(diff))
    diff_min = -diff_max

    axs[0, 0].imshow(u_in[idx, :, :, 0], cmap='viridis', vmin=s_min, vmax=s_max, aspect="equal",
                         extent=[x_test.min(), x_test.max(), y_test.min(), y_test.max()])
    axs[0, 0].set_title('Source')
    axs[1, 0].set_axis_off()
    axs[2, 0].set_axis_off()
    for j in range(len(diff_test)):
        s_min = jnp.min(s_test[0, :, :, j, 0])
        s_max = jnp.max(s_test[0, :, :, j, 0])
        diff_max = jnp.max(jnp.abs(diff[0, :, :, j, 0]))
        diff_min = -diff_max
        im1 = axs[0, j + 1].imshow(s_test[0, :, :, j, 0], cmap='viridis', vmin=s_min, vmax=s_max,
                                 aspect="equal", extent=[x_test.min(), x_test.max(), y_test.min(), y_test.max()])
        im2 = axs[1, j + 1].imshow(s_pred[0, :, :, j, 0], cmap='viridis', vmin=s_min, vmax=s_max,
                                 aspect="equal", extent=[x_test.min(), x_test.max(), y_test.min(), y_test.max()])
        im3 = axs[2, j + 1].imshow(diff[0, :, :, j, 0], cmap='coolwarm', vmin=diff_min, vmax=diff_max,
                                 aspect="equal", extent=[x_test.min(), x_test.max(), y_test.min(), y_test.max()])
        for i, im in enumerate([im1, im2, im3]):
            divider = make_axes_locatable(axs[i, j + 1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')


        axs[0, j + 1].set_title(f'True \n $k$={diff_test[j][0]:.1e}', fontsize=10)
        axs[1, j + 1].set_title(f'Pred \n $k$={diff_test[j][0]:.1e}', fontsize=10)
        axs[2, j + 1].set_title(f'Diff \n error={err_diff[j]:.2e}', fontsize=10)


    plt.suptitle(f'Sample {idx}, epoch {epoch},$\mathcal{{L}}_2$: {error_s:.3e}')

    plt.tight_layout()

    plot_dir = os.path.join(result_dir, f'vis/{epoch:06d}/{idx}/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(os.path.join(result_dir, plot_dir), 'pred.png'))
    plt.close(fig)


def main_routine(args):
    # Check if separable network is used
    if not args.separable:
        raise ValueError('Needs separable DeepONet for separable example')

    # Load data
    path = os.path.join(os.getcwd(), 'data/poisson/poisson.mat')
    data = scipy.io.loadmat(path)
    u_in = jnp.array(data['random_field'])
    # Overwrite input channels
    args.branch_cnn_input_size = u_in.shape[1:]
    # CNN needs channel dimension
    u_in = u_in[:, :, :, jnp.newaxis]
    s_ref = jnp.array(data['solution'])
    diff_test = jnp.array(data['diff']).reshape(-1, 1)
    h = data['h'][0][0]
    x_test = jnp.linspace(0, 1, h).reshape(-1, 1)
    y_test = jnp.linspace(0, 1, h).reshape(-1, 1)

    # Prepare the training data
    # get the bounds for the diffusivity from the test data
    diff_lower_bound = jnp.log10(jnp.min(diff_test))-1
    diff_upper_bound = jnp.log10(jnp.max(diff_test))+1
    x_train = x_test
    y_train = y_test
    diff_train = jnp.logspace(diff_lower_bound, diff_upper_bound, args.p_diff_train).reshape(-1, 1)

    # split the data into training and test data
    u_train = u_in[:args.n_train]

    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, 2)

    test_idx = jnp.arange(args.n_train, u_in.shape[0])
    test_idx_list = jnp.split(test_idx, args.batch_size)

    # Choose Plot for visualization
    viz_idx = jax.random.choice(keys[2], test_idx, (8,), replace=False)

    # create the data generator
    train_data = DataGenerator(u_train, x_train, y_train, diff_train, args.batch_size, keys[0])
    data_it = iter(train_data)

    # create the model
    args, model, model_fn, params = setup_deeponet(args, keys[1])

    # Define the optimizer with optax (ADAM)
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
        f.write('epoch,loss,loss_bcs_value,loss_res_value,err_val,runtime\n')

    # Save per_example error if save_pred is True
    if args.save_pred:
        err_file = os.path.join(result_dir, 'individual_error.csv')
        # header
        hdr = "epoch,"
        for idx in test_idx:
            hdr += f"err_idx_{idx},"

        # Save arguments
        with open(err_file, 'a') as f:
            f.write(hdr + '\n')

    # Initial visualization
    if args.vis_iter > 0:
        for k_i in viz_idx:
            # Visualize test example
            visualize(model_fn, params, u_in, s_ref, x_test, y_test,
                      diff_test, k_i, 0, result_dir)

    # Iterations
    pbar = tqdm.trange(args.epochs)

    # Training loop
    for it in pbar:

        if it == 1:
            # start timer and exclude first iteration (compile time)
            start = time.time()

        # Fetch data
        data_batch = next(data_it)

        # Do Step
        loss, params, opt_state = step(optimizer, loss_fn, model_fn, opt_state,
                                       params, [], [], data_batch)

        if it % args.log_iter == 0:
            # Compute losses
            loss, loss_bc_value, loss_res_value = loss_fn(model_fn, params, [], [],
                                                                data_batch, return_individual_losses=True)

            # compute error over test data (split into batches to avoid memory issues)
            errors = []
            for test_idx_i in test_idx_list:
                errors.append(jax.vmap(get_error, in_axes=(None, None, None, None,
                                                           None, None, None, 0))(model_fn, params, u_in,
                                                                                       s_ref, x_test, y_test,
                                                                                       diff_test,
                                                                                       test_idx_i))

            errors = jnp.array(errors).flatten()
            err_val = jnp.mean(errors)

            if args.save_pred:
                pred_path = os.path.join(result_dir, f'pred/{it + 1 + offset_epoch}/')
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                err = f"{it + 1 + offset_epoch},"
                for idx in test_idx:
                    err_i, s_pred, _ = get_error(model_fn, params, u_in, s_ref, x_test, y_test,
                                                    diff_test, idx, return_data=True)
                    err += f"{err_i},"
                    #jnp.save(os.path.join(pred_path, f'pred_{idx}.npy'), s_pred)  # Uncomment for saved predictions
                with open(err_file, 'a') as f:
                    f.write(err + '\n')

            # Print losses
            pbar.set_postfix({'l': f'{loss:.2e}',
                              'l_bc': f'{loss_bc_value:.2e}',
                              'l_r': f'{loss_res_value:.2e}',
                              'e': f'{err_val:.2e}'})

            # get runtime
            if it == 0:
                runtime = 0
            else:
                runtime = time.time() - start

            # Save results
            with open(log_file, 'a') as f:
                f.write(f'{it + 1 + offset_epoch}, {loss}, '
                        f'{loss_bc_value}, {loss_res_value}, {err_val}, {runtime}\n')

        # Visualize result
        if args.vis_iter > 0:
            if (it + 1) % args.vis_iter == 0:
                for k_i in viz_idx:
                    # Visualize test example
                    visualize(model_fn, params, u_in, s_ref, x_test, y_test,
                              diff_test, k_i, it+offset_epoch+1, result_dir)

        # Save checkpoint
        mngr.save(
            it + 1 + offset_epoch,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(params),
                metadata=ocp.args.JsonSave({'total_iterations': it + 1}),
            ),
        )

    runtime = time.time() - start

    mngr.wait_until_finished()

    # Compute losses
    loss, loss_bc_value, loss_res_value = loss_fn(model_fn, params, [], [],
                                                  data_batch, return_individual_losses=True)

    # compute error over test data (split into batches to avoid memory issues)
    errors = []
    for test_idx_i in test_idx_list:
        errors.append(jax.vmap(get_error, in_axes=(None, None, None, None,
                                                   None, None, None, 0))(model_fn, params, u_in,
                                                                         s_ref, x_test, y_test,
                                                                         diff_test,
                                                                         test_idx_i))

    errors = jnp.array(errors).flatten()
    err_val = jnp.mean(errors)

    # Save results
    with open(log_file, 'a') as f:
        f.write(f'{it + 1 + offset_epoch}, {loss}, '
                f'{loss_bc_value}, {loss_res_value}, {err_val}, {runtime}\n')

    if args.save_pred:
        pred_path = os.path.join(result_dir, f'pred/{it + 1 + offset_epoch}/')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        err = f"{it + 1 + offset_epoch},"
        for idx in test_idx:
            err_i, s_pred, _ = get_error(model_fn, params, u_in, s_ref, x_test, y_test,
                                         diff_test, idx, return_data=True)
            err += f"{err_i},"
            # jnp.save(os.path.join(pred_path, f'pred_{idx}.npy'), s_pred)  # Uncomment for saved predictions
        with open(err_file, 'a') as f:
            f.write(err + '\n')

    return err_val


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    
    # model settings
    parser.add_argument('--num_outputs', type=int, default=1, help='number of outputs')
    parser.add_argument('--hidden_dim', type=int, default=400,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')
    parser.add_argument('--separable', dest='separable', default=True, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=80, help='hidden tensor dimension in separable DeepONets')

    # Branch settings
    parser.add_argument('--branch_cnn', dest='branch_cnn', default=True, action='store_true',)
    parser.add_argument('--branch_cnn_blocks', nargs="+", action='append',
                        default=[[16, 2, 2, "relu"], ["avg_pool", 2, 2, 2, 2], [32, 2, 2, "relu"],
                                 ["avg_pool", 2, 2, 2, 2], [64, 2, 2, "relu"], ["avg_pool", 2, 2, 2, 2],
                                 [400, "relu"], [400, "relu"]],
                        help='branch cnn blocks, list of length 4 are Conv2D blocks (features, kernel_size_1, '
                             'kernel_size 2, ac_fun); list of length 5 are Pool2D blocks (pool_type, kernel_size_1, '
                             'kernel_size_2, stride_1, stride_2); list of length 2 are Dense blocks (features, ac_fun);'
                             'flatten will be performed before first dense layer; '
                             'no conv/pool block after first Dense allowed')
    parser.add_argument('--branch_cnn_input_channels', type=int, default=1,
                        help='hidden branch layer sizes')
    parser.add_argument('--branch_cnn_input_size', type=int, nargs="+", default=[1,],
                        help='number of sensors for branch network, also called >>m<<')
    parser.add_argument('--split_branch', dest='split_branch', default=False, action='store_true',
                        help='split branch outputs into n groups for n outputs')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=[50, 50, 50, 50, 50, 50],
                        help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=3,
                        help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=False, action='store_true',
                        help='split trunk outputs into j groups for j outputs')

    # Training settings
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=60000, help='training epochs')
    parser.add_argument('--lr_scheduler', type=str, default='exponential_decay',
                        choices=['constant', 'exponential_decay'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule_steps', type=int, default=5000, help='decay steps for lr scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='decay rate for lr scheduler')

    # result directory
    parser.add_argument('--result_dir', type=str, default='results/poisson/separable/',
                        help='a directory to save results, relative to cwd')
    # log settings
    parser.add_argument('--log_iter', type=int, default=100, help='iteration to save loss and error')
    parser.add_argument('--save_pred', dest='save_pred', default=False, action='store_true',
                        help='save predictions at log_iter')
    parser.add_argument('--vis_iter', type=int, default=10000, help='iteration to save visualization')

    # Problem / Data Settings
    parser.add_argument('--n_train', type=int, default=2500,
                        help='number of train samples, here: number of branch inputs')
    parser.add_argument('--p_diff_train', type=int, default=21,
                        help='number of diffusivities for evaluating the PDE residual and BCs')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')

    # Checkpoint settings
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint file for restoring, uses latest checkpoint')
    parser.add_argument('--checkpoint_iter', type=int, default=5000,
                        help='iteration of checkpoint file')
    parser.add_argument('--checkpoints_to_keep', type=int, default=1,
                        help='number of checkpoints to keep')

    args_in = parser.parse_args()

    main_routine(args_in)
