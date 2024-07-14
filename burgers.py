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

from models import setup_deeponet, mse, mse_single, apply_net, step


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
def generate_one_bcs_training_data(u0, p=100):
    t_bc = jnp.linspace(0, 1, p)[:, None]
    x_bc1 = jnp.zeros((p, 1))
    x_bc2 = jnp.ones((p, 1))

    y1 = jnp.hstack([t_bc, x_bc1])  # shape = (P, 2)
    y2 = jnp.hstack([t_bc, x_bc2])  # shape = (P, 2)

    u = jnp.tile(u0, (p, 1))
    y = jnp.hstack([y1, y2])  # shape = (P, 4)
    s = jnp.zeros((p, 1))

    return u, y, s


# Generate res training data corresponding to one input sample
def generate_one_res_training_data(u0, p=1000):
    p_grid = jnp.sqrt(p).astype(jnp.int32) # sqrt to get square grid
    t_res_i = jnp.linspace(0, 1, p_grid)
    x_res_i = jnp.linspace(0, 1, p_grid)

    t_res, x_res = jnp.meshgrid(t_res_i, x_res_i)

    u = jnp.tile(u0, (p, 1))
    y = jnp.hstack([t_res.flatten()[:, None], x_res.flatten()[:, None]])
    s = jnp.zeros((p, 1))

    return u, y, s


# Generate test data corresponding to one input sample
def generate_one_test_data(usol, idx, p_test=101):

    u = usol[idx]
    u0 = u[0, :]

    t = jnp.linspace(0, 1, p_test)
    x = jnp.linspace(0, 1, p_test)
    t_mesh, x_mesh = jnp.meshgrid(t, x)

    s = u.T.flatten()
    u = jnp.tile(u0, (p_test ** 2, 1))
    y = jnp.hstack([t_mesh.flatten()[:, None], x_mesh.flatten()[:, None]])

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


def loss_bcs(model_fn, params, bcs_batch):
    # Fetch data
    inputs, _ = bcs_batch
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
    inputs, _ = batch
    u, y = inputs
    # Compute forward pass
    t = y[:, 0]
    x = y[:, 1]

    # Residual PDE
    s = apply_net(model_fn, params, u, t, x)
    v_x = jnp.ones(x.shape)
    v_t = jnp.ones(t.shape)

    s_t = jax.vjp(lambda t: apply_net(model_fn, params, u, t, x), t)[1](v_t)[0]
    s_x = jax.vjp(lambda x: apply_net(model_fn, params, u, t, x), x)[1](v_x)[0]
    s_xx = jax.jvp(lambda x: jax.vjp(lambda x: apply_net(model_fn, params, u, t, x), x)[1](v_x)[0], (x,), (v_x,))[1]

    pred = s_t + s * s_x - 0.01 * s_xx
    # Compute loss
    loss = mse_single(pred)
    return loss


def loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch):
    loss_ics_i = loss_ics(model_fn, params, ics_batch)
    loss_bcs_i = loss_bcs(model_fn, params, bcs_batch)
    loss_res_i = loss_res(model_fn, params, res_batch)
    loss_value = 20.0 * loss_ics_i + loss_bcs_i + loss_res_i
    return loss_value


def get_error(model_fn, params, u_sol, idx, p_err=101, return_data=False):
    u_test, y_test, s_test = generate_one_test_data(u_sol, idx, p_err)

    t_test = y_test[:, 0]
    x_test = y_test[:, 1]

    s_pred = apply_net(model_fn, params, u_test, t_test, x_test)
    error = jnp.linalg.norm(s_test - s_pred) / jnp.linalg.norm(s_test)

    if return_data:
        return error, s_pred
    else:
        return error


def visualize(args, model_fn, params, result_dir, epoch, usol, idx, test=False):
    # Generate data, and obtain error
    error_s, s_pred = get_error(model_fn, params, usol, idx, args.p_test, return_data=True)

    u = usol[idx].T

    t = jnp.linspace(0, 1, args.p_test)
    x = jnp.linspace(0, 1, args.p_test)

    # Reshape s_pred
    s_pred = s_pred.reshape(t.shape[0], x.shape[0])

    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(u, interpolation="nearest", vmin=jnp.amin(u), vmax=jnp.amax(u),
               extent=[t.min(), t.max(), x.max(), x.min()],
               origin='upper', aspect='auto', cmap='viridis')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Exact u')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.imshow(s_pred, interpolation="nearest", vmin=jnp.amin(u), vmax=jnp.amax(u),
               extent=[t.min(), t.max(), x.max(), x.min()],
               origin='upper', aspect='auto', cmap='viridis')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Predicted u')
    plt.colorbar()
    plt.tight_layout()

    u_diff = u-s_pred
    plt.subplot(1, 3, 3)
    plt.imshow(u_diff, interpolation="nearest", vmin=-abs(u_diff).max(), vmax=abs(u_diff).max(),
               extent=[t.min(), t.max(), x.max(), x.min()],
               origin='upper', aspect='auto', cmap='seismic')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Absolute error')
    plt.colorbar()

    if test:
        plt.suptitle(f'test, L2: {error_s:.3e}')
    else:
        plt.suptitle(f'train, L2: {error_s:.3e}')
    plt.tight_layout()
    plot_dir = os.path.join(result_dir, f'vis/{epoch:06d}/{idx}/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(os.path.join(result_dir, plot_dir), 'pred.png'))
    plt.close(fig)


def main_routine(args):
    if args.separable:
        raise ValueError('Needs normal DeepONet, not separable DeepONet')
    # Prepare the training data
    # Load data
    path = os.path.join(os.getcwd(), 'data/burgers/Burger.mat')  # Please use the matlab script to generate data

    data_ref = scipy.io.loadmat(path)
    u_sol = jnp.array(data_ref['output'])

    n_in = u_sol.shape[0]  # number of total input samples
    n_test = n_in - args.n_train  # number of input samples used for test
    if n_test < args.n_test:
        raise ValueError('Not enough data for testing, please reduce the number of test samples'
                         ' or increase the number of training samples.')

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
    u_bcs_train, y_bcs_train, s_bcs_train = (jax.vmap(generate_one_bcs_training_data,
                                                      in_axes=(0, None))
                                             (u0_train, args.p_bcs_train))

    u_bcs_train = u_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)
    y_bcs_train = y_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)
    s_bcs_train = s_bcs_train.reshape(args.n_train * args.p_bcs_train, -1)

    # Residual data
    u_res_train, y_res_train, s_res_train = (jax.vmap(generate_one_res_training_data,
                                                      in_axes=(0, None))
                                             (u0_train, args.p_res_train))

    u_res_train = u_res_train.reshape(args.n_train * args.p_res_train, -1)
    y_res_train = y_res_train.reshape(args.n_train * args.p_res_train, -1)
    s_res_train = s_res_train.reshape(args.n_train * args.p_res_train, -1)

    # Create data generators
    ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, args.batch_size, keys[0])
    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, args.batch_size, keys[1])
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, args.batch_size, keys[2])

    # Create test data
    test_range = jnp.arange(args.n_train, u_sol.shape[0])
    test_idx = jax.random.choice(keys[3], test_range, (args.n_test,), replace=False)
    test_idx_list = jnp.split(test_idx, 10)

    # Create model
    args, model, model_fn, params = setup_deeponet(args, keys[4])

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

    # Data
    ics_data = iter(ics_dataset)
    bcs_data = iter(bcs_dataset)
    res_data = iter(res_dataset)

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
        f.write('epoch,loss,loss_ics_value,loss_bcs_value,loss_res_value,err_val,runtime\n')

    # Choose Plots for visualization
    k_train = jax.random.randint(keys[5], shape=(1,), minval=0, maxval=args.n_train)[0]  # index
    k_test = test_idx[0]  # index

    # First visualization
    if args.vis_iter > 0:
        # Visualize train example
        visualize(args, model_fn, params, result_dir, offset_epoch, u_sol, k_train, False)
        # Visualize test example
        visualize(args, model_fn, params, result_dir, offset_epoch, u_sol, k_test, True)

    # Iterations
    pbar = tqdm.trange(args.epochs)

    # Training loop
    for it in pbar:

        if it == 1:
            # start timer and exclude first iteration (compile time)
            start = time.time()

        # Fetch data
        ics_batch = next(ics_data)
        bcs_batch = next(bcs_data)
        res_batch = next(res_data)

        # Do Step
        loss, params, opt_state = step(optimizer, loss_fn, model_fn, opt_state,
                                       params, ics_batch, bcs_batch, res_batch)

        if it % args.log_iter == 0:
            # Compute losses
            loss = loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch)
            loss_ics_value = loss_ics(model_fn, params, ics_batch)
            loss_bcs_value = loss_bcs(model_fn, params, bcs_batch)
            loss_res_value = loss_res(model_fn, params, res_batch)

            # compute error over test data (split into 10 batches to avoid memory issues)
            errors = []
            for test_idx in test_idx_list:
                errors.append(jax.vmap(get_error, in_axes=(None, None, None, 0, None))(model_fn, params, u_sol, test_idx,
                                                                                       args.p_test))
            errors = jnp.array(errors).flatten()
            err_val = jnp.mean(errors)

            # Print losses
            pbar.set_postfix({'l': f'{loss:.2e}',
                              'l_ic': f'{loss_ics_value:.2e}',
                              'l_bc': f'{loss_bcs_value:.2e}',
                              'l_r': f'{loss_res_value:.2e}',
                              'e': f'{err_val:.2e}'})

            # get runtime
            if it == 0:
                runtime = 0
            else:
                runtime = time.time() - start

            # Save results
            with open(log_file, 'a') as f:
                f.write(f'{it+1+offset_epoch}, {loss}, {loss_ics_value}, '
                        f'{loss_bcs_value}, {loss_res_value}, {err_val}, {runtime}\n')

        # Visualize result
        if args.vis_iter > 0:
            if (it + 1) % args.vis_iter == 0:
                # Visualize train example
                visualize(args, model_fn, params, result_dir, it+1+offset_epoch, u_sol, k_train, False)
                # Visualize test example
                visualize(args, model_fn, params, result_dir, it+1+offset_epoch, u_sol, k_test, True)

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
        f.write(f'{it + 1 + offset_epoch}, {loss}, {loss_ics_value}, '
                f'{loss_bcs_value}, {loss_res_value}, {err_val}, {runtime}\n')


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--num_outputs', type=int, default=1, help='number of outputs')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')
    parser.add_argument('--separable', dest='separable', default=False, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=0, help='hidden tensor dimension in separable DeepONets')

    # Branch settings
    parser.add_argument('--branch_layers', type=int, nargs="+", default=[100, 100, 100],
                        help='hidden branch layer sizes')
    parser.add_argument('--n_sensors', type=int, default=101,
                        help='number of sensors for branch network, also called >>m<<')
    parser.add_argument('--branch_input_features', type=int, default=1,
                        help='number of input features per sensor to branch network')
    parser.add_argument('--split_branch', dest='split_branch', default=False, action='store_true',
                        help='split branch outputs into n groups for n outputs')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=[100, 100, 100],
                        help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=2,
                        help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=False, action='store_true',
                        help='split trunk outputs into j groups for j outputs')

    # Training settings
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100000, help='training epochs')
    parser.add_argument('--lr_scheduler', type=str, default='exponential_decay',
                        choices=['constant', 'exponential_decay'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule_steps', type=int, default=2000, help='decay steps for lr scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for lr scheduler')

    # result directory
    parser.add_argument('--result_dir', type=str, default='results/burgers/normal/',
                        help='a directory to save results, relative to cwd')

    # log settings
    parser.add_argument('--log_iter', type=int, default=100, help='iteration to save loss and error')
    parser.add_argument('--vis_iter', type=int, default=1000, help='iteration to save visualization')

    # Problem / Data Settings
    parser.add_argument('--n_train', type=int, default=1000,
                        help='number of input samples used for training')
    parser.add_argument('--n_test', type=int, default=1000, help='number of samples used for testing')
    parser.add_argument('--p_ics_train', type=int, default=101,
                        help='number of locations for evaluating the initial condition')
    parser.add_argument('--p_bcs_train', type=int, default=100,
                        help='number of locations for evaluating the boundary condition')
    parser.add_argument('--p_res_train', type=int, default=2500,
                        help='number of locations for evaluating the PDE residual')
    parser.add_argument('--p_test', type=int, default=101,
                        help='number of locations for evaluating the error')
    parser.add_argument('--batch_size', type=int, default=100000, help='batch size')

    # Checkpoint settings
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint file for restoring, uses latest checkpoint')
    parser.add_argument('--checkpoint_iter', type=int, default=5000,
                        help='iteration of checkpoint file')
    parser.add_argument('--checkpoints_to_keep', type=int, default=1,
                        help='number of checkpoints to keep')

    args_in = parser.parse_args()

    main_routine(args_in)
