import jax
import jax.numpy as jnp
from torch.utils import data
from functools import partial
import tqdm
import time
import optax
import os
import argparse
import matplotlib.pyplot as plt
import shutil
import orbax.checkpoint as ocp

from models import setup_deeponet, mse, mse_single, step, hvp_fwdfwd
from models import apply_net_sep as apply_net

# This file implements an example for Biot's theory of consolidation using separable DeepONets
# We sample the displacement at z=0 in t=[0,1] as branch input
# trunk input is [t, z] and stacked output is [u, p]

class DataGenerator(data.Dataset):
    # IC has same t for all samples in y
    def __init__(self, u, t, z, s, batch_size, gen_key):
        self.u = u
        self.t = t
        self.z = z
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
        z = self.z
        t = self.t
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, (t, z))
        outputs = s
        return inputs, outputs


# Generate test data corresponding to one input sample
def generate_one_test_data(usol, idx, p_test=101):

    s = usol[idx]
    u0 = s[0, :, 0]  # u0 at z=0

    t = jnp.linspace(0, 1, p_test).reshape(p_test, 1)
    z = jnp.linspace(0, 1, p_test).reshape(p_test, 1)
    y = jnp.hstack([t, z])

    u = u0.reshape(1,-1)

    # s needs reshaping for easier error computation & plotting by transposing axis 0 and 1
    s = jnp.transpose(s, (1, 0, 2))

    return u, y, s


def loss_ics(model_fn, params, ics_batch):
    inputs, outputs = ics_batch
    u, y = inputs

    # Compute forward pass
    t = y[0]
    z = y[1]
    s_pred = apply_net(model_fn, params, u, t, z)

    # Compute loss
    loss_ic_u = mse(outputs[:, :, 0].flatten(), s_pred[:, :, :, 0].flatten())
    loss_ic_p = mse(outputs[:, :, 1].flatten(), s_pred[:, :, :, 1].flatten())

    return loss_ic_u + loss_ic_p


def apply_net_squeeze(model_fn, params, u, t, z):
    # Helper function to squeeze the output in gradient computations
    return apply_net(model_fn, params, u, t, z).squeeze()


def loss_bcs(model_fn, params, bcs_batch):
    # Fetch data
    inputs, outputs = bcs_batch
    u, y = inputs
    t_bc, z = y
    z_bc_1, z_bc_2 = z

    # Assert shapes here for easier gradient computation
    z_bc_1 = jnp.reshape(z_bc_1, (-1, 1))
    z_bc_2 = jnp.reshape(z_bc_2, (-1, 1))
    t_bc = jnp.reshape(t_bc, (-1, 1))

    # Compute forward pass
    s_bc1_pred = apply_net(model_fn, params, u, t_bc, z_bc_1)
    s_bc2_pred = apply_net(model_fn, params, u, t_bc, z_bc_2)

    v_z = jnp.ones(z_bc_2.shape)
    s_z_bc2_pred = jax.jvp(lambda z: apply_net(model_fn, params, u, t_bc, z)[:, :, :, 1], (z_bc_2,), (v_z,))[1]

    # Compute loss
    loss_s_bc_1_u = mse(s_bc1_pred[:, :, :, 0].flatten(), outputs[:, :, 0].flatten())  # u(t, 0) = u0
    loss_s_bc_1_p = mse(s_bc1_pred[:, :, :, 1].flatten(), outputs[:, :, 1].flatten())  # p(t, 0) = 0
    loss_s_bc_2_u = mse(s_bc2_pred[:, :, :, 0].flatten(), outputs[:, :, 2].flatten())  # u(t, 1) = 0
    loss_s_bc_2_p = mse(s_z_bc2_pred.flatten(), outputs[:, :, 3].flatten())  # p_z(t, 1) = 0

    return loss_s_bc_1_u + loss_s_bc_1_p + loss_s_bc_2_u + loss_s_bc_2_p


# Define residual loss
def loss_res(model_fn, params, batch):
    # parameters
    lamda = 1
    mu = 1
    k = 1
    rho = 1
    g = 1

    # Fetch data
    inputs, _ = batch
    u, y = inputs
    t, z = y

    # Residual PDE
    v_z = jnp.ones(z.shape)
    v_t = jnp.ones(t.shape)

    # Compute gradients
    _, u_zz = hvp_fwdfwd(lambda z: apply_net(model_fn, params, u, t, z)[:, :, :, 0], (z,), (v_z,), True)
    p_z, p_zz = hvp_fwdfwd(lambda z: apply_net(model_fn, params, u, t, z)[:, :, :, 1], (z,), (v_z,), True)

    # Compute the mixed derivative u_tz
    u_tz = jax.jvp(lambda z: jax.jvp(lambda t: apply_net(model_fn, params, u, t, z)[:, :, :, 0], (t,), (v_t,))[1], (z,), (v_z,))[1]

    # Compute PDEs
    pred_1 = (lamda + 2 * mu) * u_zz - p_z
    pred_2 = u_tz - (k / (rho * g)) * p_zz

    # Compute loss
    loss_pred_1 = mse_single(pred_1)
    loss_pred_2 = mse_single(pred_2)

    return loss_pred_1 + loss_pred_2


def loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch):
    loss_ics_i = loss_ics(model_fn, params, ics_batch)
    loss_bcs_i = loss_bcs(model_fn, params, bcs_batch)
    loss_res_i = loss_res(model_fn, params, res_batch)
    loss_value = loss_ics_i + loss_bcs_i + loss_res_i
    return loss_value


def get_error(model_fn, params, u_sol, idx, p_err=101, return_data=False, per_dim = False):
    u_test, y_test, s_test = generate_one_test_data(u_sol, idx, p_err)

    t_test = y_test[:, 0].reshape(-1, 1)
    z_test = y_test[:, 1].reshape(-1, 1)

    s_pred = apply_net(model_fn, params, u_test, t_test, z_test)
    s_pred = s_pred[0, :, :, :]  # first example as only one branch input

    if per_dim:
        error_u = jnp.linalg.norm(s_test[:, 0] - s_pred[:, 0]) / jnp.linalg.norm(s_test[:, 0])
        error_p = jnp.linalg.norm(s_test[:, 1] - s_pred[:, 1]) / jnp.linalg.norm(s_test[:, 1])
        error = [error_u, error_p]
    else:
        error = jnp.linalg.norm(s_test - s_pred) / jnp.linalg.norm(s_test)

    if return_data:
        return error, s_pred
    else:
        return error


def visualize(args, model_fn, params, result_dir, epoch, usol, idx, test=False):
    # Generate data, and obtain error
    error_s, s_pred = get_error(model_fn, params, usol, idx, args.p_test, return_data=True, per_dim=True)

    u = usol[idx, :, :, 0]
    p = usol[idx, :, :, 1]

    t = jnp.linspace(0, 1, args.p_test)
    z = jnp.linspace(0, 1, args.p_test)

    # Reshape s_pred
    u_pred = s_pred[:, :, 0].T
    p_pred = s_pred[:, :, 1].T

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(2, 3, 1)
    plt.imshow(u, interpolation='nearest', vmin=jnp.amin(u), vmax=jnp.amax(u), \
               extent=[t.min(), t.max(), z.max(), z.min()], \
               origin='upper', aspect='auto', cmap='viridis')
    cbar = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$z$')
    plt.title('$u(z,t)$')

    plt.subplot(2, 3, 2)
    plt.imshow(u_pred, interpolation='nearest', vmin=jnp.amin(u), vmax=jnp.amax(u), \
               extent=[t.min(), t.max(), z.max(), z.min()], \
               origin='upper', aspect='auto', cmap='viridis')
    cbar = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$z$')
    plt.title('$\hat{u}(z,t)$')

    plt.subplot(2, 3, 3)
    plt.imshow(u_pred - u, interpolation='nearest', \
               extent=[t.min(), t.max(), z.max(), z.min()], \
               origin='upper', aspect='auto', cmap='seismic',
               vmax=abs(u_pred - u).max(), vmin=-abs(u_pred - u).max())
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    plt.xlabel('$t$')
    plt.ylabel('$z$')
    plt.title('$u-\hat{u}$')

    plt.subplot(2, 3, 4)
    plt.imshow(p, interpolation='nearest', vmin=jnp.amin(p), vmax=jnp.amax(p), \
               extent=[t.min(), t.max(), z.max(), z.min()], \
               origin='upper', aspect='auto', cmap='plasma')
    cbar = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$z$')
    plt.title('$p(z,t)$')

    plt.subplot(2, 3, 5)
    plt.imshow(p_pred, interpolation='nearest', vmin=jnp.amin(p), vmax=jnp.amax(p), \
               extent=[t.min(), t.max(), z.max(), z.min()], \
               origin='upper', aspect='auto', cmap='plasma')
    cbar = plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$z$')
    plt.title('$\hat{p}(z,t)$')


    plt.subplot(2, 3, 6)
    plt.imshow(p_pred - p, interpolation='nearest', \
               extent=[t.min(), t.max(), z.max(), z.min()], \
               origin='upper', aspect='auto', cmap='PuOr',
               vmax=abs(p_pred - p).max(), vmin=-abs(p_pred - p).max())
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    plt.xlabel('$t$')
    plt.ylabel('$z$')
    plt.title('$p-\hat{p}$')

    if test:
        plt.suptitle(f'test, L2: u:{error_s[0]:.3e}, p:{error_s[1]:.3e}')
    else:
        plt.suptitle(f'train, L2: u:{error_s[0]:.3e}, p:{error_s[1]:.3e}')
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

    # Prepare the training data
    # Load data
    path = os.path.join(os.getcwd(), 'data/biot/Y.npy')  # Please use the matlab script to generate data

    u_sol = jnp.load(path)

    n_in = u_sol.shape[0]  # number of total input samples
    n_test = n_in - args.n_train  # number of input samples used for test
    if n_test < args.n_test:
        raise ValueError('Not enough data for testing, please reduce the number of test samples'
                         ' or increase the number of training samples.')

    u0_train = u_sol[:args.n_train, 0, :, 0]  # input samples
    p0_train = u_sol[:args.n_train, -1, 0, 1]  # initial pressure

    # Split key for IC, BC, Residual data, and model init
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)

    # ICs data
    t_ics_train = jnp.zeros((1, 1))
    s_u_ics_train = jnp.zeros((args.n_train, args.p_ics_train))
    z_ics_train = jnp.linspace(0, 1, args.p_ics_train).reshape(-1, 1)
    s_p_ics_train = jnp.tile(p0_train, (args.p_ics_train, 1)).T
    s_ics_train = jnp.dstack([s_u_ics_train, s_p_ics_train])
    ics_dataset = DataGenerator(u0_train, t_ics_train, z_ics_train, s_ics_train, args.batch_size, keys[0])

    # BCs data
    s_bcs_train = jnp.dstack([u0_train, jnp.zeros((args.n_train, args.p_bcs_train)),
                              jnp.zeros((args.n_train, args.p_bcs_train)),
                              jnp.zeros((args.n_train, args.p_bcs_train))])
    t_bcs_train = jnp.linspace(0, 1, args.p_bcs_train).reshape(-1, 1)
    z_bc1_train = jnp.zeros((1, 1))
    z_bc2_train = jnp.ones((1, 1))
    z_bc_train = (z_bc1_train, z_bc2_train)

    # Create data generator
    bcs_dataset = DataGenerator(u0_train, t_bcs_train, z_bc_train, s_bcs_train, args.batch_size, keys[1])

    # Residual data
    # generate keys for Residuals
    s_res_train = jnp.zeros((1, 1))
    t_res_train = jnp.linspace(0, 1, args.p_res_train).reshape(-1, 1)
    z_res_train = jnp.linspace(0, 1, args.p_res_train).reshape(-1, 1)

    # Create data generators
    res_dataset = DataGenerator(u0_train, t_res_train, z_res_train, s_res_train, args.batch_size, keys[2])

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

    # Save per_example error if save_pred is True
    if args.save_pred:
        err_file = os.path.join(result_dir, 'individual_error.csv')
        # header
        hdr = "epoch,"
        for idx in test_idx:
            hdr += f"err_idx_{idx},"

        # Save arguments
        with open(err_file, 'a') as f:
            f.write(hdr+'\n')

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
            for test_idx_i in test_idx_list:
                errors.append(jax.vmap(get_error, in_axes=(None, None, None, 0, None))(model_fn, params, u_sol,
                                                                                       test_idx_i, args.p_test))
            errors = jnp.array(errors).flatten()
            err_val = jnp.mean(errors)

            if args.save_pred:
                pred_path = os.path.join(result_dir, f'pred/{it + 1 + offset_epoch}/')
                if not os.path.exists(pred_path):
                    os.makedirs(pred_path)
                err = f"{it + 1 + offset_epoch},"
                for idx in test_idx:
                    err_i, s_pred = get_error(model_fn, params, u_sol, idx, args.p_test, return_data=True)
                    err += f"{err_i},"
                    jnp.save(os.path.join(pred_path, f'pred_{idx}.npy'), s_pred)
                with open(err_file, 'a') as f:
                    f.write(err + '\n')

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

    runtime = time.time() - start

    mngr.wait_until_finished()

    # Compute losses
    loss = loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch)
    loss_ics_value = loss_ics(model_fn, params, ics_batch)
    loss_bcs_value = loss_bcs(model_fn, params, bcs_batch)
    loss_res_value = loss_res(model_fn, params, res_batch)

    # compute error over test data (split into 10 batches to avoid memory issues)
    errors = []
    for test_idx_i in test_idx_list:
        errors.append(jax.vmap(get_error, in_axes=(None, None, None, 0, None))(model_fn, params, u_sol,
                                                                               test_idx_i, args.p_test))
    errors = jnp.array(errors).flatten()
    err_val = jnp.mean(errors)

    # Save results
    with open(log_file, 'a') as f:
        f.write(f'{it + 1 + offset_epoch}, {loss}, {loss_ics_value}, '
                f'{loss_bcs_value}, {loss_res_value}, {err_val}, {runtime}\n')

    if args.save_pred:
        pred_path = os.path.join(result_dir, f'pred/{it + 1 + offset_epoch}/')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        err = f"{it + 1 + offset_epoch},"
        for idx in test_idx:
            err_i, s_pred = get_error(model_fn, params, u_sol, idx, args.p_test, return_data=True)
            err += f"{err_i},"
            jnp.save(os.path.join(pred_path, f'pred_{idx}.npy'), s_pred)
        with open(err_file, 'a') as f:
            f.write(err + '\n')


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--num_outputs', type=int, default=2, help='number of outputs')
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')
    parser.add_argument('--separable', dest='separable', default=True, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=20, help='hidden tensor dimension in separable DeepONets')

    # Branch settings
    parser.add_argument('--branch_layers', type=int, nargs="+", default=[100, 100, 100, 100, 100, 100],
                        help='hidden branch layer sizes')
    parser.add_argument('--n_sensors', type=int, default=101,
                        help='number of sensors for branch network, also called >>m<<')
    parser.add_argument('--branch_input_features', type=int, default=1,
                        help='number of input features per sensor to branch network')
    parser.add_argument('--split_branch', dest='split_branch', default=False, action='store_true',
                        help='split branch outputs into n groups for n outputs')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=[50, 50, 50, 50, 50, 50],
                        help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=2,
                        help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=False, action='store_true',
                        help='split trunk outputs into j groups for j outputs')

    # Training settings
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=150000, help='training epochs')
    parser.add_argument('--lr_scheduler', type=str, default='exponential_decay',
                        choices=['constant', 'exponential_decay'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule_steps', type=int, default=10000, help='decay steps for lr scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='decay rate for lr scheduler')

    # result directory
    parser.add_argument('--result_dir', type=str, default='results/biot/separable/',
                        help='a directory to save results, relative to cwd')

    # log settings
    parser.add_argument('--log_iter', type=int, default=100, help='iteration to save loss and error')
    parser.add_argument('--save_pred', dest='save_pred', default=False, action='store_true',
                        help='save predictions at log_iter')
    parser.add_argument('--vis_iter', type=int, default=50000, help='iteration to save visualization')

    # Problem / Data Settings
    parser.add_argument('--n_train', type=int, default=1000,
                        help='number of input samples used for training')
    parser.add_argument('--n_test', type=int, default=1000, help='number of samples used for testing')
    parser.add_argument('--p_ics_train', type=int, default=101,
                        help='number of locations for evaluating the initial condition')
    parser.add_argument('--p_bcs_train', type=int, default=101,
                        help='number of locations for evaluating the boundary condition')
    parser.add_argument('--p_res_train', type=int, default=50,
                        help='number of locations for evaluating the PDE residual')
    parser.add_argument('--p_test', type=int, default=101,
                        help='number of locations for evaluating the error')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')

    # Checkpoint settings
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint file for restoring, uses latest checkpoint')
    parser.add_argument('--checkpoint_iter', type=int, default=50000,
                        help='iteration of checkpoint file')
    parser.add_argument('--checkpoints_to_keep', type=int, default=1,
                        help='number of checkpoints to keep')

    args_in = parser.parse_args()

    main_routine(args_in)