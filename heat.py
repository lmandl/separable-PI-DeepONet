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
def generate_one_ics_training_data(t_ic, p=101):
    t_0 = jnp.zeros((1, 1))
    x_0 = jnp.linspace(0, 1, p+2)[1:-1]  # exclude boundaries
    y_0 = jnp.linspace(0, 1, p+2)[1:-1]  # exclude boundaries
    c_0 = jnp.logspace(1, 0, p)

    t_mesh, x_mesh, y_mesh, c_mesh = jnp.meshgrid(t_0.flatten(), x_0.flatten(),
                                                  y_0.flatten(), c_0.flatten(), indexing='ij')

    y = jnp.vstack([t_mesh.flatten(), x_mesh.flatten(), y_mesh.flatten(), c_mesh.flatten()]).T

    u = jnp.tile(t_ic, (p ** 3, 1))
    s = u

    return u, y, s


# Generate bcs training data corresponding to one input sample
def generate_one_bcs_training_data(t_ic, p=100):
    # Note: This has 4 times the data as other arrays and thus needs longer to cycle through
    t_bc = jnp.linspace(0, 1, p)

    x_bc_train_11 = jnp.zeros((1, 1))
    x_bc_train_12 = jnp.ones((1, 1))
    x_bc_train_2 = jnp.linspace(0, 1, p)

    y_bc_train_1 = jnp.linspace(0, 1, p)
    y_bc_train_21 = jnp.zeros((1, 1))
    y_bc_train_22 = jnp.ones((1, 1))

    c_bc_train = jnp.logspace(1, 0, p)

    x_mesh_11, y_mesh_11, t_mesh_11, c_mesh_11 = jnp.meshgrid(x_bc_train_11.flatten(), y_bc_train_1.flatten(),
                                                              t_bc, c_bc_train.flatten(), indexing='ij')
    x_mesh_12, y_mesh_12, t_mesh_12, c_mesh_12 = jnp.meshgrid(x_bc_train_12.flatten(), y_bc_train_1.flatten(),
                                                              t_bc, c_bc_train.flatten(), indexing='ij')
    x_mesh_21, y_mesh_21, t_mesh_21, c_mesh_21 = jnp.meshgrid(x_bc_train_2.flatten(), y_bc_train_21.flatten(),
                                                              t_bc, c_bc_train.flatten(), indexing='ij')
    x_mesh_22, y_mesh_22, t_mesh_22, c_mesh_22 = jnp.meshgrid(x_bc_train_2.flatten(), y_bc_train_22.flatten(),
                                                              t_bc, c_bc_train.flatten(), indexing='ij')

    y1 = jnp.vstack([t_mesh_11.flatten(), x_mesh_11.flatten(), y_mesh_11.flatten(), c_mesh_11.flatten()]).T
    y2 = jnp.vstack([t_mesh_12.flatten(), x_mesh_12.flatten(), y_mesh_12.flatten(), c_mesh_12.flatten()]).T
    y3 = jnp.vstack([t_mesh_21.flatten(), x_mesh_21.flatten(), y_mesh_21.flatten(), c_mesh_21.flatten()]).T
    y4 = jnp.vstack([t_mesh_22.flatten(), x_mesh_22.flatten(), y_mesh_22.flatten(), c_mesh_22.flatten()]).T

    u = jnp.tile(t_ic, (4 * p ** 3, 1))  # (4*p**3, 1)

    y = jnp.vstack([y1, y2, y3, y4])  # (4*p**3, 1)

    s = jnp.zeros((4 * p**3, 1))  # (4*p**3, 1)

    return u, y, s


# Generate res training data corresponding to one input sample
def generate_one_res_training_data(t_ic, p=50):
    t_res_i = jnp.linspace(0, 1, p)
    x_res_i = jnp.linspace(0, 1, p)
    y_res_i = jnp.linspace(0, 1, p)
    c_res_i = jnp.logspace(1, 0, p)

    t_res, x_res, y_res, c_res = jnp.meshgrid(t_res_i, x_res_i, y_res_i,
                                              c_res_i, indexing='ij')

    u = jnp.tile(t_ic, (p**4, 1))
    y = jnp.vstack([t_res.flatten(), x_res.flatten(), y_res.flatten(), c_res.flatten()]).T
    s = jnp.zeros((p**4, 1))  # needed for code compatibility with data generator

    return u, y, s


# Generate test data corresponding to one input sample
def generate_one_test_data(u_sol, c_ref, t_ic, idx, p_err=101):

    u = u_sol[idx]
    u0 = t_ic[idx].reshape(-1, 1)

    c_i = c_ref[idx].reshape(-1, 1)

    t_i = jnp.linspace(0, 1, p_err)
    x_i = jnp.linspace(0, 1, p_err)
    y_i = jnp.linspace(0, 1, p_err)
    t_mesh, x_mesh, y_mesh, c_mesh = jnp.meshgrid(t_i.flatten(), x_i.flatten(), y_i.flatten(),
                                                  c_i.flatten(), indexing='ij')

    s = u.T.flatten()
    u = jnp.tile(u0, (p_err ** 3, 1))
    y = jnp.vstack([t_mesh.flatten(), x_mesh.flatten(), y_mesh.flatten(), c_mesh.flatten()]).T

    return u, y, s


def loss_icbc(model_fn, params, data_batch):
    inputs, outputs = data_batch
    u, y_vec = inputs

    # Compute forward pass
    t = y_vec[:, 0]
    x = y_vec[:, 1]
    y = y_vec[:, 2]
    c = y_vec[:, 3]

    s_pred = apply_net(model_fn, params, u, t, x, y, c)

    # Compute loss
    loss_ic = mse(outputs.flatten(), s_pred)
    return loss_ic


# Define residual loss
def loss_res(model_fn, params, batch):
    # Fetch data
    inputs, _ = batch
    u, y_vec = inputs
    # Compute forward pass
    t = y_vec[:, 0]
    x = y_vec[:, 1]
    y = y_vec[:, 2]
    c = y_vec[:, 3]

    # Residual PDE
    v_x = jnp.ones(x.shape)
    v_t = jnp.ones(t.shape)
    v_y = jnp.ones(y.shape)

    s_t = jax.vjp(lambda t: apply_net(model_fn, params, u, t, x, y, c), t)[1](v_t)[0]
    s_xx = jax.jvp(lambda x: jax.vjp(lambda x: apply_net(model_fn, params, u, t, x, y, c), x)[1](v_x)[0], (x,), (v_x,))[1]
    s_yy = jax.jvp(lambda y: jax.vjp(lambda y: apply_net(model_fn, params, u, t, x, y, c), y)[1](v_y)[0], (y,), (v_y,))[1]

    c_squared = jnp.square(c)

    pred = s_t - c_squared * (s_xx + s_yy)

    # Compute loss
    loss = mse_single(pred)
    return loss


def loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch):
    loss_ics_i = loss_icbc(model_fn, params, ics_batch)
    loss_bcs_i = loss_icbc(model_fn, params, bcs_batch)
    loss_res_i = loss_res(model_fn, params, res_batch)
    loss_value = loss_ics_i + loss_bcs_i + 4 * loss_res_i
    return loss_value


def get_error(model_fn, params, u_sol, c_ref, t_ic, idx, p_err=101, return_data=False):
    u_test, y_test_vec, s_test = generate_one_test_data(u_sol, c_ref, t_ic, idx, p_err)

    t_test = y_test_vec[:, 0]
    x_test = y_test_vec[:, 1]
    y_test = y_test_vec[:, 2]
    c_test = y_test_vec[:, 3]

    s_pred = apply_net(model_fn, params, u_test, t_test, x_test, y_test, c_test)
    error = jnp.linalg.norm(s_test - s_pred) / jnp.linalg.norm(s_test)

    if return_data:
        return error, s_pred
    else:
        return error


def visualize(args, model_fn, params, result_dir, epoch, usol, c_ref, t_ic, idx, test=False):

    # Generate data, and obtain error
    error_s, s_pred, u = get_error(model_fn, params, usol, c_ref, t_ic, idx, args.p_test, return_data=True)

    c = c_ref[idx]

    t_0 = t_ic[idx]

    t = jnp.linspace(0, 1, args.p_test)
    x = jnp.linspace(0, 1, args.p_test)
    y = jnp.linspace(0, 1, args.p_test)

    diff_sq = jnp.square(u - s_pred)

    fig = plt.figure(figsize=(9, 9))

    # Plt 1: central over x
    plt.subplot(3, 3, 4)
    plt.imshow(u[:, args.p_test // 2, :], interpolation="nearest", vmin=jnp.amin(u), vmax=jnp.amax(u),
               extent=[t.min(), t.max(), y.max(), y.min()],
               origin='upper', aspect='auto', cmap='viridis')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Exact $T$ at $x=0.5$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 3, 1)
    plt.imshow(s_pred[:, args.p_test // 2, :], interpolation="nearest", vmin=jnp.amin(u), vmax=jnp.amax(u),
               extent=[t.min(), t.max(), y.max(), y.min()],
               origin='upper', aspect='auto', cmap='viridis')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Pred. $T$ at $x=0.5$')
    plt.colorbar()
    plt.tight_layout()


    plt.subplot(3, 3, 7)
    plt.imshow(diff_sq[:, args.p_test // 2, :], interpolation="nearest", vmin=0, vmax=jnp.amax(diff_sq),
               extent=[t.min(), t.max(), y.max(), y.min()],
               origin='upper', aspect='auto', cmap='Reds')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Sq. Diff. at $x=0.5$')
    plt.colorbar()
    plt.tight_layout()

    # plt 2: central over y
    plt.subplot(3, 3, 5)
    plt.imshow(u[:, :, args.p_test // 2], interpolation="nearest", vmin=jnp.amin(u), vmax=jnp.amax(u),
               extent=[t.min(), t.max(), x.max(), x.min()],
               origin='upper', aspect='auto', cmap='viridis')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Exact $T$ at $y=0.5$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 3, 2)
    plt.imshow(s_pred[:, :, args.p_test // 2], interpolation="nearest", vmin=jnp.amin(u), vmax=jnp.amax(u),
               extent=[t.min(), t.max(), x.max(), x.min()],
               origin='upper', aspect='auto', cmap='viridis')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Pred. $T$ at $y=0.5$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 3, 8)
    plt.imshow(diff_sq[:, :, args.p_test // 2], interpolation="nearest", vmin=0, vmax=jnp.amax(diff_sq),
               extent=[t.min(), t.max(), x.max(), x.min()],
               origin='upper', aspect='auto', cmap='Reds')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Sq. Diff. at $y=0.5$')
    plt.colorbar()
    plt.tight_layout()

    # t slice at t=0
    plt.subplot(3, 3, 6)
    plt.imshow(u[0, :, :], interpolation="nearest", vmin=jnp.amin(u), vmax=jnp.amax(u),
               extent=[x.min(), x.max(), y.max(), y.min()],
               origin='upper', aspect='auto', cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Exact $T$ at $t=0.0$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 3, 3)
    plt.imshow(s_pred[0, :, :], interpolation="nearest", vmin=jnp.amin(u), vmax=jnp.amax(u),
               extent=[x.min(), x.max(), y.max(), y.min()],
               origin='upper', aspect='auto', cmap='viridis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pred. $T$ at $t=0.0$')
    plt.colorbar()
    plt.tight_layout()

    plt.subplot(3, 3, 9)
    plt.imshow(diff_sq[0, :, :], interpolation="nearest", vmin=0, vmax=jnp.amax(diff_sq),
               extent=[x.min(), x.max(), y.max(), y.min()],
               origin='upper', aspect='auto', cmap='Reds')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sq. Diff. at $t=0.0$')
    plt.colorbar()
    plt.tight_layout()

    if test:
        plt.suptitle(f'test, c: {c:.2e}, $T_0$: {t_0:.1e}, $\mathcal{{L}}_2$: {error_s:.3e}')
    else:
        plt.suptitle(f'train, c: {c:.2e}, $T_0$: {t_0:.1e}, $\mathcal{{L}}_2$: {error_s:.3e}')
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
    path = os.path.join(os.getcwd(), 'data/heat/heat_const.mat')  # Please use the matlab script to generate data

    data_ref = scipy.io.loadmat(path)
    u_sol = jnp.array(data_ref['results'])
    c_ref = jnp.array(data_ref['c_vec']).flatten()  # We use c^2 instead of alpha in this code
    t_ic = jnp.array(data_ref['t_ic']).flatten()

    c_2_min_power = -1
    c_2_max_power = 0

    n_in = u_sol.shape[0]  # number of total input samples

    # Split key for IC, BC, Residual data, and model init
    seed = args.seed
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 7)

    # ICs data
    u_train = jnp.linspace(0, 1, args.n_train)[:, None]  # set temperatures

    # ICs data
    u_ics_train, y_ics_train, s_ics_train = (jax.vmap(generate_one_ics_training_data,
                                                      in_axes=(0, None))
                                             (u_train, args.p_ics_train))
    u_ics_train = u_ics_train.reshape(args.n_train * args.p_ics_train**3, -1)
    y_ics_train = y_ics_train.reshape(args.n_train * args.p_ics_train**3, -1)
    s_ics_train = s_ics_train.reshape(args.n_train * args.p_ics_train**3, -1)

    unique_rows = jnp.unique(y_ics_train, axis=0)

    # BCs data
    u_bcs_train, y_bcs_train, s_bcs_train = (jax.vmap(generate_one_bcs_training_data,
                                                      in_axes=(0, None))
                                             (u_train, args.p_bcs_train))

    u_bcs_train = u_bcs_train.reshape(args.n_train * 4 * args.p_bcs_train**3, -1)
    y_bcs_train = y_bcs_train.reshape(args.n_train * 4 * args.p_bcs_train**3, -1)
    s_bcs_train = s_bcs_train.reshape(args.n_train * 4 * args.p_bcs_train**3, -1)

    # Residual data
    u_res_train, y_res_train, s_res_train = (jax.vmap(generate_one_res_training_data,
                                                      in_axes=(0, None))
                                             (u_train, args.p_res_train))

    u_res_train = u_res_train.reshape(args.n_train * args.p_res_train**4, -1)
    y_res_train = y_res_train.reshape(args.n_train * args.p_res_train**4, -1)
    s_res_train = s_res_train.reshape(args.n_train * args.p_res_train**4, -1)

    # Calculate batch_sizes
    fac = 100  # factor for memory
    batch_size_ics = (args.n_train * args.p_ics_train**3) // fac
    batch_size_bcs = (args.n_train * 4 * args.p_bcs_train**3) // fac
    batch_size_res = (args.n_train * args.p_res_train**4) // fac
    # longer training and scheduler steps
    args.lr_schedule_steps = args.lr_schedule_steps * fac
    args.epochs = args.epochs * fac
    print("Adjusting batch sizes and training length for memory constraints:")
    print(f"Batch sizes: ics: {batch_size_ics}, bcs: {batch_size_bcs}, res: {batch_size_res} with factor {fac}")
    print(f"Training for {args.epochs} epochs with lr scheduler steps {args.lr_schedule_steps}")

    # Create data generators
    ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size_ics, keys[0])
    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size_bcs, keys[1])
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size_res, keys[2])

    # Create test data
    test_range = jnp.arange(0, u_sol.shape[0])
    test_idx = jax.random.choice(keys[4], test_range, (n_in,), replace=False)
    test_idx_list = jnp.split(test_idx, 25)

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
    k_test = [test_idx[i] for i in [0, 1, 2, 3, 4]]  # index

    # Initial visualization
    if args.vis_iter > 0:
        for k_i in k_test:
            # Visualize test example
            visualize(args, model_fn, params, result_dir, offset_epoch, u_sol, c_ref, t_ic, k_i, True)

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
            loss_ics_value = loss_icbc(model_fn, params, ics_batch)
            loss_bcs_value = loss_icbc(model_fn, params, bcs_batch)
            loss_res_value = loss_res(model_fn, params, res_batch)

            # compute error over test data (split into batches to avoid memory issues)
            errors = []
            for test_idx in test_idx_list:
                errors.append(jax.vmap(get_error, in_axes=(None, None, None,
                                                           None, None, 0, None))(model_fn, params, u_sol, c_ref,
                                                                                 t_ic, test_idx, args.p_test))
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
                f.write(f'{it + 1 + offset_epoch}, {loss}, {loss_ics_value}, '
                        f'{loss_bcs_value}, {loss_res_value}, {err_val}, {runtime}\n')
        # Visualize result
        if args.vis_iter > 0:
            if (it + 1) % args.vis_iter == 0:
                for k_i in k_test:
                    # Visualize test example
                    visualize(args, model_fn, params, result_dir, it + 1 + offset_epoch, u_sol, c_ref, t_ic, k_i,
                              True)

        # Save checkpoint
        mngr.save(
            it+1+offset_epoch,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(params),
                metadata=ocp.args.JsonSave({'total_iterations': it+1}),
            ),
        )

    # Save results
    runtime = time.time() - start

    mngr.wait_until_finished()

    loss = loss_fn(model_fn, params, ics_batch, bcs_batch, res_batch)
    loss_ics_value = loss_icbc(model_fn, params, ics_batch)
    loss_bcs_value = loss_icbc(model_fn, params, bcs_batch)
    loss_res_value = loss_res(model_fn, params, res_batch)

    # compute error over test data (split into batches to avoid memory issues)
    errors = []
    for test_idx in test_idx_list:
        errors.append(jax.vmap(get_error, in_axes=(None, None, None,
                                                   None, None, 0, None))(model_fn, params, u_sol, c_ref,
                                                                         t_ic, test_idx, args.p_test))
    errors = jnp.array(errors).flatten()
    err_val = jnp.mean(errors)

    with open(log_file, 'a') as f:
        f.write(f'{it + 1 + offset_epoch}, {loss}, {loss_ics_value}, '
                f'{loss_bcs_value}, {loss_res_value}, {err_val}, {runtime}\n')


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--num_outputs', type=int, default=1, help='number of outputs')
    parser.add_argument('--hidden_dim', type=int, default=50,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')
    parser.add_argument('--separable', dest='separable', default=False, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=50, help='hidden tensor dimension in separable DeepONets')

    # Branch settings
    parser.add_argument('--branch_layers', type=int, nargs="+", default=[50, 50, 50, 50, 50, 50],
                        help='hidden branch layer sizes')
    parser.add_argument('--n_sensors', type=int, default=1,
                        help='number of sensors for branch network, also called >>m<<')
    parser.add_argument('--branch_input_features', type=int, default=1,
                        help='number of input features per sensor to branch network')
    parser.add_argument('--split_branch', dest='split_branch', default=False, action='store_true',
                        help='split branch outputs into n groups for n outputs')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=[50, 50, 50, 50, 50, 50],
                        help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=4,
                        help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=False, action='store_true',
                        help='split trunk outputs into j groups for j outputs')

    # Training settings
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100000, help='training epochs')
    parser.add_argument('--lr_scheduler', type=str, default='exponential_decay',
                        choices=['constant', 'exponential_decay'], help='learning rate scheduler')
    parser.add_argument('--lr_schedule_steps', type=int, default=500, help='decay steps for lr scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for lr scheduler')

    # result directory
    parser.add_argument('--result_dir', type=str, default='results/heat/normal/',
                        help='a directory to save results, relative to cwd')

    # log settings
    parser.add_argument('--log_iter', type=int, default=100, help='iteration to save loss and error')
    parser.add_argument('--vis_iter', type=int, default=0, help='iteration to save visualization')

    # Problem / Data Settings
    parser.add_argument('--p_ics_train', type=int, default=51,
                        help='number of locations for evaluating the initial condition')
    parser.add_argument('--p_bcs_train', type=int, default=51,
                        help='number of locations for evaluating the boundary condition')
    parser.add_argument('--p_res_train', type=int, default=31,
                        help='number of locations for evaluating the PDE residual per dimension')
    parser.add_argument('--p_test', type=int, default=101,
                        help='number of locations for evaluating the error')
    parser.add_argument('--n_train', type=int, default=25,
                        help='number of training samples. needed for data generation')

    # Checkpoint settings
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='path to checkpoint file for restoring, uses latest checkpoint')
    parser.add_argument('--checkpoint_iter', type=int, default=5000,
                        help='iteration of checkpoint file')
    parser.add_argument('--checkpoints_to_keep', type=int, default=1,
                        help='number of checkpoints to keep')

    args_in = parser.parse_args()

    main_routine(args_in)
