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


def main_routine(args):
    # Check if separable network is used
    if not args.separable:
        raise ValueError('Needs separable DeepONet for separable example')

    # TODO: Load and Prepare the training data

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
    parser.add_argument('--separable', dest='separable', default=True, action='store_true',
                        help='use separable DeepONets')
    parser.add_argument('--r', type=int, default=0, help='hidden tensor dimension in separable DeepONets')

    # Branch settings
    parser.add_argument('--branch_cnn', dest='branch_cnn', default=False, action='store_true',)
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