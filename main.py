import argparse
import shutil
import os
import jax
import jax.numpy as jnp
import optax
from tqdm import trange
import time

from data import load_data, generate_data
from utils import train_error, update_model, loss_and_grad
from models import UnstackedDeepONet, StackedDeepONet


def main(args):

    # Start workflow
    print("Starting. Assembling data...")

    # Set result directory
    result_dir = os.path.join(os.getcwd(), args.result_dir)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # Create config file
    config_file = open(os.path.join(result_dir, 'config.txt'), 'w')

    # Write argument name and value to config file
    for arg in vars(args):
        config_file.write(arg + ': ' + str(getattr(args, arg)) + '\n')

    # Create log_file
    log_file = open(os.path.join(result_dir, 'log.txt'), 'w')

    # Set data directory
    data_dir = os.path.join(os.getcwd(), args.data_dir)
    eqn = args.problem

    # Set random seed and key
    key = jax.random.PRNGKey(args.seed)

    # Split key
    key, subkey = jax.random.split(key)

    # Generate data
    # Note:: train and test data has form [[branch_input, trunk_input], output]
    if args.generate_data:
        train_data, test_data = generate_data(args, subkey)
    else:
        train_data, test_data = load_data(data_dir, eqn)
    print("Data assembled. Initializing model...")

    # Overriding split_trunk and split_branch if num_outputs is 1
    if args.num_outputs == 1:
        args.split_trunk = False
        args.split_branch = False

    # Throw error if split_trunk and split_branch are both False and num_outputs is greater than 1
    if not args.split_trunk and not args.split_branch and args.num_outputs > 1:
        raise ValueError('split_trunk and split_branch cannot both be False when num_outputs > 1')

    # Initialize model and params
    # make sure trunk_layers and branch_layers are lists
    args.trunk_layers = [args.trunk_layers] if isinstance(args.trunk_layers, int) else args.trunk_layers
    args.branch_layers = [args.branch_layers] if isinstance(args.branch_layers, int) else args.branch_layers

    # add input and output features to trunk and branch layers
    # split trunk if split_trunk is True
    if args.split_trunk:
        trunk_layers = [args.trunk_input_features]+args.trunk_layers+[args.num_outputs*args.hidden_dim]
    else:
        trunk_layers = [args.trunk_input_features]+args.trunk_layers+[args.hidden_dim]

    # Stacked or unstacked DeepONet
    # dimensions for splits are calculated once here
    # branch input features are multiplied by n_sensors as each input function is evaluated at n_sensors
    # they are than stacked as vector
    if args.stacked_do:
        # add input and output features to branch layers for stacked DeepONet, which has one output feature
        # if split_branch is True, the output features are split into n groups for n outputs but layer sizes are kept
        branch_layers = [args.n_sensors*args.branch_input_features]+args.branch_layers+[1]
        # build model
        if args.split_branch:
            # If branches are split, we need to multiply the hidden_dim by the number of outputs
            model = StackedDeepONet(branch_layers, trunk_layers, args.split_branch, args.split_trunk,
                                    args.num_outputs, args.hidden_dim * args.num_outputs)
        else:
            model = StackedDeepONet(branch_layers, trunk_layers, args.split_branch, args.split_trunk,
                                    args.num_outputs, args.hidden_dim)
    else:
        # add input and output features to branch layers for unstacked DeepONet
        # if split_branch is True, the output features are split into n groups for n outputs
        if args.split_branch:
            branch_layers = ([args.n_sensors*args.branch_input_features] +
                             args.branch_layers + [args.num_outputs*args.hidden_dim])
        else:
            branch_layers = ([args.n_sensors*args.branch_input_features] +
                             args.branch_layers + [args.hidden_dim])
        # build model
        model = UnstackedDeepONet(branch_layers, trunk_layers, args.split_branch, args.split_trunk,
                                  args.num_outputs)
    # Initialize parameters
    params = model.init(key, jnp.ones(args.n_sensors), jnp.ones(args.trunk_input_features))

    # model function
    model_fn = jax.jit(model.apply)

    # Log total params to config file
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    config_file.write('total_params: ' + str(args.total_params) + '\n')

    print("Model initialized. Training model...")

    # Define optimizer with optax (ADAM)
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    # List of TODOs
    # TODO: Check Point
    # TODO: Visualization

    for epoch in trange(args.epochs):

        if epoch == 1:
            # exclude compile time
            start = time.time()

        # Train model
        loss, gradient = loss_and_grad(model_fn, params, train_data[0], train_data[1])
        params, state = update_model(optimizer, gradient, params, opt_state)

        # Log loss
        if epoch % args.log_iter == 0:
            error = train_error(model_fn, params, test_data[0], test_data[1])
            log_file.write(f'Epoch: {epoch}, train loss: {loss}, test error: {error}\n')
            print(f'Epoch: {epoch}/{args.epochs} --> train loss: {loss:.8f}, test error: {error:.8f}')

    print("Training done")
    # training done
    runtime = time.time() - start
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime / (args.epochs - 1) * 1000):.2f}ms/iter.)')


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument('--problem', type=str, default='antiderivative_unaligned', help='equation / problem to solve')
    parser.add_argument('--use_equation', dest='use_equation', default=False, action='store_true',
                        help='use equation for physics-informed DeepONet, if false only data is used')

    parser.add_argument('--problem_hyperparam', type=dict,
                        default={"alpha": [1e0, 1e2],
                                 "beta": [1e-2, 1e0],
                                 "a": [1e-1, 1e1],
                                 "length": 1,
                                 "t_end": 1},
                        help='hyperparameters for equation setup, e.g. domain, ranges, boundary conditions, etc.')
    parser.add_argument('--num_outputs', type=int, default=1, help='number of outputs')
    parser.add_argument('--hidden_dim', type=int, default=20,
                        help='latent layer size in DeepONet, also called >>p<<, multiples are used for splits')
    parser.add_argument('--stacked_deeponet', dest='stacked_do', default=False, action='store_true',
                        help='use stacked DeepONet, if false use unstacked DeepONet')

    # Data settings
    parser.add_argument('--generate_data', dest='generate_data', default=False, action='store_true',
                        help='generate data or use stored data in data_dir, requires data_generator')

    parser.add_argument('--save_data', dest='save_data', default=False, action='store_true',
                        help='save generated data to data_dir, requires data_generator')
    parser.add_argument('--data_dir', type=str, default='./data/antiderivative_unaligned',
                        help='a directory where train and test data are saved, relative to cwd')
    parser.add_argument('--train_samples', type=int, default=10000, help='number of training samples to generate')
    parser.add_argument('--test_samples', type=int, default=10000, help='number of test samples to generate')

    # Branch settings
    parser.add_argument('--branch_layers', type=int, nargs="+", default=40, help='hidden branch layer sizes')
    parser.add_argument('--n_sensors', type=int, default=100, help='number of sensors for branch network')
    parser.add_argument('--branch_input_features', type=int, default=1,
                        help='number of input features to branch network')
    parser.add_argument('--split_branch', dest='split_branch', default=True, action='store_false',
                        help='split branch outputs into n groups for n outputs')
    # TODO: later change to default=False

    # Trunk settings
    parser.add_argument('--trunk_layers', type=int, nargs="+", default=40, help='hidden trunk layer sizes')
    parser.add_argument('--trunk_input_features', type=int, default=1, help='number of input features to trunk network')
    parser.add_argument('--split_trunk', dest='split_trunk', default=True, action='store_false',
                        help='split trunk outputs into n groups for n outputs')
    # TODO: later change to default=False

    # Training settings
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='training epochs')

    # result directory
    parser.add_argument('--result_dir', type=str, default='./result/antiderivative',
                        help='a directory to save results, relative to cwd')

    # log settings
    parser.add_argument('--log_iter', type=int, default=100, help='iteration to save loss and error')

    args_in = parser.parse_args()

    print('Project in Development')

    main(args_in)
