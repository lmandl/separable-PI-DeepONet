import argparse
import shutil
import os
import jax
import optax
from tqdm import trange
import time

from data import load_data, generate_data
from utils import train_step, train_error
from models import DeepONet


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
    # Note:: train and test data has form [[trunk_input, branch_input], output]
    if args.generate_data:
        train_data, test_data = generate_data(args, subkey)
    else:
        train_data, test_data = load_data(data_dir, eqn)

    print("Data assembled. Initializing model...")

    # Initialize model and params
    branch_layers = [args.branch_input_features]+args.branch_layers+[args.hidden_dim]
    trunk_layers = [args.trunk_input_features]+args.trunk_layers+[args.hidden_dim]
    model = DeepONet(branch_layers, args.branch_activation, trunk_layers, args.trunk_activation, args.use_bias,
                     args.num_outputs)

    params = model.init(key, train_data[0])

    # Log total params to config file
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    config_file.write('total_params: ' + str(args.total_params) + '\n')

    print("Model initialized. Training model...")

    # Define optimizer with optax (ADAM)
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    opt_state, params, loss = train_step(optimizer, model, opt_state, params, train_data[0], train_data[1])

    # List of TODOs
    # TODO: Define jits
    # TODO: Check Point
    # TODO: Visualization

    for epoch in trange(args.epochs):

        if epoch == 1:
            # exclude compile time
            start = time.time()

        # Train model
        opt_state, params, loss = train_step(optimizer, model, opt_state, params, train_data[0], train_data[1])

        # Log loss
        if epoch % args.log_iter == 0:
            error = train_error(model, params, test_data[0], test_data[1])
            log_file.write(f'Epoch: {epoch}, Loss: {loss}, Error: {error}\n')
            print(f'Epoch: {epoch}, Loss: {loss}, Error: {error}\n')

    print("Training done")
    # training done
    runtime = time.time() - start
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime / (args.epochs - 1) * 1000):.2f}ms/iter.)')


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # result directory
    parser.add_argument('--result_dir', type=str, default='./result/antiderivative',
                        help='a directory to save results, relative to cwd')

    # model settings
    parser.add_argument('--use_equation', type=bool, default=False,
                        help='use equation for physics-informed DeepONet, if false only data is used')
    parser.add_argument('--problem', type=str, default='antiderivative_aligned', help='equation / problem to solve')
    parser.add_argument('--problem_hyperparam', type=dict,
                        default={"alpha": [1e0, 1e2],
                                 "beta": [1e-2, 1e0],
                                 "a": [1e-1, 1e1],
                                 "length": 1,
                                 "t_end": 1},
                        help='hyperparameters for equation setup, e.g. domain, ranges, boundary conditions, etc.')
    parser.add_argument('--num_outputs', type=int, default=1, help='number of outputs')
    parser.add_argument('--use_bias', type=bool, default=True, help='use bias in DeepONet')
    parser.add_argument('--hidden_dim', type=int, default=40, help='latent layer size in DeepONet')

    # Data settings
    parser.add_argument('--generate_data', type=bool, default=False,
                        help='generate data or use stored data in data_dir, requires data_generator')
    parser.add_argument('--save_data', type=bool, default=False,
                        help='save generated data to data_dir, requires data_generator')
    parser.add_argument('--data_dir', type=str, default='./data/antiderivative_aligned',
                        help='a directory where train and test data are saved, relative to cwd')
    parser.add_argument('--train_samples', type=int, default=10000, help='number of training samples to generate')
    parser.add_argument('--test_samples', type=int, default=10000, help='number of test samples to generate')

    # Branch settings
    parser.add_argument('--branch_layers', type=list, default=[40], help='hidden branch layer sizes')
    parser.add_argument('--branch_activation', type=str, default='tanh', help='branch activation function')
    parser.add_argument('--branch_input_features', type=int, default=100,
                        help='number of input features to branch network')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=list, default=[40], help='hidden trunk layer sizes')
    parser.add_argument('--trunk_activation', type=str, default='tanh', help='trunk activation function')
    parser.add_argument('--trunk_input_features', type=int, default=1, help='number of input features to trunk network')

    # Training settings
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    # log settings
    parser.add_argument('--log_iter', type=int, default=2500, help='iteration to save loss and error')

    print('Project in Development')

    main(parser.parse_args())
