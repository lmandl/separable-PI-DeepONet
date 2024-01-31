import argparse
import shutil
import os
import jax

from data.data_loader import load_data
from data.data_generator import generate_data

from models import DeepONet


def main(args):
    # Set result directory
    result_dir = os.path.join(os.getcwd(), args.result_dir)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir)

    # Create config file
    config_file = open(os.path.join(result_dir, 'config.txt'), 'w')

    # Write args to config file
    for args in vars(args):
        config_file.write('{}: {}\n'.format(args, getattr(args, args)))

    # Create log_file
    log_file = open(os.path.join(result_dir, 'log.txt'), 'w')

    # Set data directory
    data_dir = os.path.join(os.getcwd(), args.data_dir)
    eqn = args.equation

    # Set random seed and key
    key = jax.random.PRNGKey(args.seed)

    # Split key
    key, subkey = jax.random.split(key)

    # Generate data
    if args.generate_data:
        train_data, test_data = generate_data(args, subkey)
    else:
        train_data, test_data = load_data(data_dir, eqn)

    # Initialize model
    # TODO: initialize model
    model = DeepONet()

    # TODO: write training loop


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # result directory
    parser.add_argument('--result_dir', type=str, default='/result/antiderivative',
                        help='a directory to save results, relative to cwd')

    # model settings
    parser.add_argument('--use_equation', type=bool, default=True,
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

    # Data settings
    parser.add_argument('--generate_data', type=bool, default=False,
                        help='generate data or use stored data in data_dir, requires data_generator')
    parser.add_argument('--save_data', type=bool, default=False,
                        help='save generated data to data_dir, requires data_generator')
    parser.add_argument('--data_dir', type=str, default='/data/antiderivative_aligned',
                        help='a directory where train and test data are saved, relative to cwd')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=list, default=[128, 128, 128], help='hidden trunk layer sizes')
    parser.add_argument('--trunk_activation', type=str, default='tanh', help='trunk activation function')
    parser.add_argument('--num_trunk_inputs', type=int, default=1, help='number of inputs to trunk network')

    # Branch settings
    parser.add_argument('--branch_layers', type=list, default=[128, 128, 128], help='hidden branch layer sizes')
    parser.add_argument('--branch_activation', type=str, default='tanh', help='branch activation function')
    parser.add_argument('--num_branch_inputs', type=int, default=1, help='number of inputs to branch network')

    # Train samples
    parser.add_argument('--train_samples', type=int, default=10000, help='number of training samples')
    parser.add_argument('--test_samples', type=int, default=10000, help='number of test samples')

    # Training settings
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    # log settings
    parser.add_argument('--log_iter', type=int, default=2500, help='iteration to save loss and error')

    print('Project in Development')

    main(parser.parse_args())
