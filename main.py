import argparse
import shutil
import os
import jax

from data.data_loader import load_data
from data.data_generator import generate_data


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

    # Generate data
    if args.generate_data:
        train_data, test_data = generate_data(args)
    else:
        train_data, test_data = load_data(data_dir, eqn)

    # Initialize model
    model = DeepONet()

    # Train model
    model.train(train_data)


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()

    # result directory
    parser.add_argument('--result_dir', type=str, default='/result/biot1d',
                        help='a directory to save results, relative to cwd')

    # model settings
    """ 
    parser.add_argument('--use_equation', type=bool, default=True,
                        help='use equation for physics-informed DeepONet, if false only data is used')
    """
    parser.add_argument('--equation', type=str, default='biot1d', help='equation to solve')
    parser.add_argument('--problem_hyperparam', type=dict,
                        default={"alpha": [1e0, 1e2],
                                 "beta": [1e-2, 1e0],
                                 "a": [1e-1, 1e1],
                                 "L": [1, 2],
                                 "t_end": [1, 3],
                                 "n_z": 101,
                                 "n_t": 101},
                        help='hyperparameters for equation setup, e.g. domain, ranges, boundary conditions, etc.')
    parser.add_argument('--num_inputs', type=int, default=2, help='number of inputs')
    parser.add_argument('--num_outputs', type=int, default=2, help='number of outputs')

    # Data settings
    parser.add_argument('--generate_data', type=bool, default=True,
                        help='generate data or use stored data in data_dir, requires data_generator')
    parser.add_argument('--data_dir', type=str, default='/data/biot1d',
                        help='a directory where train and test data are saved, relative to cwd')

    # Trunk settings
    parser.add_argument('--trunk_layers', type=list, default=[128, 128, 128], help='hidden trunk layer sizes')
    parser.add_argument('--trunk_activation', type=str, default='tanh', help='trunk activation function')

    # Branch settings
    parser.add_argument('--branch_layers', type=list, default=[128, 128, 128], help='hidden branch layer sizes')
    parser.add_argument('--branch_activation', type=str, default='tanh', help='branch activation function')

    # Training settings
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    # log settings
    parser.add_argument('--log_iter', type=int, default=2500, help='iteration to save loss and error')

    print('Project in Development')

    args = parser.parse_args()

    main(args)
