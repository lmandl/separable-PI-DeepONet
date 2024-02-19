import jax.numpy as jnp
import jax

from .deeponet import DeepONet


def setup_deeponet(args, key):
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
        trunk_layers = [args.trunk_input_features] + args.trunk_layers + [args.num_outputs * args.hidden_dim]
    else:
        trunk_layers = [args.trunk_input_features] + args.trunk_layers + [args.hidden_dim]

    # Stacked or unstacked DeepONet
    # dimensions for splits are calculated once here
    # branch input features are multiplied by n_sensors as each input function is evaluated at n_sensors
    # they are than stacked as vector
    # add input and output features to branch layers for stacked/unstacked DeepONet
    # if split_branch is True, the output features are split into n groups for n outputs but layer sizes are kept
    # last layer size defines number of branches for stacked DeepONets
    if args.split_branch:
        branch_layers = ([args.n_sensors * args.branch_input_features] +
                         args.branch_layers + [args.num_outputs * args.hidden_dim])
    else:
        branch_layers = [args.n_sensors * args.branch_input_features] + args.branch_layers + [args.hidden_dim]

    # Convert list to tuples
    trunk_layers = tuple(trunk_layers)
    branch_layers = tuple(branch_layers)

    # build model
    model = DeepONet(branch_layers, trunk_layers, args.split_branch, args.split_trunk, args.stacked_do,
                     args.num_outputs)

    # Initialize parameters
    params = model.init(key, jnp.ones(args.n_sensors), jnp.ones(args.trunk_input_features))

    # model function
    model_fn = jax.jit(model.apply)
    # model_fn = model.apply
    return args, model, model_fn, params
