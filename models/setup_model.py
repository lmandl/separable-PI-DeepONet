import jax.numpy as jnp
import jax

from .deeponet import DeepONet, SeparableDeepONet


def setup_deeponet(args, key):
    # Overriding split_trunk and split_branch if num_outputs is 1
    if args.num_outputs == 1:
        args.split_trunk = False
        args.split_branch = False

    # if split_trunk is True and separable_trunk is false
    # then the hidden_dim is multiplied by the output_dim
    # if split_trunk is false and separable_trunk is false
    # then the trunk output is the same as the hidden_dim
    # if split_trunk is false and separable_trunk is true
    # then the output of the separable_trunk is hidden_dim*r
    # if split_trunk is true and separable_trunk is true
    # then the output of the separable_trunk is hidden_dim*r*output_dim
    # the multiplication with r is done in the deeponet class

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
    # dimensions for splits are calculated once here before creating the model
    # add input and output features to branch layers for unstacked DeepONet
    # if split_branch is True, the output features are split into n groups for n outputs
    if args.split_branch:
        branch_layers = ([args.n_sensors * args.branch_input_features] +
                         args.branch_layers + [args.num_outputs * args.hidden_dim])
    else:
        branch_layers = [args.n_sensors * args.branch_input_features] + args.branch_layers + [args.hidden_dim]

    # Convert list to tuples
    trunk_layers = tuple(trunk_layers)
    branch_layers = tuple(branch_layers)

    # build model
    if not args.separable:
        model = DeepONet(branch_layers, trunk_layers, args.split_branch, args.split_trunk, args.stacked_do,
                         args.num_outputs)
    else:
        model = SeparableDeepONet(branch_layers, trunk_layers, args.split_branch, args.split_trunk, args.stacked_do,
                                  args.num_outputs, args.r)

    # Initialize parameters
    params = model.init(key, jnp.ones(shape=(1, args.n_sensors * args.branch_input_features)),
                        jnp.ones(shape=(1, args.trunk_input_features)))

    # model function
    model_fn = jax.jit(model.apply)

    return args, model, model_fn, params
