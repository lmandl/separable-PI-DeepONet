import jax.numpy as jnp
import jax

from .deeponet import DeepONet, SeparableDeepONet, SeparableBranchCNNDeepONet

def setup_deeponet(args, key):
    # Overriding split_trunk and split_branch if num_outputs is 1
    if args.num_outputs == 1:
        args.split_trunk = False
        args.split_branch = False

    if args.num_outputs > 1 and args.split_trunk is False and args.split_branch is False:
        raise ValueError('split_trunk and split_branch cannot both be False for multi-output models')

    # compatability for version without branch CNNs
    if args.branch_cnn is None:
        args.branch_cnn = False
    else:
        # Check if arguments are valid
        for layer in args.branch_cnn_blocks:
            dense = False
            if len(layer) == 4:
                # Conv block
                if dense:
                    raise ValueError('Conv block following dense block is not allowed')
                if layer[-1] not in ["relu", "tanh", "sigmoid"]:
                    raise ValueError('Invalid activation function for branch_cnn_blocks')
            elif len(layer) == 5:
                # Pool block
                if dense:
                    raise ValueError('Pool block following dense block is not allowed')
                if layer[0] not in ["avg_pool", "max_pool"]:
                    raise ValueError('Unknown pooling operation for branch_cnn_blocks')
            elif len(layer) == 2:
                # Dense block
                dense = True
                if layer[-1] not in ["relu", "tanh", "sigmoid"]:
                    raise ValueError('Invalid activation function for branch_cnn_blocks')
            else:
                raise ValueError('Invalid branch_cnn_blocks')
        args.branch_layers = []

    if args.branch_cnn and args.split_branch:
        raise ValueError('split_branch with branch_cnn is not supported at the moment')

    if args.branch_cnn and not args.separable:
        raise ValueError('branch_cnn is only supported with separable DeepONets at the moment')

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
    if not args.branch_cnn:
        args.branch_layers = [args.branch_layers] if isinstance(args.branch_layers, int) else args.branch_layers

    # add output features to trunk and branch layers
    # split trunk if split_trunk is True
    if args.split_trunk:
        trunk_layers = args.trunk_layers + [args.num_outputs * args.hidden_dim]
    else:
        trunk_layers = args.trunk_layers + [args.hidden_dim]

    # Stacked or unstacked DeepONet
    # dimensions for splits are calculated once here before creating the model
    # add output features to branch layers for unstacked DeepONet
    # if split_branch is True, the output features are split into n groups for n outputs
    if not args.branch_cnn:
        if args.split_branch:
            branch_layers = args.branch_layers + [args.num_outputs * args.hidden_dim]
        else:
            branch_layers = args.branch_layers + [args.hidden_dim]
    else:
        # add hidden_dim to the last layer of branch_cnn_blocks as dense layer
        if args.split_branch:
            # Append last layer and activation function (None) to branch_cnn_blocks
            branch_cnn_blocks = args.branch_cnn_blocks.append[[args.hidden_dim * args.num_outputs, None]]
        else:
            branch_cnn_blocks = args.branch_cnn_blocks.append[[args.hidden_dim, None]]

    # Convert list to tuples
    trunk_layers = tuple(trunk_layers)
    branch_layers = tuple(branch_layers)

    # build model
    if not args.separable:
        model = DeepONet(branch_layers, trunk_layers, args.split_branch, args.split_trunk, args.stacked_do,
                         args.num_outputs)
        params = model.init(key, jnp.ones(shape=(1, args.n_sensors * args.branch_input_features)),
                            jnp.ones(shape=(1, args.trunk_input_features)))
    else:
        if args.branch_cnn:
            model = SeparableBranchCNNDeepONet(branch_cnn_blocks, trunk_layers, args.split_branch, args.split_trunk,
                                               args.stacked_do, args.num_outputs, args.r)
            #TODO: Initialize parameters
        else:
            model = SeparableDeepONet(branch_layers, trunk_layers, args.split_branch, args.split_trunk,
                                      args.stacked_do, args.num_outputs, args.r)
            # Initialize parameters
            # init of trunk needs args.trunk_input_features * jnp.ones(shape=(1, 1)) but not as list
            input_features = [jnp.ones(shape=(1, 1)) for _ in range(args.trunk_input_features)]

            params = model.init(key, jnp.ones(shape=(1, args.n_sensors * args.branch_input_features)),
                                *input_features)

    # Print model from parameters
    print('--- model_summary ---')
    # count total params
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'total params: {args.total_params}')
    print('--- model_summary ---')

    # model function
    model_fn = jax.jit(model.apply)

    return args, model, model_fn, params
