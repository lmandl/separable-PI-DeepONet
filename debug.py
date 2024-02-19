import jax.numpy as jnp
from models import DeepONet
import jax
import optax
from tqdm import trange
from utils import loss_and_grad, update_model

if __name__ == '__main__':
    split_branch = True
    split_trunk = True
    batch_size = 3
    input_dim_trunk = 4
    n_sensors = 10
    input_dim_branch = 101
    output_dim = 2
    hidden_dim = 5
    stacked_do = False
    separable_trunk = True
    r = 5

    # Overriding split_trunk and split_branch if num_outputs is 1
    if output_dim == 1:
        split_trunk = False
        split_branch = False

    # if split_trunk is True and separable_trunk is false
    # then the hidden_dim is multiplied by the output_dim
    # if split_trunk is false and separable_trunk is false
    # then the trunk output is the same as the hidden_dim
    # if split_trunk is false and separable_trunk is true
    # then the output of the separable_trunk is hidden_dim*r
    # if split_trunk is true and separable_trunk is true
    # then the output of the separable_trunk is hidden_dim*r*output_dim
    # the multiplication with r is done in the deeponet class

    if split_trunk:
        trunk_layers = [input_dim_trunk]+[5]+[output_dim*hidden_dim]
    else:
        trunk_layers = [input_dim_trunk]+[5]+[hidden_dim]

    # Stacked or unstacked DeepONet
    # dimensions for splits are calculated once here before creating the model
    # add input and output features to branch layers for unstacked DeepONet
    # if split_branch is True, the output features are split into n groups for n outputs
    if split_branch:
        branch_layers = [n_sensors * input_dim_branch]+[5]+[output_dim*hidden_dim]
    else:
        branch_layers = [n_sensors * input_dim_branch]+[5]+[hidden_dim]
    # build actual model

    trunk_layers = tuple(trunk_layers)
    branch_layers = tuple(branch_layers)

    model = DeepONet(branch_layers, trunk_layers, split_branch, split_trunk, stacked_do,
                     output_dim)

    # Set random seed and key
    key = jax.random.PRNGKey(1337)

    # Split key
    key, subkey = jax.random.split(key)

    # Initialize parameters
    params = model.init(subkey, jnp.ones(input_dim_branch), jnp.ones(input_dim_trunk))
    print(params)

    # Split key
    key, subkey = jax.random.split(key)

    # Generate inputs
    branch_input = jax.random.uniform(key, shape=(batch_size, input_dim_branch))
    trunk_input = jax.random.uniform(key, shape=(batch_size, input_dim_trunk))

    out = model.apply(params, branch_input, trunk_input)
    print(out)
    fake_out = jax.random.uniform(key, shape=out.shape)

    # Fake train the model
    # Define optimizer with optax (ADAM)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    # List of TODOs
    # TODO: Check Point
    # TODO: Visualization
    model_fn = model.apply

    # Note:: train and test data has form [[branch_input, trunk_input], output]
    train_data = ((branch_input, trunk_input), fake_out)

    for epoch in trange(100):

        # Train model
        loss, gradient = loss_and_grad(model_fn, params, train_data[0], train_data[1])
        params, opt_state = update_model(optimizer, gradient, params, opt_state)
