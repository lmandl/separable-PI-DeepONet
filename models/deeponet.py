import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple


class DeepONet(nn.Module):
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    split_branch: bool = False
    split_trunk: bool = False
    stacked: bool = False
    output_dim: int = 1

    @nn.compact
    def __call__(self, branch_x, trunk_x):

        init = nn.initializers.glorot_normal()

        # Branch network
        # if stacked, then we have multiple branches
        # Branch networks
        if self.stacked:
            branch_out = []
            for j in range(self.branch_layers[-1]):
                for i, fs in enumerate(self.branch_layers[:-1]):
                    branch_x = nn.Dense(fs, kernel_init=init, name=f"branch_{j}_{i}")(branch_x)
                    branch_x = nn.activation.tanh(branch_x)
                branch_x = nn.Dense(1, name=f"branch_{j}_{i+1}", kernel_init=init)(branch_x)
                # no output activation
                branch_out.append(branch_x)

            # transform list of the output into [batch_size, p*output_dim]
            # Combine the outputs of each branch into a JAX array
            branch_x = jnp.reshape(jnp.array(branch_out), (-1, len(branch_out)))

        # otherwise, we have a single branch
        else:
            for i, fs in enumerate(self.branch_layers[:-1]):
                branch_x = nn.Dense(fs, kernel_init=init, name=f"branch_{i}")(branch_x)
                branch_x = nn.activation.tanh(branch_x)
            branch_x = nn.Dense(self.branch_layers[-1], name=f"branch_{i+1}", kernel_init=init)(branch_x)
            # no output activation

        # reshape the output
        # reshape from [batch_size, p*output_dim] to [batch_size, p, output_dim] if split_branch is True
        if self.split_branch:
            branch_x = jnp.reshape(branch_x, (-1, branch_x.shape[1] // self.output_dim, self.output_dim))
        else:
        # reshape from [batch_size, p] to [batch_size, p, 1] if split_branch is false
            branch_x = jnp.reshape(branch_x, (-1, branch_x.shape[1], 1))

        # Trunk network
        for i, fs in enumerate(self.trunk_layers):
            trunk_x = nn.Dense(fs, kernel_init=init, name=f"trunk_{i}")(trunk_x)
            trunk_x = nn.activation.tanh(trunk_x)

        # reshape the output
        # reshape from [batch_size, p*output_dim] to [batch_size, p, output_dim] if split_trunk is True
        if self.split_trunk:
            trunk_x = jnp.reshape(trunk_x, (-1, trunk_x.shape[1] // self.output_dim, self.output_dim))
        else:
            # reshape from [batch_size, p] to [batch_size, p, 1] if split_trunk is false
            trunk_x = jnp.reshape(trunk_x, (-1, trunk_x.shape[1], 1))

        # Compute the final output
        # Input shapes:
        # branch: [batch_size, p, output_dim]
        # trunk: [batch_size, p, output_dim]
        # output: [batch_size, output_dim]
        result = jnp.einsum('ijk,ijk->ik', branch_x, trunk_x)

        # Add bias
        bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
        result += bias

        return result


class SeparableDeepONet(nn.Module):
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    split_branch: bool = False
    split_trunk: bool = False
    stacked: bool = False
    output_dim: int = 1
    r: int = 128

    @nn.compact
    def __call__(self, branch_x, *trunk_x):

        init = nn.initializers.glorot_normal()

        # Trunk input has shape [input_1, input_2, ..., input_dim]
        # where each input has shape [N_i, 1] and N_i is the number of samples in the i-th input
        trunk_x = [*trunk_x]

        # Branch network
        # if stacked, then we have multiple branches
        # Branch networks
        if self.stacked:
            branch_out = []
            for j in range(self.branch_layers[-1]):
                branch_x_j = branch_x
                for i, fs in enumerate(self.branch_layers[:-1]):
                    branch_x_j = nn.Dense(fs, kernel_init=init, name=f"branch_{j}_{i}")(branch_x_j)
                    branch_x_j = nn.activation.tanh(branch_x_j)
                branch_x_j = nn.Dense(1, name=f"branch_{j}_{i+1}", kernel_init=init)(branch_x_j)
                # no output activation
                branch_out.append(branch_x_j)

            # transform list of the output into [batch_size, p*output_dim]
            # Combine the outputs of each branch into a JAX array
            branch_x = jnp.reshape(jnp.array(branch_out), (-1, len(branch_out)))

        # otherwise, we have a single branch
        else:
            for i, fs in enumerate(self.branch_layers[:-1]):
                branch_x = nn.Dense(fs, kernel_init=init, name=f"branch_{i}")(branch_x)
                branch_x = nn.activation.tanh(branch_x)
            branch_x = nn.Dense(self.branch_layers[-1], name=f"branch_{i + 1}", kernel_init=init)(branch_x)
            # no output activation

        # reshape the output
        # reshape from [batch_size, p*output_dim] to [batch_size, p, output_dim] if split_branch is True
        if self.split_branch:
            branch_x = jnp.reshape(branch_x, (-1, branch_x.shape[1] // self.output_dim, self.output_dim))
        else:
            # reshape from [batch_size, p] to [batch_size, p, 1] if split_branch is false
            branch_x = jnp.reshape(branch_x, (-1, branch_x.shape[1], 1))

        # Trunk network
        # we use a separable trunk, so we have one MLP for each trunk input dimension
        # output has rank r*output_dim
        # see also https://github.com/stnamjef/SPINN
        outputs = []

        for j, x_i in enumerate(trunk_x):
            # for each input dimension, we have a separate trunk
            for i, fs in enumerate(self.trunk_layers[:-1]):
                x_i = nn.Dense(fs, kernel_init=init, name=f"trunk_{i}_{j}")(x_i)
                x_i = nn.activation.tanh(x_i)
            x_i = nn.Dense(self.trunk_layers[-1]*self.r, name=f"trunk_{i+1}_{j}", kernel_init=init)(x_i)
            x_i = nn.activation.tanh(x_i)
            # Note: SPINN does not use output activation

            # reshape the output
            # Note: batch sizes are not necessarily the same for each input dimension
            x_i = jnp.reshape(x_i, (-1, self.r, self.trunk_layers[-1], 1))

            # reshape the output
            # reshape from [batch_size, r*p*output_dim] to [batch_size, r, p, output_dim] if split_trunk is True
            if self.split_trunk:
                x_i = jnp.reshape(x_i, (-1, self.r, self.trunk_layers[-1] // self.output_dim, self.output_dim))
            else:
                # reshape from [batch_size, p] to [batch_size, p, 1] if split_trunk is false
                x_i = jnp.reshape(x_i, (-1, self.r, self.trunk_layers[-1], 1))

            outputs += [x_i]

        # we now calculate the einsum for each input dimension over p
        # note that batch_sizes may differ in the trunks, hence we use the adaptive einsum
        trunk_outs = trunk_r_einsum(outputs)

        # trunk_outs has shape [batch_size1, batch_size2,..., batch_sizeN, p, output_dim]
        # We have different input batches for the trunk and branch and then obtain a tensor of shape
        # [batch_branch, batch_size1, batch_size2,..., batch_sizeN, output_dim]

        # Create arguments for einsum
        # branch_x has shape [batch_size_b, p, output_dim]
        # trunk_outs has shape [batch_size1, batch_size2,..., batch_sizeN, p, output_dim]
        # Create einsum arguments (input_batches_branch=0, input_batches_trunk =1,2,..,N, p=N+1, output_dim=N+2)
        # Note: swapping of axis might lead to unwanted behavior such as transposing the output
        n_trunk = len(trunk_outs.shape)-2
        trunk_out_args = [i+1 for i in range(n_trunk)] + [n_trunk+1, n_trunk+2]
        result_args = [0] + [i+1 for i in range(n_trunk)] + [n_trunk+2]

        # apply einsum
        result = jnp.einsum(branch_x, [0, n_trunk+1, n_trunk+2], trunk_outs, trunk_out_args, result_args)

        # Add bias
        bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
        result += bias

        return result

class SeparableBranchCNNDeepONet(nn.Module):
    cnn_branch_layers: Tuple
    trunk_layers: Sequence[int]
    split_branch: bool = False
    split_trunk: bool = False
    stacked: bool = False
    output_dim: int = 1
    r: int = 128

    # TODO: Could be combined with the SeparableDeepONet class to avoid code duplication by restructuring the class

    @nn.compact
    def __call__(self, branch_x, *trunk_x):
        init = nn.initializers.glorot_normal()

        # Trunk input has shape [input_1, input_2, ..., input_dim]
        # where each input has shape [N_i, 1] and N_i is the number of samples in the i-th input
        trunk_x = [*trunk_x]

        # Branch network
        # if stacked, then we have multiple branches
        # Branch networks
        if self.stacked:
            branch_out = []
            debug = self.cnn_branch_layers[-1][0]
            for j in range(self.cnn_branch_layers[-1][0]):
                branch_x_j = branch_x
                for i, fs in enumerate(self.cnn_branch_layers):
                    if len(fs) == 4:
                        # Conv Block
                        branch_x_j = nn.Conv(fs[0], kernel_size=(fs[1], fs[2]), name=f"branch_{j}_{i}")(branch_x_j)
                        if fs[3] == "tanh":
                            branch_x_j = nn.activation.tanh(branch_x_j)
                        elif fs[3] == "relu":
                            branch_x_j = nn.activation.relu(branch_x_j)
                        elif fs[3] == "sigmoid":
                            branch_x_j = nn.activation.sigmoid(branch_x_j)
                        else:
                            raise ValueError("Unknown activation function")
                    elif len(fs) == 5:
                        # Pooling Block
                        if fs[0] == "avg_pool":
                            branch_x_j = nn.avg_pool(branch_x_j, window_shape=(fs[1], fs[2]), strides=(fs[3], fs[4]))
                        elif fs[0] == "max_pool":
                            branch_x_j = nn.max_pool(branch_x_j, window_shape=(fs[1], fs[2]), strides=(fs[3], fs[4]))
                        else:
                            raise ValueError("Unknown pooling type")
                    elif len(fs) == 2:
                        # Dense Block
                        branch_x_j = nn.Dense(fs[0], kernel_init=init, name=f"branch_{j}_{i}")(branch_x_j)
                        if fs[1] == "tanh":
                            branch_x_j = nn.activation.tanh(branch_x_j)
                        elif fs[1] == "relu":
                            branch_x_j = nn.activation.relu(branch_x_j)
                        elif fs[1] == "sigmoid":
                            branch_x_j = nn.activation.sigmoid(branch_x_j)
                        elif fs[1] == "None":
                            # No activation for the output layer
                            branch_x_j = branch_x_j
                        else:
                            raise ValueError("Unknown activation function")
                    elif len(fs) == 1:
                        # Flatten Block
                        branch_x_j = branch_x_j.reshape((branch_x_j.shape[0], -1))
                    else:
                        raise ValueError("Unknown layer type")
                branch_x_j = nn.Dense(1, name=f"branch_{j}_{i+1}", kernel_init=init)(branch_x_j)
                # no output activation
                branch_out.append(branch_x_j)
            # transform list of the output into [batch_size, p*output_dim]
            # Combine the outputs of each branch into a JAX array
            branch_x = jnp.reshape(jnp.array(branch_out), (-1, len(branch_out)))

        # otherwise, we have a single branch
        else:
            for i, fs in enumerate(self.cnn_branch_layers):
                if len(fs) == 4:
                    # Conv Block
                    branch_x = nn.Conv(fs[0], kernel_size=(fs[1], fs[2]), name=f"branch_{i}")(branch_x)
                    if fs[3] == "tanh":
                        branch_x = nn.activation.tanh(branch_x)
                    elif fs[3] == "relu":
                        branch_x = nn.activation.relu(branch_x)
                    elif fs[3] == "sigmoid":
                        branch_x = nn.activation.sigmoid(branch_x)
                    else:
                        raise ValueError("Unknown activation function")
                elif len(fs) == 5:
                    # Pooling Block
                    if fs[0] == "avg_pool":
                        branch_x = nn.avg_pool(branch_x, window_shape=(fs[1], fs[2]), strides=(fs[3], fs[4]))
                    elif fs[0] == "max_pool":
                        branch_x = nn.max_pool(branch_x, window_shape=(fs[1], fs[2]), strides=(fs[3], fs[4]))
                    else:
                        raise ValueError("Unknown pooling type")
                elif len(fs) == 2:
                    # Dense Block
                    branch_x = nn.Dense(fs[0], kernel_init=init, name=f"branch_{i}")(branch_x)
                    if fs[1] == "tanh":
                        branch_x = nn.activation.tanh(branch_x)
                    elif fs[1] == "relu":
                        branch_x = nn.activation.relu(branch_x)
                    elif fs[1] == "sigmoid":
                        branch_x = nn.activation.sigmoid(branch_x)
                    elif fs[1] == "None":
                        # No activation for the output layer
                        branch_x = branch_x
                    else:
                        raise ValueError("Unknown activation function")
                elif len(fs) == 1:
                    # Flatten Block to [batch_size, -1]
                    branch_x = branch_x.reshape((branch_x.shape[0], -1))
                else:
                    raise ValueError("Unknown layer type")

        # reshape the output
        # reshape from [batch_size, p*output_dim] to [batch_size, p, output_dim] if split_branch is True
        if self.split_branch:
            branch_x = jnp.reshape(branch_x, (-1, branch_x.shape[1] // self.output_dim, self.output_dim))
        else:
            # reshape from [batch_size, p] to [batch_size, p, 1] if split_branch is false
            branch_x = jnp.reshape(branch_x, (-1, branch_x.shape[1], 1))

        # Trunk network
        # we use a separable trunk, so we have one MLP for each trunk input dimension
        # output has rank r*output_dim
        # see also https://github.com/stnamjef/SPINN
        outputs = []

        for j, x_i in enumerate(trunk_x):
            # for each input dimension, we have a separate trunk
            for i, fs in enumerate(self.trunk_layers[:-1]):
                x_i = nn.Dense(fs, kernel_init=init, name=f"trunk_{i}_{j}")(x_i)
                x_i = nn.activation.tanh(x_i)
            x_i = nn.Dense(self.trunk_layers[-1] * self.r, name=f"trunk_{i + 1}_{j}", kernel_init=init)(x_i)
            x_i = nn.activation.tanh(x_i)
            # Note: SPINN does not use output activation

            # reshape the output
            # Note: batch sizes are not necessarily the same for each input dimension
            x_i = jnp.reshape(x_i, (-1, self.r, self.trunk_layers[-1], 1))

            # reshape the output
            # reshape from [batch_size, r*p*output_dim] to [batch_size, r, p, output_dim] if split_trunk is True
            if self.split_trunk:
                x_i = jnp.reshape(x_i, (-1, self.r, self.trunk_layers[-1] // self.output_dim, self.output_dim))
            else:
                # reshape from [batch_size, p] to [batch_size, p, 1] if split_trunk is false
                x_i = jnp.reshape(x_i, (-1, self.r, self.trunk_layers[-1], 1))

            outputs += [x_i]

        # we now calculate the einsum for each input dimension over p
        # note that batch_sizes may differ in the trunks, hence we use the adaptive einsum
        trunk_outs = trunk_r_einsum(outputs)

        # trunk_outs has shape [batch_size1, batch_size2,..., batch_sizeN, p, output_dim]
        # We have different input batches for the trunk and branch and then obtain a tensor of shape
        # [batch_branch, batch_size1, batch_size2,..., batch_sizeN, output_dim]

        # Create arguments for einsum
        # branch_x has shape [batch_size_b, p, output_dim]
        # trunk_outs has shape [batch_size1, batch_size2,..., batch_sizeN, p, output_dim]
        # Create einsum arguments (input_batches_branch=0, input_batches_trunk =1,2,..,N, p=N+1, output_dim=N+2)
        # Note: swapping of axis might lead to unwanted behavior such as transposing the output
        n_trunk = len(trunk_outs.shape) - 2
        trunk_out_args = [i + 1 for i in range(n_trunk)] + [n_trunk + 1, n_trunk + 2]
        result_args = [0] + [i + 1 for i in range(n_trunk)] + [n_trunk + 2]

        # apply einsum
        result = jnp.einsum(branch_x, [0, n_trunk + 1, n_trunk + 2], trunk_outs, trunk_out_args, result_args)

        # Add bias
        bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
        result += bias

        return result

def trunk_r_einsum(list_of_trunk_outs):
    # Input is a list of N trunk outputs, one for each input dimension in the trunk
    # Each trunk output has shape [batch_size, r, p, output_dim]
    # r, p, and output_dim are the same for all trunk outputs
    # batch_size may differ between trunk outputs
    # it returns a single output of shape [batch_size1, batch_size2,..., batch_sizeN, p, output_dim]

    i_dim = len(list_of_trunk_outs)

    # Create einsum arguments (input_batches=0,1,..,N-1, r=N, p=N+1, output_dim=N+2)

    einsum_args = [[elem, [i, i_dim, i_dim+1, i_dim+2]] for i, elem in enumerate(list_of_trunk_outs)]

    # Flatten the list of lists into a single list
    einsum_args = [item for sublist in einsum_args for item in sublist]

    # Create output dimension [batch_size1, batch_size2,..., batch_sizeN, p, output_dim]
    output_dim = [i for i in range(i_dim)] + [i_dim+1, i_dim+2]

    # Unpack the list of lists into separate arguments for einsum
    # We use einsum in the alternative way: einsum(op0, sublist0, op1, sublist1, ..., [sublistout])
    result = jnp.einsum(*einsum_args, output_dim)

    return result
