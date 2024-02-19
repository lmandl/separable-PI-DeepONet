import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class DeepONet(nn.Module):
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    split_branch: bool = False
    split_trunk: bool = False
    stacked: bool = False
    output_dim: int = 1
    n_branches: int = 1

    @nn.compact
    def __call__(self, branch_x, trunk_x):

        init = nn.initializers.glorot_normal()

        # Make sure the input is 2D (also during init)
        if len(branch_x.shape) == 1:
            branch_x = jnp.reshape(branch_x, (1, -1))
        if len(trunk_x.shape) == 1:
            trunk_x = jnp.reshape(trunk_x, (1, -1))

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
    n_branches: int = 1
    r: int = 128
    # Note: Work in Progress, split_trunk and separable DeepONet are currently not compatible

    @nn.compact
    def __call__(self, branch_x, trunk_x):

        init = nn.initializers.glorot_normal()

        # Make sure the input is 2D (also during init)
        if len(branch_x.shape) == 1:
            branch_x = jnp.reshape(branch_x, (1, -1))
        if len(trunk_x.shape) == 1:
            trunk_x = jnp.reshape(trunk_x, (1, -1))

        # Branch network
        # if stacked, then we have multiple branches
        # Branch networks
        if self.stacked:
            branch_out = []
            for j in range(self.branch_layers[-1]*self.r):
                for i, fs in enumerate(self.branch_layers[:-1]):
                    branch_x = nn.Dense(fs, kernel_init=init, name=f"branch_{j}_{i}")(branch_x)
                    branch_x = nn.activation.tanh(branch_x)
                branch_x = nn.Dense(1, name=f"branch_{j}_{i + 1}", kernel_init=init)(branch_x)
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
            branch_x = nn.Dense(self.branch_layers[-1]*self.r, name=f"branch_{i + 1}", kernel_init=init)(branch_x)
            # no output activation

        # reshape the output
        # reshape from [batch_size, r*p*output_dim] to [batch_size, r, p, output_dim] if split_branch is True
        if self.split_branch:
            branch_x = jnp.reshape(branch_x, (-1, self.r,
                                              branch_x.shape[1] // (self.r*self.output_dim), self.output_dim))
        else:
            # Reshape from [batch_size, r*p] to [batch_size, r, p, 1] if split_branch is false
            branch_x = jnp.reshape(branch_x, (-1, self.r, branch_x.shape[1] // self.r, 1))

        # Trunk network
        # we use a separable trunk, so we have one MLP for each trunk input dimension
        # output has rank r*output_dim
        # see also https://github.com/stnamjef/SPINN
        outputs = []

        dim = self.trunk_layers[0]
        # split inputs along the last axis so that every input dim is passed through a separate trunk
        trunk_x = jnp.split(trunk_x, dim, axis=-1)
        for j, x_i in enumerate(trunk_x):
            # for each output dimension, we have a separate trunk
            for i, fs in enumerate(self.trunk_layers[1:-1]):
                x_i = nn.Dense(fs, kernel_init=init, name=f"trunk_{i}_{j}")(x_i)
                x_i = nn.activation.tanh(x_i)
            x_i = nn.Dense(self.trunk_layers[-1]*self.r, name=f"trunk_{i+1}_{j}", kernel_init=init)(x_i)
            x_i = nn.activation.tanh(x_i)
            # Do we need activation of last layer? SPINN does not use it
            # reshape the output

            if self.split_trunk:
                # reshape from [batch_size, r*p*output_dim] to [batch_size, r, p, output_dim]
                x_i = jnp.reshape(x_i, (-1, self.r, x_i.shape[1] // (self.r*self.output_dim), self.output_dim))
            else:
                # reshape from [batch_size, r*p] to [batch_size, r, p, 1]
                x_i = jnp.reshape(x_i, (-1, self.r, x_i.shape[1] // self.r, 1))

            outputs += [x_i]

        # we now calculate the einsum for each input dimension
        net_outs = []
        for trunk_out in outputs:
            # Input shapes:
            # branch: [batch_size, r, p, output_dim]
            # trunk: [batch_size, r, p, output_dim]
            # output: [batch_size, r, output_dim]
            net_outs += [jnp.einsum('ijkl,ijkl->ijl', branch_x, trunk_out)]

        # at this point we have a list of outputs, one for each input dimension
        # The shape of each output is [batch_size, r, output_dim]
        # The final output will be of size [[batch_size]*input_dim, output_dim]
        result = output_einsum_sep(net_outs)

        # Add bias
        bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
        result += bias

        return result


def output_einsum_sep(list_of_outputs):
    # TODO: Reduce arguments by inferring lengths from input shapes
    # Output einsum (outer product over trunk input) for separable DeepONet
    # Then summation over r
    # Similar to SPINNs
    raise NotImplementedError
