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
    # TODO: check vmap use for batch input

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
            for j in range(self.n_branches):
                for i, fs in enumerate(self.branch_layers[:-1]):
                    branch_x = nn.Dense(fs, kernel_init=init, name=f"branch_{j}_{i}")(branch_x)
                    branch_x = nn.activation.tanh(branch_x)
                branch_x = nn.Dense(self.branch_layers[-1], name=f"branch_{j}_{i+1}", kernel_init=init)(branch_x)
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

        # Trunk network
        for i, fs in enumerate(self.trunk_layers):
            trunk_x = nn.Dense(fs, kernel_init=init, name=f"trunk_{i}")(trunk_x)
            trunk_x = nn.activation.tanh(trunk_x)
        # reshape the output
        # reshape from [batch_size, p*output_dim] to [batch_size, p, output_dim] if split_trunk is True
        if self.split_trunk:
            trunk_x = jnp.reshape(trunk_x, (-1, trunk_x.shape[1] // self.output_dim, self.output_dim))

        # Compute the final output
        result = output_einsum(self.split_branch, self.split_trunk, branch_x, trunk_x)

        # Add bias
        bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
        result += bias

        return result


def output_einsum(split_branch, split_trunk, branch, trunk):
    # Input shapes:
    # branch: [batch_size, p, output_dim] if split_branch is True
    # branch: [batch_size, output_dim] if split_branch is False
    # trunk: [batch_size, p, output_dim] if split_trunk is True
    # trunk: [batch_size, output_dim] if split_trunk is False
    # output: [batch_size, output_dim]
    if split_branch and split_trunk:
        return jnp.einsum('ijk,ijk->ik', branch, trunk)
    elif split_branch and not split_trunk:
        return jnp.einsum('ijk,ij->ik', branch, trunk)
    elif split_trunk and not split_branch:
        return jnp.einsum('ij,ijk->ik', branch, trunk)
    else:
        return jnp.einsum('ij,ij->i', branch, trunk)
