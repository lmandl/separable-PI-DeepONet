import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class UnstackedDeepONet(nn.Module):
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, branch_x, trunk_x):

        init = nn.initializers.glorot_normal()

        # Branch network
        for i, fs in enumerate(self.branch_layers[:-1]):
            branch_x = nn.Dense(fs, kernel_init=init, name=f"branch_{i}")(branch_x)
            branch_x = nn.activation.tanh(branch_x)
        branch_x = nn.Dense(self.branch_layers[-1], name=f"branch_{i+1}", kernel_init=init)(branch_x)
        # no output activation

        # Trunk network
        for i, fs in enumerate(self.trunk_layers):
            trunk_x = nn.Dense(fs, kernel_init=init, name=f"trunk_{i}")(trunk_x)
            trunk_x = nn.activation.tanh(trunk_x)

        # Compute the final output
        result = jnp.sum(branch_x*trunk_x, axis=-1)

        # Add bias
        bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
        result += bias

        return result


class StackedDeepONet(nn.Module):
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    output_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, branch_x, trunk_x):

        init = nn.initializers.glorot_normal()

        # Branch networks
        branch_out = []
        for j in range(self.hidden_dim):
            for i, fs in enumerate(self.branch_layers[:-1]):
                branch_x = nn.Dense(fs, kernel_init=init, name=f"branch_{j}_{i}")(branch_x)
                branch_x = nn.activation.tanh(branch_x)
            branch_x = nn.Dense(self.branch_layers[-1], name=f"branch_{j}_{i+1}", kernel_init=init)(branch_x)
            # no output activation
            branch_out.append(branch_x)

        # Combine the outputs of each branch into a JAX array
        branch_out = jnp.stack(branch_out, axis=-1).squeeze()

        # Trunk network
        for i, fs in enumerate(self.trunk_layers):
            trunk_x = nn.Dense(fs, kernel_init=init, name=f"trunk_{i}")(trunk_x)
            trunk_x = nn.activation.tanh(trunk_x)

        # Compute the final output
        result = jnp.sum(branch_out * trunk_x, axis=-1)

        # Add bias
        bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
        result += bias

        return result
