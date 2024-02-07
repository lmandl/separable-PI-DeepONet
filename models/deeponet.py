import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class DeepONet(nn.Module):
    # TODO: Adapt to stacked / unstacked version
    branch_layers: Sequence[int]
    trunk_layers: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, branch_x, trunk_x):

        # Branch network
        init = nn.initializers.glorot_normal()
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
