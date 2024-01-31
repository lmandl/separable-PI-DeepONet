import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class DeepONet(nn.Module):
    branch_layers: Sequence[int]
    branch_ac: str
    trunk_layers: Sequence[int]
    trunk_ac: str
    use_bias: bool
    output_dim: int

    def setup(self):
        keys = jax.random.split(self.key, 2)
        self.trunk = FNN(features=self.trunk_layers, activation=self.trunk_ac, output_activation=True, key=keys[0])
        self.branch = FNN(features=self.branch_layers, activation=self.branch_ac, output_activation=False, key=keys[1])
        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.ouput_dim,))

    def __call__(self, branch_input, trunk_input):
        branch_output = self.branch(branch_input)
        trunk_output = self.trunk(trunk_input)

        result = jnp.sum(branch_output*trunk_output)

        if self.use_bias:
            result += self.bias

        return result


class FNN(nn.Module):
    """
    see https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html
    """
    features: Sequence[int]
    activation: str
    output_activation: bool

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

        if self.activation == "tanh":
            self.ac_fun = nn.tanh
        elif self.activation == "swish":
            self.ac_fun = nn.swish
        elif self.activation == "relu":
            self.ac_fun = nn.relu
        elif self.activation == "sigmoid":
            self.ac_fun = nn.sigmoid
        else:
            raise NotImplementedError

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = self.ac_fun(x)
        if self.output_activation:
            x = self.ac_fun(x)
        return x
