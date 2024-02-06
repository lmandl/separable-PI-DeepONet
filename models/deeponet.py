import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class DeepONet(nn.Module):
    # TODO: Adapt to stacked / unstacked version
    branch_layers: tuple
    branch_ac: str
    trunk_layers: tuple
    trunk_ac: str
    use_bias: bool
    output_dim: int

    def setup(self):
        self.branch = FNN(features=self.branch_layers, activation=self.branch_ac, output_activation=False)
        self.trunk = FNN(features=self.trunk_layers, activation=self.trunk_ac, output_activation=True)
        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.output_dim,))

    def __call__(self, inputs):
        branch_input, trunk_input = inputs
        branch_output = self.branch(branch_input)
        trunk_output = self.trunk(trunk_input)
        # TODO: Check formats and shapes
        result = jnp.sum(branch_output*trunk_output, axis=-1)

        if self.use_bias:
            result += self.bias

        return result


class FNN(nn.Module):
    """
    see https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html
    """
    features: tuple
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


def train_step(optimizer, model, x, y):
    def loss_fn(model_in):
        y_pred = model_in(x)
        return jnp.mean((y - y_pred) ** 2)

    loss, grad = jax.value_and_grad(loss_fn)(model)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss
