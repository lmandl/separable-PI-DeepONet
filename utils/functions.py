import jax.numpy as jnp
import jax
import optax
from functools import partial


def relative_l2(u_gt, u):
    """
    Computes the relative l2 error between u and u_gt
    """
    # if relative_l2 is called with a list of arrays, we stack them
    if isinstance(u, list) and isinstance(u_gt, list):
        u = jnp.dstack([jnp.array(elem) for elem in u])
        u_gt = jnp.dstack([jnp.array(elem) for elem in u_gt])
    rel_l2 = jnp.linalg.norm(u - u_gt) / jnp.linalg.norm(u_gt)
    return rel_l2


def mse(u, u_gt):
    """
    Computes the mean squared error between u and u_gt
    """
    # if mse is called with a list of arrays, we stack them
    if isinstance(u, list) and isinstance(u_gt, list):
        u = jnp.dstack([jnp.array(elem) for elem in u])
        u_gt = jnp.dstack([jnp.array(elem) for elem in u_gt])
    return jnp.mean((u - u_gt) ** 2)


def mse_loss(y_true, y_pred):
    """
    short version for call in loss functions
    """
    return jnp.mean((y_true - y_pred) ** 2)


# single update function
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state


@partial(jax.jit, static_argnums=(0,))
def loss_and_grad(model_fn, params, x, y):
    def loss_fn(params_loss):
        y_pred = model_fn(params_loss, x[0], x[1])
        mse_val = mse_loss(y, y_pred)
        return mse_val
    return jax.value_and_grad(loss_fn)(params)


@partial(jax.jit, static_argnums=(0,))
def train_error(model_fn, params, x, y):
    y_pred = model_fn(params, x[0], x[1])
    rel_l2 = relative_l2(y, y_pred)
    return rel_l2
