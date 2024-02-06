import jax.numpy as jnp
import jax
import optax


def relative_l2(u_gt, u):
    """
    Computes the relative l2 error between u and u_gt
    """
    # if relative_l2 is called with a list of arrays, we stack them
    if isinstance(u, list) and isinstance(u_gt, list):
        u = jnp.dstack([jnp.array(elem) for elem in u])
        u_gt = jnp.dstack([jnp.array(elem) for elem in u_gt])
    return jnp.linalg.norm(u - u_gt) / jnp.linalg.norm(u_gt)


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


def train_step(optimizer, model, opt_state_fn, params_fn, x, y):
    def loss_fn(params_loss):
        y_pred = model.apply(params_loss, x)
        return mse_loss(y, y_pred)

    loss, grads = jax.value_and_grad(loss_fn)(params_fn)
    updates, opt_state_fn = optimizer.update(grads, opt_state_fn)
    params_fn = optax.apply_updates(params_fn, updates)
    return opt_state_fn, params_fn, loss


def train_error(model, params, x, y):
    y_pred = model.apply(params, x)
    return relative_l2(y, y_pred)
