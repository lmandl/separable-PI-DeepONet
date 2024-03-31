import jax.numpy as jnp
import jax
from jax import jvp, vjp
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


def mse(y_true, y_pred):
    """
    Computes the mean squared error between u and u_gt
    """
    return jnp.mean((y_true - y_pred) ** 2)


@partial(jax.jit, static_argnums=(0,))
def apply_net(model_fn, params, branch_input, *trunk_in):
    # Define forward pass that takes series of trunk inputs
    # if entries in trunk_in are squeezable, squeeze them
    trunk_in = [jnp.squeeze(entry) if entry is not list or tuple else entry for entry in trunk_in]
    trunk_input = jnp.stack(trunk_in, axis=-1)
    out = model_fn(params, branch_input, trunk_input)
    # Reshape to vector for single output for easier gradient computation
    out = jnp.squeeze(out)
    return out


# single update function
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state


@partial(jax.jit, static_argnums=(0, 1, 2))
def step(optimizer, loss_fn, model_fn, opt_state, params_step, ics_batch, bcs_batch, res_batch):
    loss, gradient = jax.value_and_grad(loss_fn, argnums=1)(model_fn, params_step, ics_batch, bcs_batch, res_batch)
    updates, opt_state = optimizer.update(gradient, opt_state)
    params_step = optax.apply_updates(params_step, updates)

    return loss, params_step, opt_state


@partial(jax.jit, static_argnums=(0,))
def train_error(model_fn, params, x, y):
    y_pred = model_fn(params, x[0], x[1])
    rel_l2 = relative_l2(y, y_pred)
    return rel_l2


# Following functions taken from https://github.com/stnamjef/SPINN

# forward over forward
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out