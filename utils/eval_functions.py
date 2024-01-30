import jax.numpy as jnp


def relative_l2(u, u_gt):
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
