import jax.numpy as jnp
from jax import vmap


PRNGKey = jnp.array

def adj(a: jnp.ndarray) -> jnp.ndarray:
    """Returns adjoint matrix.

    Args:
        a: complex valued tensor of shape (..., n1, n2)

    Returns:
        complex valued tensor of shape (..., n2, n1)"""

    matrix_shape = a.shape[-2:]
    bs_shape = a.shape[:-2]
    a = a.reshape((-1, *matrix_shape))
    a = a.transpose((0, 2, 1))
    a = a.reshape((*bs_shape, matrix_shape[1], matrix_shape[0]))
    a = a.conj()
    return a

def transp(a: jnp.ndarray) -> jnp.ndarray:
    """Returns transposed matrix.

    Args:
        a: tensor of shape (..., n1, n2)

    Returns:
        tensor of shape (..., n2, n1)"""

    matrix_shape = a.shape[-2:]
    bs_shape = a.shape[:-2]
    a = a.reshape((-1, *matrix_shape))
    a = a.transpose((0, 2, 1))
    a = a.reshape((*bs_shape, matrix_shape[1], matrix_shape[0]))
    return a

def diag_part(a: jnp.ndarray) -> jnp.ndarray:
    """Returns the diagonal part of a matrix.

    Args:
        a: tensor of shape (..., n, n).

    Returns:
        tensor of shape (..., n)."""

    bs_shape = a.shape[:-2]
    matrix_shape = a.shape[-2:]
    a = vmap(jnp.diag)(a.reshape((-1, *matrix_shape)))
    a = a.reshape((*bs_shape, -1))
    return a
