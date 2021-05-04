import jax.numpy as jnp
from typing import Tuple
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


def ab_decomposition(u: jnp.ndarray,
                     v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Decompose vector v as follows
    v = u @ a + u_orth @ b. If vector v is a tangent,
    then a matrix is  skew-hermitian.

    Args:
        u: array like of shape (..., n, m).
        v: array like of shape (..., n, m).

    Returns:
        elements of decomposition a, b, and u_orth."""

    n, m = u.shape[-2:]
    tail = u.shape[:-2]
    u = u.reshape((-1, n, m))
    v = v.reshape((-1, n, m))
    u_orth = vmap(lambda x: jnp.linalg.qr(x, mode='complete')[0])(u)[..., m:]
    a = u.conj().transpose((0, 2, 1)) @ v
    b = u_orth.conj().transpose((0, 2, 1)) @ v
    a = a.reshape((*tail, -1, m))
    b = b.reshape((*tail, -1, m))
    u_orth = u_orth.reshape((*tail, n, -1))
    return a, b, u_orth


def sylvester_solve(a: jnp.ndarray,
                    rho: jnp.ndarray,
                    eps: float=1e-6) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solves Sylvester equation x @ rho + rho @ x = 2 * a.

    Args:
        a: array like of shape (..., m, m).
        rho: array like of shape (..., m, m).
        eps: float value regularizing inversion of eigenvalues.

    Returns:
        array like of shape (..., m, m), solution of the equation
        and array like of shape (..., m, m), inverse preconditioner."""

    lmbd, u = jnp.linalg.eigh(rho)
    lmbd_inv = lmbd / (lmbd ** 2 + eps ** 2)
    rho_inv = (adj(u) * lmbd_inv) @ u
    
    dlmbd = lmbd[..., jnp.newaxis, :] + lmbd[..., jnp.newaxis]
    dlmbd = dlmbd / 2
    dlmbd_inv = dlmbd / (dlmbd ** 2 + eps ** 2)
    
    return u @ (dlmbd_inv * (adj(u) @ a @ u)) @ adj(u), rho_inv
