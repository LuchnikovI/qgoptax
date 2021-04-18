import jax.numpy as jnp
import jax
from jax import random
from ..utils import adj, transp, diag_part, PRNGKey
from typing import Tuple


class StiefelManifold:

    def __init__(self,
                 retraction='svd',
                 metric='euclidean'):

        list_of_metrics = ['euclidean', 'canonical']
        list_of_retractions = ['svd', 'cayley', 'qr']

        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")
        if retraction not in list_of_retractions:
            raise ValueError("Incorrect retraction")
        
        self._retraction = retraction
        self._metric = metric

    def __repr__(self):

        return 'StiefelManifold(retraction={}, metric={})'.format(self.retraction, self.metric)

    def __str__(self):

        return 'StiefelManifold(retraction={}, metric={})'.format(self.retraction, self.metric)

    def inner(self,
              u: jnp.ndarray,
              vec1: jnp.ndarray,
              vec2: jnp.ndarray) -> jnp.ndarray:
        """Returns manifold wise inner product of vectors from
        a tangent space.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold.
            vec1: complex valued tensor of shape (..., n, p),
                a set of tangent vectors from the complex
                Stiefel manifold.
            vec2: complex valued tensor of shape (..., n, p),
                a set of tangent vectors from the complex
                Stiefel manifold.

        Returns:
            complex valued tensor of shape (..., 1, 1),
            manifold wise inner product

        Note:
            The complexity for the 'euclidean' metric is O(pn),
            the complexity for the 'canonical' metric is O(np^2)"""

        if self._metric == 'euclidean':
            s_sq = (vec1.conj() * vec2).sum(keepdims=True, axis=(-2, -1))
        elif self._metric == 'canonical':
            s_sq_1 = (vec1.conj() * vec2).sum(keepdims=True, axis=(-2, -1))
            vec1_dag_u = adj(vec1) @ u
            u_dag_vec2 = adj(u) @ vec2
            s_sq_2 = (u_dag_vec2 * transp(vec1_dag_u)).sum(axis=(-2, -1), keepdims=True)
            s_sq = s_sq_1 - 0.5 * s_sq_2
        return jnp.real(s_sq).astype(dtype=u.dtype)

    def egrad_to_rgrad(self,
                       u: jnp.ndarray,
                       egrad: jnp.ndarray) -> jnp.ndarray:
        """Returns the Riemannian gradient from an Euclidean gradient.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold.
            egrad: complex valued tensor of shape (..., n, p),
                a set of Euclidean gradients.

        Returns:
            complex valued tensor of shape (..., n, p),
            the set of Reimannian gradients.

        Note:
            The complexity is O(np^2)"""

        if self._metric == 'euclidean':
            return 0.5 * u @ (adj(u) @ egrad - adj(egrad) @ u) +\
                             egrad - u @ (adj(u) @ egrad)

        elif self._metric == 'canonical':
            return egrad - u @ (adj(egrad) @ u)

    def proj(self,
             u: jnp.ndarray,
             vec: jnp.ndarray) -> jnp.ndarray:
        """Returns projection of vectors on the tangen space
        of the complex Stiefel manifold.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold.
            vec: complex valued tensor of shape (..., n, p),
                a set of vectors to be projected.

        Returns:
            complex valued tensor of shape (..., n, p),
            a set of projected vectors

        Note:
            the complexity is O(np^2)"""

        return 0.5 * u @ (adj(u) @ vec - adj(vec) @ u) + vec - u @ (adj(u) @ vec)

    def retraction(self, 
                   u: jnp.ndarray,
                   vec: jnp.ndarray) -> jnp.ndarray:
        """Transports a set of points from the complex Stiefel
        manifold via a retraction map.

        Args:
            u: complex valued tensor of shape (..., n, p), a set
                of points to be transported.
            vec: complex valued tensor of shape (..., n, p),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n, p),
            a set of transported points.

        Note:
            The complexity for the 'svd' retraction is O(np^2),
            the complexity for the 'cayley' retraction is O(n^3),
            the complexity for the 'qr' retraction is O(np^2)"""

        if self._retraction == 'svd':
            new_u = u + vec
            v, _, w = jnp.linalg.svd(new_u, full_matrices=False)
            return v @ w

        elif self._retraction == 'cayley':
            W = vec @ adj(u) - 0.5 * u @ (adj(u) @ vec @ adj(u))
            W = W - adj(W)
            W_shape = W.shape
            Id = jnp.eye(W_shape[-1], dtype=W.dtype)
            return jnp.linalg.inv(Id - W / 2) @ (Id + W / 2) @ u

        elif self._retraction == 'qr':
            new_u = u + vec
            q, r = jnp.linalg.qr(new_u)
            diag = diag_part(r)
            sign = jax.numpy.sign(diag)[..., jnp.newaxis, :]
            return q * sign

    def vector_transport(self,
                         u: jnp.ndarray,
                         vec1: jnp.ndarray,
                         vec2: jnp.ndarray) -> jnp.ndarray:
        """Returns a vector tranported along an another vector
        via vector transport.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold, starting points.
            vec1: complex valued tensor of shape (..., n, p),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n, p),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n, p),
            a set of transported vectors.

        Note:
            The complexity for the 'svd' retraction is O(np^2),
            the complexity for the 'cayley' retraction is O(n^3),
            the complexity for the 'qr' retraction is O(np^2)"""

        new_u = self.retraction(u, vec2)
        return self.proj(new_u, vec1)

    def retraction_transport(self,
                             u: jnp.ndarray,
                             vec1: jnp.ndarray,
                             vec2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Performs a retraction and a vector transport simultaneously.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold, starting points.
            vec1: complex valued tensor of shape (..., n, p),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n, p),
                a set of direction vectors.

        Returns:
            two complex valued tensors of shape (..., n, p),
            a set of transported points and vectors."""

        new_u = self.retraction(u, vec2)
        return new_u, self.proj(new_u, vec1)

    def random(self,
               key: PRNGKey,
               shape: tuple,
               dtype=jnp.complex64) -> jnp.ndarray:
        """Returns a set of points from the complex Stiefel
        manifold generated randomly.

        Args:
            key: PRNGKey,
            shape: tuple of integer numbers (..., n, p),
                shape of a generated matrix.
            dtype: type of an output tensor, can be
                either jnp.complex64 or jnp.complex128.

        Returns:
            complex valued tensor of shape (..., n, p),
            a generated matrix."""

        list_of_dtypes = [jnp.complex64, jnp.complex128]

        if dtype not in list_of_dtypes:
            raise ValueError("Incorrect dtype")

        #TODO random generation with proper dtype
        u = random.normal(key, (*shape, 2))
        u = jax.lax.complex(u[..., 0], u[..., 1])
        u, _ = jnp.linalg.qr(u)
        return u.astype(dtype)

    def random_tangent(self,
                       key,
                       u: jnp.ndarray) -> jnp.ndarray:
        """Returns a set of random tangent vectors to points from
        the complex Stiefel manifold.

        Args:
            u: complex valued tensor of shape (..., n, p), points
                from the complex Stiefel manifold.

        Returns:
            complex valued tensor, set of tangent vectors to u."""

        u_shape = u.shape
        vec = random.normal(key, (*u_shape, 2))
        vec = jax.lax.complex(vec[..., 0], vec[..., 1])
        vec = vec.astype(dtype=u.dtype)
        vec = self.proj(u, vec)
        return vec

    def is_in_manifold(self,
                       u: jnp.ndarray,
                       tol=1e-5) -> jnp.ndarray:
        """Checks if a point is in the Stiefel manifold or not.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a point to be checked.
            tol: small real value showing tolerance.

        Returns:
            bolean tensor of shape (...)."""

        u_shape = u.shape
        Id = jnp.eye(u_shape[-1], dtype=u.dtype)
        udagu = adj(u) @ u
        diff = Id - udagu
        diff_norm = jnp.linalg.norm(diff, axis=(-2, -1))
        udagu_norm = jnp.linalg.norm(udagu, axis=(-2, -1))
        Id_norm = jnp.linalg.norm(Id, axis=(-2, -1))
        rel_diff = jnp.abs(diff_norm / jnp.sqrt(Id_norm * udagu_norm))
        return tol > rel_diff
