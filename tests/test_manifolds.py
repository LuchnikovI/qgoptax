import jax
import jax.numpy as jnp
from jax import random


class CheckManifolds():

    def __init__(self, key, m, descr, shape, tol):

        self.m = m  # example of a manifold
        self.descr = descr
        self.shape = shape  # shape of a tensor
        self.u = m.random(key, shape)  # point from a manifold
        self.v1 = m.random_tangent(key, self.u)  # first tangent vector
        self.v2 = m.random_tangent(key, self.u)  # second tangent vector
        self.zero = self.u * 0.  # zero vector
        self.tol = tol  # tolerance of a test
        self.key = random.split(key)[0]  # PRNGKey

    def _proj_of_tangent(self):
        """
        Checking m.proj: Projection of a tangent vector should remain the same
        after application of the proj method.

        Args:

        Returns:
            jnp scalar, maximum value of error"""

        err = jnp.linalg.norm(self.v1 - self.m.proj(self.u, self.v1), axis=(-2, -1))
        err = jnp.real(err).max()
        return err

    def _inner_proj_matching(self):
        """Checking matching between m.inner and m.proj
        (suitable only for embedded manifolds with globally defined metric)

        Args:

        Returns:
            jnp scalar, maximum value of error"""

        xi = random.normal(self.key, (*self.shape, 2))
        xi = jax.lax.complex(xi[..., 0], xi[..., 1])

        xi_proj = self.m.proj(self.u, xi)
        first_inner = self.m.inner(self.u, xi_proj, self.v1)
        second_inner = self.m.inner(self.u, xi, self.v1)
        err = jnp.abs(first_inner - second_inner)
        err = err[..., 0, 0]
        err = err.max()
        return err

    def _retraction(self):
        """
        Checking retraction
        Page 46, Nikolas Boumal, An introduction to optimization on smooth
        manifolds.
        1) Rx(0) = x (Identity mapping)
        2) DRx(o)[v] = v : introduce v->t*v, calculate err=dRx(tv)/dt|_{t=0}-v
        3) Presence of a new point in a manifold

        Args:

        Returns:
            list with three tf scalars. First two scalars give maximum
            violation of first two conditions, third scalar shows wether
            any point
        """

        t = 1e-8  # dt for numerical derivative

        # transition along zero vector (first cond)
        err1 = self.u - self.m.retraction(self.u, self.zero)
        err1 = jnp.real(jnp.linalg.norm(err1, axis=(-2, -1)))

        # third order approximation of differential of retraction (second cond)
        retr_forward = self.m.retraction(self.u, t * self.v1)
        retr_forward_two_steps = self.m.retraction(self.u, 2 * t * self.v1)
        retr_back = self.m.retraction(self.u, -t * self.v1)
        dretr = (-2 * retr_back - 3 * self.u + 6 * retr_forward - retr_forward_two_steps)
        dretr = dretr / (6 * t)
        err2 = jnp.real(jnp.linalg.norm(dretr - self.v1, axis=(-2, -1)))

        # presence of a new point in a manifold (third cond)
        err3 = self.m.is_in_manifold(self.m.retraction(self.u, self.v1),
                                                                tol=self.tol)
        err1 = err1.max()
        err2 = err2.max()
        err3 = jnp.any(err3)
        return err1, err2, err3

    def _vector_transport(self):
        """Checking vector transport.
        Page 264, Nikolas Boumal, An introduction to optimization on smooth
        manifolds.
        1) transported vector lies in a new tangent space
        2) VT(x,0)[v] is the identity mapping on TxM.

        Args:

        Returns:
            list with two tf scalars that give maximum
            violation of two conditions."""

        vt = self.m.vector_transport(self.u, self.v1, self.v2)
        err1 = vt - self.m.proj(self.m.retraction(self.u, self.v2), vt)
        err1 = jnp.real(jnp.linalg.norm(err1, axis=(-2, -1)))

        err2 = self.v1 - self.m.vector_transport(self.u, self.v1, self.zero)
        err2 = jnp.real(jnp.linalg.norm(err2, axis=(-2, -1)))
        err1 = err1.max()
        err2 = err2.max()
        return err1, err2

    def _egrad_to_rgrad(self):
        """Checking egrad_to_rgrad method.
        1) rgrad is in the tangent space of a manifold's point
        2) <v1 egrad> = <v1 rgrad>_m (matching between egrad and rgrad)

        Args:

        Returns:
            list with two tf scalars that give maximum
            violation of two conditions
        """

        # vector that plays the role of a gradient
        xi = random.normal(self.key, (*self.u.shape, 2))
        xi = xi[..., 0] + 1j * xi[..., 1]


        # rgrad
        rgrad = self.m.egrad_to_rgrad(self.u, xi)

        err1 = rgrad - self.m.proj(self.u, rgrad)
        err1 = jnp.real(jnp.linalg.norm(err1, axis=(-2, -1)))

        err2 = (self.v1.conj() * xi).sum(axis=(-2, -1)) - self.m.inner(self.u, self.v1, rgrad)[..., 0, 0]

        err2 = jnp.abs(jnp.real(err2))

        err1 = err1.max()
        err2 = err2.max()
        return err1, err2

    def checks(self):
        # TODO after checking: rewrite with asserts
        """
        Routine for pytest: checking tolerance of manifold functions
        """
        err = self._proj_of_tangent()
        assert err < self.tol, "Projection error for:{}.\
                    ".format(self.descr)
        err = self._inner_proj_matching()
        assert err < self.tol, "Inner/proj error for:{}.\
                    ".format(self.descr)

        err1, err2, err3 = self._retraction()
        assert err1 < self.tol, "Retraction (Rx(0) != x) error for:{}.\
                    ".format(self.descr)
        assert err2 < self.tol, "Retraction (DRx(o)[v] != v) error for:{}.\
                    ".format(self.descr)
        assert err3 == True, "Retraction (not in manifold) error for:{}.\
                    ".format(self.descr)

        err1, err2 = self._vector_transport()
        assert err1 < self.tol, "Vector transport (not in a TMx) error for:{}.\
                    ".format(self.descr)
        assert err2 < self.tol, "Vector transport (VT(x,0)[v] != v) error for:\
                    {}.".format(self.descr)

        err1, err2 = self._egrad_to_rgrad()
        assert err1 < self.tol, "Rgrad (not in a TMx) error for:{}.\
                ".format(self.descr)
        assert err2 < self.tol, "Rgrad (<v1 egrad> != inner<v1 rgrad>) \
                error for:{}.".format(self.descr)
