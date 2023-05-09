from qgoptax.optimizers.utils import Manifold, Params
from typing import Callable
from jax.tree_util import tree_map
import jax.numpy as jnp
from jax import jvp, grad
from jax.scipy.sparse.linalg import cg


class NaturalRGD:
    def __init__(self, manifold: Manifold,
                 dist: Callable[[Params, Params], jnp.ndarray],
                 learning_rate: jnp.ndarray,
                 name='NaturalRGD',
                 orth_penalty=1e6,
                 penalty=1e-2):
        self.manifold = manifold
        self.name = name
        self.metric_action = lambda u, v: jvp(grad(lambda x: dist(u, x)), (u,), (v,))[0]
        self.total_penalty = lambda u, v: tree_map(lambda x, y: orth_penalty * (y - manifold.proj(x, y)) + penalty * y, u, v)
        self.tree_proj = lambda u, v: tree_map(manifold.proj, u, v)
        self.A = lambda u, v: self.tree_proj(u, self.metric_action(u, self.tree_proj(u, v)))
        self.penalized_A = lambda u, v: tree_map(lambda x, y: x+y, self.A(u, v), self.total_penalty(u, v))
        self.learning_rate = learning_rate
        

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


    def update(self, grads: Params,
               params: Params,
               x0=None,
               tol=1e-05,
               atol=0.0,
               maxiter=None,
               M=None) -> Params:
        """Returns riemannian natural gradient update.

        Args:
            eps: float scalar, regularizer of cg method
            grads: pytree with tensors
            params: pytree with tensors.
        
        Other args: x0, tol, atol, maxiter and M are arguments of 
            jax.scipy.sparse.linalg.cg function that is used
            to find riemannian gradient with natural metric

        Returns:
            new parameters"""

        A = lambda v: self.penalized_A(params, v)
        proj_grads = tree_map(lambda u, v: self.manifold.proj(u, v.conj()), params, grads)
        rgrad = cg(A, proj_grads, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)[0]
        params = tree_map(lambda x, y: self.manifold.retraction(x, -self.learning_rate*y), params, rgrad)
        return params
