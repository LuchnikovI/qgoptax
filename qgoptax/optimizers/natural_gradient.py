from qgoptax.optimizers.utils import Manifold, Params
from typing import Callable
from jax.tree_util import tree_multimap, tree_map
import jax.numpy as jnp
from jax import jvp, grad
from jax.scipy.sparse.linalg import cg


class NaturalRGD:
    def __init__(self, manifold: Manifold,
                 dist: Callable[[Params, Params], jnp.ndarray],
                 learning_rate: jnp.ndarray,
                 name='NaturalRGD'):
        self.manifold = manifold
        self.name = name
        metric = lambda u, v: jvp(grad(lambda x: dist(u, x)), u, v)[1]
        self.A = lambda u, v: manifold.proj(u, metric(u, manifold.proj(u, v)))
        self.learning_rate = learning_rate
        

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


    def update(self, grads: Params,
               params: Params,
               eps=1e-2,
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

        A = lambda x: self.A(x) + eps * x
        A = lambda x: tree_multimap(A, params, (x,))
        rgrad = cg(A, grads, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)
        rgrad = tree_map(lambda x: self.manifold.proj(params, x), rgrad)
        params = tree_multimap(lambda x, y: x - self.learning_rate * y, params, (rgrad,))
        return params

