from qgoptax.optimizers.utils import Manifold, Params
from typing import Callable
from jax.tree_util import tree_multimap
import jax.numpy as jnp
from jax import jvp, grad
from jax.scipy.sparse.linalg import cg


class NaturalRGD:
    def __init__(self, manifold: Manifold,
                 dist: Callable[[Params, Params], jnp.ndarray],
                 learning_rate: jnp.ndarray,
                 name='NaturalRGD',
                 penalty=1e6):
        self.manifold = manifold
        self.name = name
        metric = lambda u, v: jvp(grad(lambda x: dist(u, x)), (u,), (v,))[1]
        orthogonal_penalty = lambda u, v: tree_multimap(lambda x, y: penalty*(y - manifold.proj(x, y)), u, v)
        self.A = lambda u, v: tree_multimap(lambda x, y: x + y, tree_multimap(manifold.proj, u, metric(u, tree_multimap(manifold.proj, u, v))), orthogonal_penalty(u, v))
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

        A = lambda x: tree_multimap(lambda a, b: a + eps * b, self.A(params, x), x)
        preprocessing = lambda x: tree_multimap(lambda u, v: self.manifold.proj(u, v.conj()), params, x)
        proj_grads = preprocessing(grads)
        rgrad = cg(A, proj_grads, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)[0]
        #rgrad = tree_proj(rgrad)
        params = tree_multimap(lambda x, y: self.manifold.retraction(x, -self.learning_rate*y), params, rgrad)
        return params
