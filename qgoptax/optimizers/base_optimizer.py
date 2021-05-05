from qgoptax.optimizers.utils import Manifold, Params, transpose_pytree
from typing import Tuple, Union
from jax.tree_util import tree_map, tree_multimap
import jax.numpy as jnp


class Optimizer:
    def __init__(self,
                 manifold: Manifold,
                 name: str,
                 use_precond: bool=False):
        self.manifold = manifold
        self.name = name
        self.use_precond = manifold._use_precond

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def init(self, params: Params) -> Tuple[Params]:
        """Initializes the state of an optimizer.

        Args:
            params: pytree with tensors.

        Returns:
            state of an optimizer."""

        return (
            jnp.array(0, dtype=jnp.int32),
            *transpose_pytree(tree_map(self._create_state, params)),
        )

    def update(
        self,
        grads: Params,
        state: Tuple[Params],
        params: Params,
        precond: Union[None, Params]=None
    ) -> Tuple[Params, Tuple[Params]]:
        """Returns update direction and new state of an optimizer.

        Args:
            grads: pytree with tensors
            state: pytree with tensors.
            params: pytree with tensors.
            precond: optional pytree with tensors

        Returns:
            two pytrees with tensors: new params and new state"""

        iter = state[0]
        if self.use_precond:
            params, state = transpose_pytree(
                tree_multimap(
                    lambda x, z, p, *y: self._apply(iter, x, y, z, p, use_precond=True), grads, params, precond, *state[1:]
                )
            )
        else:
            params, state = transpose_pytree(
                tree_multimap(
                    lambda x, z, *y: self._apply(iter, x, y, z, None, use_precond=False), grads, params, *state[1:]
                )
            )
        state = transpose_pytree(state)
        return params, (iter + 1, *state)

    def _apply(
        self,
        iter: jnp.ndarray,
        grad: jnp.ndarray,
        state: Tuple[jnp.ndarray],
        param: jnp.ndarray,
        precond: Union[None, jnp.ndarray]=None
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
        """Applies update to tensor.

        Args:
            iter: number of iteration.
            grad: complex valued tensor.
            state: complex valued tensor.
            param: complex valued tensor.

        Returns:
            two complex valued tensors: new param and new state."""
        pass

    def _create_state(self, param: jnp.ndarray) -> Tuple[jnp.ndarray]:
        """Returns state for the corresponding param.

        Args:
            param: complex valued tensor.

        Returns:
            tuple with tensors representing a state"""
        pass
