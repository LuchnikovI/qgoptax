from functools import Any
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp


# Params can be an arbitrary PyTree of complex valued tensors
Params = Any


# An example of a manifold
Manifold = Any


def is_leaf(x):
    if isinstance(x, tuple):
        if len(x) == 0:
            return True
        elif isinstance(x[0], jnp.ndarray):
            return True
        else:
            return False
    else:
        return False


def transpose_pytree(pytree: Params) -> Params:
    """Transpose pytree.

    Args:
        pytree: pytree of tuples of tensors

    Returns:
        tuple of pytrees with tensors"""

    flatten_tree, treedef = tree_flatten(pytree, is_leaf)
    return tuple(map(lambda x: tree_unflatten(treedef, x), zip(*flatten_tree)))
