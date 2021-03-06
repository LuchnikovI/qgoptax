from qgoptax.optimizers.base_optimizer import Optimizer
from qgoptax.optimizers.utils import Manifold
from typing import Tuple, Union
import jax.numpy as jnp


class RSGD(Optimizer):
    """Riemannian gradient descent and gradient descent with momentum
    optimizers.

    Args:
        manifold: object of the class Manifold, marks a particular manifold.
        learning_rate: floating point number. A learning rate.
            Defaults to 0.01.
        momentum: floating point value, the momentum. Defaults to 0
            (Standard GD)."""

    def __init__(
        self, manifold: Manifold,
        learning_rate=0.01,
        momentum=0.0,
        name="RSGD"
    ):

        super().__init__(manifold, name)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_momentum = False

        if isinstance(momentum, jnp.ndarray) or momentum > 0:
            self.use_momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

    def _create_state(self, param: jnp.ndarray) -> Tuple[jnp.ndarray]:
        if self.use_momentum:
            return (jnp.zeros_like(param),)
        else:
            return None

    def _apply(
        self,
        iter: jnp.ndarray,
        grad: jnp.ndarray,
        state: Tuple[jnp.ndarray],
        param: jnp.ndarray,
        precond: Union[None, jnp.ndarray]=None,
        use_precond=False
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
        if use_precond:
            rgrad = self.manifold.egrad_to_rgrad(param, grad.conj(), precond)
        else:
            rgrad = self.manifold.egrad_to_rgrad(param, grad.conj())
        if self.use_momentum:
            momentum = self.momentum * state[0] + (1 - self.momentum) * rgrad
            param, momentum = self.manifold.retraction_transport(
                param, momentum, -self.learning_rate * momentum
            )
            return param, (momentum,)
        else:
            param = self.manifold.retraction(param, -self.learning_rate * rgrad)
            return param, state
