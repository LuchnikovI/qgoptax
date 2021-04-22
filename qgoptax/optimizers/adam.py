from qgoptax.optimizers.base_optimizer import Optimizer
from qgoptax.optimizers.utils import Manifold
from typing import Tuple
import jax.numpy as jnp


class RAdam(Optimizer):
    """Riemannain Adam and AMSGrad optimizers. Returns a new optimizer.

    Args:
        manifold: object of the class Manifold, marks a particular manifold.
        learning_rate: real number. A learning rate. Defaults to 0.05.
        beta1: real number. An exponential decay rate for the first moment.
            Defaults to 0.9.
        beta2: real number. An exponential decay rate for the second moment.
            Defaults to 0.999.
        eps: real number. Regularization coeffitient. Defaults to 1e-8.
        ams: boolean number. Use ams (AMSGrad) or not.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'RAdam'."""

    def __init__(
        self,
        manifold: Manifold,
        learning_rate=0.05,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        ams=False,
        name="RAdam",
    ):
        super().__init__(manifold, name)
        if isinstance(beta1, (int, float)) and (beta1 < 0 or beta1 > 1):
            raise ValueError("`beta1` must be between [0, 1].")
        if isinstance(beta2, (int, float)) and (beta2 < 0 or beta2 > 1):
            raise ValueError("`beta2` must be between [0, 1].")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ams = ams

    def _create_state(self, param: jnp.ndarray) -> Tuple[jnp.ndarray]:
        state = (
            jnp.zeros_like(param),
            jnp.zeros((*param.shape[:-2], 1, 1), dtype=param.dtype),
        )
        if self.ams:
            state = (*state, jnp.zeros((*param.shape[:-2], 1, 1), dtype=param.dtype))
        return state

    def _apply(
        self,
        iter: jnp.ndarray,
        grad: jnp.ndarray,
        state: Tuple[jnp.ndarray],
        param: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
        rgrad = self.manifold.egrad_to_rgrad(param, grad.conj())
        momentum = self.beta1 * state[0] + (1 - self.beta1) * rgrad
        v = self.beta2 * state[1] + (1 - self.beta2) * self.manifold.inner(
            param, rgrad, rgrad
        )
        if self.ams:
            v_hat = jax.lax.complex(jnp.maximum(jnp.real(v), jnp.real(state[2])), jnp.imag(v))

        # Bias correction
        lr_corr = (
            self.learning_rate
            * jnp.sqrt(1 - self.beta2 ** (iter + 1))
            / (1 - self.beta1 ** (iter + 1))
        )

        if self.ams:
            search_dir = -lr_corr * momentum / (jnp.sqrt(v_hat) + self.eps)
            param, momentum = self.manifold.retraction_transport(
                param, momentum, search_dir
            )
            return param, (momentum, v, v_hat)
        else:
            search_dir = -lr_corr * momentum / (jnp.sqrt(v) + self.eps)
            param, momentum = self.manifold.retraction_transport(
                param, momentum, search_dir
            )
            return param, (momentum, v)
