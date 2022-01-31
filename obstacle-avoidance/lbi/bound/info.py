import jax
import jax.numpy as jnp

from lbi import sensing

from typing import Callable, Union
from jax.scipy.special import logsumexp

from lbi.sensing import Workspace, Lidar


def estimate_mi(state_samples: jnp.ndarray,
                output_samples: jnp.ndarray,
                log_output_given_state: Callable[[jnp.ndarray, jnp.ndarray], float],
                num_samples: int) -> float:
    """
    See Section 2.5 (Inequality (13)) of "On Variational Bounds of Mutual Information."

    :param env:
    :param state_samples:
    :param output_samples:
    :return:
    """

    @jax.jit
    def log_outputs_given_state(outputs: jnp.ndarray, state: jnp.ndarray):
        return jax.vmap(lambda o: log_output_given_state(o, state))(outputs)

    log_outputs_given_states = jax.vmap(lambda s: log_outputs_given_state(output_samples, s))(state_samples).T

    log_conds = jnp.diag(log_outputs_given_states)
    log_marginals = logsumexp(log_outputs_given_states - jnp.diag(jnp.inf * jnp.ones(num_samples)),
                              b=1 / (num_samples - 1),
                              axis=1)

    return jnp.maximum((log_conds - log_marginals).mean(), 0)


@jax.jit
def estimate_mi_bounds(lidar: Lidar, num_samples: int) -> tuple[float, float]:
    lidar_log_pdf_lb = sensing.lidar_logpdf_lowerbound(lidar)
    lidar_log_pdf_ub = sensing.lidar_logpdf_upperbound(lidar)

    return (jnp.maximum(lidar_log_pdf_lb - lidar_log_pdf_ub, 0.0), lidar_log_pdf_ub - lidar_log_pdf_lb / num_samples)


@jax.jit
def mi_ci(lidar: Lidar, confidence: float, num_samples_per_batch: Union[int, jnp.ndarray], num_batches: int) -> float:
    lb, ub = estimate_mi_bounds(lidar, num_samples_per_batch)
    d = (ub - lb)**2

    return jnp.sqrt((-d * jnp.log(0.5 - confidence / 2.0)) / (2 * num_batches))
