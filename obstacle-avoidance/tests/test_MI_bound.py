import unittest

import jax
import jax.random as rnd
import jax.numpy as jnp

from jax.scipy import stats
from lbi.bound import estimate_mi


class BoundTestCase(unittest.TestCase):
    def test_estimate_mi_gaussian_channel(self):
        num_samples = 5000

        state_mean = 0.0
        state_std = 1.0
        noise_std = 1.0

        output_std = jnp.sqrt(state_std**2 + noise_std**2)
        noise_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * (noise_std**2)) + 0.5
        output_entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * (output_std**2)) + 0.5

        mi_analytic = output_entropy - noise_entropy

        @jax.jit
        def log_output_given_state(output: float, state: float) -> float:
            return stats.norm.logpdf(output, loc=state, scale=noise_std)

        key = rnd.PRNGKey(0)
        _, state_key, noise_key = rnd.split(key, 3)

        state_samples = state_std * rnd.normal(state_key, (num_samples, )) + state_mean
        noise_samples = noise_std * rnd.normal(noise_key, (num_samples, ))
        output_samples = state_samples + noise_samples

        mi_est = estimate_mi(state_samples.reshape((-1, 1)), output_samples.reshape((-1, 1)), log_output_given_state)

        self.assertTrue(jnp.allclose(mi_est, mi_analytic, rtol=0.0, atol=0.01))


if __name__ == '__main__':
    unittest.main()
