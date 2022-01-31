"""
LIDAR model from Chapter 6.3 of Probabilistic Robotics by Thrun et al.
"""

import jax
import jax.numpy as jnp
import jax.random as rnd

import lbi.geom as geom

from jax.scipy import stats
from jax.scipy.special import logsumexp
from lbi.sensing.types import Workspace, Lidar, LidarScan
from typing import Literal


@jax.jit
def ray_workspace_distance(ray: geom.Ray, workspace: Workspace) -> float:
    distances = jnp.append(
        jax.vmap(lambda o: geom.ray_circle_distance(ray, o))(workspace.obstacles),
        geom.ray_aabb_distance(ray, workspace.aabb))

    return jnp.min(distances)


@jax.jit
def sample_obs_dist(key: jnp.ndarray, ray: geom.Ray, workspace: Workspace, lidar: Lidar) -> float:
    local_noise_key, failure_key = rnd.split(key, 2)
    exact_distance = ray_workspace_distance(ray, workspace)
    distance_to_endpoint = jnp.minimum(exact_distance, lidar.max_distance)

    distance_with_local_noise = distance_to_endpoint + lidar.local_noise_std * rnd.normal(local_noise_key)
    failure = rnd.bernoulli(failure_key, jnp.array(lidar.failure_rate)).astype(float)

    return jax.lax.cond(failure,
                        lambda _: lidar.max_distance,
                        lambda _: jnp.clip(distance_with_local_noise, 0, lidar.max_distance),
                        None)


@jax.jit
def ray_logpdf(distance: float, ray: geom.Ray, workspace: Workspace, lidar: Lidar) -> float:
    exact_distance = jnp.minimum(lidar.max_distance, ray_workspace_distance(ray, workspace))
    valid_distance_indicator = jnp.array((0.0 <= distance) & (distance <= lidar.max_distance), float)
    max_dist_indicator = jnp.array(distance == lidar.max_distance, float)

    @jax.jit
    def local_noise_logpdf(obs: float, exact: float, lidar: Lidar) -> float:
        lower_logcdf = stats.norm.logcdf(0.0, loc=exact, scale=lidar.local_noise_std)
        upper_logcdf = stats.norm.logcdf(lidar.max_distance, loc=exact, scale=lidar.local_noise_std)
        log_normalization = logsumexp(jnp.array([upper_logcdf, lower_logcdf]), b=jnp.array([1.0, -1.0]))

        return stats.norm.logpdf(obs, loc=exact, scale=lidar.local_noise_std) - log_normalization

    @jax.jit
    def max_dist_logprob(obs: float, exact: float, lidar: Lidar) -> float:
        max_dist_from_local_noise_logpdf = 1 - stats.norm.cdf(
            lidar.max_distance, loc=exact, scale=lidar.local_noise_std)

        log_probs = jnp.array([lidar.failure_rate, jnp.log(1 - lidar.failure_rate) + max_dist_from_local_noise_logpdf])

        return logsumexp(log_probs)

    local_noise_logpdf(distance, exact_distance, lidar)

    valid_logpdf = jax.lax.cond(max_dist_indicator,
                                lambda x: max_dist_logprob(*x),
                                lambda x: local_noise_logpdf(*x), (distance, exact_distance, lidar))

    return jax.lax.cond(valid_distance_indicator, lambda _: valid_logpdf, lambda _: -jnp.inf, None)


@jax.jit
def ray_logpdf_lowerbound(lidar: Lidar) -> float:
    failure_lb = jnp.log(lidar.failure_rate)

    normalization = stats.norm.cdf(lidar.max_distance, loc=lidar.max_distance, scale=lidar.local_noise_std) -\
                    stats.norm.cdf(0.0, loc=lidar.max_distance, scale=lidar.local_noise_std)
    noise_lb = stats.norm.logpdf(0.0, loc=lidar.max_distance, scale=lidar.local_noise_std) - jnp.log(normalization)

    return jnp.minimum(noise_lb, failure_lb)


@jax.jit
def ray_logpdf_upperbound(lidar: Lidar) -> float:
    failure_ub = jnp.log(lidar.failure_rate)

    normalization = stats.norm.cdf(lidar.max_distance, loc=lidar.max_distance, scale=lidar.local_noise_std) -\
                    stats.norm.cdf(0.0, loc=lidar.max_distance, scale=lidar.local_noise_std)
    noise_ub = stats.norm.logpdf(lidar.max_distance, loc=lidar.max_distance,
                                 scale=lidar.local_noise_std) - jnp.log(normalization)

    return jnp.maximum(noise_ub, failure_ub)


@jax.jit
def lidar_logpdf(obs: LidarScan, config: geom.Point, workspace: Workspace, lidar: Lidar) -> float:
    return jnp.sum(
        jax.vmap(lambda dist, dir: ray_logpdf(dist, geom.ray(config, dir), workspace, lidar))(obs,
                                                                                              lidar.ray_directions))


def lidar_logpdf_lowerbound(lidar: Lidar) -> float:
    return ray_logpdf_lowerbound(lidar) * lidar.num_rays


def lidar_logpdf_upperbound(lidar: Lidar) -> float:
    return ray_logpdf_upperbound(lidar) * lidar.num_rays


def sample_lidar(key: jnp.ndarray, config: geom.Point, workspace: Workspace, lidar: Lidar) -> LidarScan:
    keys = rnd.split(key, lidar.num_rays)

    return jax.vmap(
        lambda ray_key, ray_dir: jnp.array(sample_obs_dist(ray_key, geom.ray(config, ray_dir), workspace, lidar)))(
            keys, lidar.ray_directions)


def lidar(num_rays,
          angle_range: tuple[float, float],
          max_distance: float = jnp.inf,
          local_noise_std: float = 0.0,
          failure_rate: float = 0.0,
          angle_units: Literal['radians', 'degrees'] = 'radians') -> Lidar:
    if failure_rate > 0 and jnp.isinf(max_distance):
        raise ValueError(f'Failure rate was positive ({failure_rate} > 0.0), but max_distance was not set.')

    if angle_units == 'degrees':
        angle_range = (2 * jnp.pi * angle_range[0] / 360.0, 2 * jnp.pi * angle_range[1] / 360.0)

    angles = jnp.linspace(angle_range[0], angle_range[1], num_rays)
    ray_directions = jax.vmap(lambda angle: geom.unitvector(jnp.cos(angle), jnp.sin(angle)))(angles)

    return Lidar(num_rays, ray_directions, max_distance, local_noise_std, failure_rate)
