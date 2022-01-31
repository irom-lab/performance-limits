import jax
import jax.numpy as jnp

import lbi.geom as geom

from typing import NamedTuple


class Workspace(NamedTuple):
    aabb: geom.AxisAlignedBoundingBox
    obstacles: geom.CircleArray


class Lidar(NamedTuple):
    num_rays: int
    ray_directions: geom.UnitVectorArray
    max_distance: float
    local_noise_std: float
    failure_rate: float


@jax.jit
def state_to_workspace(circle_centroids: jnp.ndarray,
                       radius: float,
                       xlim: tuple[float, float],
                       ylim: tuple[float, float]) -> Workspace:
    aabb = geom.aabb(xlim, ylim)
    circles = geom.CircleArray(geom.PointArray(circle_centroids[0, :], circle_centroids[1, :]),
                               radius * jnp.ones_like(circle_centroids[0, :]))

    return Workspace(aabb, circles)


LidarScan = jnp.ndarray
