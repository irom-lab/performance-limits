import jax
import jax.numpy as jnp

from lbi.geom.types import Ray, Circle, Point, point, Plane, AxisAlignedBoundingBox, plane, unitvector


@jax.jit
def dist_to_pt(d: float, origin: jnp.ndarray, dir: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.cond(d < 0, lambda _: jnp.array([jnp.inf, jnp.inf]), lambda _: origin + d * dir, None)


@jax.jit
def ray_circle_intersections(ray: Ray, circle: Circle) -> Point:
    center = jnp.array(circle.center)
    origin = jnp.array(ray.origin)
    direction = jnp.array(ray.direction)
    radius = circle.radius

    @jax.jit
    def compute_intersections(args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Point:
        origin, direction, center, desc = args

        d_pos = (jnp.sqrt(desc) - (direction @ (origin - center)))
        d_neg = -(jnp.sqrt(desc) + (direction @ (origin - center)))

        pt_pos = dist_to_pt(d_pos, origin, direction)
        pt_neg = dist_to_pt(d_neg, origin, direction)

        return point(jnp.array([pt_pos[0], pt_neg[0]]).flatten(), jnp.array([pt_pos[1], pt_neg[1]]).flatten())

    desc = ((direction @ (origin - center))**2 - (jnp.linalg.norm(origin - center, ord=2)**2 - radius**2))
    compute_intersections((origin, direction, center, desc))

    return jax.lax.cond(desc < 0,
                        lambda _: point(jnp.array([jnp.inf, jnp.inf]), jnp.array([jnp.inf, jnp.inf])),
                        compute_intersections, (origin, direction, center, desc))


@jax.jit
def ray_circle_distance(ray: Ray, circle: Circle) -> float:
    intersections = ray_circle_intersections(ray, circle)

    return jnp.linalg.norm(jnp.stack(intersections, axis=-1) - jnp.array(ray.origin).reshape(1, -1), ord=2,
                           axis=1).min(axis=-1)


@jax.jit
def ray_plane_intersection(ray: Ray, plane: Plane) -> Point:
    """
    References:
        https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

    :param ray:
    :param plane_normal:
    :return:
    """
    origin = jnp.array(ray.origin)
    direction = jnp.array(ray.direction)
    point = jnp.array(plane.point)
    normal = jnp.array(plane.normal)

    @jax.jit
    def compute_intersection(origin: jnp.ndarray, direction: jnp.ndarray, point: jnp.ndarray,
                             normal: jnp.ndarray) -> Point:
        d = (normal @ (point - origin)) / (normal @ direction)
        intersection = origin + d * direction

        return jax.lax.cond((jnp.abs(direction @ normal) < 1e-8) | (d < 0),
                            lambda int: Point(jnp.inf, jnp.inf),
                            lambda int: Point(int[0], int[1]),
                            intersection)

    return compute_intersection(origin, direction, point, normal)


@jax.jit
def ray_plane_distance(ray: Ray, plane: Plane) -> jnp.ndarray:
    intersections = jnp.stack(ray_plane_intersection(ray, plane), axis=-1)
    origin = jnp.array(ray.origin).reshape((1, -1))

    return jnp.linalg.norm(intersections - origin, ord=2, axis=1)


@jax.jit
def ray_aabb_distance(ray: Ray, aabb: AxisAlignedBoundingBox) -> Point:
    xmin, xmax = aabb.xlim
    ymin, ymax = aabb.ylim

    normals = jnp.array([
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    plane_pts = jnp.array([
        [xmin, ymin],
        [xmin, ymin],
        [xmax, ymax],
        [xmax, ymax],
    ])

    return jnp.min(
        jax.vmap(lambda pt, n: ray_plane_distance(ray, plane(point(pt[0], pt[1]), unitvector(n[0], n[1]))))(plane_pts,
                                                                                                            normals))
