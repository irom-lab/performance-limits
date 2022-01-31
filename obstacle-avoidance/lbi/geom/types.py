import jax
import jax.numpy as jnp
from typing import NamedTuple, Union


class Point(NamedTuple):
    x: float
    y: float


class PointArray(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


class UnitVector(NamedTuple):
    x: float
    y: float


class UnitVectorArray(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


class Vector(NamedTuple):
    x: float
    y: float


class VectorArray(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


class Circle(NamedTuple):
    center: Point
    radius: float


class CircleArray(NamedTuple):
    center: PointArray
    radius: jnp.ndarray


class Ray(NamedTuple):
    origin: Point
    direction: UnitVector


class RayArray(NamedTuple):
    origin: PointArray
    direction: UnitVectorArray


class AxisAlignedBoundingBox(NamedTuple):
    xlim: tuple[float, float]
    ylim: tuple[float, float]


class Plane(NamedTuple):
    point: Point
    normal: UnitVector


@jax.jit
def point(x: float, y: float) -> Point:
    return Point(x, y)


@jax.jit
def unitvector(x: float, y: float):
    vec = jnp.array([x, y])
    unitvec = vec / jnp.linalg.norm(vec, ord=2, axis=-1)

    return UnitVector(unitvec[0], unitvec[1])


@jax.jit
def vector(x: float, y: float) -> Vector:
    return Vector(x, y)


@jax.jit
def ray(origin: Point, direction: UnitVector) -> Ray:
    return Ray(origin, direction)


@jax.jit
def circle(center: Point, radius: float) -> Circle:
    return Circle(center, radius)


@jax.jit
def plane(point: Point, normal: UnitVector) -> Plane:
    return Plane(point, normal)


@jax.jit
def aabb(xlim: Union[tuple[float, float], list[float]],
         ylim: Union[tuple[float, float], list[float]]) -> AxisAlignedBoundingBox:  # yapf: disable
    return AxisAlignedBoundingBox(tuple(xlim), tuple(ylim))
