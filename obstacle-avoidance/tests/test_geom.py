import unittest

import jax
import jax.numpy as jnp

import lbi.geom as geom


class GeomTestCase(unittest.TestCase):
    def test_ray_circle_collisions(self):
        ray = geom.ray(geom.point(1, 1), geom.unitvector(-1, -1))
        circle = geom.circle(geom.point(0.0, 0.0), 1.0)

        collisions = geom.ray_circle_intersections(ray, circle)

        self.assertTrue(jnp.allclose(jnp.linalg.norm(jnp.stack(collisions, axis=-1), ord=2, axis=-1), 1.0))

    def test_ray_circle_distance(self):
        circle = geom.circle(geom.point(0.0, 0.0), 1.0)

        ray = geom.ray(geom.point(2, 0), geom.unitvector(-1, 0))
        distance = geom.ray_circle_distance(ray, circle)
        self.assertTrue(jnp.allclose(distance, 1.0))

        ray = geom.ray(geom.point(-2, 0), geom.unitvector(1, 0))
        distance = geom.ray_circle_distance(ray, circle)
        self.assertTrue(jnp.allclose(distance, 1.0))

    def test_ray_plane_distance(self):
        with jax.disable_jit():
            ray = geom.ray(geom.point(1.0, 0.0), geom.unitvector(-1.0, 1.0))
            plane = geom.plane(geom.point(0, 0), geom.unitvector(1.0, 0.0))

            collisions = geom.ray_plane_intersection(ray, plane)
            distances = geom.ray_plane_distance(ray, plane)

        pass

    def test_ray_aabb_distance(self):
        x = jnp.array([1.0, -1.0, 0.0, 0.0])
        y = jnp.array([0.0, 0.0, 1.0, -1.0])
        aabb = geom.aabb((0.0, 1.0), (0.0, 1.0))

        dist = jax.vmap(
            lambda x, y: geom.ray_aabb_distance(geom.ray(geom.point(0.8, 0.8), geom.unitvector(x, y)), aabb))(x, y)

        self.assertTrue(jnp.allclose(dist, jnp.array([0.2, 0.8, 0.2, 0.8])))


if __name__ == '__main__':
    unittest.main()
