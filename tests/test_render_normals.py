"""Regression: STL triangle soup must shade smoothly without changing geometry."""

import copy
import math

import numpy as np

from myo.render_normals import smooth_corner_normals


def test_soup_seams_are_smooth_but_sharp_and_back_faces_stay_separate():
    # Two triangles share positions, not indices; their normals differ by 30 deg.
    vertices = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                [0., 0., 0.], [0., -math.cos(math.pi / 6), .5], [1., 0., 0.]]
    triangles = [[0, 1, 2], [3, 4, 5]]
    before = copy.deepcopy((vertices, triangles))
    normals = smooth_corner_normals(vertices, triangles)
    np.testing.assert_allclose(normals[0], normals[3], atol=1e-12)
    np.testing.assert_allclose(normals[1], normals[5], atol=1e-12)
    assert normals[0][1] > 0  # Shading actually changed from per-face (0, 0, 1).
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-12)
    assert (vertices, triangles) == before
    # A 90-deg edge and an opposite face must not blend through a thin surface.
    hard_vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    hard = smooth_corner_normals(hard_vertices, [[0, 1, 2], [0, 3, 1], [2, 1, 0]])
    np.testing.assert_allclose(hard[0], [0, 0, 1])
    np.testing.assert_allclose(hard[3], [0, 1, 0])
    np.testing.assert_allclose(hard[8], [0, 0, -1])
