"""Tests for the point-to-mesh distance backends."""

from __future__ import annotations

import numpy as np
import pytest

from blitzkugel.core.distance import NumpyDistance, create_distance_query, open3d_available
from blitzkugel.core.mesh import make_box

BACKENDS = ["numpy"] + (["open3d"] if open3d_available() else [])


@pytest.fixture
def unit_box():
    return make_box(0, 0, 0, 1, 1, 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_known_distances(unit_box, backend: str) -> None:
    q = create_distance_query(unit_box, backend)
    pts = np.array(
        [
            [0.5, 0.5, 3.0],  # above the top face
            [2.0, 2.0, 0.5],  # next to vertical edge x=1,y=1
            [-1.0, -1.0, -1.0],  # beyond the corner
            [0.5, 0.5, 0.5],  # centre (unsigned distance to the faces)
        ]
    )
    expected = [2.0, np.sqrt(2.0), np.sqrt(3.0), 0.5]
    np.testing.assert_allclose(q.distance(pts), expected, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_closest_points(unit_box, backend: str) -> None:
    q = create_distance_query(unit_box, backend)
    cp, ids, dist = q.closest(np.array([[0.25, 0.75, 5.0], [3.0, 0.5, 0.5]]))
    np.testing.assert_allclose(cp, [[0.25, 0.75, 1.0], [1.0, 0.5, 0.5]], atol=1e-5)
    normals = unit_box.face_normals()[ids]
    np.testing.assert_allclose(normals, [[0, 0, 1], [1, 0, 0]], atol=1e-9)
    np.testing.assert_allclose(dist, [4.0, 2.0], atol=1e-5)


def test_numpy_matches_open3d_random() -> None:
    if not open3d_available():
        pytest.skip("open3d nicht installiert")
    mesh = make_box(-3, -1, 0, 4, 2, 5)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-10, 10, size=(2000, 3))
    a = NumpyDistance(mesh).distance(pts)
    b = create_distance_query(mesh, "open3d").distance(pts)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_large_coordinates_keep_precision() -> None:
    """Geodetic coordinates (≈ 10⁶ m) must not lose precision in the float32 backend."""
    if not open3d_available():
        pytest.skip("open3d nicht installiert")
    off = np.array([500_000.0, 5_800_000.0, 0.0])
    mesh = make_box(0, 0, 0, 1, 1, 1).translated(off)
    q = create_distance_query(mesh, "open3d")
    d = q.distance(np.array([off + [0.5, 0.5, 1.25]]))
    assert d[0] == pytest.approx(0.25, abs=1e-3)
