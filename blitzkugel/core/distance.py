"""Unsigned point-to-mesh distance queries.

Two interchangeable backends:

* :class:`Open3DDistance` – BVH based (``open3d.t.geometry.RaycastingScene``),
  fast enough for millions of query points against large meshes.
* :class:`NumpyDistance` – exact brute force in NumPy. Only suitable for small
  meshes (tests, simple demo geometry) but has no extra dependency.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from .mesh import FloatArray, IntArray, TriMesh

log = logging.getLogger(__name__)


class DistanceQuery(Protocol):
    """Unsigned distance from arbitrary points to a fixed triangle mesh."""

    name: str

    def distance(self, points: FloatArray) -> FloatArray:
        """Distances (P,) of ``points`` (P, 3) to the mesh."""
        ...

    def closest(self, points: FloatArray) -> tuple[FloatArray, IntArray, FloatArray]:
        """Closest mesh points (P, 3), triangle ids (P,) and distances (P,)."""
        ...


class Open3DDistance:
    """Distance queries using Open3D's RaycastingScene.

    Open3D works in float32. The mesh is therefore shifted to its bounding box
    centre so that large absolute coordinates (e.g. geodetic models) do not
    lose precision.
    """

    name = "open3d"

    def __init__(self, mesh: TriMesh) -> None:
        import open3d as o3d  # noqa: PLC0415 – optional dependency

        self._o3d = o3d
        lo, hi = mesh.bounds
        self._origin = (lo + hi) / 2.0
        verts = (mesh.vertices - self._origin).astype(np.float32)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(
            o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(mesh.faces.astype(np.uint32), dtype=o3d.core.Dtype.UInt32),
        )

    def _tensor(self, points: FloatArray):  # noqa: ANN202
        local = (np.asarray(points, dtype=np.float64) - self._origin).astype(np.float32)
        return self._o3d.core.Tensor(np.ascontiguousarray(local), dtype=self._o3d.core.Dtype.Float32)

    def distance(self, points: FloatArray) -> FloatArray:
        if len(points) == 0:
            return np.zeros(0)
        return self._scene.compute_distance(self._tensor(points)).numpy().astype(np.float64)

    def closest(self, points: FloatArray) -> tuple[FloatArray, IntArray, FloatArray]:
        if len(points) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0)
        res = self._scene.compute_closest_points(self._tensor(points))
        cp = res["points"].numpy().astype(np.float64) + self._origin
        ids = res["primitive_ids"].numpy().astype(np.int64)
        dist = np.linalg.norm(cp - np.asarray(points, dtype=np.float64), axis=1)
        return cp, ids, dist


class NumpyDistance:
    """Exact brute-force closest-point computation (Ericson, Real-Time Collision Detection 5.1.5)."""

    name = "numpy"

    #: Upper bound for (points × triangles) evaluated per vectorised block.
    BLOCK_ELEMENTS = 1_000_000

    def __init__(self, mesh: TriMesh) -> None:
        tri = mesh.vertices[mesh.faces]
        self._a = tri[:, 0]
        self._b = tri[:, 1]
        self._c = tri[:, 2]
        if mesh.n_faces > 20_000:
            log.warning("NumPy-Distanzberechnung mit %d Dreiecken ist sehr langsam – open3d installieren.",
                        mesh.n_faces)

    def distance(self, points: FloatArray) -> FloatArray:
        return self.closest(points)[2]

    def closest(self, points: FloatArray) -> tuple[FloatArray, IntArray, FloatArray]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        n_pts, n_tri = len(pts), len(self._a)
        best_d2 = np.full(n_pts, np.inf)
        best_pt = np.zeros((n_pts, 3))
        best_id = np.zeros(n_pts, dtype=np.int64)
        tri_block = max(1, min(n_tri, self.BLOCK_ELEMENTS))
        for t0 in range(0, n_tri, tri_block):
            t1 = min(n_tri, t0 + tri_block)
            pt_block = max(1, self.BLOCK_ELEMENTS // (t1 - t0))
            for p0 in range(0, n_pts, pt_block):
                p1 = min(n_pts, p0 + pt_block)
                cp = _closest_on_triangles(
                    pts[p0:p1, None, :], self._a[None, t0:t1], self._b[None, t0:t1], self._c[None, t0:t1]
                )
                d2 = np.sum((cp - pts[p0:p1, None, :]) ** 2, axis=2)
                local = np.argmin(d2, axis=1)
                rows = np.arange(p1 - p0)
                cand = d2[rows, local]
                better = cand < best_d2[p0:p1]
                idx = np.nonzero(better)[0]
                best_d2[p0 + idx] = cand[idx]
                best_pt[p0 + idx] = cp[idx, local[idx]]
                best_id[p0 + idx] = t0 + local[idx]
        return best_pt, best_id, np.sqrt(best_d2)


def _dot(u: FloatArray, v: FloatArray) -> FloatArray:
    return np.einsum("...k,...k->...", u, v)


def _closest_on_triangles(p: FloatArray, a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    """Vectorised closest point on triangle(s) abc to point(s) p (broadcasting)."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def safe_div(num: FloatArray, den: FloatArray) -> FloatArray:
        return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=np.abs(den) > 1e-300)

    # Interior (face) region as default; overwritten by vertex/edge regions in
    # reverse order of Ericson's sequential tests so that earlier tests win.
    denom = va + vb + vc
    v = safe_div(vb, denom)[..., None]
    w = safe_div(vc, denom)[..., None]
    result = a + ab * v + ac * w

    t_bc = safe_div(d4 - d3, (d4 - d3) + (d5 - d6))[..., None]
    result = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * t_bc, result)
    t_ac = safe_div(d2, d2 - d6)[..., None]
    result = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * t_ac, result)
    result = np.where(((d6 >= 0) & (d5 <= d6))[..., None], np.broadcast_to(c, result.shape), result)
    t_ab = safe_div(d1, d1 - d3)[..., None]
    result = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * t_ab, result)
    result = np.where(((d3 >= 0) & (d4 <= d3))[..., None], np.broadcast_to(b, result.shape), result)
    result = np.where(((d1 <= 0) & (d2 <= 0))[..., None], np.broadcast_to(a, result.shape), result)
    return result


def open3d_available() -> bool:
    """True if the open3d package can be imported."""
    try:
        import open3d  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def create_distance_query(mesh: TriMesh, backend: str = "auto") -> DistanceQuery:
    """Create a distance query for ``mesh``.

    Args:
        mesh: Non-empty triangle mesh.
        backend: ``"auto"`` (open3d if installed, else numpy), ``"open3d"`` or ``"numpy"``.
    """
    if mesh.is_empty:
        raise ValueError("Das Netz enthält keine Dreiecke.")
    if backend == "auto":
        backend = "open3d" if open3d_available() else "numpy"
    log.info("Distanz-Backend: %s (%d Dreiecke)", backend, mesh.n_faces)
    if backend == "open3d":
        return Open3DDistance(mesh)
    if backend == "numpy":
        return NumpyDistance(mesh)
    raise ValueError(f"Unbekanntes Distanz-Backend: {backend}")
