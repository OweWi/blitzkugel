"""Triangle mesh data structure and simple primitives (used by tests and demos)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class TriMesh:
    """Indexed triangle mesh in metres.

    Attributes:
        vertices: (N, 3) vertex coordinates in metres.
        faces: (M, 3) vertex indices per triangle.
        body_ids: optional (M,) index of the source body per triangle.
        body_names: names of the source bodies (index = body id).
    """

    vertices: FloatArray
    faces: IntArray
    body_ids: IntArray | None = None
    body_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if self.body_ids is not None:
            self.body_ids = np.asarray(self.body_ids, dtype=np.int64).reshape(-1)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.n_faces == 0

    @property
    def bounds(self) -> tuple[FloatArray, FloatArray]:
        """Axis-aligned bounding box as (min_xyz, max_xyz)."""
        if self.n_vertices == 0:
            raise ValueError("Leeres Netz hat keine Bounding-Box.")
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def scaled(self, factor: float) -> TriMesh:
        """Return a copy with all coordinates multiplied by ``factor``."""
        return TriMesh(self.vertices * factor, self.faces.copy(), self.body_ids, list(self.body_names))

    def translated(self, offset: FloatArray | tuple[float, float, float]) -> TriMesh:
        """Return a copy translated by ``offset``."""
        return TriMesh(
            self.vertices + np.asarray(offset, dtype=np.float64),
            self.faces.copy(),
            self.body_ids,
            list(self.body_names),
        )

    def face_normals(self) -> FloatArray:
        """Unit normals per triangle (zero vector for degenerate triangles)."""
        tri = self.vertices[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, length, out=np.zeros_like(n), where=length > 0)

    @staticmethod
    def merge(meshes: list[TriMesh]) -> TriMesh:
        """Concatenate several meshes; each input becomes its own body."""
        verts: list[FloatArray] = []
        faces: list[IntArray] = []
        bodies: list[IntArray] = []
        names: list[str] = []
        offset = 0
        for i, m in enumerate(meshes):
            verts.append(m.vertices)
            faces.append(m.faces + offset)
            bodies.append(np.full(m.n_faces, i, dtype=np.int64))
            names.append(m.body_names[0] if m.body_names else f"Körper {i + 1}")
            offset += m.n_vertices
        if not verts:
            return TriMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        return TriMesh(np.vstack(verts), np.vstack(faces), np.concatenate(bodies), names)


def make_box(
    xmin: float, ymin: float, zmin: float, xmax: float, ymax: float, zmax: float, name: str = "Quader"
) -> TriMesh:
    """Closed axis-aligned box with outward-facing triangles."""
    v = np.array(
        [
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front (-y)
            [2, 3, 7], [2, 7, 6],  # back (+y)
            [1, 2, 6], [1, 6, 5],  # right (+x)
            [3, 0, 4], [3, 4, 7],  # left (-x)
        ],
        dtype=np.int64,
    )
    return TriMesh(v, f, body_names=[name])


def make_rod(x: float, y: float, z0: float, height: float, width: float = 0.02, name: str = "Fangstange") -> TriMesh:
    """Thin vertical square rod (air-termination rod) centred at (x, y)."""
    h = width / 2.0
    return make_box(x - h, y - h, z0, x + h, y + h, z0 + height, name=name)
