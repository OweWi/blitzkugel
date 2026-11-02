"""Rolling sphere method (Blitzkugelverfahren) according to DIN EN 62305-3.

Both evaluation modes share the same discrete formulation:

1. A regular grid of candidate sphere centres is laid around the model.
2. The unsigned distance ``d`` of every centre to the triangle mesh is computed.
3. A centre is *valid* if the sphere does not penetrate the structure
   (``d >= r``) or the ground (``z >= ground + r``; ensured by the grid).
4. Only valid centres that are connected to the top of the grid are kept
   (flood fill), so spheres in closed cavities do not count.
5. Reachable centres with ``d < r + tol`` are contact positions; the closest
   mesh point of each contact position is a possible point of strike.

Mode B ("3D-Hülle") uses 3D connectivity, i.e. the sphere can move freely in
space. Mode A ("Richtungen") uses a grid aligned with a horizontal rolling
direction and connectivity only *within* each vertical plane: the sphere
centre stays in its plane and rolls from the start side over ground and
structure to the opposite side, touching everything on the outer contour.
Rolling direction θ and θ+180° therefore yield identical contact sets.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .distance import DistanceQuery, create_distance_query
from .mesh import FloatArray, IntArray, TriMesh
from .progress import CancelCallback, ProgressCallback, ProgressScope, never_cancel, no_progress

log = logging.getLogger(__name__)


class ProtectionClass(Enum):
    """Lightning protection level (LPL / Schutzklasse) with rolling sphere diameter in m."""

    I = 40.0
    II = 60.0
    III = 90.0
    IV = 120.0

    @property
    def diameter(self) -> float:
        return float(self.value)

    @property
    def radius(self) -> float:
        return float(self.value) / 2.0

    @classmethod
    def from_diameter(cls, diameter: float, tol: float = 1e-6) -> ProtectionClass | None:
        """Return the class with exactly this diameter, or None for custom values."""
        for pc in cls:
            if abs(pc.diameter - diameter) <= tol:
                return pc
        return None


class Mode(Enum):
    """Evaluation mode."""

    DIRECTIONS = "A"  #: Rolling along horizontal directions in vertical planes
    HULL_3D = "B"  #: Full 3D evaluation
    ADAPTIVE = "C"  #: 3D evaluation, coarse-to-fine refinement near the contact band


@dataclass
class SphereParams:
    """Parameters of the rolling sphere evaluation (all lengths in metres).

    Attributes:
        radius: Rolling sphere radius r.
        grid_step: Spacing of the sphere centre grid.
        ground_z: Height of the ground plane.
        contact_tolerance: Max. ``d - r`` for a contact position; ``None`` = automatic
            (half the grid cell diagonal).
        plane_spacing: Mode A only: spacing of the parallel planes; ``None`` = ``grid_step``.
        coarse_step: Mode C only: grid spacing of the first (coarsest) stage; rounded up to
            ``grid_step · 2^k``. ``None`` = automatic (about r/4).
        feature_blocks: Mode C only: blocks whose nearest geometry is a single point, edge
            or plane are not refined further; their contact positions are placed directly
            on that feature (see :func:`roll_adaptive`). ``False`` = exact octree.
        max_grid_points: Safety limit for the number of grid points per evaluation.
    """

    radius: float
    grid_step: float
    ground_z: float = 0.0
    contact_tolerance: float | None = None
    plane_spacing: float | None = None
    coarse_step: float | None = None
    feature_blocks: bool = True
    max_grid_points: int = 150_000_000

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Der Kugelradius muss positiv sein.")
        if self.grid_step <= 0:
            raise ValueError("Die Rasterweite muss positiv sein.")
        if self.contact_tolerance is not None and self.contact_tolerance <= 0:
            raise ValueError("Die Kontakttoleranz muss positiv sein.")
        if self.plane_spacing is not None and self.plane_spacing <= 0:
            raise ValueError("Der Ebenenabstand muss positiv sein.")
        if self.coarse_step is not None and self.coarse_step <= 0:
            raise ValueError("Die grobe Rasterweite muss positiv sein.")

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def effective_plane_spacing(self) -> float:
        return self.plane_spacing if self.plane_spacing is not None else self.grid_step

    @property
    def refinement_levels(self) -> int:
        """Mode C: number of halvings from the coarse to the fine grid step."""
        if self.coarse_step is None:
            ratio = self.radius / 4.0 / self.grid_step
            return int(min(8, max(0, math.floor(math.log2(ratio))))) if ratio >= 1 else 0
        ratio = self.coarse_step / self.grid_step
        return int(min(12, max(0, math.ceil(math.log2(ratio) - 1e-9)))) if ratio > 1 else 0

    @property
    def effective_coarse_step(self) -> float:
        """Mode C: actual coarse step ``grid_step · 2^levels``."""
        return self.grid_step * (1 << self.refinement_levels)

    def tolerance(self, mode: Mode) -> float:
        """Contact tolerance actually used for ``mode``."""
        if self.contact_tolerance is not None:
            return self.contact_tolerance
        h = self.grid_step
        third = self.effective_plane_spacing if mode is Mode.DIRECTIONS else h
        return 0.5 * math.sqrt(2.0 * h * h + third * third)


@dataclass
class DirectionResult:
    """Result for one rolling direction (or for the 3D hull)."""

    label: str
    angle_deg: float | None
    strike_points: FloatArray
    triangle_ids: IntArray
    contact_centers: FloatArray
    n_grid_points: int
    #: Closest mesh point of every contact centre (not de-duplicated, float32);
    #: dense where many sphere positions touch the same spot (edges, corners).
    contact_points: FloatArray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    #: Number of fine grid positions each contact point stands for (``None`` = 1 each);
    #: mode C places one position per feature block instead of a dense band.
    contact_weights: FloatArray | None = None


@dataclass
class StrikeResult:
    """Combined result of a rolling sphere evaluation."""

    mode: Mode
    params: SphereParams
    directions: list[DirectionResult]
    face_hits: IntArray
    backend: str
    elapsed_s: float
    notes: list[str] = field(default_factory=list)

    @property
    def strike_points(self) -> FloatArray:
        """All strike points of all directions (may contain duplicates across directions)."""
        if not self.directions:
            return np.zeros((0, 3))
        return np.vstack([d.strike_points for d in self.directions])

    @property
    def direction_index(self) -> IntArray:
        """Index into :attr:`directions` for every row of :attr:`strike_points`."""
        if not self.directions:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(
            [np.full(len(d.strike_points), i, dtype=np.int64) for i, d in enumerate(self.directions)]
        )

    @property
    def n_strike_points(self) -> int:
        return sum(len(d.strike_points) for d in self.directions)

    @property
    def contact_points(self) -> FloatArray:
        """Contact points of all directions (not de-duplicated)."""
        parts = [d.contact_points for d in self.directions if d.n_grid_points > 0]
        return np.vstack(parts) if parts else np.zeros((0, 3), dtype=np.float32)

    @property
    def contact_weights(self) -> FloatArray:
        """Weight of every row of :attr:`contact_points`."""
        parts = [d.contact_weights if d.contact_weights is not None else np.ones(len(d.contact_points))
                 for d in self.directions if d.n_grid_points > 0]
        return np.concatenate(parts) if parts else np.zeros(0)


# --------------------------------------------------------------------------- helpers


def penetration_depth(radius: float, spacing: float) -> float:
    """Penetration depth p of the sphere between two points at distance ``spacing``.

    ``p = r - sqrt(r² - (d/2)²)`` (DIN EN 62305-3, Anhang A).
    """
    half = spacing / 2.0
    if half > radius:
        return math.inf
    return radius - math.sqrt(radius * radius - half * half)


def _aligned_range(lo: float, hi: float, step: float) -> FloatArray:
    """Grid values that are multiples of ``step`` covering [lo, hi]."""
    start = math.floor(lo / step) * step
    stop = math.ceil(hi / step) * step
    n = int(round((stop - start) / step)) + 1
    return start + step * np.arange(n, dtype=np.float64)


def _z_range(mesh: TriMesh, params: SphereParams, tol: float) -> FloatArray:
    """Centre heights from ``ground + r`` up to safely above the model."""
    _, hi = mesh.bounds
    z0 = params.ground_z + params.radius
    z_top = max(float(hi[2]) + params.radius + tol + params.grid_step, z0 + params.grid_step)
    n = int(math.ceil((z_top - z0) / params.grid_step)) + 1
    return z0 + params.grid_step * np.arange(n, dtype=np.float64)


def _bbox_lower_bound(points: FloatArray, lo: FloatArray, hi: FloatArray) -> FloatArray:
    """Lower bound of the distance from points to anything inside the box [lo, hi]."""
    delta = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    return np.linalg.norm(delta, axis=1)


def _direction_frame(angle_deg: float) -> tuple[FloatArray, FloatArray]:
    a = math.radians(angle_deg)
    u = np.array([math.cos(a), math.sin(a), 0.0])
    v = np.array([-math.sin(a), math.cos(a), 0.0])
    return u, v


def _dedupe(points: FloatArray, tri_ids: IntArray, cell: float) -> tuple[FloatArray, IntArray]:
    """Remove near-duplicate points (one representative per cube of edge ``cell``)."""
    if len(points) == 0:
        return points, tri_ids
    keys = np.floor(points / cell).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx.sort()
    return points[idx], tri_ids[idx]


def contact_density(
    surface_points: FloatArray, contact_points: FloatArray, radius: float, weights: FloatArray | None = None
) -> FloatArray:
    """Number of contact points within ``radius`` of each surface point (heatmap values).

    Args:
        surface_points: (V, 3) points on the model surface, e.g. vertices of a refined mesh.
        contact_points: (K, 3) contact points (:attr:`StrikeResult.contact_points`).
        radius: Search radius, typically 1–2 grid steps.
        weights: Optional (K,) weights (:attr:`StrikeResult.contact_weights`); default 1.
    """
    if len(contact_points) == 0 or len(surface_points) == 0:
        return np.zeros(len(surface_points))
    contact_points = np.asarray(contact_points, dtype=np.float64)
    tree = cKDTree(contact_points)
    density = np.asarray(tree.query_ball_point(surface_points, radius, return_length=True), dtype=np.float64)
    if weights is not None:
        # Few points carry a weight != 1: add their surplus via the surface tree.
        heavy = np.nonzero(weights != 1.0)[0]
        if len(heavy):
            hits = cKDTree(np.asarray(surface_points, dtype=np.float64)).query_ball_point(
                contact_points[heavy], radius)
            counts = np.fromiter((len(x) for x in hits), dtype=np.int64, count=len(hits))
            if counts.sum():
                np.add.at(density, np.concatenate(hits).astype(np.int64),
                          np.repeat(weights[heavy] - 1.0, counts))
    return density


def estimate_grid_points(
    mesh: TriMesh, params: SphereParams, mode: Mode, angles: Sequence[float] = (0.0,)
) -> int:
    """Number of grid points the evaluation will examine (for UI warnings)."""
    tol = params.tolerance(mode)
    nz = len(_z_range(mesh, params, tol))
    margin = params.radius + tol + params.grid_step
    lo, hi = mesh.bounds
    if mode in (Mode.HULL_3D, Mode.ADAPTIVE):
        nx = (hi[0] - lo[0] + 2 * margin) / params.grid_step + 1
        ny = (hi[1] - lo[1] + 2 * margin) / params.grid_step + 1
        full = int(nx * ny * nz)
        if mode is Mode.HULL_3D:
            return full
        # Coarse stage plus a band around the sphere-centre hull; each finer stage
        # evaluates 8 children per boundary block (empirical factor ~1.3 · A / h²).
        r = params.radius
        dx, dy, dz = hi - lo
        area = (dx + 2 * r) * (dy + 2 * r) + 2 * (dx + dy + 4 * r) * (dz + r)
        n_coarse = full / 8 ** params.refinement_levels
        band = 1.3 * area / params.grid_step**2 if params.refinement_levels else 0.0
        if params.feature_blocks and params.refinement_levels:
            # Feature blocks of edge ~s_max cover the hull; fine work remains on the
            # model surface (upper bound: all of it is touched).
            s_max = max(math.sqrt(8.0 * r * tol), params.grid_step)
            tri = mesh.vertices[mesh.faces]
            surface = 0.5 * float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                                 axis=1).sum())
            band = min(band, 40.0 * area / s_max**2 + surface / params.grid_step**2)
        return int(min(full, n_coarse + band))
    total = 0
    for angle in _unique_axes(angles):
        u, v = _direction_frame(angle)
        t = mesh.vertices @ u
        s = mesh.vertices @ v
        nt = (t.max() - t.min() + 2 * margin) / params.grid_step + 1
        ns = (s.max() - s.min() + 2 * margin) / params.effective_plane_spacing + 1
        total += int(nt * ns * nz)
    return total


def _unique_axes(angles: Sequence[float]) -> list[float]:
    """Angles that need an own computation (θ and θ+180° are equivalent)."""
    seen: list[float] = []
    for a in angles:
        if not any(abs(((a - b) % 180.0)) < 1e-9 or abs(((a - b) % 180.0) - 180.0) < 1e-9 for b in seen):
            seen.append(a)
    return seen


# --------------------------------------------------------------------------- core grid evaluation


def _evaluate_grid(
    query: DistanceQuery,
    mesh: TriMesh,
    params: SphereParams,
    tol: float,
    origin_u: FloatArray,
    origin_v: FloatArray,
    t_vals: FloatArray,
    s_vals: FloatArray,
    z_vals: FloatArray,
    in_plane_only: bool,
    progress: ProgressScope,
) -> FloatArray:
    """Evaluate a (t, s, z) grid and return the reachable contact centres (K, 3).

    World position of grid node (i, j, k): ``t[i]*u + s[j]*v + z[k]*e_z``.
    """
    nt, ns, nz = len(t_vals), len(s_vals), len(z_vals)
    n_total = nt * ns * nz
    if n_total > params.max_grid_points:
        raise MemoryError(
            f"Das Raster hätte {n_total:,} Punkte (Limit {params.max_grid_points:,}). "
            "Bitte Rasterweite vergrößern."
        )
    r = params.radius
    lo, hi = mesh.bounds
    dist = np.full((nt, ns, nz), np.inf, dtype=np.float32)

    # Horizontal positions of one layer, shape (nt*ns, 3) with z = 0.
    tt, ss = np.meshgrid(t_vals, s_vals, indexing="ij")
    layer_xy = tt.reshape(-1, 1) * origin_u + ss.reshape(-1, 1) * origin_v

    for k, z in enumerate(z_vals):
        pts = layer_xy.copy()
        pts[:, 2] = z
        # Nodes farther than r + tol from the bounding box can neither be invalid nor in contact.
        need = _bbox_lower_bound(pts, lo, hi) < r + tol
        if np.any(need):
            layer = np.full(len(pts), np.inf, dtype=np.float32)
            layer[need] = query.distance(pts[need])
            dist[:, :, k] = layer.reshape(nt, ns)
        progress((k + 1) / nz, f"Abstände berechnen: Ebene {k + 1}/{nz}")

    valid = dist >= r
    if in_plane_only:
        # Connect neighbours along t and z only -> each vertical plane is independent.
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, :] = True
        structure[:, 1, 1] = True
    else:
        structure = ndimage.generate_binary_structure(3, 1)
    labels, _ = ndimage.label(valid, structure=structure)
    top_labels = np.unique(labels[:, :, -1])
    top_labels = top_labels[top_labels > 0]
    reachable = np.isin(labels, top_labels)
    contact = reachable & (dist < r + tol)
    log.debug("Raster %dx%dx%d: %d gültig, %d erreichbar, %d Kontakt",
              nt, ns, nz, int(valid.sum()), int(reachable.sum()), int(contact.sum()))

    i, j, k = np.nonzero(contact)
    centers = t_vals[i, None] * origin_u + s_vals[j, None] * origin_v
    centers[:, 2] = z_vals[k]
    return centers


def _contacts_to_strikes(
    query: DistanceQuery, centers: FloatArray, params: SphereParams, chunk: int = 500_000
) -> tuple[FloatArray, IntArray]:
    """Closest mesh points (and their triangles) of the contact centres."""
    pts: list[FloatArray] = []
    ids: list[IntArray] = []
    for c0 in range(0, len(centers), chunk):
        p, t, _ = query.closest(centers[c0 : c0 + chunk])
        pts.append(p)
        ids.append(t)
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    points = np.vstack(pts)
    tri_ids = np.concatenate(ids)
    above_ground = points[:, 2] >= params.ground_z - 1e-6
    return points[above_ground], tri_ids[above_ground]


# --------------------------------------------------------------------------- public API


def roll_3d(
    mesh: TriMesh,
    params: SphereParams,
    progress: ProgressCallback = no_progress,
    cancel: CancelCallback = never_cancel,
    query: DistanceQuery | None = None,
) -> StrikeResult:
    """Mode B: full 3D evaluation of all sphere positions around the model."""
    t_start = time.perf_counter()
    scope = ProgressScope(progress, 0.0, 1.0, cancel)
    query = query or create_distance_query(mesh)
    tol = params.tolerance(Mode.HULL_3D)
    lo, hi = mesh.bounds
    margin = params.radius + tol + params.grid_step
    x_vals = _aligned_range(lo[0] - margin, hi[0] + margin, params.grid_step)
    y_vals = _aligned_range(lo[1] - margin, hi[1] + margin, params.grid_step)
    z_vals = _z_range(mesh, params, tol)
    log.info("Modus B: r=%.2f m, h=%.3f m, tol=%.3f m, Raster %dx%dx%d",
             params.radius, params.grid_step, tol, len(x_vals), len(y_vals), len(z_vals))

    centers = _evaluate_grid(
        query, mesh, params, tol, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
        x_vals, y_vals, z_vals, in_plane_only=False, progress=scope.sub(0.0, 0.9),
    )
    scope(0.9, "Einschlagpunkte bestimmen")
    raw_points, raw_ids = _contacts_to_strikes(query, centers, params)
    face_hits = np.bincount(raw_ids, minlength=mesh.n_faces)
    points, tri_ids = _dedupe(raw_points, raw_ids, params.grid_step / 2.0)
    direction = DirectionResult("3D", None, points, tri_ids, centers, len(x_vals) * len(y_vals) * len(z_vals),
                                raw_points.astype(np.float32))
    scope(1.0, "Fertig")
    return StrikeResult(Mode.HULL_3D, params, [direction], face_hits, query.name,
                        time.perf_counter() - t_start)


def roll_directions(
    mesh: TriMesh,
    params: SphereParams,
    angles_deg: Sequence[float],
    progress: ProgressCallback = no_progress,
    cancel: CancelCallback = never_cancel,
    plane_offsets: Sequence[float] | None = None,
    query: DistanceQuery | None = None,
) -> StrikeResult:
    """Mode A: roll the sphere along horizontal directions in parallel vertical planes.

    Args:
        mesh: Structure in metres.
        params: Sphere parameters.
        angles_deg: Rolling directions, measured counter-clockwise from +x in degrees.
        progress: Progress callback.
        cancel: Cancellation callback.
        plane_offsets: Explicit plane offsets ``s`` (distance along the horizontal normal
            of the direction); default is a regular set covering the model.
        query: Optional pre-built distance query.
    """
    if not angles_deg:
        raise ValueError("Mindestens eine Rollrichtung auswählen.")
    t_start = time.perf_counter()
    scope = ProgressScope(progress, 0.0, 1.0, cancel)
    query = query or create_distance_query(mesh)
    tol = params.tolerance(Mode.DIRECTIONS)
    margin = params.radius + tol + params.grid_step
    z_vals = _z_range(mesh, params, tol)
    face_hits = np.zeros(mesh.n_faces, dtype=np.int64)
    notes: list[str] = []
    computed: dict[float, DirectionResult] = {}
    results: list[DirectionResult] = []
    n = len(angles_deg)

    for idx, angle in enumerate(angles_deg):
        label = f"{angle % 360:g}°"
        twin = next((a for a in computed if abs((angle - a) % 180.0) < 1e-9
                     or abs((angle - a) % 180.0 - 180.0) < 1e-9), None)
        if twin is not None:
            src = computed[twin]
            notes.append(f"{label} entspricht {src.label} (Gegenrichtung) – Ergebnis übernommen.")
            results.append(DirectionResult(label, angle, src.strike_points, src.triangle_ids,
                                           src.contact_centers, 0, src.contact_points))
            continue

        sub = scope.sub(idx / n, (idx + 1) / n)
        u, v = _direction_frame(angle)
        t_proj = mesh.vertices @ u
        s_proj = mesh.vertices @ v
        t_vals = _aligned_range(t_proj.min() - margin, t_proj.max() + margin, params.grid_step)
        if plane_offsets is not None:
            s_vals = np.asarray(sorted(plane_offsets), dtype=np.float64)
        else:
            s_vals = _aligned_range(s_proj.min() - margin, s_proj.max() + margin, params.effective_plane_spacing)
        log.info("Modus A, Richtung %s: %d Ebenen x %d x %d", label, len(s_vals), len(t_vals), len(z_vals))

        centers = _evaluate_grid(
            query, mesh, params, tol, u, v, t_vals, s_vals, z_vals,
            in_plane_only=True, progress=sub.sub(0.0, 0.9),
        )
        sub(0.9, f"Richtung {label}: Einschlagpunkte bestimmen")
        raw_points, raw_ids = _contacts_to_strikes(query, centers, params)
        face_hits += np.bincount(raw_ids, minlength=mesh.n_faces)
        points, tri_ids = _dedupe(raw_points, raw_ids, params.grid_step / 2.0)
        res = DirectionResult(label, angle, points, tri_ids, centers, len(t_vals) * len(s_vals) * len(z_vals),
                              raw_points.astype(np.float32))
        computed[angle] = res
        results.append(res)

    scope(1.0, "Fertig")
    return StrikeResult(Mode.DIRECTIONS, params, results, face_hits, query.name,
                        time.perf_counter() - t_start, notes)


# --------------------------------------------------------------------------- mode C: adaptive


_CHILD_OFFSETS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)
_FACE_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)


def _thousands(n: int) -> str:
    """German thousands separator: 1234567 -> '1.234.567'."""
    return f"{n:,}".replace(",", ".")


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


def _linear_keys(idx: IntArray, dims: IntArray) -> IntArray:
    return (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]


def _block_corners(idx: IntArray, n: int) -> IntArray:
    """Fine-node coordinates of the 8 corner nodes of each block, shape (B, 8, 3)."""
    return idx[:, None, :] * n + _CHILD_OFFSETS[None] * (n - 1)


def _closest_chunked(
    query: DistanceQuery, points: FloatArray, chunk: int = 1_000_000
) -> tuple[FloatArray, IntArray, FloatArray]:
    parts = [query.closest(points[c0 : c0 + chunk]) for c0 in range(0, len(points), chunk)]
    if not parts:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0)
    return (np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]).astype(np.float64))


#: Feature kinds of mode C feature blocks.
_POINT, _EDGE, _PLANE = 0, 1, 2


def _classify_features(
    query: DistanceQuery, world: FloatArray, centres: FloatArray, h: float
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Find blocks whose nearest geometry is a single point, edge or plane.

    The closest mesh points of the 8 corners and the centre of each block are
    fitted by a point, a line or a plane (deviation ≤ h/2). The fit is accepted
    only if it also explains all 9 distances within h/2.

    Args:
        world: (B, 8, 3) corner positions; centres: (B, 3) block centres.

    Returns:
        kind (-1 = none, :data:`_POINT`, :data:`_EDGE`, :data:`_PLANE`), anchor (a point of the
        feature) and direction (edge axis or plane normal pointing towards the block).
    """
    samples = np.concatenate([world, centres[:, None]], axis=1)
    p, _, d = _closest_chunked(query, samples.reshape(-1, 3))
    p = p.reshape(samples.shape)
    d = d.reshape(samples.shape[:2])
    anchor = p.mean(axis=1)
    q = p - anchor[:, None]
    _, vecs = np.linalg.eigh(np.einsum("bsi,bsj->bij", q, q))  # eigenvectors in columns, ascending
    thin = np.abs(np.einsum("bsi,bij->bsj", q, vecs)).max(axis=1) <= h / 2
    kind = np.full(len(samples), -1, dtype=np.int64)
    kind[thin[:, 0]] = _PLANE
    kind[thin[:, 0] & thin[:, 1]] = _EDGE
    kind[thin.all(axis=1)] = _POINT

    normal, axis = vecs[:, :, 0], vecs[:, :, 2]
    rel = samples - anchor[:, None]
    to_point = np.linalg.norm(rel, axis=2)
    along = np.einsum("bsi,bi->bs", rel, axis)
    model = np.select(
        [kind[:, None] == _EDGE, kind[:, None] == _PLANE],
        [np.sqrt(np.maximum(to_point**2 - along**2, 0.0)), np.abs(np.einsum("bsi,bi->bs", rel, normal))],
        to_point,
    )
    kind[np.abs(model - d).max(axis=1) > h / 2] = -1

    direction = np.where((kind == _PLANE)[:, None], normal, axis)
    rows = np.arange(len(direction))
    flip = np.where(kind == _PLANE, np.einsum("bi,bi->b", centres - anchor, direction) < 0,
                    direction[rows, np.abs(direction).argmax(axis=1)] < 0)
    direction[flip] *= -1.0
    return kind, anchor, direction


def _feature_contacts(
    query: DistanceQuery,
    lo_c: FloatArray,
    hi_c: FloatArray,
    kind: IntArray,
    anchor: FloatArray,
    direction: FloatArray,
    n: int,
    h: float,
    r: float,
    tol: float,
    z_min: float,
) -> tuple[FloatArray, FloatArray, IntArray, FloatArray, int]:
    """Contact positions of feature blocks, placed directly on their feature.

    Point: one position; edge: one per ``h`` along the edge; plane: one per ``h²``
    of the plane inside the block. Each position is moved to distance
    ``r + tol/2`` from its closest mesh point and kept if it is a contact.

    Args:
        lo_c, hi_c: (B, 3) world positions of the first and last fine node of each block.

    Returns:
        Centres, closest points, triangle ids, weights (number of fine grid
        positions each centre stands for) and the number of distance queries.
    """
    off = r + tol / 2.0
    # Weight 1 per h² of contact surface (about one fine position, the band is tol ≈ h
    # thick). A surface crossing a cube of edge s cuts on average 2/3 s².
    cut = 2.0 / 3.0 * n * n
    mid = (lo_c + hi_c) / 2.0
    cand: list[FloatArray] = []
    weights: list[FloatArray] = []

    sel = kind == _POINT
    u = mid[sel] - anchor[sel]
    cand.append(anchor[sel] + u / np.linalg.norm(u, axis=1, keepdims=True) * off)
    weights.append(np.full(int(sel.sum()), cut))

    sel = np.nonzero(kind == _EDGE)[0]
    if len(sel):
        a, base = direction[sel], anchor[sel]
        span = np.stack([lo_c[sel], hi_c[sel]], axis=1)  # (B, 2, 3)
        corners = span[:, _CHILD_OFFSETS, [0, 1, 2]]  # (B, 8, 3)
        t = np.einsum("bsi,bi->bs", corners, a)
        k_lo = np.ceil(t.min(axis=1) / h).astype(np.int64)
        count = np.maximum(np.floor(t.max(axis=1) / h).astype(np.int64) - k_lo + 1, 0)
        rep = np.repeat(np.arange(len(sel)), count)
        k = k_lo[rep] + np.arange(len(rep)) - np.repeat(np.cumsum(count) - count, count)
        perp = mid[sel] - base
        perp -= np.einsum("bi,bi->b", perp, a)[:, None] * a
        perp /= np.linalg.norm(perp, axis=1, keepdims=True)
        shift = k * h - np.einsum("bi,bi->b", base, a)[rep]
        cand.append(base[rep] + shift[:, None] * a[rep] + perp[rep] * off)
        weights.append((cut / np.maximum(count, 1))[rep])

    for b in np.nonzero(kind == _PLANE)[0]:
        nrm = direction[b]
        ref = np.array([0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        v1 = np.cross(nrm, ref)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(nrm, v1)
        corners = np.stack([lo_c[b], hi_c[b]])[_CHILD_OFFSETS, [0, 1, 2]]
        ranges = [np.arange(math.ceil((corners @ v).min() / h), math.floor((corners @ v).max() / h) + 1) * h
                  for v in (v1, v2)]
        aa, bb = np.meshgrid(*ranges, indexing="ij")
        pts = aa.reshape(-1, 1) * v1 + bb.reshape(-1, 1) * v2 + nrm * (anchor[b] @ nrm + off)
        inside = np.all((pts >= lo_c[b] - h / 2) & (pts < hi_c[b] + h / 2), axis=1)
        cand.append(pts[inside])
        weights.append(np.ones(int(inside.sum())))

    centres = np.vstack(cand)
    w = np.concatenate(weights)
    p, _, d = _closest_chunked(query, centres)
    ok = d > 0
    centres = p[ok] + (centres[ok] - p[ok]) * (off / d[ok])[:, None]
    p, tri, d = _closest_chunked(query, centres)
    keep = (d >= r) & (d < r + tol) & (centres[:, 2] >= z_min - 1e-9)
    return centres[keep], p[keep], tri[keep], w[ok][keep], 2 * len(w)


def roll_adaptive(
    mesh: TriMesh,
    params: SphereParams,
    progress: ProgressCallback = no_progress,
    cancel: CancelCallback = never_cancel,
    query: DistanceQuery | None = None,
) -> StrikeResult:
    """Mode C: 3D evaluation with coarse-to-fine refinement (octree).

    The fine grid of sphere centres (spacing ``params.grid_step``) is covered by
    cubic blocks of ``2^L`` fine nodes per axis. For every block only the
    distance ``d`` at its centre is computed. Since the distance function is
    1-Lipschitz, all nodes of a block with half diagonal ``e`` have distances in
    ``[d - e, d + e]``:

    * ``d - e >= r + tol`` – block is *free*: all nodes valid, none in contact,
    * ``d + e < r`` – block is *solid*: all nodes invalid,
    * otherwise the block is split into 8 children and examined on the next stage.

    With ``params.feature_blocks`` (default), a block of edge ≤ ``sqrt(8·r·tol)``
    that would be split is first tested for a single nearest feature: if the
    closest mesh points of its corners and centre lie on one point, edge or plane
    and explain the measured distances, it is a *feature block*. It is not split;
    instead contact positions are placed directly at distance ``r + tol/2`` from
    the feature (1, ``s/h`` or ``(s/h)²`` positions instead of ``(s/h)³`` nodes).
    The contact surface of large spheres is mostly such domes over corners and
    cylinders around edges, so the cost then grows with the touched model area
    instead of the much larger hull area.

    Free blocks of all sizes, feature blocks and the valid fine nodes then form a
    graph (face neighbours); the components connected to the top of the domain
    are reachable. Without feature blocks the result is identical to
    :func:`roll_3d` with the fine grid step; with them the strike points agree to
    about one grid step.
    """
    t_start = time.perf_counter()
    scope = ProgressScope(progress, 0.0, 1.0, cancel)
    query = query or create_distance_query(mesh)
    h, r = params.grid_step, params.radius
    tol = params.tolerance(Mode.ADAPTIVE)
    levels = params.refinement_levels
    m = 1 << levels
    lo, hi = mesh.bounds

    # Fine node (i, j, k) sits at ((ix0 + i)·h, (iy0 + j)·h, ground + r + k·h) – the same
    # alignment as the grid of roll_3d. The domain is a whole number of coarse blocks.
    margin = r + tol + h
    ix0 = math.floor((lo[0] - margin) / h)
    iy0 = math.floor((lo[1] - margin) / h)
    z0 = params.ground_z + r
    dims = np.array([
        _round_up(math.ceil((hi[0] + margin) / h) - ix0 + 1, m),
        _round_up(math.ceil((hi[1] + margin) / h) - iy0 + 1, m),
        _round_up(max(math.ceil((hi[2] + r + tol + h - z0) / h), 0) + 2, m),
    ], dtype=np.int64)
    n_full = int(np.prod(dims))
    top_dims = dims // m
    if int(np.prod(top_dims)) > params.max_grid_points:
        raise MemoryError("Das grobe Raster ist zu groß. Bitte die grobe Rasterweite vergrößern.")
    log.info("Modus C: r=%.2f m, %d Stufen %.3g m -> %.3g m, tol=%.3f m, feines Raster %dx%dx%d",
             r, levels + 1, h * m, h, tol, *dims)

    def to_world(c: FloatArray) -> FloatArray:
        return np.column_stack([(ix0 + c[:, 0]) * h, (iy0 + c[:, 1]) * h, z0 + c[:, 2] * h])

    # A feature hidden between the 9 samples of a block can be at most about s²/(8r)
    # closer than the fitted feature; blocks up to s_max keep that error below tol.
    s_max = math.sqrt(8.0 * r * tol) if params.feature_blocks else 0.0

    # ---- stage-wise classification ------------------------------------------------
    valid_blocks: dict[int, IntArray] = {}  # level -> block indices of valid final blocks
    features: dict[int, tuple[IntArray, IntArray, FloatArray, FloatArray]] = {}  # level -> feature blocks
    contact_flag = np.zeros(0, dtype=bool)  # for the valid fine nodes (level 0)
    n_queries = 0
    idx = np.indices(top_dims).reshape(3, -1).T.astype(np.int64)
    for level in range(levels, -1, -1):
        stage = levels - level
        sub = scope.sub(0.85 * stage / (levels + 1), 0.85 * (stage + 1) / (levels + 1))
        n = 1 << level
        half_diag = (n - 1) * h * math.sqrt(3.0) / 2.0
        centers = to_world(idx * n + (n - 1) / 2.0)
        d = np.full(len(idx), np.inf)
        need = np.nonzero(_bbox_lower_bound(centers, lo, hi) - half_diag < r + tol)[0]
        chunk = 1_000_000
        for c0 in range(0, len(need), chunk):
            sel = need[c0 : c0 + chunk]
            d[sel] = query.distance(centers[sel])
            sub((c0 + len(sel)) / len(need),
                f"Stufe {stage + 1}/{levels + 1} (Raster {n * h:g} m): {_thousands(len(need))} Abstände")
        n_queries += len(need)
        if level == 0:
            valid = d >= r
            valid_blocks[0] = idx[valid]
            contact_flag = d[valid] < r + tol
            break
        free = d - half_diag >= r + tol
        mixed = ~free & (d + half_diag >= r)
        if n * h <= s_max and mixed.any():
            sub(1.0, f"Stufe {stage + 1}/{levels + 1}: Kanten und Flächen erkennen")
            cand = np.nonzero(mixed)[0]
            corners = to_world(_block_corners(idx[cand], n).reshape(-1, 3)).reshape(-1, 8, 3)
            kind, anchor, direction = _classify_features(query, corners, centers[cand], h)
            n_queries += 9 * len(cand)
            hit = kind >= 0
            features[level] = (idx[cand[hit]], kind[hit], anchor[hit], direction[hit])
            mixed[cand[hit]] = False
        valid_blocks[level] = idx[free]
        idx = (idx[mixed][:, None, :] * 2 + _CHILD_OFFSETS[None]).reshape(-1, 3)
        if len(idx) > params.max_grid_points:
            raise MemoryError("Zu viele Blöcke im Kontaktbereich. Bitte die feine Rasterweite vergrößern.")
        if len(idx) == 0:
            break

    # ---- reachability on the block graph ------------------------------------------
    # Nodes per level: free blocks first, then feature blocks. A feature block is only
    # partly valid, but its valid part (outside a ball, cylinder or half-space of
    # radius r) is connected and touches its free neighbours.
    scope(0.86, "Erreichbarkeit bestimmen")
    empty = np.zeros((0, 3), dtype=np.int64)
    nodes = {lv: np.vstack([valid_blocks.get(lv, empty), features[lv][0] if lv in features else empty])
             for lv in set(valid_blocks) | set(features)}
    present = [lv for lv in sorted(nodes) if len(nodes[lv])]
    offsets: dict[int, int] = {}
    lookup: dict[int, tuple[IntArray, IntArray]] = {}
    n_nodes = 0
    for lv in present:
        keys = _linear_keys(nodes[lv], dims >> lv)
        order = np.argsort(keys)
        lookup[lv] = (keys[order], order)
        offsets[lv] = n_nodes
        n_nodes += len(keys)

    def locate(points: IntArray) -> IntArray:
        """Node id of the final block containing each fine node (-1 if none)."""
        out = np.full(len(points), -1, dtype=np.int64)
        inside = np.nonzero(np.all((points >= 0) & (points < dims), axis=1))[0]
        for lv in present:
            keys_sorted, order = lookup[lv]
            q = _linear_keys(points[inside] >> lv, dims >> lv)
            pos = np.minimum(np.searchsorted(keys_sorted, q), len(keys_sorted) - 1)
            hit = keys_sorted[pos] == q
            out[inside[hit]] = offsets[lv] + order[pos[hit]]
        return out

    # Each block looks one fine node beyond each of its faces. Aligned blocks of equal or
    # larger size cover the whole face, so every adjacency is found from the smaller side.
    src: list[IntArray] = []
    dst: list[IntArray] = []
    seeds: list[IntArray] = []
    for lv in present:
        blocks = nodes[lv]
        n = 1 << lv
        ids = offsets[lv] + np.arange(len(blocks), dtype=np.int64)
        base = blocks * n
        for face_dir in _FACE_DIRS:
            probe = base + np.where(face_dir > 0, n, face_dir)
            nb = locate(probe)
            ok = nb >= 0
            src.append(ids[ok])
            dst.append(nb[ok])
        seeds.append(ids[(blocks[:, 2] + 1) * n == dims[2]])
    scope(0.88, "Erreichbarkeit bestimmen")

    reachable = np.zeros(n_nodes, dtype=bool)
    if n_nodes:
        rows = np.concatenate(src) if src else np.zeros(0, dtype=np.int64)
        cols = np.concatenate(dst) if dst else np.zeros(0, dtype=np.int64)
        graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_nodes, n_nodes))
        _, labels = connected_components(graph, directed=False)
        reachable = np.isin(labels, np.unique(labels[np.concatenate(seeds)]))
    if 0 in offsets and len(valid_blocks.get(0, empty)):
        leaf_ids = offsets[0] + np.arange(len(valid_blocks[0]))
        centers = to_world(valid_blocks[0][contact_flag & reachable[leaf_ids]].astype(np.float64))
    else:
        centers = np.zeros((0, 3))

    scope(0.9, "Einschlagpunkte bestimmen")
    raw_points, raw_ids = _contacts_to_strikes(query, centers, params)
    weights = np.ones(len(raw_points))
    n_feature = 0
    for lv, (blocks, kind, anchor, direction) in features.items():
        ids = offsets[lv] + len(valid_blocks.get(lv, empty)) + np.arange(len(blocks))
        sel = reachable[ids]
        if not sel.any():
            continue
        n = 1 << lv
        f_centers, f_points, f_ids, f_weights, n_q = _feature_contacts(
            query, to_world(blocks[sel] * n), to_world(blocks[sel] * n + (n - 1)), kind[sel], anchor[sel],
            direction[sel], n, h, r, tol, z0)
        n_queries += n_q
        n_feature += int(sel.sum())
        above = f_points[:, 2] >= params.ground_z - 1e-6
        centers = np.vstack([centers, f_centers[above]])
        raw_points = np.vstack([raw_points, f_points[above]])
        raw_ids = np.concatenate([raw_ids, f_ids[above]])
        weights = np.concatenate([weights, f_weights[above]])
    face_hits = np.bincount(raw_ids, minlength=mesh.n_faces)
    points, tri_ids = _dedupe(raw_points, raw_ids, h / 2.0)
    direction = DirectionResult("3D adaptiv", None, points, tri_ids, centers, n_queries,
                                raw_points.astype(np.float32), weights)
    share = 100.0 * n_queries / n_full
    note = (f"Adaptiv: {levels + 1} Stufen {h * m:g} m → {h:g} m, {_thousands(n_queries)} "
            f"Abstandsberechnungen statt {_thousands(n_full)} beim vollen feinen Raster ({share:.2g} %)")
    if n_feature:
        note += f", {_thousands(n_feature)} Blöcke direkt an Ecke, Kante oder Fläche ausgewertet"
    note += "."
    log.info(note)
    scope(1.0, "Fertig")
    return StrikeResult(Mode.ADAPTIVE, params, [direction], face_hits, query.name,
                        time.perf_counter() - t_start, [note])
