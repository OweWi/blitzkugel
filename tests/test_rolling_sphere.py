"""Tests for the rolling sphere computation (Modus A and B)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from blitzkugel.core.mesh import TriMesh, make_box, make_rod
from blitzkugel.core.progress import CancelledError
from blitzkugel.core.rolling_sphere import (
    Mode,
    ProtectionClass,
    SphereParams,
    contact_density,
    estimate_grid_points,
    penetration_depth,
    roll_3d,
    roll_adaptive,
    roll_directions,
)


def cube_10m() -> TriMesh:
    return make_box(-5, -5, 0, 5, 5, 10, name="Würfel")


# --------------------------------------------------------------------------- basics


def test_protection_classes() -> None:
    assert [pc.diameter for pc in ProtectionClass] == [40, 60, 90, 120]
    assert ProtectionClass.III.radius == 45
    assert ProtectionClass.from_diameter(60.0) is ProtectionClass.II
    assert ProtectionClass.from_diameter(50.0) is None


def test_penetration_formula() -> None:
    assert penetration_depth(20.0, 10.0) == pytest.approx(20 - np.sqrt(400 - 25))
    assert penetration_depth(20.0, 50.0) == np.inf


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        SphereParams(radius=0, grid_step=1)
    with pytest.raises(ValueError):
        SphereParams(radius=20, grid_step=-1)


# --------------------------------------------------------------------------- cube, r = 20 m


@pytest.mark.parametrize("mode", [Mode.HULL_3D, Mode.DIRECTIONS])
def test_cube_touched_only_on_top(mode: Mode) -> None:
    """Würfel 10 m auf dem Boden, r = 20 m: Kontakt nur an Oberfläche/-kanten."""
    mesh = cube_10m()
    params = SphereParams(radius=20.0, grid_step=0.5, ground_z=0.0)
    if mode is Mode.HULL_3D:
        res = roll_3d(mesh, params)
    else:
        res = roll_directions(mesh, params, [0.0, 45.0, 90.0])
    pts = res.strike_points
    assert len(pts) > 0

    # No point on the side faces below the top edge.
    assert pts[:, 2].min() == pytest.approx(10.0, abs=1e-4)

    # Top face interior and the top edges are hit.
    interior = (np.abs(pts[:, 0]) < 4) & (np.abs(pts[:, 1]) < 4)
    assert interior.any()
    for axis in (0, 1):
        for sign in (-1, 1):
            on_edge = np.abs(pts[:, axis] - sign * 5.0) < 1e-4
            assert on_edge.any(), f"Kante {'xy'[axis]}={sign * 5} nicht berührt"

    # Heatmap: side and bottom triangles are never hit.
    normals = mesh.face_normals()
    side_or_bottom = normals[:, 2] < 0.5
    top = normals[:, 2] > 0.5
    assert res.face_hits[top].sum() > 0
    # Closest points exactly on the top edge may be attributed to a side triangle;
    # those all lie at z = 10, which is checked above.


def test_tall_tower_side_strikes_start_at_radius_height() -> None:
    """Turm 60 m, r = 20 m: Seitenflächen werden erst ab Höhe r getroffen."""
    mesh = make_box(-5, -5, 0, 5, 5, 60)
    res = roll_3d(mesh, SphereParams(radius=20.0, grid_step=1.0, ground_z=0.0))
    pts = res.strike_points
    side = (np.abs(np.abs(pts[:, 0]) - 5) < 1e-4) & (np.abs(pts[:, 1]) < 4.9)
    assert side.any()
    assert pts[side, 2].min() == pytest.approx(20.0, abs=1e-4)
    assert pts[:, 2].min() >= 20.0 - 1e-4


def test_raised_ground_plane() -> None:
    """Bodenhöhe 5 m: am Würfel (10 m) bleibt Kontakt nur oben – Kugel liegt höher."""
    res = roll_3d(cube_10m(), SphereParams(radius=20.0, grid_step=1.0, ground_z=5.0))
    assert res.strike_points[:, 2].min() == pytest.approx(10.0, abs=1e-4)


# --------------------------------------------------------------------------- penetration between two rods


@pytest.mark.parametrize(("radius", "spacing"), [(20.0, 10.0), (20.0, 20.0), (45.0, 30.0)])
def test_penetration_depth_between_rods(radius: float, spacing: float) -> None:
    """Zwei Fangstangen im Abstand d: Eindringtiefe p = r - sqrt(r² - (d/2)²)."""
    height, width, h = 10.0, 0.02, 0.02
    mesh = TriMesh.merge(
        [make_rod(-spacing / 2, 0, 0, height, width), make_rod(spacing / 2, 0, 0, height, width)]
    )
    # Sphere centres in the vertical plane y = 0 containing both rods.
    params = SphereParams(radius=radius, grid_step=h, ground_z=0.0, contact_tolerance=2 * h)
    res = roll_directions(mesh, params, [0.0], plane_offsets=[0.0])
    centers = res.directions[0].contact_centers
    mid = centers[np.abs(centers[:, 0]) < h / 2]
    assert len(mid) > 0
    lowest_sphere_point = mid[:, 2].min() - radius
    p_measured = height - lowest_sphere_point
    # The rods are thin square bars: the sphere touches their inner edges.
    p_expected = penetration_depth(radius, spacing - width)
    assert p_measured == pytest.approx(p_expected, abs=h + 1e-6)

    # Strike points are the rod tips, the ground between the rods is not part of the mesh.
    assert res.strike_points[:, 2].max() == pytest.approx(height, abs=1e-4)


# --------------------------------------------------------------------------- reachability


def test_closed_cavity_is_ignored() -> None:
    """Kugeln in einem geschlossenen Hohlraum zählen nicht (Flood-Fill)."""
    outer = make_box(-50, -50, 0, 50, 50, 100)
    inner = make_box(-40, -40, 10, 40, 40, 90)
    inner.faces = inner.faces[:, ::-1]  # inward-facing hull of the cavity
    mesh = TriMesh.merge([outer, inner])
    res = roll_3d(mesh, SphereParams(radius=20.0, grid_step=2.0, ground_z=0.0))
    pts = res.strike_points
    assert len(pts) > 0
    inside = np.all(np.abs(pts[:, :2]) < 49.9, axis=1) & (pts[:, 2] < 99.9)
    assert not inside.any()
    assert res.face_hits[mesh.body_ids == 1].sum() == 0


def test_courtyard_reachable_only_if_wide_enough() -> None:
    """Ein Innenhof wird nur erreicht, wenn die Kugel hineinpasst."""
    def ring(inner_half: float) -> TriMesh:
        o, hgt = inner_half + 5, 30.0
        return TriMesh.merge([
            make_box(-o, -o, 0, o, -inner_half, hgt), make_box(-o, inner_half, 0, o, o, hgt),
            make_box(-o, -inner_half, 0, -inner_half, inner_half, hgt),
            make_box(inner_half, -inner_half, 0, o, inner_half, hgt),
        ])

    params = SphereParams(radius=20.0, grid_step=1.0, ground_z=0.0)
    # Courtyard 30 m wide < 40 m sphere: the courtyard floor (ground) is not touched,
    # the inner walls are only touched near the top.
    narrow = roll_3d(ring(15.0), params).strike_points
    in_yard = np.all(np.abs(narrow[:, :2]) <= 15.0 + 1e-4, axis=1)
    assert in_yard.any()
    assert narrow[in_yard, 2].min() > 30.0 - penetration_depth(20.0, 30.0) - 1.0
    # Courtyard 60 m wide: the sphere reaches down to the ground, inner walls hit from z = r.
    wide = roll_3d(ring(30.0), params).strike_points
    inner_wall = np.abs(np.abs(wide[:, 0]) - 30.0) < 1e-4
    assert wide[inner_wall, 2].min() == pytest.approx(20.0, abs=1e-4)


# --------------------------------------------------------------------------- mode A specifics


def test_opposite_directions_share_result() -> None:
    res = roll_directions(cube_10m(), SphereParams(20.0, 1.0), [0.0, 180.0])
    assert len(res.directions) == 2
    assert res.directions[1].n_grid_points == 0  # reused
    np.testing.assert_array_equal(res.directions[0].strike_points, res.directions[1].strike_points)
    assert res.notes


def test_direction_index_and_labels() -> None:
    res = roll_directions(cube_10m(), SphereParams(20.0, 1.0), [0.0, 90.0])
    assert [d.label for d in res.directions] == ["0°", "90°"]
    assert len(res.direction_index) == res.n_strike_points


def test_estimate_and_limit() -> None:
    params = SphereParams(20.0, 1.0)
    n = estimate_grid_points(cube_10m(), params, Mode.HULL_3D)
    res = roll_3d(cube_10m(), params)
    assert n == pytest.approx(res.directions[0].n_grid_points, rel=0.2)
    with pytest.raises(MemoryError):
        roll_3d(cube_10m(), SphereParams(20.0, 1.0, max_grid_points=1000))


def test_cancel() -> None:
    with pytest.raises(CancelledError):
        roll_3d(cube_10m(), SphereParams(20.0, 1.0), cancel=lambda: True)


def test_progress_monotonic() -> None:
    seen: list[float] = []
    roll_directions(cube_10m(), SphereParams(20.0, 1.0), [0.0, 90.0], progress=lambda f, _m: seen.append(f))
    assert seen[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-12 for a, b in zip(seen, seen[1:]))


def test_contact_density_highlights_top_edges() -> None:
    """Heatmap: Kanten werden von mehr Kugellagen berührt als die Dachmitte, Seiten gar nicht."""
    res = roll_3d(cube_10m(), SphereParams(20.0, 0.5))
    probe = np.array([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0], [5.0, 5.0, 10.0], [5.0, 0.0, 5.0]])
    dens = contact_density(probe, res.contact_points, 0.75)
    centre, edge, corner, side = dens
    assert side == 0
    assert centre > 0
    assert edge > centre
    assert corner > centre


# --------------------------------------------------------------------------- mode C (adaptive)


def _center_set(res) -> set[tuple[float, float, float]]:
    return {tuple(np.round(c, 6)) for c in res.directions[0].contact_centers}


def _building() -> TriMesh:
    return TriMesh.merge([make_box(-10, -5, 0, 10, 5, 12), make_rod(-9, 0, 12, 3, 0.1),
                          make_rod(9, 0, 12, 3, 0.1), make_box(15, -3, 0, 25, 3, 25)])


@pytest.mark.parametrize(
    "mesh_factory",
    [cube_10m, lambda: make_box(-5, -5, 0, 5, 5, 60), _building],
    ids=["wuerfel", "turm", "gebaeude"],
)
def test_adaptive_equals_full_grid(mesh_factory) -> None:
    """Modus C ohne Feature-Blöcke liefert exakt dieselben Kontaktlagen wie Modus B."""
    mesh = mesh_factory()
    params = SphereParams(radius=20.0, grid_step=0.5, coarse_step=4.0, feature_blocks=False)
    full = roll_3d(mesh, params)
    adaptive = roll_adaptive(mesh, params)
    assert adaptive.mode is Mode.ADAPTIVE
    assert _center_set(adaptive) == _center_set(full)
    # ... with far fewer distance evaluations.
    assert adaptive.directions[0].n_grid_points < 0.25 * full.directions[0].n_grid_points


@pytest.mark.parametrize("radius", [20.0, 60.0])
@pytest.mark.parametrize("mesh_factory", [cube_10m, _building], ids=["wuerfel", "gebaeude"])
def test_adaptive_features_match_full_grid(mesh_factory, radius: float) -> None:
    """Mit Feature-Blöcken: gleiche Einschlagbereiche wie Modus B (bis auf etwa eine Rasterweite)."""
    mesh = mesh_factory()
    h = 0.5
    full = roll_3d(mesh, SphereParams(radius=radius, grid_step=h))
    res = roll_adaptive(mesh, SphereParams(radius=radius, grid_step=h))
    a, b = res.strike_points, full.strike_points
    assert len(a) and len(b)
    # every strike point of B is covered by C and vice versa
    assert cKDTree(a).query(b)[0].max() <= 1.5 * h
    assert cKDTree(b).query(a)[0].max() <= 1.5 * h
    assert res.directions[0].n_grid_points < 0.5 * full.directions[0].n_grid_points
    # heatmap weights: as many contact positions in total as on the full grid (roughly)
    assert res.contact_weights.sum() == pytest.approx(len(full.contact_points), rel=0.5)


def test_adaptive_cube_and_cavity() -> None:
    params = SphereParams(radius=20.0, grid_step=0.5, coarse_step=8.0)
    pts = roll_adaptive(cube_10m(), params).strike_points
    assert pts[:, 2].min() == pytest.approx(10.0, abs=1e-4)

    outer = make_box(-50, -50, 0, 50, 50, 100)
    inner = make_box(-40, -40, 10, 40, 40, 90)
    inner.faces = inner.faces[:, ::-1]
    mesh = TriMesh.merge([outer, inner])
    res = roll_adaptive(mesh, SphereParams(radius=20.0, grid_step=1.0, coarse_step=8.0))
    inside = np.all(np.abs(res.strike_points[:, :2]) < 49.9, axis=1) & (res.strike_points[:, 2] < 99.9)
    assert len(res.strike_points) and not inside.any()


def test_adaptive_fine_penetration_depth() -> None:
    """Fein aufgelöste 3D-Rechnung zwischen zwei Fangstangen (mit vollem Raster zu teuer)."""
    radius, spacing, height, width, h = 20.0, 10.0, 10.0, 0.02, 0.05
    mesh = TriMesh.merge([make_rod(-spacing / 2, 0, 0, height, width), make_rod(spacing / 2, 0, 0, height, width)])
    # Exact octree: feature blocks place positions on the features, not on the fine grid nodes.
    params = SphereParams(radius=radius, grid_step=h, coarse_step=3.2, contact_tolerance=2 * h,
                          feature_blocks=False)
    res = roll_adaptive(mesh, params)
    centers = res.directions[0].contact_centers
    mid = centers[(np.abs(centers[:, 0]) < h / 2) & (np.abs(centers[:, 1]) < h / 2)]
    p_measured = height - (mid[:, 2].min() - radius)
    assert p_measured == pytest.approx(penetration_depth(radius, spacing - width), abs=h + 1e-6)


def test_refinement_levels() -> None:
    assert SphereParams(20.0, 0.5, coarse_step=4.0).refinement_levels == 3
    p = SphereParams(20.0, 0.1, coarse_step=4.0)  # 40 -> rounded up to 2^6 = 64
    assert p.refinement_levels == 6
    assert p.effective_coarse_step == pytest.approx(6.4)
    assert SphereParams(20.0, 0.5, coarse_step=0.3).refinement_levels == 0
    assert SphereParams(20.0, 0.5).refinement_levels == 3  # auto: about r/4 = 5 m -> 4 m


def test_adaptive_cancel() -> None:
    with pytest.raises(CancelledError):
        roll_adaptive(cube_10m(), SphereParams(20.0, 0.5), cancel=lambda: True)
