"""pyvista scene construction, standard views and image export.

The scene is always built by :func:`build_scene` so the interactive Qt view
and the off-screen image export show exactly the same content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv

from ..core.mesh import FloatArray, TriMesh
from ..core.rolling_sphere import Mode, ProtectionClass, StrikeResult, contact_density

log = logging.getLogger(__name__)

#: Well distinguishable colours for up to 8 rolling directions (strike points use red otherwise).
DIRECTION_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a6a6", "#f032e6", "#9a6324",
]
STRIKE_COLOR = "#e00000"

#: name -> (label, view direction from focal point to camera, view-up vector)
STANDARD_VIEWS: dict[str, tuple[str, tuple[float, float, float], tuple[float, float, float]]] = {
    "iso": ("Isometrisch", (1.0, -1.0, 0.8), (0.0, 0.0, 1.0)),
    "oben": ("Oben", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "vorne": ("Vorne", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "hinten": ("Hinten", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "links": ("Links", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "rechts": ("Rechts", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}


@dataclass
class DisplayOptions:
    """What to show and how."""

    model_opacity: float = 0.35
    model_color: str = "#b4b4b4"
    show_edges: bool = False
    point_size: float = 9.0
    color_by_direction: bool = True
    show_legend: bool = True
    show_info: bool = True
    show_heatmap: bool = False
    show_spheres: bool = False
    n_spheres: int = 12
    show_ground: bool = True
    background: str = "white"


@dataclass
class SceneData:
    """Everything that is displayed."""

    mesh: TriMesh | None = None
    ground_z: float = 0.0
    result: StrikeResult | None = None
    title: str = ""
    extra_info: list[str] = field(default_factory=list)
    heatmap_cache: tuple[tuple[int, int], pv.PolyData] | None = field(default=None, repr=False)


def to_polydata(mesh: TriMesh) -> pv.PolyData:
    """Convert a :class:`TriMesh` into pyvista PolyData."""
    faces = np.hstack([np.full((mesh.n_faces, 1), 3, dtype=np.int64), mesh.faces]).ravel()
    return pv.PolyData(mesh.vertices, faces)


def info_lines(data: SceneData) -> list[str]:
    """Text block with sphere diameter, protection class and mode."""
    lines: list[str] = []
    if data.title:
        lines.append(data.title)
    res = data.result
    if res is not None:
        p = res.params
        pc = ProtectionClass.from_diameter(p.diameter)
        pc_text = f"Schutzklasse {pc.name}" if pc else "benutzerdefiniert"
        lines.append(f"Blitzkugel D = {p.diameter:g} m (r = {p.radius:g} m), {pc_text}")
        if res.mode is Mode.HULL_3D:
            lines.append("Modus B: 3D-Hülle")
        elif res.mode is Mode.ADAPTIVE:
            # VTK's default font has no arrow glyph.
            lines.append(f"Modus C: 3D adaptiv, grob {p.effective_coarse_step:g} m / fein {p.grid_step:g} m")
        else:
            lines.append("Modus A: Richtungen " + ", ".join(d.label for d in res.directions))
        lines.append(
            f"Raster {p.grid_step:g} m, Toleranz {p.tolerance(res.mode):.3g} m, Boden z = {p.ground_z:g} m, "
            f"{res.n_strike_points} Einschlagpunkte"
        )
    lines.extend(data.extra_info)
    return lines


def _sample_spheres(centers: FloatArray, radius: float, n: int) -> FloatArray:
    """Pick up to ``n`` well-spread sphere centres (one per coarse voxel, highest first)."""
    if len(centers) == 0 or n <= 0:
        return np.zeros((0, 3))
    keys = np.floor(centers / radius).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    picked = centers[idx]
    picked = picked[np.argsort(-picked[:, 2])]
    if len(picked) > n:
        picked = picked[np.linspace(0, len(picked) - 1, n).round().astype(int)]
    return picked


def build_scene(plotter: pv.Plotter, data: SceneData, options: DisplayOptions, point_scale: float = 1.0) -> None:
    """(Re)build the complete scene in ``plotter``. The camera is kept."""
    plotter.clear()
    plotter.set_background(options.background)
    if data.mesh is None or data.mesh.is_empty:
        plotter.add_text("Keine Geometrie geladen – Datei › STEP öffnen …", font_size=12, color="gray")
        return

    mesh = data.mesh
    poly = to_polydata(mesh)
    res = data.result

    heat = heatmap_polydata(data) if options.show_heatmap and res is not None else None
    if heat is not None:
        vmax = max(2, int(np.nanmax(heat.point_data["Kontakte"])))
        plotter.add_mesh(
            heat, scalars="Kontakte", cmap="inferno_r", clim=[1, vmax], log_scale=True,
            nan_color=options.model_color, opacity=max(options.model_opacity, 0.9),
            show_edges=options.show_edges,
            scalar_bar_args={"title": "Kugellagen je Bereich", "color": "black", "fmt": "%.0f"},
        )
    else:
        plotter.add_mesh(
            poly, color=options.model_color, opacity=options.model_opacity,
            show_edges=options.show_edges, smooth_shading=False,
        )

    lo, hi = mesh.bounds
    if options.show_ground:
        size = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 1.3 + (res.params.radius if res else 0.0)
        centre = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, data.ground_z)
        plotter.add_mesh(pv.Plane(center=centre, direction=(0, 0, 1), i_size=size, j_size=size),
                         color="#9fc59f", opacity=0.35, label=None)

    legend_labels: list[str] = []
    if res is not None:
        size_px = options.point_size * point_scale
        if options.color_by_direction and len(res.directions) > 1:
            for i, d in enumerate(res.directions):
                if len(d.strike_points):
                    label = f"Richtung {d.label} ({len(d.strike_points)})"
                    plotter.add_points(d.strike_points, color=DIRECTION_COLORS[i % len(DIRECTION_COLORS)],
                                       point_size=size_px, render_points_as_spheres=True, label=label)
                    legend_labels.append(label)
        else:
            pts = res.strike_points
            if len(pts):
                label = f"Einschlagpunkte ({len(pts)})"
                plotter.add_points(pts, color=STRIKE_COLOR, point_size=size_px, render_points_as_spheres=True,
                                   label=label)
                legend_labels.append(label)

        if options.show_spheres:
            centers = np.vstack([d.contact_centers for d in res.directions if len(d.contact_centers)] or
                                [np.zeros((0, 3))])
            for c in _sample_spheres(centers, res.params.radius, options.n_spheres):
                sphere = pv.Sphere(radius=res.params.radius, center=c, theta_resolution=24, phi_resolution=16)
                plotter.add_mesh(sphere, style="wireframe", color="#2060c0", opacity=0.25, line_width=0.6)

    if options.show_legend and legend_labels:
        _add_legend(plotter, legend_labels)
    if options.show_info:
        lines = info_lines(data)
        if lines:
            plotter.add_text("\n".join(lines), position="upper_left", font_size=9, color="black")
    plotter.add_axes(color="black")


def _add_legend(plotter: pv.Plotter, labels: list[str]) -> None:
    """Legend in the upper right corner, sized to its entries.

    Positioned explicitly because pyvista's ``loc`` placement lets long or
    single entries overflow the window.
    """
    w = 0.0085 * max(len(label) for label in labels) + 0.03
    h = 0.04 * len(labels)
    legend = plotter.add_legend(bcolor="white", face="circle", loc=None, size=(w, h))
    legend.SetPosition(0.99 - w, 0.99 - h)
    legend.SetPosition2(w, h)


def focus_bounds(data: SceneData) -> list[float] | None:
    """Camera bounds: the model (without the ground plane), slightly enlarged."""
    if data.mesh is None or data.mesh.is_empty:
        return None
    lo, hi = data.mesh.bounds
    pad = 0.05 * float(np.max(hi - lo))
    return [lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, min(lo[2], data.ground_z), hi[2] + pad]


def heatmap_polydata(data: SceneData, max_triangles: int = 400_000) -> pv.PolyData | None:
    """Refined model surface with the contact density as point scalars ``"Kontakte"``.

    CAD tessellations often consist of few large triangles, so the surface is
    subdivided to roughly the grid step before sampling the density. The result
    is cached per evaluation result.
    """
    res = data.result
    if data.mesh is None or res is None or len(res.contact_points) == 0:
        return None
    key = (id(data.mesh), id(res))
    if data.heatmap_cache is not None and data.heatmap_cache[0] == key:
        return data.heatmap_cache[1]
    poly = to_polydata(data.mesh)
    area = float(poly.area)
    edge = max(2.0 * res.params.grid_step, float(np.sqrt(4.0 * area / max_triangles)))
    try:
        fine = poly.subdivide_adaptive(max_edge_len=edge, max_n_tris=max_triangles)
    except Exception:  # noqa: BLE001 – fall back to the original tessellation
        log.exception("Netzverfeinerung für Heatmap fehlgeschlagen")
        fine = poly
    density = contact_density(np.asarray(fine.points), res.contact_points, 1.5 * res.params.grid_step,
                              res.contact_weights)
    # NaN instead of 0 so that untouched areas get the neutral model colour (log scale has no 0).
    fine.point_data["Kontakte"] = np.where(density > 0, density, np.nan).astype(np.float64)
    data.heatmap_cache = (key, fine)
    return fine


def apply_view(plotter: pv.Plotter, name: str, bounds: list[float] | None = None) -> None:
    """Set one of :data:`STANDARD_VIEWS`, framing ``bounds`` (default: all actors).

    The orthogonal views use parallel projection, the isometric view perspective.
    """
    _, vec, up = STANDARD_VIEWS[name]
    if name == "iso":
        plotter.disable_parallel_projection()
    else:
        plotter.enable_parallel_projection()
    plotter.view_vector(vec, viewup=up)
    plotter.reset_camera(bounds=bounds)
    plotter.render()


def export_views(
    data: SceneData,
    options: DisplayOptions,
    out_dir: str | Path,
    basename: str,
    views: list[str],
    size: tuple[int, int] = (1920, 1080),
) -> list[Path]:
    """Render ``views`` off-screen and save them as PNG files.

    Returns:
        The written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    try:
        build_scene(plotter, data, options, point_scale=max(1.0, size[0] / 1600))
        written: list[Path] = []
        for name in views:
            apply_view(plotter, name, focus_bounds(data))
            path = out_dir / f"{basename}_{name}.png"
            plotter.screenshot(str(path))
            written.append(path)
            log.info("Bild gespeichert: %s", path)
        return written
    finally:
        plotter.close()


def export_camera_view(
    data: SceneData,
    options: DisplayOptions,
    camera: pv.Camera,
    path: str | Path,
    size: tuple[int, int] = (1920, 1080),
) -> Path:
    """Render the scene off-screen with a copy of ``camera`` (e.g. the interactive view's camera)."""
    path = Path(path)
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    try:
        build_scene(plotter, data, options, point_scale=max(1.0, size[0] / 1600))
        plotter.camera = camera.copy()
        plotter.screenshot(str(path))
        log.info("Bild gespeichert: %s", path)
        return path
    finally:
        plotter.close()
