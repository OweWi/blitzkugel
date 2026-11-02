"""STEP import and tessellation with OCP (OpenCascade bindings from CadQuery).

OpenCascade converts all STEP length units to millimetres on import. The
importer reads the unit declared in the file for information and converts the
tessellated result to metres.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.mesh import TriMesh
from ..core.progress import CancelCallback, ProgressCallback, ProgressScope, never_cancel, no_progress

log = logging.getLogger(__name__)

#: Length units in metres (keys: names returned by OpenCascade and short names for the UI).
UNIT_TO_METRE: dict[str, float] = {
    "millimetre": 1e-3, "millimeter": 1e-3, "mm": 1e-3,
    "centimetre": 1e-2, "centimeter": 1e-2, "cm": 1e-2,
    "decimetre": 1e-1, "dm": 1e-1,
    "metre": 1.0, "meter": 1.0, "m": 1.0,
    "kilometre": 1e3, "km": 1e3,
    "inch": 0.0254, "in": 0.0254,
    "foot": 0.3048, "ft": 0.3048,
}


class StepImportError(Exception):
    """Raised when a STEP file cannot be read or contains no usable geometry."""


@dataclass
class TessellationSettings:
    """Tessellation accuracy.

    Attributes:
        linear_deflection_mm: Max. chordal deviation between surface and triangles in mm.
        angular_deflection_deg: Max. angle between neighbouring triangle normals in degrees.
    """

    linear_deflection_mm: float = 20.0
    angular_deflection_deg: float = 15.0


@dataclass
class StepModel:
    """A loaded STEP file (B-Rep in OpenCascade millimetres)."""

    path: Path
    shape: object  # OCP.TopoDS.TopoDS_Shape
    file_units: list[str] = field(default_factory=list)
    bodies: list[tuple[str, object]] = field(default_factory=list)

    @property
    def file_unit(self) -> str:
        """First declared length unit of the file (``"millimetre"`` if none)."""
        return self.file_units[0] if self.file_units else "millimetre"

    def file_unit_to_metre(self) -> float:
        return UNIT_TO_METRE.get(self.file_unit.lower(), 1e-3)


def load_step(path: str | Path) -> StepModel:
    """Read a STEP file and collect its bodies.

    Raises:
        StepImportError: if the file is missing, defective or empty.
    """
    from OCP.collections import Sequence_TCollection_AsciiString  # noqa: PLC0415
    from OCP.IFSelect import IFSelect_RetDone  # noqa: PLC0415
    from OCP.STEPControl import STEPControl_Reader  # noqa: PLC0415

    path = Path(path)
    if not path.is_file():
        raise StepImportError(f"Datei nicht gefunden: {path}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise StepImportError(f"Die STEP-Datei konnte nicht gelesen werden (Status {status}).")

    lengths, angles, solid_angles = (Sequence_TCollection_AsciiString() for _ in range(3))
    try:
        reader.FileUnits(lengths, angles, solid_angles)
        units = [lengths.Value(i + 1).ToCString() for i in range(len(lengths))]
    except Exception:  # noqa: BLE001 – unit info is optional
        log.exception("Einheiten konnten nicht ermittelt werden")
        units = []

    n_roots = reader.TransferRoots()
    shape = reader.OneShape()
    if n_roots == 0 or shape is None or shape.IsNull():
        raise StepImportError("Die STEP-Datei enthält keine übertragbare Geometrie.")

    bodies = _collect_bodies(shape)
    if not bodies:
        raise StepImportError("Die STEP-Datei enthält keine Flächen (leere Geometrie).")
    log.info("STEP geladen: %s, Einheit(en) %s, %d Körper", path.name, units or "?", len(bodies))
    return StepModel(path, shape, units, bodies)


def _collect_bodies(shape: object) -> list[tuple[str, object]]:
    """Solids as separate bodies plus loose faces/shells as one extra body."""
    from OCP.BRep import BRep_Builder  # noqa: PLC0415
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID  # noqa: PLC0415
    from OCP.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCP.TopoDS import TopoDS_Compound  # noqa: PLC0415

    bodies: list[tuple[str, object]] = []
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        bodies.append((f"Körper {len(bodies) + 1}", exp.Current()))
        exp.Next()

    loose = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(loose)
    n_loose = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID)
    while exp.More():
        builder.Add(loose, exp.Current())
        n_loose += 1
        exp.Next()
    if n_loose:
        bodies.append((f"Einzelflächen ({n_loose})", loose))
    return bodies


def tessellate(
    model: StepModel,
    settings: TessellationSettings,
    unit_override: str | None = None,
    progress: ProgressCallback = no_progress,
    cancel: CancelCallback = never_cancel,
) -> TriMesh:
    """Tessellate all bodies of ``model`` into one triangle mesh in metres.

    Args:
        model: Loaded STEP model.
        settings: Tessellation accuracy.
        unit_override: Interpret the file coordinates in this unit (key of
            :data:`UNIT_TO_METRE`) instead of the declared one. Useful for files
            with a wrong unit declaration.
        progress: Progress callback.
        cancel: Cancellation callback.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: PLC0415
    from OCP.BRepTools import BRepTools  # noqa: PLC0415

    scope = ProgressScope(progress, 0.0, 1.0, cancel)
    scale = 1e-3  # OpenCascade millimetres -> metres
    if unit_override:
        # mm -> raw numbers as written in the file -> metres in the overridden unit
        scale = 1e-3 / model.file_unit_to_metre() * UNIT_TO_METRE[unit_override]

    meshes: list[TriMesh] = []
    n = len(model.bodies)
    for i, (name, body) in enumerate(model.bodies):
        scope(i / n, f"Tesselliere {name} ({i + 1}/{n})")
        BRepTools.Clean_s(body)
        BRepMesh_IncrementalMesh(
            body, settings.linear_deflection_mm, False, math.radians(settings.angular_deflection_deg), True
        )
        verts, faces = _triangulation_arrays(body)
        if len(faces):
            meshes.append(TriMesh(verts * scale, faces, body_names=[name]))
        else:
            log.warning("%s lieferte keine Dreiecke", name)
    scope(1.0, "Tessellierung abgeschlossen")
    mesh = TriMesh.merge(meshes)
    if mesh.is_empty:
        raise StepImportError("Die Tessellierung lieferte keine Dreiecke (leere Geometrie).")
    log.info("Tessellierung: %d Knoten, %d Dreiecke, Skalierung %g", mesh.n_vertices, mesh.n_faces, scale)
    return mesh


def _triangulation_arrays(shape: object) -> tuple[np.ndarray, np.ndarray]:
    """Collect the triangulation of all faces of ``shape`` (coordinates in mm)."""
    from OCP.BRep import BRep_Tool  # noqa: PLC0415
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED  # noqa: PLC0415
    from OCP.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCP.TopLoc import TopLoc_Location  # noqa: PLC0415
    from OCP.TopoDS import TopoDS  # noqa: PLC0415

    to_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face  # OCP < 8 / >= 8
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = to_face(exp.Current())
        exp.Next()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        nodes = np.array(
            [tri.Node(k).Transformed(trsf).Coord() for k in range(1, tri.NbNodes() + 1)], dtype=np.float64
        )
        tris = np.array([tri.Triangle(k).Get() for k in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
        if face.Orientation() == TopAbs_REVERSED:
            tris = tris[:, ::-1]
        verts.append(nodes)
        faces.append(tris + offset)
        offset += len(nodes)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(verts), np.vstack(faces)


def load_and_tessellate(
    path: str | Path,
    settings: TessellationSettings | None = None,
    unit_override: str | None = None,
    progress: ProgressCallback = no_progress,
    cancel: CancelCallback = never_cancel,
) -> tuple[StepModel, TriMesh]:
    """Convenience: :func:`load_step` followed by :func:`tessellate`."""
    scope = ProgressScope(progress, 0.0, 1.0, cancel)
    scope(0.0, "STEP-Datei lesen")
    model = load_step(path)
    mesh = tessellate(model, settings or TessellationSettings(), unit_override, scope.sub(0.3, 1.0), cancel)
    return model, mesh
