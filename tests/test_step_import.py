"""Tests for the STEP import (skipped if OCP is not installed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("OCP")

from blitzkugel.geometry.step_import import (  # noqa: E402
    StepImportError,
    TessellationSettings,
    load_and_tessellate,
    load_step,
)


def _write_step(path: Path, unit: str) -> None:
    """Write two solids (box 10x20x30 mm³ and a cylinder) with the given STEP unit."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCP.TopoDS import TopoDS_Compound

    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    builder.Add(comp, BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape())
    builder.Add(comp, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(50, 0, 0), gp_Dir(0, 0, 1)), 5.0, 40.0).Shape())
    Interface_Static.SetCVal_s("write.step.unit", unit)
    writer = STEPControl_Writer()
    writer.Transfer(comp, STEPControl_AsIs)
    writer.Write(str(path))


@pytest.mark.parametrize("unit", ["MM", "M"])
def test_load_converts_to_metres(tmp_path: Path, unit: str) -> None:
    f = tmp_path / "test.step"
    _write_step(f, unit)
    model, mesh = load_and_tessellate(f, TessellationSettings(0.5, 10.0))
    assert model.file_unit in {"millimetre", "metre"}
    assert len(model.bodies) == 2
    lo, hi = mesh.bounds
    np.testing.assert_allclose(lo, [0.0, -0.005, 0.0], atol=1e-4)
    np.testing.assert_allclose(hi, [0.055, 0.020, 0.040], atol=1e-4)
    assert set(np.unique(mesh.body_ids)) == {0, 1}


def test_unit_override(tmp_path: Path) -> None:
    f = tmp_path / "test.step"
    _write_step(f, "MM")
    model = load_step(f)
    from blitzkugel.geometry.step_import import tessellate

    mesh = tessellate(model, TessellationSettings(0.5, 10.0), unit_override="m")
    assert mesh.bounds[1][2] == pytest.approx(40.0, abs=1e-3)


def test_finer_tessellation_gives_more_triangles(tmp_path: Path) -> None:
    f = tmp_path / "test.step"
    _write_step(f, "MM")
    model = load_step(f)
    from blitzkugel.geometry.step_import import tessellate

    coarse = tessellate(model, TessellationSettings(2.0, 30.0))
    fine = tessellate(model, TessellationSettings(0.05, 5.0))
    assert fine.n_faces > coarse.n_faces


def test_outward_orientation(tmp_path: Path) -> None:
    f = tmp_path / "test.step"
    _write_step(f, "MM")
    _, mesh = load_and_tessellate(f)
    box = mesh.body_ids == 0
    normals = mesh.face_normals()[box]
    centres = mesh.vertices[mesh.faces[box]].mean(axis=1)
    outward = np.einsum("ij,ij->i", normals, centres - np.array([0.005, 0.01, 0.015]))
    assert np.all(outward > 0)


def test_missing_and_defective_files(tmp_path: Path) -> None:
    with pytest.raises(StepImportError):
        load_step(tmp_path / "gibt_es_nicht.step")
    bad = tmp_path / "defekt.step"
    bad.write_text("das ist keine STEP-Datei")
    with pytest.raises(StepImportError):
        load_step(bad)
