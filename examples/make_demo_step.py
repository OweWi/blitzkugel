"""Create a demo STEP file (building with chimney, gable roof and air-termination rods, in mm).

Usage: python examples/make_demo_step.py [out.step]
"""

from __future__ import annotations

import sys

from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopoDS import TopoDS_Compound


def build() -> TopoDS_Compound:
    m = 1000.0  # model in millimetres
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    # Main building 30 x 12 x 9 m with a gable roof (ridge at 14 m)
    builder.Add(comp, BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 30 * m, 12 * m, 9 * m).Shape())
    poly = BRepBuilderAPI_MakePolygon(gp_Pnt(0, 0, 9 * m), gp_Pnt(0, 12 * m, 9 * m), gp_Pnt(0, 6 * m, 14 * m), True)
    roof = BRepPrimAPI_MakePrism(BRepBuilderAPI_MakeFace(poly.Wire()).Face(), gp_Vec(30 * m, 0, 0)).Shape()
    builder.Add(comp, roof)
    # Chimney (cylinder) and two air-termination rods on the ridge
    builder.Add(comp, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(22 * m, 8 * m, 9 * m), gp_Dir(0, 0, 1)),
                                               0.6 * m, 7 * m).Shape())
    for x in (2 * m, 28 * m):
        builder.Add(comp, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, 6 * m, 14 * m), gp_Dir(0, 0, 1)),
                                                   0.01 * m, 2 * m).Shape())
    # Neighbouring tower 8 x 8 x 32 m
    builder.Add(comp, BRepPrimAPI_MakeBox(gp_Pnt(40 * m, 0, 0), 8 * m, 8 * m, 32 * m).Shape())
    return comp


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_gebaeude.step"
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = STEPControl_Writer()
    writer.Transfer(build(), STEPControl_AsIs)
    writer.Write(out)
    print(f"geschrieben: {out}")


if __name__ == "__main__":
    main()
