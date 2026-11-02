"""GUI-independent computation: mesh, distance queries and rolling sphere method."""

from .mesh import TriMesh, make_box, make_rod
from .rolling_sphere import (
    Mode,
    ProtectionClass,
    SphereParams,
    StrikeResult,
    penetration_depth,
    roll_3d,
    roll_adaptive,
    roll_directions,
)

__all__ = [
    "Mode",
    "ProtectionClass",
    "SphereParams",
    "StrikeResult",
    "TriMesh",
    "make_box",
    "make_rod",
    "penetration_depth",
    "roll_3d",
    "roll_adaptive",
    "roll_directions",
]
