"""CSV export of strike points."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from ..core.rolling_sphere import StrikeResult

log = logging.getLogger(__name__)

HEADER = ["x_m", "y_m", "z_m", "richtung", "dreieck_id"]


def export_strike_points_csv(result: StrikeResult, path: str | Path, delimiter: str = ";") -> int:
    """Write all strike points as CSV (coordinates in metres, decimal point).

    Columns: x, y, z, rolling direction (e.g. ``"45°"`` or ``"3D"``), triangle id.
    The default delimiter ``;`` lets German Excel open the file directly.

    Returns:
        Number of rows written.
    """
    path = Path(path)
    rows = 0
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(HEADER)
        for direction in result.directions:
            for (x, y, z), tri in zip(direction.strike_points, direction.triangle_ids):
                writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", direction.label, int(tri)])
                rows += 1
    log.info("%d Einschlagpunkte nach %s exportiert", rows, path)
    return rows
