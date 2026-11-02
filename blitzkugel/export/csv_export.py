"""CSV export of strike points."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from ..core.rolling_sphere import StrikeResult

log = logging.getLogger(__name__)

HEADER = ["x_m", "y_m", "z_m", "richtung", "dreieck_id"]


def export_strike_points_csv(
    result: StrikeResult, path: str | Path, delimiter: str = ";", decimal_comma: bool = False
) -> int:
    """Write all strike points as CSV (coordinates in metres).

    Columns: x, y, z, rolling direction (e.g. ``"45°"`` or ``"3D"``), triangle id.
    The default delimiter ``;`` lets German Excel open the file directly; with
    ``decimal_comma`` it also reads the numbers correctly (``12,3456``).

    Returns:
        Number of rows written.
    """
    path = Path(path)
    if decimal_comma and delimiter == ",":
        raise ValueError("Dezimalkomma und Trennzeichen ',' schließen sich aus.")

    def num(v: float) -> str:
        text = f"{v:.4f}"
        return text.replace(".", ",") if decimal_comma else text

    rows = 0
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(HEADER)
        for direction in result.directions:
            for (x, y, z), tri in zip(direction.strike_points, direction.triangle_ids):
                writer.writerow([num(x), num(y), num(z), direction.label, int(tri)])
                rows += 1
    log.info("%d Einschlagpunkte nach %s exportiert", rows, path)
    return rows
