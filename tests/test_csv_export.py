"""Tests for the CSV export."""

from __future__ import annotations

import csv
from pathlib import Path

from blitzkugel.core.mesh import make_box
from blitzkugel.core.rolling_sphere import SphereParams, roll_directions
from blitzkugel.export.csv_export import HEADER, export_strike_points_csv


def test_csv_roundtrip(tmp_path: Path) -> None:
    res = roll_directions(make_box(-5, -5, 0, 5, 5, 10), SphereParams(20.0, 1.0), [0.0, 90.0])
    out = tmp_path / "punkte.csv"
    n = export_strike_points_csv(res, out)
    assert n == res.n_strike_points
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    assert rows[0] == HEADER
    assert len(rows) == n + 1
    assert {r[3] for r in rows[1:]} == {"0°", "90°"}
    assert all(float(r[2]) >= 10.0 - 1e-3 for r in rows[1:])
