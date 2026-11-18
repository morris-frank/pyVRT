"""End-to-end run on a synthetic field: both interpolators and the zoning complete."""

import json
import random

import pytest

from pyvrt.cli import make_arg_parser
from pyvrt.config import build_config
from pyvrt.pipeline import run


@pytest.mark.parametrize("method", ["kriging", "svm"])
def test_synthetic_field_produces_rasters(tmp_path, method):
    rng = random.Random(0)
    x0, y0, size = 5.0, 52.0, 0.004  # a ~300 m square field
    field = [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]
    polygons = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"field_id": "f1"},
                "geometry": {"type": "Polygon", "coordinates": [field]},
            }
        ],
    }
    points = {"type": "FeatureCollection", "features": []}
    for _ in range(30):
        x, y = x0 + rng.random() * size, y0 + rng.random() * size
        points["features"].append(
            {
                "type": "Feature",
                "properties": {"field_id": "f1", "boron": (x - x0) * 500 + rng.random()},
                "geometry": {"type": "Point", "coordinates": [x, y]},
            }
        )
    (tmp_path / "points.geojson").write_text(json.dumps(points))
    (tmp_path / "fields.geojson").write_text(json.dumps(polygons))

    args = make_arg_parser().parse_args(
        [
            "--points", str(tmp_path / "points.geojson"),
            "--polygons", str(tmp_path / "fields.geojson"),
            "--join-key", "field_id",
            "--target", "boron",
            "--method", method,
            "--output-dir", str(tmp_path / "out"),
            "--no-report",
            "--set", "grid.cells_per_axis=20",
        ]
    )  # fmt: skip
    summary = run(build_config(args))

    assert summary["keys_total"] == 1
    assert list((tmp_path / "out").rglob("*.tif"))
