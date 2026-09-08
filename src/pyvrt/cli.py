from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import build_config


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GeoJSON-based field interpolation runner.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--points", help="Point GeoJSON path.")
    parser.add_argument("--polygons", help="Polygon GeoJSON path.")
    parser.add_argument("--join-key", help="Property name used to join points to polygons.")
    parser.add_argument("--target", help="Target attribute to interpolate.")
    parser.add_argument(
        "--covariates",
        help="Comma-separated covariates as name or name:weight (for example elev:1.0,slope:0.7).",
    )
    parser.add_argument("--num-zones", type=int, help="Desired number of management zones.")
    parser.add_argument("--output-dir", help="Output directory.")
    parser.add_argument(
        "--output-shapefiles",
        action="store_true",
        help="Also export sampled point layers as ESRI Shapefiles.",
    )
    parser.add_argument("--report-pdf", help="PDF filename for the combined report inside output-dir.")
    parser.add_argument("--no-report", action="store_true", help="Disable PDF report generation.")
    parser.add_argument(
        "--satellite-mode",
        choices=["xyz", "none", "auto"],
        help="Satellite background mode for report pages.",
    )
    parser.add_argument(
        "--method",
        choices=["kriging", "svm", "both"],
        help="Interpolation method to run.",
    )
    parser.add_argument(
        "--set",
        action="append",
        help="Override any config value via dotted key=value. Example: --set kriging.n_neighbors=12",
    )
    return parser


def main() -> int:
    parser = make_arg_parser()
    args = parser.parse_args()
    try:
        config = build_config(args)
        from .pipeline import run

        summary = run(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(Path(config["output_dir"]).resolve()),
                "keys": summary["keys_total"],
                "report_pdf": summary.get("report_pdf"),
            }
        )
    )
    return 0
