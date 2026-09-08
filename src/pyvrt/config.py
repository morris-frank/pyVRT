from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "input": {
        "points_geojson": None,
        "polygons_geojson": None,
        "join_key": None,
        "target": None,
    },
    "method": "both",
    "output_dir": "output",
    "output_shapefiles": False,
    "output_epsg": 4326,
    "covariates": [],
    "grid": {
        "pixel_size_x": None,
        "pixel_size_y": None,
        "cells_per_axis": 100,
        "auto_refine_if_empty": True,
        "max_auto_refine_steps": 6,
        "max_grid_cells": 2000000,
    },
    "kriging": {
        "models": ["linear", "linear-sill", "exponential", "spherical", "gaussian"],
        "model": None,
        "n_neighbors": 16,
        "search_radius": None,
        "max_distance_factor": 0.6,
        "active_distance": None,
        "lag_distance": None,
    },
    "svm": {
        "n_neighbors": 16,
        "search_radius": None,
        "idw_power": 1.0,
        "C_values": [0.001, 0.01, 0.1, 1, 10],
        "gamma_values": [0.001, 0.01, 0.1, 1, 10],
        "kfold_min": 2,
        "kfold_max": 5,
        "default_C": 1.0,
        "default_gamma": 0.1,
    },
    "zones": {
        "num_zones": 3,
        "max_iterations": 100,
        "fuzzy_coefficient": 1.25,
        "error_tolerance": 1.0e-5,
        "normalize": True,
    },
    "report": {
        "enabled": True,
        "pdf_filename": "project_report.pdf",
        "dpi": 180,
        "logo_path": None,
        "theme": {
            "bg": "#f5f3f0",
            "fg": "#29332e",
            "font_family": "Inter",
        },
        "satellite": {
            "mode": "xyz",
            "xyz_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "zoom": 17,
            "alpha": 0.95,
            "timeout_seconds": 15,
        },
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_literal(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    ref = config
    for part in parts[:-1]:
        if part not in ref or not isinstance(ref[part], dict):
            ref[part] = {}
        ref = ref[part]
    ref[parts[-1]] = value


def parse_covariates(covariates_arg: Optional[str]) -> List[Dict[str, Any]]:
    if not covariates_arg:
        return []
    items: List[Dict[str, Any]] = []
    for token in covariates_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            name, weight = token.split(":", 1)
            items.append({"name": name.strip(), "weight": float(weight)})
        else:
            items.append({"name": token, "weight": 1.0})
    return items


def validate_required(cfg: Dict[str, Any]) -> None:
    missing = []
    for field in ("points_geojson", "polygons_geojson", "join_key", "target"):
        if not cfg["input"].get(field):
            missing.append(f"input.{field}")
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config_base_dir = Path.cwd()

    if args.config:
        cfg_path = Path(args.config).resolve()
        config_base_dir = cfg_path.parent
        with cfg_path.open("r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        config = deep_merge(config, user_cfg)

    if args.points:
        config["input"]["points_geojson"] = args.points
    if args.polygons:
        config["input"]["polygons_geojson"] = args.polygons
    if args.join_key:
        config["input"]["join_key"] = args.join_key
    if args.target:
        config["input"]["target"] = args.target
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.output_shapefiles:
        config["output_shapefiles"] = True
    if args.method:
        config["method"] = args.method
    if args.num_zones is not None:
        config["zones"]["num_zones"] = int(args.num_zones)
    if args.covariates:
        config["covariates"] = parse_covariates(args.covariates)
    if args.report_pdf:
        config["report"]["pdf_filename"] = args.report_pdf
    if args.no_report:
        config["report"]["enabled"] = False
    if args.satellite_mode:
        config["report"]["satellite"]["mode"] = args.satellite_mode

    for kv in args.set or []:
        if "=" not in kv:
            raise ValueError(f"Invalid --set format '{kv}'. Expected key=value.")
        key, raw = kv.split("=", 1)
        set_nested(config, key.strip(), parse_literal(raw.strip()))

    logo_path = config.get("report", {}).get("logo_path")
    if logo_path:
        path = Path(str(logo_path))
        if not path.is_absolute():
            config["report"]["logo_path"] = str((config_base_dir / path).resolve())

    sat_mode = str(config.get("report", {}).get("satellite", {}).get("mode", "xyz")).lower()
    if sat_mode == "local":
        config["report"]["satellite"]["mode"] = "xyz"

    validate_required(config)
    return config
