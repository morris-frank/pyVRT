from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class GeoJSONData:
    raw: dict[str, Any]
    features: list[dict[str, Any]]
    epsg: int


def load_geojson(path: Path) -> GeoJSONData:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if raw.get("type") == "FeatureCollection":
        features = raw.get("features", [])
    elif raw.get("type") == "Feature":
        features = [raw]
    else:
        raise ValueError(f"Unsupported GeoJSON type in {path}: {raw.get('type')}")

    epsg = 4326
    crs = raw.get("crs", {})
    if isinstance(crs, dict):
        name = (crs.get("properties") or {}).get("name", "")
        match = re.search(r"EPSG:(\d+)", str(name), flags=re.IGNORECASE)
        if match:
            epsg = int(match.group(1))
    return GeoJSONData(raw=raw, features=features, epsg=epsg)


def load_points_dataframe(features: list[dict[str, Any]], join_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feat.get("properties") or {}
        if join_key not in props:
            continue
        row = dict(props)
        row["CoordX_SM"] = float(coords[0])
        row["CoordY_SM"] = float(coords[1])
        rows.append(row)

    if not rows:
        raise ValueError("No valid point features found with the join key.")
    return pd.DataFrame(rows)


def load_polygon_groups(
    features: list[dict[str, Any]], join_key: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        props = feat.get("properties") or {}
        if join_key not in props:
            continue
        key = str(props[join_key])
        grouped.setdefault(key, []).append(feat)
    return grouped
