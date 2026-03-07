#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from osgeo import gdal, osr
from PIL import Image
from scipy import spatial
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, scale

from krig import kriging, semivariogram
from skfuzzy.cluster import _cmeans

if "MPLCONFIGDIR" not in os.environ:
    mpl_default = Path.home() / ".matplotlib"
    if not mpl_default.exists() or not os.access(mpl_default, os.W_OK):
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="matplotlib-")
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = Path.home() / ".cache"
    if not xdg_cache.exists() or not os.access(xdg_cache, os.W_OK):
        os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="xdg-cache-")


gdal.UseExceptions()
_XYZ_FETCH_DISABLED = False


DEFAULT_CONFIG: Dict[str, Any] = {
    "input": {
        "points_geojson": None,
        "polygons_geojson": None,
        "join_key": None,
        "target": None,
    },
    "method": "both",  # "kriging", "svm", "both"
    "output_dir": "output",
    "output_epsg": 4326,
    "covariates": [],  # [{"name": "...", "weight": 1.0}, ...]
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
        "model": None,  # optional fixed model
        "n_neighbors": 16,
        "search_radius": None,  # default: max distance between sample points
        "max_distance_factor": 0.6,
        "active_distance": None,
        "lag_distance": None,
    },
    "svm": {
        "n_neighbors": 16,
        "search_radius": None,  # default: max distance between sample points
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
            "mode": "xyz",  # "xyz", "none"
            "xyz_url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "zoom": 17,
            "alpha": 0.95,
            "timeout_seconds": 15,
        },
    },
}


@dataclass
class GeoJSONData:
    raw: Dict[str, Any]
    features: List[Dict[str, Any]]
    epsg: int


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


def safe_name(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "key"


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


def get_polygon_rings(geometry: Dict[str, Any]) -> List[List[np.ndarray]]:
    """
    Returns list of polygons; each polygon is [outer_ring, hole_ring1, ...].
    """
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    polygons: List[List[np.ndarray]] = []
    if gtype == "Polygon":
        poly: List[np.ndarray] = []
        for ring in coords:
            arr = np.asarray(ring, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                poly.append(arr[:, :2])
        if poly:
            polygons.append(poly)
    elif gtype == "MultiPolygon":
        for polygon in coords:
            poly = []
            for ring in polygon:
                arr = np.asarray(ring, dtype=float)
                if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                    poly.append(arr[:, :2])
            if poly:
                polygons.append(poly)
    return polygons


def polygons_bbox(geometries: Iterable[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for geom in geometries:
        for poly in get_polygon_rings(geom):
            for ring in poly:
                xs.extend(ring[:, 0].tolist())
                ys.extend(ring[:, 1].tolist())
    if not xs or not ys:
        raise ValueError("No valid polygon coordinates found for bbox calculation.")
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def points_inside_polygons(points_xy: np.ndarray, geometries: Iterable[Dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(points_xy.shape[0], dtype=bool)
    for geom in geometries:
        for poly in get_polygon_rings(geom):
            outer = MplPath(poly[0])
            in_poly = outer.contains_points(points_xy, radius=1e-12)
            if len(poly) > 1:
                for hole in poly[1:]:
                    in_poly &= ~MplPath(hole).contains_points(points_xy, radius=1e-12)
            mask |= in_poly
    return mask


def build_grid(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    pixel_x: float,
    pixel_y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(xmin, xmax, pixel_x, dtype=float)
    ys = np.arange(ymin, ymax, pixel_y, dtype=float)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Grid has zero cells. Adjust pixel sizes or input bounds.")
    xv, yv = np.meshgrid(xs + (pixel_x / 2.0), ys + (pixel_y / 2.0))
    grid_points = np.column_stack([xv.ravel(), yv.ravel()])
    return xs, ys, grid_points


def safe_grid_build(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    pixel_x: float,
    pixel_y: float,
    max_cells: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, grid = build_grid(xmin, ymin, xmax, ymax, pixel_x, pixel_y)
    n_cells = int(grid.shape[0])
    if n_cells > int(max_cells):
        raise ValueError(
            f"Grid too large ({n_cells} cells) for pixel size " f"({pixel_x}, {pixel_y}). Increase pixel size or lower refinement."
        )
    return xs, ys, grid


def write_ogr_vrt(
    path: Path,
    csv_path: Path,
    layer_name: str,
    x_field: str,
    y_field: str,
    z_field: str,
) -> None:
    content = (
        "<OGRVRTDataSource>\n"
        f'  <OGRVRTLayer name="{layer_name}">\n'
        f"    <SrcDataSource>{csv_path.name}</SrcDataSource>\n"
        "    <GeometryType>wkbPoint</GeometryType>\n"
        f'    <GeometryField encoding="PointFromColumns" x="{x_field}" y="{y_field}" z="{z_field}"/>\n'
        "  </OGRVRTLayer>\n"
        "</OGRVRTDataSource>\n"
    )
    path.write_text(content, encoding="utf-8")


def write_geotiff(
    output_tif: Path,
    grid_array: np.ndarray,
    xmin: float,
    ymax: float,
    pixel_x: float,
    pixel_y: float,
    epsg: int,
    nodata: float,
    dtype: int = gdal.GDT_Float32,
) -> None:
    ny, nx = grid_array.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(output_tif), nx, ny, 1, dtype)
    if ds is None:
        raise RuntimeError(f"Failed to create GeoTIFF: {output_tif}")
    geotransform = [xmin, pixel_x, 0.0, ymax, 0.0, -pixel_y]
    ds.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(np.flipud(grid_array))
    band.FlushCache()
    ds.FlushCache()
    ds = None


def write_raster_vrt(output_vrt: Path, source_tif: Path) -> None:
    gdal.Translate(str(output_vrt), str(source_tif), format="VRT")


def idw(distances: np.ndarray, values: np.ndarray, power: float) -> float:
    dist_pow = np.power(distances, power)
    numerator = np.sum(values / dist_pow)
    denominator = np.sum(1.0 / dist_pow)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def plugin_style_local_value(
    tree: spatial.cKDTree,
    source_xy: np.ndarray,
    source_values: np.ndarray,
    point: np.ndarray,
    n_neighbors: int,
    radius: float,
    idw_power: float,
) -> float:
    neigs = tree.query_ball_point(point, radius)
    if len(neigs) > n_neighbors:
        k = n_neighbors + 1
    elif len(neigs) < 2:
        k = min(3, len(source_xy))
    elif len(neigs) < len(source_xy):
        k = len(neigs) + 1
    else:
        k = len(neigs)
    k = max(1, min(k, len(source_xy)))

    distances, points_idx = tree.query(point, k=k)
    distances = np.atleast_1d(np.asarray(distances, dtype=float))
    points_idx = np.atleast_1d(np.asarray(points_idx, dtype=int))
    local_values = source_values[points_idx]

    if n_neighbors == 1:
        return float(local_values[0])

    if distances[0] == 0.0 and len(distances) > 1:
        distances = distances[1:]
        local_values = local_values[1:]
    elif len(distances) > 1:
        distances = distances[:-1]
        local_values = local_values[:-1]

    if len(distances) == 0:
        return float(local_values[0])
    return idw(distances, local_values, idw_power)


def build_covariate_surface(
    sample_xy: np.ndarray,
    sample_values: np.ndarray,
    query_xy: np.ndarray,
    n_neighbors: int,
    search_radius: float,
    idw_power: float,
) -> np.ndarray:
    tree = spatial.cKDTree(sample_xy)
    out = np.zeros(query_xy.shape[0], dtype=float)
    for idx, point in enumerate(query_xy):
        out[idx] = plugin_style_local_value(
            tree=tree,
            source_xy=sample_xy,
            source_values=sample_values,
            point=point,
            n_neighbors=n_neighbors,
            radius=search_radius,
            idw_power=idw_power,
        )
    return out


def choose_kfolds(n_samples: int, kmin: int, kmax: int) -> int:
    k = int(round(n_samples / 20.0))
    k = max(kmin, k)
    k = min(kmax, k)
    return min(k, n_samples)


def fit_svm_params(
    features: np.ndarray,
    labels: np.ndarray,
    c_values: List[float],
    gamma_values: List[float],
    kfolds: int,
    default_c: float,
    default_gamma: float,
) -> Tuple[float, float]:
    if features.shape[0] < 3 or kfolds < 2:
        return default_c, default_gamma
    param_grid = {"C": c_values, "gamma": gamma_values}
    norm = StandardScaler().fit(features)
    features_norm = norm.transform(features)
    search = GridSearchCV(svm.SVR(kernel="rbf"), param_grid=param_grid, cv=kfolds)
    search.fit(features_norm, labels)
    params = search.best_params_
    return float(params["C"]), float(params["gamma"])


def compute_fpi_nce(u_transposed: np.ndarray, k_classes: int) -> Tuple[float, float]:
    n = u_transposed.shape[0]
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        for j in range(k_classes):
            uij = float(u_transposed[i, j])
            sum1 += uij * uij
            sum2 += uij * math.log(max(uij, np.finfo(float).eps))
    fpi = 1.0 - (k_classes * (((1.0 / n) * sum1) - 1.0) / (k_classes - 1.0))
    nce = ((-1.0 / n) * sum2) / math.log(k_classes)
    return fpi, nce


def fit_variogram_params(
    xy_df: pd.DataFrame,
    z_series: pd.Series,
    cfg: Dict[str, Any],
) -> Tuple[str, List[float], Dict[str, Any]]:
    sem = semivariogram.Semivariogram(xy_df, z_series)
    max_dist = float(sem.max_dist)
    min_dist = float(sem.min_dist)
    active_distance = cfg.get("active_distance")
    if active_distance is None:
        active_distance = float(cfg.get("max_distance_factor", 0.6)) * max_dist
    lag_distance = cfg.get("lag_distance")
    if lag_distance is None:
        idx = min(max(len(z_series), 0), len(sem.var["lag"]) - 1)
        lag_distance = float(sem.var["lag"].iloc[idx])
    lag_distance = max(float(lag_distance), min_dist)

    sem.Exp_Semiv(float(lag_distance), float(active_distance))
    use_models = cfg.get("models") or ["linear", "linear-sill", "exponential", "spherical", "gaussian"]
    while True:
        try:
            fitted = sem.Fit(use_models)
            break
        except ValueError as exc:
            if "is infeasible" in str(exc):
                lag_distance += 1.0
                sem.Exp_Semiv(float(lag_distance), float(active_distance))
            else:
                raise

    fixed_model = cfg.get("model")
    if fixed_model:
        model = fixed_model
    else:
        model = min(fitted.keys(), key=lambda m: fitted[m][3])
    params = [float(fitted[model][0]), float(fitted[model][1]), float(fitted[model][2])]
    stats = {
        "model": model,
        "nugget": params[0],
        "range": params[1],
        "sill": params[2],
        "active_distance": float(active_distance),
        "lag_distance": float(lag_distance),
        "max_dist": max_dist,
        "min_dist": min_dist,
        "rss": float(fitted[model][3]),
        "r2": float(fitted[model][4]),
    }
    return model, params, stats


def ring_area(ring: np.ndarray) -> float:
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def polygon_area_coord2(geometry: Dict[str, Any]) -> float:
    area = 0.0
    for poly in get_polygon_rings(geometry):
        if not poly:
            continue
        outer = abs(ring_area(poly[0]))
        holes = sum(abs(ring_area(hole)) for hole in poly[1:])
        area += max(0.0, outer - holes)
    return float(area)


def field_area_coord2(geometries: Iterable[Dict[str, Any]]) -> float:
    return float(sum(polygon_area_coord2(g) for g in geometries))


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "range": np.nan, "cv_pct": np.nan}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    rng = float(np.max(arr) - np.min(arr))
    cv_pct = float((std / mean) * 100.0) if mean != 0 else np.nan
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": mean,
        "std": std,
        "range": rng,
        "cv_pct": cv_pct,
    }


def _osr_transform_xy(x: float, y: float, src_epsg: int, dst_epsg: int) -> Tuple[float, float]:
    if src_epsg == dst_epsg:
        return float(x), float(y)
    src = osr.SpatialReference()
    src.ImportFromEPSG(int(src_epsg))
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(int(dst_epsg))
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, dst)
    out = tr.TransformPoint(float(x), float(y))
    return float(out[0]), float(out[1])


def _transform_bbox(bbox: Tuple[float, float, float, float], src_epsg: int, dst_epsg: int) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    corners = [
        _osr_transform_xy(xmin, ymin, src_epsg, dst_epsg),
        _osr_transform_xy(xmin, ymax, src_epsg, dst_epsg),
        _osr_transform_xy(xmax, ymin, src_epsg, dst_epsg),
        _osr_transform_xy(xmax, ymax, src_epsg, dst_epsg),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _read_geotiff_for_plot(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float, float], Optional[float]]:
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Could not read raster: {path}")
    gt = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    xmin = gt[0]
    xmax = gt[0] + xsize * gt[1]
    ymax = gt[3]
    ymin = gt[3] + ysize * gt[5]
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    return arr, (xmin, xmax, ymin, ymax), nodata


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n)
    return xtile, ytile


def _tile_bounds_xyz(x: int, y: int, z: int) -> Tuple[float, float, float, float]:
    n = 2**z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_left, lat_bottom, lon_right, lat_top


def _fetch_xyz_background(
    bbox: Tuple[float, float, float, float],
    epsg: int,
    xyz_url: str,
    zoom: int,
    timeout_seconds: int,
) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float], str]]:
    global _XYZ_FETCH_DISABLED
    if _XYZ_FETCH_DISABLED:
        return None
    bbox_4326 = _transform_bbox(bbox, epsg, 4326)
    xmin, ymin, xmax, ymax = bbox_4326
    if xmin >= xmax or ymin >= ymax:
        return None
    x0, y1 = _lonlat_to_tile(xmin, ymin, zoom)
    x1, y0 = _lonlat_to_tile(xmax, ymax, zoom)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    nx = x1 - x0 + 1
    ny = y1 - y0 + 1
    if nx * ny > 100:
        return None

    tile_size = 256
    canvas = Image.new("RGB", (nx * tile_size, ny * tile_size))
    for ix, xt in enumerate(range(x0, x1 + 1)):
        for iy, yt in enumerate(range(y0, y1 + 1)):
            url = xyz_url.format(z=zoom, x=xt, y=yt)
            success = False
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
                        data = resp.read()
                    tile = Image.open(BytesIO(data)).convert("RGB")
                    success = True
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                    print(f"Error fetching XYZ tile (attempt {attempt + 1}/3): {url}")
                    print(f"Error: {e}")
                    if attempt == 2:  # On last retry
                        _XYZ_FETCH_DISABLED = True
                        return None
            if not success:
                _XYZ_FETCH_DISABLED = True
                return None
            canvas.paste(tile, (ix * tile_size, iy * tile_size))

    arr = np.asarray(canvas).astype(float) / 255.0
    left, _, _, top = _tile_bounds_xyz(x0, y0, zoom)
    _, bottom, right, _ = _tile_bounds_xyz(x1, y1, zoom)
    extent4326 = (left, right, bottom, top)
    extent = _transform_bbox(extent4326, 4326, epsg)
    return arr, extent, f"XYZ z{zoom}"


def get_satellite_background(
    bbox: Tuple[float, float, float, float],
    epsg: int,
    sat_cfg: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float], str]]:
    mode = str(sat_cfg.get("mode", "auto")).lower()
    if mode in {"none"}:
        return None

    if mode in {"xyz", "auto"}:
        return _fetch_xyz_background(
            bbox=bbox,
            epsg=epsg,
            xyz_url=str(sat_cfg.get("xyz_url")),
            zoom=int(sat_cfg.get("zoom", 17)),
            timeout_seconds=int(sat_cfg.get("timeout_seconds", 15)),
        )
    return None


def draw_polygon_edges(ax: Any, geometries: Iterable[Dict[str, Any]], color: str = "black") -> None:
    for geom in geometries:
        for poly in get_polygon_rings(geom):
            for ring in poly:
                ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=1.2)


def resolve_font_family(preferred: str) -> str:
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        if preferred in available:
            return preferred
    except Exception:
        pass
    return "DejaVu Sans"


def build_field_page_report(
    pdf: PdfPages,
    key: str,
    key_out: Path,
    key_summary: Dict[str, Any],
    polygons_for_key: List[Dict[str, Any]],
    target: str,
    epsg: int,
    report_cfg: Dict[str, Any],
    logo_img: Optional[np.ndarray],
) -> None:
    theme = report_cfg.get("theme", {})
    bg_color = theme.get("bg", "#f5f3f0")
    fg_color = theme.get("fg", "#29332e")
    font_family = theme.get("font_family", "Inter")
    plt.rcParams["font.family"] = resolve_font_family(font_family)

    geoms = [(f.get("geometry") or {}) for f in polygons_for_key]
    bbox_raw = key_summary.get("bbox")
    if bbox_raw is not None and len(bbox_raw) == 4:
        bbox = tuple(bbox_raw)
    elif geoms:
        bbox = polygons_bbox(geoms)
    else:
        bbox = (0.0, 0.0, 1.0, 1.0)
    sat_cfg = report_cfg.get("satellite", {})
    sat_alpha = float(sat_cfg.get("alpha", 0.95))
    background = get_satellite_background(bbox=bbox, epsg=epsg, sat_cfg=sat_cfg)

    fig = plt.figure(figsize=(11.69, 8.27), facecolor=bg_color)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.9, 4.0, 1.6], hspace=0.2, wspace=0.14)
    axh = fig.add_subplot(grid[0, :])
    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[1, 1])
    ax3 = fig.add_subplot(grid[2, :])
    axh.axis("off")
    ax3.axis("off")

    axh.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axh.transAxes, color=fg_color, zorder=0))
    if logo_img is not None:
        axh.imshow(logo_img, extent=(0.01, 0.21, 0.12, 0.88), transform=axh.transAxes, zorder=1)
    axh.text(0.24, 0.64, "SOILYTIX FIELD REPORT", color="white", fontsize=17, fontweight="bold", transform=axh.transAxes)
    axh.text(0.24, 0.28, f"Field {key} | Interpolation variable: {target}", color="white", fontsize=11, transform=axh.transAxes)

    krig_path = key_out / "kriging.tif"
    title = "Predicted Kriging Map"
    if not krig_path.exists():
        krig_path = key_out / "svm.tif"
        title = "Predicted SVM Map"
    zones_path = key_out / "zones_kriging.tif"
    if not zones_path.exists():
        zones_path = key_out / "zones_svm.tif"
    sample_points_path = key_out / "input_points.csv"

    if background is not None:
        bg_arr, bg_extent, bg_label = background
        ax1.imshow(bg_arr, extent=bg_extent, origin="upper", alpha=sat_alpha)
        ax2.imshow(bg_arr, extent=bg_extent, origin="upper", alpha=sat_alpha)
    else:
        bg_label = "No satellite background available"

    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")

    if krig_path.exists():
        arr, extent, nodata = _read_geotiff_for_plot(krig_path)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        im1 = ax1.imshow(arr, extent=extent, origin="upper", cmap="viridis", alpha=0.65)

        # Move colorbar and label outside and aligned bottom left under imshow:
        # Use make_axes_locatable for external colorbar under ax1
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("bottom", size="4.5%", pad=0.21)
        cb1 = fig.colorbar(im1, cax=cax1, orientation="horizontal")
        cb1.set_label(f"{target}", fontsize=7, color=fg_color, labelpad=2)
        cb1.ax.tick_params(labelsize=6, colors=fg_color, pad=1)
        cb1.outline.set_edgecolor(fg_color)
        cb1.ax.xaxis.set_ticks_position("bottom")
        cb1.ax.yaxis.set_ticks_position("none")
        cax1.set_anchor("W")
        ax1.set_title(title, color=fg_color, fontsize=12, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "Kriging output not available", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Predicted Kriging Map", color=fg_color, fontsize=12, fontweight="bold")

    if sample_points_path.exists():
        try:
            sample_df = pd.read_csv(sample_points_path)
            if {"CoordX_SM", "CoordY_SM"}.issubset(sample_df.columns):
                ax1.scatter(
                    sample_df["CoordX_SM"].to_numpy(dtype=float),
                    sample_df["CoordY_SM"].to_numpy(dtype=float),
                    s=12,
                    c="white",
                    edgecolors=fg_color,
                    linewidths=0.6,
                    zorder=7,
                )
                sample_handle = Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor="white",
                    markeredgecolor=fg_color,
                    markeredgewidth=0.8,
                    markersize=5,
                    label="Sample points",
                )
                ax1.legend(
                    handles=[sample_handle],
                    loc="lower left",
                    bbox_to_anchor=(0.01, 0.09),
                    frameon=True,
                    facecolor="white",
                    edgecolor=fg_color,
                    fontsize=6.5,
                    borderpad=0.25,
                    handletextpad=0.3,
                )
        except Exception:
            pass

    if zones_path.exists():
        zarr, zextent, znodata = _read_geotiff_for_plot(zones_path)
        if znodata is not None:
            zarr = np.where(zarr == znodata, np.nan, zarr)
        valid = zarr[np.isfinite(zarr)]
        max_cls = int(np.max(valid)) if valid.size > 0 else 1
        cmap = plt.get_cmap("tab10", max(max_cls, 2))
        ax2.imshow(zarr, extent=zextent, origin="upper", cmap=cmap, alpha=0.75, vmin=1, vmax=max(max_cls, 2))
        classes = sorted({int(v) for v in valid.astype(int).tolist()}) if valid.size else [1]
        handles = [
            Patch(
                facecolor=cmap((cls - 1) / max(max_cls - 1, 1)),
                edgecolor=fg_color,
                linewidth=0.3,
                label=str(cls),
            )
            for cls in classes
        ]
        ax2.legend(
            handles=handles,
            title="Zones",
            title_fontsize=7,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.02),
            ncol=min(6, len(handles)),
            frameon=True,
            facecolor="white",
            edgecolor=fg_color,
            fontsize=6.5,
            borderpad=0.25,
            handlelength=0.9,
            handletextpad=0.3,
            columnspacing=0.6,
            labelspacing=0.2,
        )
        ax2.set_title("Management Zones", color=fg_color, fontsize=12, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Management zones output not available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Management Zones", color=fg_color, fontsize=12, fontweight="bold")

    draw_polygon_edges(ax1, geoms, color="white")
    draw_polygon_edges(ax2, geoms, color="black")

    xmin, ymin, xmax, ymax = bbox
    for ax in (ax1, ax2):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()

    obs_stats = key_summary.get("target_stats_observed", {})
    pred_stats = key_summary.get("target_stats_kriging", {})
    area_coord2 = key_summary.get("field_area_coord2", np.nan)
    summary_lines = [
        f"Field: {key}",
        "",
        f"Area (coordinate units^2): {area_coord2:.3f}" if np.isfinite(area_coord2) else "Area: n/a",
        f"Observed {target}: n={obs_stats.get('count', 0)}, min={obs_stats.get('min', np.nan):.3f}, max={obs_stats.get('max', np.nan):.3f}, "
        f"mean={obs_stats.get('mean', np.nan):.3f}, std={obs_stats.get('std', np.nan):.3f}, CV%={obs_stats.get('cv_pct', np.nan):.2f}",
    ]
    if pred_stats:
        summary_lines.append(
            f"Kriging predicted {target}: n={pred_stats.get('count', 0)}, min={pred_stats.get('min', np.nan):.3f}, max={pred_stats.get('max', np.nan):.3f}, "
            f"range={pred_stats.get('range', np.nan):.3f}, mean={pred_stats.get('mean', np.nan):.3f}, std={pred_stats.get('std', np.nan):.3f}, CV%={pred_stats.get('cv_pct', np.nan):.2f}"
        )
    summary_lines.append(f"CRS: EPSG:{epsg}")
    ax3.add_patch(
        plt.Rectangle(
            (0.0, 0.02),
            1.0,
            0.96,
            transform=ax3.transAxes,
            facecolor="#ffffff",
            edgecolor=fg_color,
            linewidth=1.2,
        )
    )
    ax3.text(0.02, 0.9, "\n".join(summary_lines), va="top", ha="left", fontsize=10, color=fg_color)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.05)
    pdf.savefig(fig, dpi=int(report_cfg.get("dpi", 180)))
    plt.close(fig)


def generate_project_pdf_report(
    output_dir: Path,
    summary: Dict[str, Any],
    polygon_groups: Dict[str, List[Dict[str, Any]]],
    join_key: str,
    target: str,
    epsg: int,
    report_cfg: Dict[str, Any],
) -> Path:
    pdf_filename = str(report_cfg.get("pdf_filename", "project_report.pdf"))
    pdf_path = output_dir / pdf_filename
    keys_sorted = sorted(summary["keys"].keys())
    if Path(report_cfg.get("logo_path")).exists():
        logo_img = plt.imread(report_cfg.get("logo_path"))
    else:
        logo_img = None
    with PdfPages(str(pdf_path)) as pdf:
        for key in keys_sorted:
            key_summary = summary["keys"][key]
            key_out = output_dir / f"{safe_name(join_key)}={safe_name(key)}"
            build_field_page_report(
                pdf=pdf,
                key=key,
                key_out=key_out,
                key_summary=key_summary,
                polygons_for_key=polygon_groups.get(key, []),
                target=target,
                epsg=epsg,
                report_cfg=report_cfg,
                logo_img=logo_img,
            )
    return pdf_path


def load_points_dataframe(features: List[Dict[str, Any]], join_key: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
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


def load_polygon_groups(features: List[Dict[str, Any]], join_key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
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
        p = Path(str(logo_path))
        if not p.is_absolute():
            config["report"]["logo_path"] = str((config_base_dir / p).resolve())
    sat_mode = str(config.get("report", {}).get("satellite", {}).get("mode", "xyz")).lower()
    if sat_mode == "local":
        config["report"]["satellite"]["mode"] = "xyz"

    validate_required(config)
    return config


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    points_geo = load_geojson(Path(config["input"]["points_geojson"]))
    polygons_geo = load_geojson(Path(config["input"]["polygons_geojson"]))

    epsg = int(config.get("output_epsg") or polygons_geo.epsg or points_geo.epsg or 4326)
    join_key = config["input"]["join_key"]
    target = config["input"]["target"]
    method = str(config.get("method", "both")).lower()
    run_kriging = method in {"both", "kriging"}
    run_svm = method in {"both", "svm"}

    points_df = load_points_dataframe(points_geo.features, join_key=join_key)
    polygon_groups = load_polygon_groups(polygons_geo.features, join_key=join_key)
    point_keys = set(points_df[join_key].astype(str).tolist())
    polygon_keys = set(polygon_groups.keys())
    common_keys = sorted(point_keys.intersection(polygon_keys))

    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "input_points": str(Path(config["input"]["points_geojson"]).resolve()),
        "input_polygons": str(Path(config["input"]["polygons_geojson"]).resolve()),
        "join_key": join_key,
        "target": target,
        "method": method,
        "output_epsg": epsg,
        "keys_total": len(common_keys),
        "keys": {},
    }

    covariates_cfg = config.get("covariates") or []
    covariates: List[Tuple[str, float]] = []
    for item in covariates_cfg:
        if isinstance(item, dict) and "name" in item:
            covariates.append((str(item["name"]), float(item.get("weight", 1.0))))
    variable_names: List[str] = []
    for name in [target] + [c for c, _ in covariates]:
        if name not in variable_names:
            variable_names.append(name)

    for key in common_keys:
        key_safe = safe_name(key)
        key_out = output_dir / f"{safe_name(join_key)}={key_safe}"
        key_out.mkdir(parents=True, exist_ok=True)

        poly_geoms = [(f.get("geometry") or {}) for f in polygon_groups[key]]
        xmin, ymin, xmax, ymax = polygons_bbox(poly_geoms)
        area_coord2 = field_area_coord2(poly_geoms)

        pixel_x = config["grid"].get("pixel_size_x")
        pixel_y = config["grid"].get("pixel_size_y")
        if pixel_x is None or pixel_y is None:
            cells = max(2, int(config["grid"].get("cells_per_axis", 100)))
            pixel_x = (xmax - xmin) / float(cells)
            pixel_y = (ymax - ymin) / float(cells)
        pixel_x = float(pixel_x)
        pixel_y = float(pixel_y)
        if pixel_x <= 0 or pixel_y <= 0:
            raise ValueError("Grid pixel sizes must be > 0.")

        max_cells = int(config["grid"].get("max_grid_cells", 2000000))
        xs, ys, full_grid_xy = safe_grid_build(xmin, ymin, xmax, ymax, pixel_x, pixel_y, max_cells)
        inside_mask = points_inside_polygons(full_grid_xy, poly_geoms)
        clipped_grid_xy = full_grid_xy[inside_mask]

        if clipped_grid_xy.size == 0 and bool(config["grid"].get("auto_refine_if_empty", True)):
            refine_steps = int(config["grid"].get("max_auto_refine_steps", 6))
            for _ in range(refine_steps):
                pixel_x /= 2.0
                pixel_y /= 2.0
                xs, ys, full_grid_xy = safe_grid_build(xmin, ymin, xmax, ymax, pixel_x, pixel_y, max_cells)
                inside_mask = points_inside_polygons(full_grid_xy, poly_geoms)
                clipped_grid_xy = full_grid_xy[inside_mask]
                if clipped_grid_xy.size > 0:
                    break

        key_points = points_df[points_df[join_key].astype(str) == key].copy()
        needed_cols = ["CoordX_SM", "CoordY_SM", target] + [c for c, _ in covariates]
        missing_cols = [col for col in needed_cols if col not in key_points.columns]
        if missing_cols:
            summary["keys"][key] = {"status": "skipped", "reason": f"missing columns: {missing_cols}"}
            continue

        key_points = key_points[needed_cols].dropna().reset_index(drop=True)
        key_points.to_csv(key_out / "input_points.csv", index=False)

        if key_points.empty:
            summary["keys"][key] = {"status": "skipped", "reason": "no valid points after dropna"}
            continue
        if clipped_grid_xy.size == 0:
            summary["keys"][key] = {"status": "skipped", "reason": "no grid cells inside polygon geometry"}
            continue

        n_points = len(key_points)
        xy_df = key_points[["CoordX_SM", "CoordY_SM"]]
        xy = xy_df.to_numpy(dtype=float)
        z = key_points[target].to_numpy(dtype=float)

        key_summary: Dict[str, Any] = {
            "status": "ok",
            "n_points": int(n_points),
            "n_polygons": int(len(poly_geoms)),
            "field_area_coord2": area_coord2,
            "bbox": [xmin, ymin, xmax, ymax],
            "pixel_size_x": pixel_x,
            "pixel_size_y": pixel_y,
            "grid_total_cells": int(full_grid_xy.shape[0]),
            "grid_clipped_cells": int(clipped_grid_xy.shape[0]),
            "files": [],
            "target_stats_observed": compute_stats(z),
        }

        grid_info = pd.DataFrame(
            {
                "CoordX_SM": full_grid_xy[:, 0],
                "CoordY_SM": full_grid_xy[:, 1],
                "inside_polygon": inside_mask.astype(int),
            }
        )
        grid_info.to_csv(key_out / "grid_points.csv", index=False)
        key_summary["files"].append("grid_points.csv")

        max_dist = float(semivariogram.Semivariogram(xy_df, pd.Series(z)).max_dist)

        zone_covariate_surfaces: Dict[str, np.ndarray] = {}
        if covariates:
            svm_n_neighbors = int(config["svm"].get("n_neighbors", 16))
            svm_n_neighbors = max(1, min(svm_n_neighbors, n_points))
            svm_radius = config["svm"].get("search_radius")
            if svm_radius is None:
                svm_radius = max_dist
            svm_radius = float(svm_radius)
            idw_power = float(config["svm"].get("idw_power", 1.0))
            for cov_name, weight in covariates:
                surface = build_covariate_surface(
                    sample_xy=xy,
                    sample_values=key_points[cov_name].to_numpy(dtype=float),
                    query_xy=clipped_grid_xy,
                    n_neighbors=svm_n_neighbors,
                    search_radius=svm_radius,
                    idw_power=idw_power,
                )
                zone_covariate_surfaces[cov_name] = surface * float(weight)

        if run_kriging:
            if n_points < 4:
                key_summary["kriging"] = {"status": "skipped", "reason": "requires at least 4 points"}
            else:
                krig_pred: Optional[np.ndarray] = None
                try:
                    model, var_params, var_stats = fit_variogram_params(
                        xy_df=xy_df,
                        z_series=pd.Series(z),
                        cfg=config["kriging"],
                    )
                except Exception as exc:
                    sample_var = float(np.var(z))
                    var_params = [0.0, max(max_dist / 3.0, 1e-6), max(sample_var, 1e-6)]
                    model = "spherical"
                    var_stats = {
                        "model": model,
                        "nugget": var_params[0],
                        "range": var_params[1],
                        "sill": var_params[2],
                        "fallback": True,
                        "reason": str(exc),
                    }

                krig_neighbors = int(config["kriging"].get("n_neighbors", 16))
                krig_neighbors = max(4, min(krig_neighbors, n_points))
                krig_radius = config["kriging"].get("search_radius")
                if krig_radius is None:
                    krig_radius = max_dist
                krig_radius = float(krig_radius)

                ok = kriging.OrdinaryKriging(
                    xy=xy_df,
                    z=pd.Series(z),
                    variogram_model=model,
                    variogram_parameters=var_params,
                )
                krig_pred, _ = ok.execute(
                    clipped_grid_xy,
                    n_closest_points=krig_neighbors,
                    radius=krig_radius,
                )
                krig_points = pd.DataFrame({"CoordX_SM": clipped_grid_xy[:, 0], "CoordY_SM": clipped_grid_xy[:, 1], target: krig_pred})
                krig_points.to_csv(key_out / "kriging_points.csv", index=False)
                write_ogr_vrt(
                    path=key_out / "kriging_points.ogr.vrt",
                    csv_path=key_out / "kriging_points.csv",
                    layer_name="kriging_points",
                    x_field="CoordX_SM",
                    y_field="CoordY_SM",
                    z_field=target,
                )

                full = np.full(full_grid_xy.shape[0], -999999999.0, dtype=float)
                full[inside_mask] = krig_pred
                arr2d = full.reshape(len(ys), len(xs))
                write_geotiff(
                    output_tif=key_out / "kriging.tif",
                    grid_array=arr2d,
                    xmin=xmin,
                    ymax=ymax,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    epsg=epsg,
                    nodata=-999999999.0,
                    dtype=gdal.GDT_Float32,
                )
                write_raster_vrt(output_vrt=key_out / "kriging.vrt", source_tif=key_out / "kriging.tif")

                key_summary["files"].extend(["kriging_points.csv", "kriging_points.ogr.vrt", "kriging.tif", "kriging.vrt"])
                key_summary["kriging"] = {
                    "status": "ok",
                    "variogram": var_stats,
                    "n_neighbors": krig_neighbors,
                    "search_radius": krig_radius,
                }
                key_summary["target_stats_kriging"] = compute_stats(krig_pred)

        svm_zone_target: Optional[np.ndarray] = None
        if run_svm:
            if n_points < 3:
                key_summary["svm"] = {"status": "skipped", "reason": "requires at least 3 points"}
            else:
                svm_n_neighbors = int(config["svm"].get("n_neighbors", 16))
                svm_n_neighbors = max(1, min(svm_n_neighbors, n_points))
                svm_radius = config["svm"].get("search_radius")
                if svm_radius is None:
                    svm_radius = max_dist
                svm_radius = float(svm_radius)
                idw_power = float(config["svm"].get("idw_power", 1.0))

                train_features = [xy[:, 0], xy[:, 1]]
                grid_features = [clipped_grid_xy[:, 0], clipped_grid_xy[:, 1]]
                feature_names = ["CoordX_SM", "CoordY_SM"]

                for cov_name, weight in covariates:
                    cov_values = key_points[cov_name].to_numpy(dtype=float)
                    cov_train = build_covariate_surface(
                        sample_xy=xy,
                        sample_values=cov_values,
                        query_xy=xy,
                        n_neighbors=svm_n_neighbors,
                        search_radius=svm_radius,
                        idw_power=idw_power,
                    )
                    cov_grid = build_covariate_surface(
                        sample_xy=xy,
                        sample_values=cov_values,
                        query_xy=clipped_grid_xy,
                        n_neighbors=svm_n_neighbors,
                        search_radius=svm_radius,
                        idw_power=idw_power,
                    )
                    cov_train *= float(weight)
                    cov_grid *= float(weight)
                    train_features.append(cov_train)
                    grid_features.append(cov_grid)
                    feature_names.append(f"{cov_name}_{float(weight):g}")

                X_train = np.column_stack(train_features).astype(float)
                X_grid = np.column_stack(grid_features).astype(float)
                y_train = z.astype(float)

                kfolds = choose_kfolds(
                    n_samples=len(y_train),
                    kmin=int(config["svm"].get("kfold_min", 2)),
                    kmax=int(config["svm"].get("kfold_max", 5)),
                )
                c_best, gamma_best = fit_svm_params(
                    features=X_train,
                    labels=y_train,
                    c_values=[float(v) for v in config["svm"].get("C_values", [0.001, 0.01, 0.1, 1, 10])],
                    gamma_values=[float(v) for v in config["svm"].get("gamma_values", [0.001, 0.01, 0.1, 1, 10])],
                    kfolds=kfolds,
                    default_c=float(config["svm"].get("default_C", 1.0)),
                    default_gamma=float(config["svm"].get("default_gamma", 0.1)),
                )

                scaler = StandardScaler().fit(X_train)
                X_train_n = scaler.transform(X_train)
                X_grid_n = scaler.transform(X_grid)
                model = svm.SVR(kernel="rbf", C=c_best, gamma=gamma_best)
                model.fit(X_train_n, y_train)
                svm_pred = model.predict(X_grid_n)
                svm_zone_target = svm_pred.copy()

                train_df = pd.DataFrame(X_train, columns=feature_names)
                train_df[target] = y_train
                train_df.to_csv(key_out / "svm_train_features.csv", index=False)

                grid_df = pd.DataFrame(X_grid, columns=feature_names)
                grid_df[target] = svm_pred
                grid_df.to_csv(key_out / "svm_grid_features.csv", index=False)

                svm_points = pd.DataFrame({"CoordX_SM": clipped_grid_xy[:, 0], "CoordY_SM": clipped_grid_xy[:, 1], target: svm_pred})
                svm_points.to_csv(key_out / "svm_points.csv", index=False)
                write_ogr_vrt(
                    path=key_out / "svm_points.ogr.vrt",
                    csv_path=key_out / "svm_points.csv",
                    layer_name="svm_points",
                    x_field="CoordX_SM",
                    y_field="CoordY_SM",
                    z_field=target,
                )

                full = np.full(full_grid_xy.shape[0], -999999999.0, dtype=float)
                full[inside_mask] = svm_pred
                arr2d = full.reshape(len(ys), len(xs))
                write_geotiff(
                    output_tif=key_out / "svm.tif",
                    grid_array=arr2d,
                    xmin=xmin,
                    ymax=ymax,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    epsg=epsg,
                    nodata=-999999999.0,
                    dtype=gdal.GDT_Float32,
                )
                write_raster_vrt(output_vrt=key_out / "svm.vrt", source_tif=key_out / "svm.tif")

                key_summary["files"].extend(
                    [
                        "svm_train_features.csv",
                        "svm_grid_features.csv",
                        "svm_points.csv",
                        "svm_points.ogr.vrt",
                        "svm.tif",
                        "svm.vrt",
                    ]
                )
                key_summary["svm"] = {
                    "status": "ok",
                    "n_neighbors": svm_n_neighbors,
                    "search_radius": svm_radius,
                    "idw_power": idw_power,
                    "kfolds": kfolds,
                    "C": c_best,
                    "gamma": gamma_best,
                }

        zone_inputs: List[Tuple[str, np.ndarray]] = []
        if run_kriging and (key_out / "kriging_points.csv").exists():
            krig_points_df = pd.read_csv(key_out / "kriging_points.csv")
            zone_inputs.append(("kriging", krig_points_df[target].to_numpy(dtype=float)))
        if run_svm and svm_zone_target is not None:
            zone_inputs.append(("svm", svm_zone_target))

        zones_cfg = config["zones"]
        num_zones = int(zones_cfg.get("num_zones", 3))
        max_iter = int(zones_cfg.get("max_iterations", 100))
        m_fuzzy = float(zones_cfg.get("fuzzy_coefficient", 1.25))
        tol = float(zones_cfg.get("error_tolerance", 1.0e-5))
        do_norm = bool(zones_cfg.get("normalize", True))

        for zone_tag, zone_target_surface in zone_inputs:
            zone_features = [zone_target_surface]
            zone_names = [f"{target}_{zone_tag}"]
            for cov_name, _weight in covariates:
                if cov_name in zone_covariate_surfaces:
                    zone_features.append(zone_covariate_surfaces[cov_name])
                    zone_names.append(cov_name)

            alldata = np.vstack(zone_features)
            if alldata.shape[0] == 1:
                alldata = np.vstack((alldata[0], alldata[0]))
                zone_names = [zone_names[0], zone_names[0]]
            if do_norm:
                alldata = scale(alldata, axis=1, with_mean=True, with_std=True, copy=True)

            _, u_orig, _, _, _, _, _ = _cmeans.cmeans(
                alldata,
                c=num_zones,
                m=m_fuzzy,
                error=tol,
                maxiter=max_iter,
            )
            classes = (np.argmax(u_orig, axis=0) + 1).astype(int)
            u_t = u_orig.T
            fpi, nce = compute_fpi_nce(u_t, num_zones)

            class_df = pd.DataFrame(
                {
                    "CoordX_SM": clipped_grid_xy[:, 0],
                    "CoordY_SM": clipped_grid_xy[:, 1],
                    "Classe": classes,
                }
            )
            class_name = f"zones_{zone_tag}_class.csv"
            class_df.to_csv(key_out / class_name, index=False)

            membership_cols = {f"Cls{i+1}": u_t[:, i] for i in range(num_zones)}
            membership_df = pd.DataFrame(
                {
                    "CoordX_SM": clipped_grid_xy[:, 0],
                    "CoordY_SM": clipped_grid_xy[:, 1],
                    **membership_cols,
                }
            )
            membership_name = f"zones_{zone_tag}_membership.csv"
            membership_df.to_csv(key_out / membership_name, index=False)

            full = np.zeros(full_grid_xy.shape[0], dtype=np.uint16)
            full[inside_mask] = classes
            arr2d = full.reshape(len(ys), len(xs))
            tif_name = f"zones_{zone_tag}.tif"
            vrt_name = f"zones_{zone_tag}.vrt"
            write_geotiff(
                output_tif=key_out / tif_name,
                grid_array=arr2d.astype(float),
                xmin=xmin,
                ymax=ymax,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                epsg=epsg,
                nodata=0.0,
                dtype=gdal.GDT_UInt16,
            )
            write_raster_vrt(output_vrt=key_out / vrt_name, source_tif=key_out / tif_name)

            key_summary["files"].extend([class_name, membership_name, tif_name, vrt_name])
            key_summary[f"zones_{zone_tag}"] = {
                "status": "ok",
                "num_zones": num_zones,
                "max_iterations": max_iter,
                "fuzzy_coefficient": m_fuzzy,
                "normalize": do_norm,
                "fpi": fpi,
                "nce": nce,
                "features": zone_names,
            }

        with (key_out / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(key_summary, fh, indent=2)
        key_summary["files"].append("summary.json")
        summary["keys"][key] = key_summary

    report_cfg = config.get("report", {})
    if bool(report_cfg.get("enabled", True)):
        report_path = generate_project_pdf_report(
            output_dir=output_dir,
            summary=summary,
            polygon_groups=polygon_groups,
            join_key=join_key,
            target=target,
            epsg=epsg,
            report_cfg=report_cfg,
        )
        summary["report_pdf"] = str(report_path)

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Smart-Map interpolation runner (no QGIS dependency).")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--points", help="Point GeoJSON path.")
    parser.add_argument("--polygons", help="Polygon GeoJSON path.")
    parser.add_argument("--join-key", help="Property name used to join points to polygons.")
    parser.add_argument("--target", help="Target attribute to interpolate.")
    parser.add_argument(
        "--covariates",
        help="Comma-separated covariates as name or name:weight (e.g. elev:1.0,slope:0.7).",
    )
    parser.add_argument("--num-zones", type=int, help="Desired number of management zones.")
    parser.add_argument("--output-dir", help="Output directory.")
    parser.add_argument("--report-pdf", help="PDF filename for combined project report (inside output-dir).")
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
        help="Override any config value via dotted key=value (repeatable). "
        "Example: --set kriging.n_neighbors=12 --set grid.pixel_size_x=5",
    )
    return parser


def main() -> int:
    parser = make_arg_parser()
    args = parser.parse_args()
    try:
        config = build_config(args)
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


if __name__ == "__main__":
    raise SystemExit(main())
