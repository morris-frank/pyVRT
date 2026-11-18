from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .runtime import configure_runtime

configure_runtime()

from matplotlib.path import Path as MplPath


def get_polygon_rings(geometry: dict[str, Any]) -> list[list[np.ndarray]]:
    polygons: list[list[np.ndarray]] = []
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        polygon: list[np.ndarray] = []
        for ring in coords:
            arr = np.asarray(ring, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                polygon.append(arr[:, :2])
        if polygon:
            polygons.append(polygon)
    elif gtype == "MultiPolygon":
        for polygon_coords in coords:
            polygon = []
            for ring in polygon_coords:
                arr = np.asarray(ring, dtype=float)
                if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                    polygon.append(arr[:, :2])
            if polygon:
                polygons.append(polygon)

    return polygons


def polygons_bbox(geometries: Iterable[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for geom in geometries:
        for poly in get_polygon_rings(geom):
            for ring in poly:
                xs.extend(ring[:, 0].tolist())
                ys.extend(ring[:, 1].tolist())
    if not xs or not ys:
        raise ValueError("No valid polygon coordinates found for bbox calculation.")
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def points_inside_polygons(
    points_xy: np.ndarray, geometries: Iterable[dict[str, Any]]
) -> np.ndarray:
    mask = np.zeros(points_xy.shape[0], dtype=bool)
    for geom in geometries:
        for poly in get_polygon_rings(geom):
            outer = MplPath(poly[0])
            in_poly = outer.contains_points(points_xy, radius=1e-12)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, grid = build_grid(xmin, ymin, xmax, ymax, pixel_x, pixel_y)
    n_cells = int(grid.shape[0])
    if n_cells > int(max_cells):
        raise ValueError(
            f"Grid too large ({n_cells} cells) for pixel size ({pixel_x}, {pixel_y}). "
            "Increase pixel size or lower refinement."
        )
    return xs, ys, grid


def ring_area(ring: np.ndarray) -> float:
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def polygon_area_coord2(geometry: dict[str, Any]) -> float:
    area = 0.0
    for poly in get_polygon_rings(geometry):
        if not poly:
            continue
        outer = abs(ring_area(poly[0]))
        holes = sum(abs(ring_area(hole)) for hole in poly[1:])
        area += max(0.0, outer - holes)
    return float(area)


def field_area_coord2(geometries: Iterable[dict[str, Any]]) -> float:
    return float(sum(polygon_area_coord2(geometry) for geometry in geometries))
