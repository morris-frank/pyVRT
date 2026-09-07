from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

gdal.UseExceptions()


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
    dataset = driver.Create(str(output_tif), nx, ny, 1, dtype)
    if dataset is None:
        raise RuntimeError(f"Failed to create GeoTIFF: {output_tif}")

    dataset.SetGeoTransform([xmin, pixel_x, 0.0, ymax, 0.0, -pixel_y])

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    dataset.SetProjection(srs.ExportToWkt())

    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(np.flipud(grid_array))
    band.FlushCache()
    dataset.FlushCache()


def write_raster_vrt(output_vrt: Path, source_tif: Path) -> None:
    gdal.Translate(str(output_vrt), str(source_tif), format="VRT")


def _shape_field_name(name: str, used: set[str]) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "field"
    if safe[0].isdigit():
        safe = f"f_{safe}"
    safe = safe[:10]

    candidate = safe
    counter = 1
    while candidate.lower() in used:
        suffix = str(counter)
        candidate = f"{safe[: max(1, 10 - len(suffix))]}{suffix}"
        counter += 1
    used.add(candidate.lower())
    return candidate


def _ogr_field_type(series: pd.Series) -> int:
    if pd.api.types.is_bool_dtype(series):
        return ogr.OFTInteger
    if pd.api.types.is_integer_dtype(series):
        return ogr.OFTInteger64
    if pd.api.types.is_float_dtype(series):
        return ogr.OFTReal
    return ogr.OFTString


def write_point_shapefile(
    output_shp: Path,
    frame: pd.DataFrame,
    x_field: str,
    y_field: str,
    epsg: int,
    layer_name: Optional[str] = None,
) -> Dict[str, str]:
    if x_field not in frame.columns or y_field not in frame.columns:
        raise ValueError(f"Missing coordinate columns '{x_field}'/'{y_field}' for shapefile export.")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("GDAL ESRI Shapefile driver is not available.")

    if output_shp.suffix.lower() != ".shp":
        output_shp = output_shp.with_suffix(".shp")
    if output_shp.exists():
        driver.DeleteDataSource(str(output_shp))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    dataset = driver.CreateDataSource(str(output_shp))
    if dataset is None:
        raise RuntimeError(f"Failed to create shapefile: {output_shp}")

    layer = dataset.CreateLayer(
        layer_name or output_shp.stem,
        srs=srs,
        geom_type=ogr.wkbPoint,
        options=["ENCODING=UTF-8"],
    )
    if layer is None:
        dataset = None
        raise RuntimeError(f"Failed to create shapefile layer: {output_shp}")

    field_names: Dict[str, str] = {}
    used_names: set[str] = set()
    for column in frame.columns:
        shape_name = _shape_field_name(column, used_names)
        field_names[column] = shape_name

        field_defn = ogr.FieldDefn(shape_name, _ogr_field_type(frame[column]))
        if field_defn.GetType() == ogr.OFTString:
            non_null = frame[column].dropna()
            width = 80 if non_null.empty else min(254, max(1, int(non_null.astype(str).str.len().max())))
            field_defn.SetWidth(width)
        layer.CreateField(field_defn)

    layer_defn = layer.GetLayerDefn()
    for row in frame.to_dict(orient="records"):
        feature = ogr.Feature(layer_defn)
        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint(float(row[x_field]), float(row[y_field]))
        feature.SetGeometry(geom)

        for column, shape_name in field_names.items():
            value = row[column]
            if pd.isna(value):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool):
                value = int(value)
            feature.SetField(shape_name, value)

        layer.CreateFeature(feature)
        geom = None
        feature = None

    dataset = None
    return field_names


def osr_transform_xy(x: float, y: float, src_epsg: int, dst_epsg: int) -> Tuple[float, float]:
    if src_epsg == dst_epsg:
        return float(x), float(y)

    src = osr.SpatialReference()
    src.ImportFromEPSG(int(src_epsg))
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(int(dst_epsg))
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)
    out = transform.TransformPoint(float(x), float(y))
    return float(out[0]), float(out[1])


def transform_bbox(
    bbox: Tuple[float, float, float, float],
    src_epsg: int,
    dst_epsg: int,
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bbox
    corners = [
        osr_transform_xy(xmin, ymin, src_epsg, dst_epsg),
        osr_transform_xy(xmin, ymax, src_epsg, dst_epsg),
        osr_transform_xy(xmax, ymin, src_epsg, dst_epsg),
        osr_transform_xy(xmax, ymax, src_epsg, dst_epsg),
    ]
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return min(xs), min(ys), max(xs), max(ys)


def read_geotiff_for_plot(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float, float], Optional[float]]:
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Could not read raster: {path}")

    gt = dataset.GetGeoTransform()
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    xmin = gt[0]
    xmax = gt[0] + xsize * gt[1]
    ymax = gt[3]
    ymin = gt[3] + ysize * gt[5]
    band = dataset.GetRasterBand(1)
    arr = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    return arr, (xmin, xmax, ymin, ymax), nodata
