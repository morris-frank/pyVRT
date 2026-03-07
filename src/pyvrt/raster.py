from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal, osr

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
