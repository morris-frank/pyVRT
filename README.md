# pyVRT

`pyVRT` is a standalone soil interpolation runner for point and field GeoJSON data. It builds kriging and/or SVM prediction rasters, derives management zones, and can assemble a compact PDF field report.

## Install

```bash
pip install .
```

`GDAL` is a real dependency, so the target environment needs a compatible GDAL runtime and Python bindings available.

## Run

```bash
pyvrt --config config.example.json
```

Or run it fully from flags:

```bash
pyvrt --points points.geojson --polygons fields.geojson --join-key field_id --target "Boron (mg/kg)" --method kriging
```

To also emit sampled point layers as ESRI Shapefiles:

```bash
pyvrt --config config.example.json --output-shapefiles
```

Outputs are written into the configured `output_dir` as per-field CSV, VRT, GeoTIFF, JSON, optional shapefile, and optional PDF files.
