<img src="icons/icon.png" align="left" width="160" hspace="16" alt="pyvrt logo">

<h3>pyvrt</h3>

<p>
  <sub>A SOILYTIX TOOL · SOIL SAMPLES IN, VARIABLE-RATE MAPS OUT</sub>
  <br>
  <strong>Interpolates point soil samples over field polygons into prediction rasters and management zones, with an optional PDF field report.</strong>
  <br>
  <br>
  <a href="https://github.com/morris-frank/pyVRT/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/morris-frank/pyVRT/ci.yml?style=flat-square&amp;label=CI&amp;labelColor=16211B&amp;color=1AB172" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/status-alpha-8EDE3D?style=flat-square&amp;labelColor=16211B" alt="Alpha">
  <a href="pyproject.toml">
    <img src="https://img.shields.io/badge/python-3.13-1AB172?style=flat-square&amp;labelColor=16211B" alt="Python 3.13">
  </a>
  <a href="mise.toml">
    <img src="https://img.shields.io/badge/gdal-3.13-8EDE3D?style=flat-square&amp;labelColor=16211B" alt="GDAL 3.13">
  </a>
</p>

<br clear="left">

- [What it does](#what-it-does)
- [Install](#install)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)

## What it does

`pyvrt` is a standalone interpolation runner for point and field GeoJSON data. It
joins sample points to field polygons by a key, interpolates one target attribute
per field, and derives management zones from the result.

- **Kriging, SVM, or both.** Kriging fits each listed variogram model and keeps
  the one with the lowest residual sum of squares, unless `kriging.model` fixes
  one; SVM grid-searches `C` and `gamma` with k-fold cross-validation.
- **Management zones.** Fuzzy c-means clusters the prediction raster into the
  configured number of zones.
- **Per-field outputs.** CSV, VRT, GeoTIFF and JSON per field, plus optional ESRI
  Shapefiles of the sampled points (`--output-shapefiles`).
- **Field report.** An optional combined PDF, with satellite tiles fetched from
  the configured XYZ server behind each map.

It reads no credentials. Its rasters are interpolations of the samples given:
their quality is bounded by sample density and the variogram fit, which the
per-field JSON records.

## Install

Needs [mise](https://mise.jdx.dev). Everything else, GDAL included, comes from
`mise.toml`.

```sh
git clone https://github.com/morris-frank/pyVRT.git
cd pyVRT
mise install        # tools, libgdal, the frozen uv sync and the git hooks
```

GDAL ships no PyPI wheels, so the `gdal` binding builds from its sdist against
the libgdal that mise installs; the two versions are pinned together in
`mise.toml` and `pyproject.toml`.

## Usage

```sh
uv run pyvrt --config config.example.json
uv run pyvrt --points points.geojson --polygons fields.geojson \
  --join-key field_id --target "Chemistry__Boron (mg/kg)" --method kriging
```

Any config value can be overridden with `--set key.path=value`, for example
`--set kriging.n_neighbors=12`. `uv run pyvrt --help` lists every flag.

## Configuration

A run is configured by a JSON file; `config.example.json` shows every key. Flags
override the file, and `--set` overrides both.

| Variable | Default | Consumed by |
|---|---|---|
| `MPLCONFIGDIR` | a fresh temp dir | matplotlib, via `src/pyvrt/runtime.py` |
| `XDG_CACHE_HOME` | a fresh temp dir | matplotlib's font cache, via `src/pyvrt/runtime.py` |

Neither is a secret; `.env.example` holds the reasoning.

## Development

`mise run check` is the definition of done and exactly what CI runs: the prek
hooks (ruff format and lint, file hygiene, the staged secret scan), the pytest
suite, `gitleaks` over the history and `osv-scanner` over `uv.lock`. Commit hooks
run the static checks per file; the push hook runs the tests.

`src/krig` and `src/skfuzzy` are vendored upstream code and are excluded from
ruff.
