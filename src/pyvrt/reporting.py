from __future__ import annotations

import math
import urllib.error
import urllib.request
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .geometry import get_polygon_rings, polygons_bbox
from .raster import read_geotiff_for_plot, transform_bbox
from .runtime import configure_runtime
from .utils import safe_name

configure_runtime()

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_XYZ_FETCH_DISABLED = False


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi)
        / 2.0
        * n
    )
    return xtile, ytile


def _tile_bounds_xyz(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    n = 2**z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_left, lat_bottom, lon_right, lat_top


def _fetch_xyz_background(
    bbox: tuple[float, float, float, float],
    epsg: int,
    xyz_url: str,
    zoom: int,
    timeout_seconds: int,
) -> tuple[np.ndarray, tuple[float, float, float, float], str] | None:
    global _XYZ_FETCH_DISABLED
    if _XYZ_FETCH_DISABLED:
        return None

    bbox_4326 = transform_bbox(bbox, epsg, 4326)
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
    for ix, xtile in enumerate(range(x0, x1 + 1)):
        for iy, ytile in enumerate(range(y0, y1 + 1)):
            url = xyz_url.format(z=zoom, x=xtile, y=ytile)
            success = False
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310 - tile URL is operator config
                        data = response.read()
                    tile = Image.open(BytesIO(data)).convert("RGB")
                    success = True
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
                    if attempt == 2:
                        _XYZ_FETCH_DISABLED = True
                        return None
            if not success:
                _XYZ_FETCH_DISABLED = True
                return None
            canvas.paste(tile, (ix * tile_size, iy * tile_size))

    arr = np.asarray(canvas).astype(float) / 255.0
    left, _, _, top = _tile_bounds_xyz(x0, y0, zoom)
    _, bottom, right, _ = _tile_bounds_xyz(x1, y1, zoom)
    extent_4326 = (left, right, bottom, top)
    extent = transform_bbox(extent_4326, 4326, epsg)
    return arr, extent, f"XYZ z{zoom}"


def get_satellite_background(
    bbox: tuple[float, float, float, float],
    epsg: int,
    sat_cfg: dict[str, Any],
) -> tuple[np.ndarray, tuple[float, float, float, float], str] | None:
    mode = str(sat_cfg.get("mode", "xyz")).lower()
    if mode == "none":
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


def draw_polygon_edges(ax: Any, geometries: Iterable[dict[str, Any]], color: str = "black") -> None:
    for geom in geometries:
        for poly in get_polygon_rings(geom):
            for ring in poly:
                ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=1.2)


def resolve_font_family(preferred: str) -> str:
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
        if preferred in available:
            return preferred
    except Exception:  # noqa: S110 - font lookup is best effort; fall back below
        pass
    return "DejaVu Sans"


def build_field_page_report(
    pdf: PdfPages,
    key: str,
    key_out: Path,
    key_summary: dict[str, Any],
    polygons_for_key: list[dict[str, Any]],
    target: str,
    epsg: int,
    report_cfg: dict[str, Any],
    logo_img: np.ndarray | None,
) -> None:
    theme = report_cfg.get("theme", {})
    bg_color = theme.get("bg", "#f5f3f0")
    fg_color = theme.get("fg", "#29332e")
    plt.rcParams["font.family"] = resolve_font_family(theme.get("font_family", "Inter"))

    geometries = [(feature.get("geometry") or {}) for feature in polygons_for_key]
    bbox_raw = key_summary.get("bbox")
    if bbox_raw is not None and len(bbox_raw) == 4:
        bbox = tuple(bbox_raw)
    elif geometries:
        bbox = polygons_bbox(geometries)
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
    axh.text(
        0.24,
        0.64,
        "SOILYTIX FIELD REPORT",
        color="white",
        fontsize=17,
        fontweight="bold",
        transform=axh.transAxes,
    )
    axh.text(
        0.24,
        0.28,
        f"Field {key} | Interpolation variable: {target}",
        color="white",
        fontsize=11,
        transform=axh.transAxes,
    )

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
        bg_arr, bg_extent, _ = background
        ax1.imshow(bg_arr, extent=bg_extent, origin="upper", alpha=sat_alpha)
        ax2.imshow(bg_arr, extent=bg_extent, origin="upper", alpha=sat_alpha)

    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")

    if krig_path.exists():
        arr, extent, nodata = read_geotiff_for_plot(krig_path)
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        im1 = ax1.imshow(arr, extent=extent, origin="upper", cmap="viridis", alpha=0.65)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("bottom", size="4.5%", pad=0.21)
        colorbar = fig.colorbar(im1, cax=cax1, orientation="horizontal")
        colorbar.set_label(f"{target}", fontsize=7, color=fg_color, labelpad=2)
        colorbar.ax.tick_params(labelsize=6, colors=fg_color, pad=1)
        colorbar.outline.set_edgecolor(fg_color)
        colorbar.ax.xaxis.set_ticks_position("bottom")
        ax1.set_title(title, color=fg_color, fontsize=12, fontweight="bold")
    else:
        ax1.text(
            0.5,
            0.5,
            "Interpolation output not available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("Predicted Map", color=fg_color, fontsize=12, fontweight="bold")

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
        except Exception:  # noqa: S110 - a legend that fails to draw must not sink the report
            pass

    if zones_path.exists():
        zarr, zextent, znodata = read_geotiff_for_plot(zones_path)
        if znodata is not None:
            zarr = np.where(zarr == znodata, np.nan, zarr)
        valid = zarr[np.isfinite(zarr)]
        max_cls = int(np.max(valid)) if valid.size > 0 else 1
        cmap = plt.get_cmap("tab10", max(max_cls, 2))
        ax2.imshow(
            zarr,
            extent=zextent,
            origin="upper",
            cmap=cmap,
            alpha=0.75,
            vmin=1,
            vmax=max(max_cls, 2),
        )
        classes = (
            sorted({int(value) for value in valid.astype(int).tolist()}) if valid.size else [1]
        )
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
        ax2.text(
            0.5,
            0.5,
            "Management zones output not available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Management Zones", color=fg_color, fontsize=12, fontweight="bold")

    draw_polygon_edges(ax1, geometries, color="white")
    draw_polygon_edges(ax2, geometries, color="black")

    xmin, ymin, xmax, ymax = bbox
    for ax in (ax1, ax2):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()

    obs_stats = key_summary.get("target_stats_observed", {})
    pred_stats = key_summary.get("target_stats_kriging") or key_summary.get("target_stats_svm", {})
    area_coord2 = key_summary.get("field_area_coord2", np.nan)
    summary_lines = [
        f"Field: {key}",
        "",
        f"Area (coordinate units^2): {area_coord2:.3f}"
        if np.isfinite(area_coord2)
        else "Area: n/a",
        f"Observed {target}: n={obs_stats.get('count', 0)}, min={obs_stats.get('min', np.nan):.3f}, max={obs_stats.get('max', np.nan):.3f}, "
        f"mean={obs_stats.get('mean', np.nan):.3f}, std={obs_stats.get('std', np.nan):.3f}, CV%={obs_stats.get('cv_pct', np.nan):.2f}",
    ]
    if pred_stats:
        summary_lines.append(
            f"Predicted {target}: n={pred_stats.get('count', 0)}, min={pred_stats.get('min', np.nan):.3f}, max={pred_stats.get('max', np.nan):.3f}, "
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
    summary: dict[str, Any],
    polygon_groups: dict[str, list[dict[str, Any]]],
    join_key: str,
    target: str,
    epsg: int,
    report_cfg: dict[str, Any],
) -> Path:
    pdf_path = output_dir / str(report_cfg.get("pdf_filename", "project_report.pdf"))
    logo_path = report_cfg.get("logo_path")
    logo_img = None
    if logo_path:
        candidate = Path(str(logo_path))
        if candidate.exists():
            logo_img = plt.imread(candidate)

    with PdfPages(str(pdf_path)) as pdf:
        for key in sorted(summary["keys"].keys()):
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
