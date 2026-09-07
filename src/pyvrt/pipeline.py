from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler, scale

from skfuzzy.cluster import _cmeans

from krig import semivariogram

from .data import load_geojson, load_points_dataframe, load_polygon_groups
from .geometry import field_area_coord2, points_inside_polygons, polygons_bbox, safe_grid_build
from .modeling import (
    build_covariate_surface,
    choose_kfolds,
    compute_fpi_nce,
    fit_svm_params,
    fit_variogram_params,
    run_ordinary_kriging,
)
from .raster import gdal, write_geotiff, write_ogr_vrt, write_point_shapefile, write_raster_vrt
from .reporting import generate_project_pdf_report
from .utils import compute_stats, safe_name

FLOAT_NODATA = -999999999.0


def _extract_covariates(config: Dict[str, Any]) -> List[Tuple[str, float]]:
    covariates: List[Tuple[str, float]] = []
    for item in config.get("covariates") or []:
        if isinstance(item, dict) and "name" in item:
            covariates.append((str(item["name"]), float(item.get("weight", 1.0))))
    return covariates


def _prepare_grid(
    poly_geoms: List[Dict[str, Any]],
    grid_cfg: Dict[str, Any],
) -> Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = polygons_bbox(poly_geoms)

    pixel_x = grid_cfg.get("pixel_size_x")
    pixel_y = grid_cfg.get("pixel_size_y")
    if pixel_x is None or pixel_y is None:
        cells = max(2, int(grid_cfg.get("cells_per_axis", 100)))
        pixel_x = (xmax - xmin) / float(cells)
        pixel_y = (ymax - ymin) / float(cells)

    pixel_x = float(pixel_x)
    pixel_y = float(pixel_y)
    if pixel_x <= 0 or pixel_y <= 0:
        raise ValueError("Grid pixel sizes must be > 0.")

    max_cells = int(grid_cfg.get("max_grid_cells", 2000000))
    xs, ys, full_grid_xy = safe_grid_build(xmin, ymin, xmax, ymax, pixel_x, pixel_y, max_cells)
    inside_mask = points_inside_polygons(full_grid_xy, poly_geoms)
    clipped_grid_xy = full_grid_xy[inside_mask]

    if clipped_grid_xy.size == 0 and bool(grid_cfg.get("auto_refine_if_empty", True)):
        for _ in range(int(grid_cfg.get("max_auto_refine_steps", 6))):
            pixel_x /= 2.0
            pixel_y /= 2.0
            xs, ys, full_grid_xy = safe_grid_build(xmin, ymin, xmax, ymax, pixel_x, pixel_y, max_cells)
            inside_mask = points_inside_polygons(full_grid_xy, poly_geoms)
            clipped_grid_xy = full_grid_xy[inside_mask]
            if clipped_grid_xy.size > 0:
                break

    return xmin, ymin, xmax, ymax, pixel_x, pixel_y, xs, ys, full_grid_xy, inside_mask, clipped_grid_xy


def _build_zone_covariate_surfaces(
    config: Dict[str, Any],
    xy: np.ndarray,
    key_points: pd.DataFrame,
    covariates: List[Tuple[str, float]],
    clipped_grid_xy: np.ndarray,
    max_dist: float,
) -> Dict[str, np.ndarray]:
    if not covariates:
        return {}

    svm_n_neighbors = int(config["svm"].get("n_neighbors", 16))
    svm_n_neighbors = max(1, min(svm_n_neighbors, len(key_points)))
    svm_radius = config["svm"].get("search_radius")
    if svm_radius is None:
        svm_radius = max_dist
    svm_radius = float(svm_radius)
    idw_power = float(config["svm"].get("idw_power", 1.0))

    surfaces: Dict[str, np.ndarray] = {}
    for cov_name, weight in covariates:
        surface = build_covariate_surface(
            sample_xy=xy,
            sample_values=key_points[cov_name].to_numpy(dtype=float),
            query_xy=clipped_grid_xy,
            n_neighbors=svm_n_neighbors,
            search_radius=svm_radius,
            idw_power=idw_power,
        )
        surfaces[cov_name] = surface * float(weight)
    return surfaces


def _write_point_outputs(
    key_out: Path,
    base_name: str,
    points_df: pd.DataFrame,
    epsg: int,
    write_shapefiles: bool,
) -> List[str]:
    csv_name = f"{base_name}.csv"
    points_df.to_csv(key_out / csv_name, index=False)

    files = [csv_name]
    if write_shapefiles:
        shp_name = f"{base_name}.shp"
        write_point_shapefile(
            output_shp=key_out / shp_name,
            frame=points_df,
            x_field="CoordX_SM",
            y_field="CoordY_SM",
            epsg=epsg,
            layer_name=base_name,
        )
        files.append(shp_name)
    return files


def _export_prediction_surface(
    key_out: Path,
    prefix: str,
    target: str,
    query_xy: np.ndarray,
    predictions: np.ndarray,
    full_grid_xy: np.ndarray,
    inside_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    xmin: float,
    ymax: float,
    pixel_x: float,
    pixel_y: float,
    epsg: int,
    write_shapefiles: bool,
) -> List[str]:
    points_name = f"{prefix}_points.csv"
    vrt_points_name = f"{prefix}_points.ogr.vrt"
    tif_name = f"{prefix}.tif"
    vrt_name = f"{prefix}.vrt"

    points_df = pd.DataFrame(
        {
            "CoordX_SM": query_xy[:, 0],
            "CoordY_SM": query_xy[:, 1],
            target: predictions,
        }
    )
    files = _write_point_outputs(
        key_out=key_out,
        base_name=f"{prefix}_points",
        points_df=points_df,
        epsg=epsg,
        write_shapefiles=write_shapefiles,
    )
    write_ogr_vrt(
        path=key_out / vrt_points_name,
        csv_path=key_out / points_name,
        layer_name=f"{prefix}_points",
        x_field="CoordX_SM",
        y_field="CoordY_SM",
        z_field=target,
    )

    full = np.full(full_grid_xy.shape[0], FLOAT_NODATA, dtype=float)
    full[inside_mask] = predictions
    arr2d = full.reshape(len(ys), len(xs))
    write_geotiff(
        output_tif=key_out / tif_name,
        grid_array=arr2d,
        xmin=xmin,
        ymax=ymax,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        epsg=epsg,
        nodata=FLOAT_NODATA,
        dtype=gdal.GDT_Float32,
    )
    write_raster_vrt(output_vrt=key_out / vrt_name, source_tif=key_out / tif_name)
    files.extend([vrt_points_name, tif_name, vrt_name])
    return files


def _run_kriging(
    config: Dict[str, Any],
    xy_df: pd.DataFrame,
    z_values: np.ndarray,
    max_dist: float,
    clipped_grid_xy: np.ndarray,
    full_grid_xy: np.ndarray,
    inside_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    xmin: float,
    ymax: float,
    pixel_x: float,
    pixel_y: float,
    epsg: int,
    key_out: Path,
    target: str,
    write_shapefiles: bool,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], List[str], Optional[Dict[str, float]]]:
    n_points = len(z_values)
    if n_points < 4:
        return {"status": "skipped", "reason": "requires at least 4 points"}, None, [], None
    try:
        model, var_params, var_stats = fit_variogram_params(
            xy_df=xy_df,
            z_series=pd.Series(z_values),
            cfg=config["kriging"],
        )
    except Exception as exc:
        sample_var = float(np.var(z_values))
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

    predictions = run_ordinary_kriging(
        xy_df=xy_df,
        z_values=z_values,
        query_xy=clipped_grid_xy,
        variogram_model=model,
        variogram_parameters=var_params,
        n_neighbors=krig_neighbors,
        radius=krig_radius,
    )
    files = _export_prediction_surface(
        key_out=key_out,
        prefix="kriging",
        target=target,
        query_xy=clipped_grid_xy,
        predictions=predictions,
        full_grid_xy=full_grid_xy,
        inside_mask=inside_mask,
        xs=xs,
        ys=ys,
        xmin=xmin,
        ymax=ymax,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        epsg=epsg,
        write_shapefiles=write_shapefiles,
    )
    result = {
        "status": "ok",
        "variogram": var_stats,
        "n_neighbors": krig_neighbors,
        "search_radius": krig_radius,
    }
    return result, predictions, files, compute_stats(predictions)


def _run_svm(
    config: Dict[str, Any],
    xy: np.ndarray,
    xy_df: pd.DataFrame,
    z_values: np.ndarray,
    max_dist: float,
    clipped_grid_xy: np.ndarray,
    full_grid_xy: np.ndarray,
    inside_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    xmin: float,
    ymax: float,
    pixel_x: float,
    pixel_y: float,
    epsg: int,
    key_out: Path,
    target: str,
    key_points: pd.DataFrame,
    covariates: List[Tuple[str, float]],
    write_shapefiles: bool,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], List[str], Optional[Dict[str, float]]]:
    n_points = len(z_values)
    if n_points < 3:
        return {"status": "skipped", "reason": "requires at least 3 points"}, None, [], None
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

    x_train = np.column_stack(train_features).astype(float)
    x_grid = np.column_stack(grid_features).astype(float)
    y_train = z_values.astype(float)

    kfolds = choose_kfolds(
        n_samples=len(y_train),
        kmin=int(config["svm"].get("kfold_min", 2)),
        kmax=int(config["svm"].get("kfold_max", 5)),
    )
    c_best, gamma_best = fit_svm_params(
        features=x_train,
        labels=y_train,
        c_values=[float(value) for value in config["svm"].get("C_values", [0.001, 0.01, 0.1, 1, 10])],
        gamma_values=[float(value) for value in config["svm"].get("gamma_values", [0.001, 0.01, 0.1, 1, 10])],
        kfolds=kfolds,
        default_c=float(config["svm"].get("default_C", 1.0)),
        default_gamma=float(config["svm"].get("default_gamma", 0.1)),
    )

    scaler = StandardScaler().fit(x_train)
    model = svm.SVR(kernel="rbf", C=c_best, gamma=gamma_best)
    model.fit(scaler.transform(x_train), y_train)
    predictions = model.predict(scaler.transform(x_grid))

    train_df = pd.DataFrame(x_train, columns=feature_names)
    train_df[target] = y_train
    files = _write_point_outputs(
        key_out=key_out,
        base_name="svm_train_features",
        points_df=train_df,
        epsg=epsg,
        write_shapefiles=write_shapefiles,
    )

    grid_df = pd.DataFrame(x_grid, columns=feature_names)
    grid_df[target] = predictions
    files.extend(
        _write_point_outputs(
            key_out=key_out,
            base_name="svm_grid_features",
            points_df=grid_df,
            epsg=epsg,
            write_shapefiles=write_shapefiles,
        )
    )
    files.extend(
        _export_prediction_surface(
            key_out=key_out,
            prefix="svm",
            target=target,
            query_xy=clipped_grid_xy,
            predictions=predictions,
            full_grid_xy=full_grid_xy,
            inside_mask=inside_mask,
            xs=xs,
            ys=ys,
            xmin=xmin,
            ymax=ymax,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            epsg=epsg,
            write_shapefiles=write_shapefiles,
        )
    )

    result = {
        "status": "ok",
        "n_neighbors": svm_n_neighbors,
        "search_radius": svm_radius,
        "idw_power": idw_power,
        "kfolds": kfolds,
        "C": c_best,
        "gamma": gamma_best,
    }
    return result, np.asarray(predictions, dtype=float), files, compute_stats(predictions)


def _run_zoning(
    key_out: Path,
    zone_tag: str,
    zone_target_surface: np.ndarray,
    clipped_grid_xy: np.ndarray,
    full_grid_xy: np.ndarray,
    inside_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    xmin: float,
    ymax: float,
    pixel_x: float,
    pixel_y: float,
    epsg: int,
    target: str,
    covariates: List[Tuple[str, float]],
    zone_covariate_surfaces: Dict[str, np.ndarray],
    zones_cfg: Dict[str, Any],
    write_shapefiles: bool,
) -> Tuple[List[str], Dict[str, Any]]:
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

    do_norm = bool(zones_cfg.get("normalize", True))
    if do_norm:
        alldata = scale(alldata, axis=1, with_mean=True, with_std=True, copy=True)

    num_zones = int(zones_cfg.get("num_zones", 3))
    max_iter = int(zones_cfg.get("max_iterations", 100))
    m_fuzzy = float(zones_cfg.get("fuzzy_coefficient", 1.25))
    tol = float(zones_cfg.get("error_tolerance", 1.0e-5))

    _, u_orig, _, _, _, _, _ = _cmeans.cmeans(
        alldata,
        c=num_zones,
        m=m_fuzzy,
        error=tol,
        maxiter=max_iter,
    )
    classes = (np.argmax(u_orig, axis=0) + 1).astype(int)
    u_transposed = u_orig.T
    fpi, nce = compute_fpi_nce(u_transposed, num_zones)

    files = _write_point_outputs(
        key_out=key_out,
        base_name=f"zones_{zone_tag}_class",
        points_df=pd.DataFrame(
            {
                "CoordX_SM": clipped_grid_xy[:, 0],
                "CoordY_SM": clipped_grid_xy[:, 1],
                "Classe": classes,
            }
        ),
        epsg=epsg,
        write_shapefiles=write_shapefiles,
    )

    files.extend(
        _write_point_outputs(
            key_out=key_out,
            base_name=f"zones_{zone_tag}_membership",
            points_df=pd.DataFrame(
                {
                    "CoordX_SM": clipped_grid_xy[:, 0],
                    "CoordY_SM": clipped_grid_xy[:, 1],
                    **{f"Cls{i + 1}": u_transposed[:, i] for i in range(num_zones)},
                }
            ),
            epsg=epsg,
            write_shapefiles=write_shapefiles,
        )
    )

    tif_name = f"zones_{zone_tag}.tif"
    vrt_name = f"zones_{zone_tag}.vrt"
    full = np.zeros(full_grid_xy.shape[0], dtype=np.uint16)
    full[inside_mask] = classes
    arr2d = full.reshape(len(ys), len(xs))
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

    result = {
        "status": "ok",
        "num_zones": num_zones,
        "max_iterations": max_iter,
        "fuzzy_coefficient": m_fuzzy,
        "normalize": do_norm,
        "fpi": fpi,
        "nce": nce,
        "features": zone_names,
    }
    files.extend([tif_name, vrt_name])
    return files, result


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    points_geo = load_geojson(Path(config["input"]["points_geojson"]))
    polygons_geo = load_geojson(Path(config["input"]["polygons_geojson"]))

    epsg = int(config.get("output_epsg") or polygons_geo.epsg or points_geo.epsg or 4326)
    join_key = config["input"]["join_key"]
    target = config["input"]["target"]
    method = str(config.get("method", "both")).lower()
    run_kriging = method in {"both", "kriging"}
    run_svm = method in {"both", "svm"}
    write_shapefiles = bool(config.get("output_shapefiles", False))

    points_df = load_points_dataframe(points_geo.features, join_key=join_key)
    polygon_groups = load_polygon_groups(polygons_geo.features, join_key=join_key)
    common_keys = sorted(set(points_df[join_key].astype(str).tolist()).intersection(polygon_groups.keys()))

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

    covariates = _extract_covariates(config)

    for key in common_keys:
        key_out = output_dir / f"{safe_name(join_key)}={safe_name(key)}"
        key_out.mkdir(parents=True, exist_ok=True)

        poly_geoms = [(feature.get("geometry") or {}) for feature in polygon_groups[key]]
        area_coord2 = field_area_coord2(poly_geoms)
        (
            xmin,
            ymin,
            xmax,
            ymax,
            pixel_x,
            pixel_y,
            xs,
            ys,
            full_grid_xy,
            inside_mask,
            clipped_grid_xy,
        ) = _prepare_grid(poly_geoms, config["grid"])

        key_points = points_df[points_df[join_key].astype(str) == key].copy()
        needed_cols = ["CoordX_SM", "CoordY_SM", target] + [name for name, _ in covariates]
        missing_cols = [col for col in needed_cols if col not in key_points.columns]
        if missing_cols:
            summary["keys"][key] = {"status": "skipped", "reason": f"missing columns: {missing_cols}"}
            continue

        key_points = key_points[needed_cols].dropna().reset_index(drop=True)
        key_summary_files = _write_point_outputs(
            key_out=key_out,
            base_name="input_points",
            points_df=key_points,
            epsg=epsg,
            write_shapefiles=write_shapefiles,
        )

        if key_points.empty:
            summary["keys"][key] = {"status": "skipped", "reason": "no valid points after dropna"}
            continue
        if clipped_grid_xy.size == 0:
            summary["keys"][key] = {"status": "skipped", "reason": "no grid cells inside polygon geometry"}
            continue

        xy_df = key_points[["CoordX_SM", "CoordY_SM"]]
        xy = xy_df.to_numpy(dtype=float)
        z_values = key_points[target].to_numpy(dtype=float)
        max_dist = float(semivariogram.Semivariogram(xy_df, pd.Series(z_values)).max_dist)

        key_summary: Dict[str, Any] = {
            "status": "ok",
            "n_points": int(len(key_points)),
            "n_polygons": int(len(poly_geoms)),
            "field_area_coord2": area_coord2,
            "bbox": [xmin, ymin, xmax, ymax],
            "pixel_size_x": pixel_x,
            "pixel_size_y": pixel_y,
            "grid_total_cells": int(full_grid_xy.shape[0]),
            "grid_clipped_cells": int(clipped_grid_xy.shape[0]),
            "files": key_summary_files,
            "target_stats_observed": compute_stats(z_values),
        }

        key_summary["files"].extend(
            _write_point_outputs(
                key_out=key_out,
                base_name="grid_points",
                points_df=pd.DataFrame(
                    {
                        "CoordX_SM": full_grid_xy[:, 0],
                        "CoordY_SM": full_grid_xy[:, 1],
                        "inside_polygon": inside_mask.astype(int),
                    }
                ),
                epsg=epsg,
                write_shapefiles=write_shapefiles,
            )
        )

        kriging_surface: Optional[np.ndarray] = None
        zone_covariate_surfaces = _build_zone_covariate_surfaces(
            config=config,
            xy=xy,
            key_points=key_points,
            covariates=covariates,
            clipped_grid_xy=clipped_grid_xy,
            max_dist=max_dist,
        )

        if run_kriging:
            kriging_result, kriging_surface, kriging_files, kriging_stats = _run_kriging(
                config=config,
                xy_df=xy_df,
                z_values=z_values,
                max_dist=max_dist,
                clipped_grid_xy=clipped_grid_xy,
                full_grid_xy=full_grid_xy,
                inside_mask=inside_mask,
                xs=xs,
                ys=ys,
                xmin=xmin,
                ymax=ymax,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                epsg=epsg,
                key_out=key_out,
                target=target,
                write_shapefiles=write_shapefiles,
            )
            key_summary["kriging"] = kriging_result
            key_summary["files"].extend(kriging_files)
            if kriging_stats is not None:
                key_summary["target_stats_kriging"] = kriging_stats

        svm_surface: Optional[np.ndarray] = None
        if run_svm:
            svm_result, svm_surface, svm_files, svm_stats = _run_svm(
                config=config,
                xy=xy,
                xy_df=xy_df,
                z_values=z_values,
                max_dist=max_dist,
                clipped_grid_xy=clipped_grid_xy,
                full_grid_xy=full_grid_xy,
                inside_mask=inside_mask,
                xs=xs,
                ys=ys,
                xmin=xmin,
                ymax=ymax,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                epsg=epsg,
                key_out=key_out,
                target=target,
                key_points=key_points,
                covariates=covariates,
                write_shapefiles=write_shapefiles,
            )
            key_summary["svm"] = svm_result
            key_summary["files"].extend(svm_files)
            if svm_stats is not None:
                key_summary["target_stats_svm"] = svm_stats

        for zone_tag, zone_target_surface in (("kriging", kriging_surface), ("svm", svm_surface)):
            if zone_target_surface is None:
                continue
            zone_files, zone_result = _run_zoning(
                key_out=key_out,
                zone_tag=zone_tag,
                zone_target_surface=zone_target_surface,
                clipped_grid_xy=clipped_grid_xy,
                full_grid_xy=full_grid_xy,
                inside_mask=inside_mask,
                xs=xs,
                ys=ys,
                xmin=xmin,
                ymax=ymax,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                epsg=epsg,
                target=target,
                covariates=covariates,
                zone_covariate_surfaces=zone_covariate_surfaces,
                zones_cfg=config["zones"],
                write_shapefiles=write_shapefiles,
            )
            key_summary["files"].extend(zone_files)
            key_summary[f"zones_{zone_tag}"] = zone_result

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
