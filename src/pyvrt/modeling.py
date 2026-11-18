from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import spatial
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from krig import kriging, semivariogram


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
    neighbors = tree.query_ball_point(point, radius)
    if len(neighbors) > n_neighbors:
        k = n_neighbors + 1
    elif len(neighbors) < 2:
        k = min(3, len(source_xy))
    elif len(neighbors) < len(source_xy):
        k = len(neighbors) + 1
    else:
        k = len(neighbors)
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
    kfolds = int(round(n_samples / 20.0))
    kfolds = max(kmin, kfolds)
    kfolds = min(kmax, kfolds)
    return min(kfolds, n_samples)


def fit_svm_params(
    features: np.ndarray,
    labels: np.ndarray,
    c_values: list[float],
    gamma_values: list[float],
    kfolds: int,
    default_c: float,
    default_gamma: float,
) -> tuple[float, float]:
    if features.shape[0] < 3 or kfolds < 2:
        return default_c, default_gamma

    norm = StandardScaler().fit(features)
    features_norm = norm.transform(features)
    search = GridSearchCV(
        svm.SVR(kernel="rbf"),
        param_grid={"C": c_values, "gamma": gamma_values},
        cv=kfolds,
    )
    search.fit(features_norm, labels)
    params = search.best_params_
    return float(params["C"]), float(params["gamma"])


def compute_fpi_nce(u_transposed: np.ndarray, k_classes: int) -> tuple[float, float]:
    n_samples = u_transposed.shape[0]
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n_samples):
        for j in range(k_classes):
            value = float(u_transposed[i, j])
            sum1 += value * value
            sum2 += value * math.log(max(value, np.finfo(float).eps))

    fpi = 1.0 - (k_classes * (((1.0 / n_samples) * sum1) - 1.0) / (k_classes - 1.0))
    nce = ((-1.0 / n_samples) * sum2) / math.log(k_classes)
    return fpi, nce


def fit_variogram_params(
    xy_df: pd.DataFrame,
    z_series: pd.Series,
    cfg: dict[str, Any],
) -> tuple[str, list[float], dict[str, Any]]:
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
    use_models = cfg.get("models") or [
        "linear",
        "linear-sill",
        "exponential",
        "spherical",
        "gaussian",
    ]

    while True:
        try:
            fitted = sem.Fit(use_models)
            break
        except ValueError as exc:
            if "is infeasible" not in str(exc):
                raise
            lag_distance += 1.0
            sem.Exp_Semiv(float(lag_distance), float(active_distance))

    fixed_model = cfg.get("model")
    model = fixed_model if fixed_model else min(fitted.keys(), key=lambda name: fitted[name][3])
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


def run_ordinary_kriging(
    xy_df: pd.DataFrame,
    z_values: np.ndarray,
    query_xy: np.ndarray,
    variogram_model: str,
    variogram_parameters: list[float],
    n_neighbors: int,
    radius: float,
) -> np.ndarray:
    estimator = kriging.OrdinaryKriging(
        xy=xy_df,
        z=pd.Series(z_values),
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
    )
    predictions, _ = estimator.execute(
        query_xy,
        n_closest_points=n_neighbors,
        radius=radius,
    )
    return np.asarray(predictions, dtype=float)
