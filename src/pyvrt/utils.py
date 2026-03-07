from __future__ import annotations

import re
from typing import Any, Dict

import numpy as np


def safe_name(value: Any) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "key"


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "range": np.nan,
            "cv_pct": np.nan,
        }

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
