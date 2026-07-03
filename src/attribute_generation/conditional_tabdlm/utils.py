"""Small shared utilities for conditional TABDLM experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(jsonable(data), handle, sort_keys=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(jsonable(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_pickle(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_dataframe(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
        return
    if path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            frame.to_pickle(path)
        return
    if path.suffix.lower() in {".pkl", ".pickle"}:
        frame.to_pickle(path)
        return
    raise ValueError(f"Unsupported table format: {path}")


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def safe_corr(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ks_statistic(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])
    if len(x) == 0 or len(y) == 0:
        return None
    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def distribution_l1(real: pd.Series, synthetic: pd.Series) -> float:
    real_counts = real.astype(str).value_counts(normalize=True)
    syn_counts = synthetic.astype(str).value_counts(normalize=True)
    index = real_counts.index.union(syn_counts.index)
    return float(np.abs(real_counts.reindex(index, fill_value=0.0) - syn_counts.reindex(index, fill_value=0.0)).sum())


def js_divergence(real: pd.Series, synthetic: pd.Series) -> float:
    real_counts = real.astype(str).value_counts(normalize=True)
    syn_counts = synthetic.astype(str).value_counts(normalize=True)
    index = real_counts.index.union(syn_counts.index)
    p = real_counts.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    q = syn_counts.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    m = 0.5 * (p + q)
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))

