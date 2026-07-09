"""Foreign-key cardinality fidelity metrics for single event tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import gini, ks_distance, normalize_value, wasserstein_1d


def fk_cardinality_metrics(real: pd.DataFrame, synthetic: pd.DataFrame, table_config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_fk: dict[str, Any] = {}
    warnings: list[str] = []
    for column, cfg in (table_config.get("columns", {}) or {}).items():
        if str((cfg or {}).get("type", "")).lower() != "foreign_key":
            continue
        parent_ids, parent_used = parent_index(cfg, real, synthetic, column)
        if not parent_used:
            warnings.append(f"parent_table_missing_for_fk:{column}")
        real_counts = child_counts(real, column, parent_ids)
        syn_counts = child_counts(synthetic, column, parent_ids)
        ks = ks_distance(real_counts, syn_counts)
        metric = {
            "similarity": float(1.0 - ks) if ks is not None else None,
            "ks_distance": ks,
            "wasserstein_distance": wasserstein_1d(real_counts, syn_counts),
            "mean_abs_count_diff": float(np.mean(np.abs(real_counts - syn_counts))) if len(real_counts) else None,
            "real_mean_cardinality": float(np.mean(real_counts)) if len(real_counts) else None,
            "synthetic_mean_cardinality": float(np.mean(syn_counts)) if len(syn_counts) else None,
            "real_gini": gini(real_counts),
            "synthetic_gini": gini(syn_counts),
            "num_parent_entities_compared": int(len(parent_ids)),
            "parent_table_used": bool(parent_used),
        }
        per_fk[column] = metric
        rows.append({"fk_column": column, **metric})
    similarities = [item["similarity"] for item in per_fk.values() if item.get("similarity") is not None]
    kss = [item["ks_distance"] for item in per_fk.values() if item.get("ks_distance") is not None]
    payload = {
        "macro_similarity": float(np.mean(similarities)) if similarities else None,
        "macro_ks": float(np.mean(kss)) if kss else None,
        "per_fk": per_fk,
        "warnings": warnings,
    }
    return payload, pd.DataFrame(rows)


def parent_index(cfg: dict[str, Any], real: pd.DataFrame, synthetic: pd.DataFrame, column: str) -> tuple[pd.Index, bool]:
    parent_path = cfg.get("parent_table_path")
    ref_col = (cfg.get("references") or {}).get("column")
    if parent_path and ref_col and Path(parent_path).exists():
        parent = pd.read_csv(parent_path, usecols=[ref_col])
        return pd.Index(parent[ref_col].map(normalize_value).dropna().unique()), True
    values = pd.concat([real[column], synthetic[column]], ignore_index=True).map(normalize_value)
    return pd.Index(values.dropna().unique()), False


def child_counts(frame: pd.DataFrame, column: str, parent_ids: pd.Index) -> np.ndarray:
    counts = frame[column].map(normalize_value).value_counts()
    return counts.reindex(parent_ids, fill_value=0).to_numpy(dtype=float)
