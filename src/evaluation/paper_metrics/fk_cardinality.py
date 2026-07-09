"""Foreign-key cardinality fidelity metrics for single event tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import gini, ks_distance, normalize_value, wasserstein_1d


def fk_cardinality_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    table_config: dict[str, Any],
    row_count_match: bool | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    per_fk: dict[str, Any] = {}
    warnings: list[str] = []
    if row_count_match is None:
        row_count_match = len(real) == len(synthetic)
    for column, cfg in (table_config.get("columns", {}) or {}).items():
        if str((cfg or {}).get("type", "")).lower() != "foreign_key":
            continue
        parent_ids, parent_used = parent_index(cfg, real, synthetic, column)
        if not parent_used:
            warnings.append(f"parent_table_missing_for_fk:{column}")
        real_counts = child_counts(real, column, parent_ids)
        syn_counts = child_counts(synthetic, column, parent_ids)
        real_norm = real_counts / max(float(len(real)), 1.0)
        syn_norm = syn_counts / max(float(len(synthetic)), 1.0)
        absolute_ks = ks_distance(real_counts, syn_counts)
        normalized_ks = ks_distance(real_norm, syn_norm)
        if row_count_match:
            ks = absolute_ks
            similarity = float(1.0 - absolute_ks) if absolute_ks is not None else None
        else:
            ks = normalized_ks
            similarity = float(1.0 - normalized_ks) if normalized_ks is not None else None
            warnings.append(f"absolute_fk_cardinality_row_count_confounded:{column}")
        metric = {
            "similarity": similarity,
            "ks_distance": ks,
            "absolute_similarity": float(1.0 - absolute_ks) if absolute_ks is not None else None,
            "absolute_ks_distance": absolute_ks,
            "normalized_similarity": float(1.0 - normalized_ks) if normalized_ks is not None else None,
            "normalized_ks_distance": normalized_ks,
            "wasserstein_distance": wasserstein_1d(real_counts, syn_counts),
            "mean_abs_count_diff": float(np.mean(np.abs(real_counts - syn_counts))) if len(real_counts) else None,
            "real_mean_cardinality": float(np.mean(real_counts)) if len(real_counts) else None,
            "synthetic_mean_cardinality": float(np.mean(syn_counts)) if len(syn_counts) else None,
            "real_normalized_mean_cardinality": float(np.mean(real_norm)) if len(real_norm) else None,
            "synthetic_normalized_mean_cardinality": float(np.mean(syn_norm)) if len(syn_norm) else None,
            "real_gini": gini(real_counts),
            "synthetic_gini": gini(syn_counts),
            "num_parent_entities_compared": int(len(parent_ids)),
            "parent_table_used": bool(parent_used),
        }
        per_fk[column] = metric
        rows.append({"fk_column": column, **metric})
    similarities = [item["similarity"] for item in per_fk.values() if item.get("similarity") is not None]
    kss = [item["ks_distance"] for item in per_fk.values() if item.get("ks_distance") is not None]
    absolute_similarities = [item["absolute_similarity"] for item in per_fk.values() if item.get("absolute_similarity") is not None]
    normalized_similarities = [item["normalized_similarity"] for item in per_fk.values() if item.get("normalized_similarity") is not None]
    payload = {
        "macro_similarity": float(np.mean(similarities)) if similarities else None,
        "macro_ks": float(np.mean(kss)) if kss else None,
        "macro_absolute_similarity": float(np.mean(absolute_similarities)) if absolute_similarities else None,
        "macro_normalized_similarity": float(np.mean(normalized_similarities)) if normalized_similarities else None,
        "row_count_match": bool(row_count_match),
        "per_fk": per_fk,
        "warnings": sorted(set(warnings)),
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
