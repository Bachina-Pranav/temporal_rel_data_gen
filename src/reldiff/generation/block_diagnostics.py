"""Block-pair diagnostics shared by temporal structure generators and evaluators."""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .continuous_time_temporal_sbm import empirical_ks_statistic


BLOCK_METADATA_WARNING = (
    "Block assignment metadata not provided; block-pair diagnostics skipped."
)


def load_block_map(path: str | Path, id_col: str, block_col: str) -> Dict[Any, int]:
    df = pd.read_csv(path)
    missing = [column for column in (id_col, block_col) if column not in df.columns]
    if missing:
        raise ValueError(f"Block assignment file {path} is missing columns: {missing}")
    return dict(zip(df[id_col], df[block_col].astype(int)))


def find_block_files(debug_dir: str | Path) -> Tuple[Optional[Path], Optional[Path]]:
    debug_dir = Path(debug_dir)
    customer_candidates = [
        debug_dir / "customer_blocks.csv",
        debug_dir / "ct_2k_sbm_temporal_stubs_customer_assignments.csv",
        debug_dir / "temporal_sbm_customer_assignments.csv",
    ]
    product_candidates = [
        debug_dir / "product_blocks.csv",
        debug_dir / "ct_2k_sbm_temporal_stubs_product_assignments.csv",
        debug_dir / "temporal_sbm_product_assignments.csv",
    ]
    customer_path = next((path for path in customer_candidates if path.exists()), None)
    product_path = next((path for path in product_candidates if path.exists()), None)
    return customer_path, product_path


def load_block_maps(
    customer_blocks_path: str | Path,
    product_blocks_path: str | Path,
    customer_col: str,
    product_col: str,
) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    return (
        load_block_map(customer_blocks_path, customer_col, "customer_block"),
        load_block_map(product_blocks_path, product_col, "product_block"),
    )


def load_block_maps_from_debug_dir(
    debug_dir: str | Path,
    customer_col: str,
    product_col: str,
) -> Tuple[Optional[Dict[Any, int]], Optional[Dict[Any, int]], Optional[Path], Optional[Path]]:
    customer_path, product_path = find_block_files(debug_dir)
    if customer_path is None or product_path is None:
        return None, None, customer_path, product_path
    customer_blocks, product_blocks = load_block_maps(
        customer_path, product_path, customer_col, product_col
    )
    return customer_blocks, product_blocks, customer_path, product_path


def missing_block_diagnostics(warn: bool = True) -> Dict[str, Any]:
    if warn:
        warnings.warn(BLOCK_METADATA_WARNING)
    return {
        "num_customer_blocks": None,
        "num_product_blocks": None,
        "num_total_blocks": None,
        "num_possible_block_pairs": None,
        "num_nonzero_block_pairs_real": None,
        "num_nonzero_block_pairs_synthetic": None,
        "block_pair_count_exact_match_rate": None,
        "block_pair_count_num_pairs_real": None,
        "block_pair_count_num_pairs_synthetic": None,
        "block_pair_count_num_pairs_union": None,
        "block_pair_count_abs_error_sum": None,
        "block_pair_count_max_abs_error": None,
        "block_pair_count_l1_relative_error": None,
        "block_pair_timestamp_ks_mean": None,
        "block_pair_timestamp_ks_median": None,
        "block_pair_timestamp_ks_weighted_mean": None,
        "block_pair_timestamp_ks_num_pairs": None,
        "block_pair_timestamp_ks_skipped_pairs": None,
        "block_pair_timestamp_ks_min_count": None,
        "block_diagnostic_warnings": [BLOCK_METADATA_WARNING],
    }


def annotate_with_blocks(
    df: pd.DataFrame,
    customer_block_map: Dict[Any, int],
    product_block_map: Dict[Any, int],
    customer_col: str,
    product_col: str,
) -> pd.DataFrame:
    annotated = df.copy()
    annotated["customer_block"] = annotated[customer_col].map(customer_block_map)
    annotated["product_block"] = annotated[product_col].map(product_block_map)
    return annotated.dropna(subset=["customer_block", "product_block"]).copy()


def block_pair_counts(
    df: pd.DataFrame,
    customer_block_map: Dict[Any, int],
    product_block_map: Dict[Any, int],
    customer_col: str,
    product_col: str,
) -> pd.Series:
    annotated = annotate_with_blocks(
        df, customer_block_map, product_block_map, customer_col, product_col
    )
    if annotated.empty:
        return pd.Series(dtype=int)
    counts = annotated.groupby(["customer_block", "product_block"]).size()
    counts.index = pd.MultiIndex.from_tuples(
        [(int(a), int(b)) for a, b in counts.index],
        names=["customer_block", "product_block"],
    )
    return counts.sort_index()


def compute_block_pair_count_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    customer_block_map: Dict[Any, int],
    product_block_map: Dict[Any, int],
    customer_col: str,
    product_col: str,
) -> Dict[str, Any]:
    real_counts = block_pair_counts(
        real_df, customer_block_map, product_block_map, customer_col, product_col
    )
    synth_counts = block_pair_counts(
        synth_df, customer_block_map, product_block_map, customer_col, product_col
    )
    union_index = real_counts.index.union(synth_counts.index)
    if len(union_index) == 0:
        return {
            "block_pair_count_exact_match_rate": None,
            "block_pair_count_num_pairs_real": 0,
            "block_pair_count_num_pairs_synthetic": 0,
            "block_pair_count_num_pairs_union": 0,
            "block_pair_count_abs_error_sum": None,
            "block_pair_count_max_abs_error": None,
            "block_pair_count_l1_relative_error": None,
        }
    real_aligned = real_counts.reindex(union_index, fill_value=0).astype(int)
    synth_aligned = synth_counts.reindex(union_index, fill_value=0).astype(int)
    abs_error = (real_aligned - synth_aligned).abs()
    real_total = int(real_aligned.sum())
    return {
        "block_pair_count_exact_match_rate": float((abs_error == 0).mean()),
        "block_pair_count_num_pairs_real": int((real_counts > 0).sum()),
        "block_pair_count_num_pairs_synthetic": int((synth_counts > 0).sum()),
        "block_pair_count_num_pairs_union": int(len(union_index)),
        "block_pair_count_abs_error_sum": int(abs_error.sum()),
        "block_pair_count_max_abs_error": int(abs_error.max()),
        "block_pair_count_l1_relative_error": float(abs_error.sum() / real_total)
        if real_total
        else None,
    }


def compute_block_pair_timestamp_ks(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    customer_block_map: Dict[Any, int],
    product_block_map: Dict[Any, int],
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    min_count: int = 5,
) -> Dict[str, Any]:
    real = annotate_with_blocks(
        real_df, customer_block_map, product_block_map, customer_col, product_col
    )
    synth = annotate_with_blocks(
        synth_df, customer_block_map, product_block_map, customer_col, product_col
    )
    real[timestamp_col] = pd.to_datetime(real[timestamp_col], errors="coerce")
    synth[timestamp_col] = pd.to_datetime(synth[timestamp_col], errors="coerce")
    real = real.dropna(subset=[timestamp_col])
    synth = synth.dropna(subset=[timestamp_col])

    real_groups = {
        (int(a), int(b)): group
        for (a, b), group in real.groupby(["customer_block", "product_block"], sort=True)
    }
    synth_groups = {
        (int(a), int(b)): group
        for (a, b), group in synth.groupby(["customer_block", "product_block"], sort=True)
    }
    block_pairs = sorted(set(real_groups) | set(synth_groups))

    ks_values = []
    weights = []
    skipped = 0
    for block_pair in block_pairs:
        real_group = real_groups.get(block_pair)
        synth_group = synth_groups.get(block_pair)
        real_count = 0 if real_group is None else len(real_group)
        synth_count = 0 if synth_group is None else len(synth_group)
        if real_count < min_count or synth_count < min_count:
            skipped += 1
            continue
        real_values = timestamp_values(real_group[timestamp_col])
        synth_values = timestamp_values(synth_group[timestamp_col])
        ks = empirical_ks_statistic(real_values, synth_values)
        if ks is None:
            skipped += 1
            continue
        ks_values.append(float(ks))
        weights.append(float(real_count))

    return {
        "block_pair_timestamp_ks_mean": float(np.mean(ks_values))
        if ks_values
        else None,
        "block_pair_timestamp_ks_median": float(np.median(ks_values))
        if ks_values
        else None,
        "block_pair_timestamp_ks_weighted_mean": float(np.average(ks_values, weights=weights))
        if ks_values
        else None,
        "block_pair_timestamp_ks_num_pairs": int(len(ks_values)),
        "block_pair_timestamp_ks_skipped_pairs": int(skipped),
        "block_pair_timestamp_ks_min_count": int(min_count),
    }


def timestamp_values(timestamps: pd.Series) -> np.ndarray:
    return pd.to_datetime(timestamps).astype("int64").to_numpy(dtype=float)


def compute_block_overview(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    customer_block_map: Dict[Any, int],
    product_block_map: Dict[Any, int],
    customer_col: str,
    product_col: str,
) -> Dict[str, Any]:
    customer_blocks = sorted(set(customer_block_map.values()))
    product_blocks = sorted(set(product_block_map.values()))
    real_counts = block_pair_counts(
        real_df, customer_block_map, product_block_map, customer_col, product_col
    )
    synth_counts = block_pair_counts(
        synth_df, customer_block_map, product_block_map, customer_col, product_col
    )
    customer_sizes = Counter(customer_block_map.values())
    product_sizes = Counter(product_block_map.values())
    return {
        "num_customer_blocks": int(len(customer_blocks)),
        "num_product_blocks": int(len(product_blocks)),
        "num_total_blocks": int(len(customer_blocks) + len(product_blocks)),
        "num_possible_block_pairs": int(len(customer_blocks) * len(product_blocks)),
        "num_nonzero_block_pairs_real": int((real_counts > 0).sum()),
        "num_nonzero_block_pairs_synthetic": int((synth_counts > 0).sum()),
        **block_size_summary("customer", customer_sizes.values()),
        **block_size_summary("product", product_sizes.values()),
    }


def block_size_summary(prefix: str, sizes: Iterable[int]) -> Dict[str, Any]:
    values = np.asarray(list(sizes), dtype=float)
    if len(values) == 0:
        return {
            f"{prefix}_block_size_min": None,
            f"{prefix}_block_size_median": None,
            f"{prefix}_block_size_max": None,
        }
    return {
        f"{prefix}_block_size_min": int(values.min()),
        f"{prefix}_block_size_median": float(np.median(values)),
        f"{prefix}_block_size_max": int(values.max()),
    }


def compute_all_block_diagnostics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    customer_block_map: Dict[Any, int],
    product_block_map: Dict[Any, int],
    customer_col: str,
    product_col: str,
    timestamp_col: str,
    min_count: int = 5,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    diagnostics.update(
        compute_block_overview(
            real_df, synth_df, customer_block_map, product_block_map, customer_col, product_col
        )
    )
    diagnostics.update(
        compute_block_pair_count_metrics(
            real_df, synth_df, customer_block_map, product_block_map, customer_col, product_col
        )
    )
    diagnostics.update(
        compute_block_pair_timestamp_ks(
            real_df,
            synth_df,
            customer_block_map,
            product_block_map,
            customer_col,
            product_col,
            timestamp_col,
            min_count=min_count,
        )
    )
    diagnostics["block_diagnostic_warnings"] = block_diagnostic_warnings(diagnostics)
    return diagnostics


def block_diagnostic_warnings(diagnostics: Dict[str, Any]) -> list[str]:
    warnings_list = []
    if (
        diagnostics.get("num_customer_blocks") == 1
        and diagnostics.get("num_product_blocks") == 1
    ):
        warnings_list.append(
            "SBM inferred only one customer block and one product block; "
            "block-pair model is effectively global."
        )
    if (
        diagnostics.get("num_nonzero_block_pairs_real") == 1
        and (diagnostics.get("num_possible_block_pairs") or 0) > 1
    ):
        warnings_list.append(
            "Only one block pair has events despite multiple inferred blocks. "
            "Check block assignment/type mapping."
        )
    if (
        diagnostics.get("block_pair_timestamp_ks_num_pairs") == 1
        and (diagnostics.get("num_nonzero_block_pairs_real") or 0) > 1
    ):
        warnings_list.append(
            "Evaluator computed timestamp KS for only one block pair despite "
            "multiple nonzero block pairs. This indicates a diagnostic bug."
        )
    if diagnostics.get("block_pair_count_exact_match_rate") is None:
        warnings_list.append(
            "Block pair count metric should not be null when block metadata exists."
        )
    return warnings_list


def block_assignment_frame(
    block_map: Dict[Any, int],
    id_col: str,
    block_col: str,
    real_df: pd.DataFrame,
    synth_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    real_counts = Counter(real_df[id_col]) if id_col in real_df.columns else Counter()
    synth_counts = (
        Counter(synth_df[id_col])
        if synth_df is not None and id_col in synth_df.columns
        else Counter()
    )
    rows = []
    for entity_id, block in sorted(block_map.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                id_col: entity_id,
                block_col: int(block),
                "real_event_count": int(real_counts.get(entity_id, 0)),
                "synthetic_event_count": int(synth_counts.get(entity_id, 0)),
            }
        )
    return pd.DataFrame(rows)


def block_pair_counts_frame(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    customer_block_map: Dict[Any, int],
    product_block_map: Dict[Any, int],
    customer_col: str,
    product_col: str,
) -> pd.DataFrame:
    real = annotate_with_blocks(
        real_df, customer_block_map, product_block_map, customer_col, product_col
    )
    synth = annotate_with_blocks(
        synth_df, customer_block_map, product_block_map, customer_col, product_col
    )
    real_counts = real.groupby(["customer_block", "product_block"]).size()
    synth_counts = synth.groupby(["customer_block", "product_block"]).size()
    union_index = real_counts.index.union(synth_counts.index)
    rows = []
    for customer_block, product_block in union_index:
        real_group = real[
            (real["customer_block"] == customer_block)
            & (real["product_block"] == product_block)
        ]
        synth_group = synth[
            (synth["customer_block"] == customer_block)
            & (synth["product_block"] == product_block)
        ]
        real_count = int(real_counts.get((customer_block, product_block), 0))
        synth_count = int(synth_counts.get((customer_block, product_block), 0))
        rows.append(
            {
                "customer_block": int(customer_block),
                "product_block": int(product_block),
                "real_event_count": real_count,
                "synthetic_event_count": synth_count,
                "abs_error": abs(real_count - synth_count),
                "real_unique_customers": int(real_group[customer_col].nunique()),
                "synthetic_unique_customers": int(synth_group[customer_col].nunique()),
                "real_unique_products": int(real_group[product_col].nunique()),
                "synthetic_unique_products": int(synth_group[product_col].nunique()),
            }
        )
    return pd.DataFrame(rows)
