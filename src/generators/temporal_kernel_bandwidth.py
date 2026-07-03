"""Temporal kernel bandwidth estimation for local desired-time sampling."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def estimate_temporal_kernel_bandwidths(
    entity_offsets: np.ndarray,
    entity_time_values: np.ndarray,
    entity_blocks: np.ndarray,
    num_blocks: int,
    default_bandwidth: float = 7.0,
    bandwidth_mode: str = "auto_block_iqr",
    bandwidth_scale: float = 0.25,
    min_bandwidth: float = 1.0,
    max_bandwidth: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate local temporal noise bandwidths from real inter-event gaps.

    All inputs are compact integer arrays from FastTemporalActivityModel. The
    estimator uses only real event times and never looks at synthetic metrics.
    """

    if bandwidth_mode not in {"auto_block_iqr", "auto_global_iqr", "fixed"}:
        raise ValueError("bandwidth_mode must be auto_block_iqr, auto_global_iqr, or fixed")
    entity_offsets = np.asarray(entity_offsets, dtype=np.int64)
    entity_time_values = np.asarray(entity_time_values, dtype=np.int32)
    entity_blocks = np.asarray(entity_blocks, dtype=np.int64)
    if len(entity_offsets) != len(entity_blocks) + 1:
        raise ValueError("entity_offsets must have length num_entities + 1")

    num_entities = int(len(entity_blocks))
    num_blocks = int(max(num_blocks, int(entity_blocks.max()) + 1 if len(entity_blocks) else 0, 1))
    default_bandwidth = float(default_bandwidth)
    bandwidth_scale = float(bandwidth_scale)
    min_bandwidth = float(min_bandwidth)
    max_bandwidth_value = None if max_bandwidth is None else float(max_bandwidth)

    if bandwidth_mode == "fixed":
        fixed = clip_bandwidth(default_bandwidth, min_bandwidth, max_bandwidth_value)
        entity_bandwidths = np.full(num_entities, fixed, dtype=float)
        block_bandwidths = {int(block): fixed for block in range(num_blocks)}
        return {
            "entity_bandwidths": entity_bandwidths,
            "block_bandwidths": block_bandwidths,
            "global_bandwidth": float(fixed),
            "diagnostics": diagnostics(entity_bandwidths, block_bandwidths, fixed, bandwidth_mode, bandwidth_scale, min_bandwidth, max_bandwidth_value),
        }

    entity_gaps: list[np.ndarray] = []
    block_gaps: Dict[int, list[np.ndarray]] = {int(block): [] for block in range(num_blocks)}
    for entity_idx in range(num_entities):
        lo = int(entity_offsets[entity_idx])
        hi = int(entity_offsets[entity_idx + 1])
        gaps = inter_event_gaps(entity_time_values[lo:hi])
        entity_gaps.append(gaps)
        if len(gaps):
            block_gaps.setdefault(int(entity_blocks[entity_idx]), []).append(gaps)

    all_gaps = concatenate_nonempty(entity_gaps)
    global_raw = robust_gap_scale(all_gaps, default_bandwidth / max(bandwidth_scale, 1e-12))
    global_bandwidth = clip_bandwidth(bandwidth_scale * global_raw, min_bandwidth, max_bandwidth_value)

    block_bandwidths: Dict[int, float] = {}
    for block in range(num_blocks):
        gaps = concatenate_nonempty(block_gaps.get(int(block), []))
        if len(gaps):
            raw = robust_gap_scale(gaps, global_raw)
            block_bandwidths[int(block)] = clip_bandwidth(bandwidth_scale * raw, min_bandwidth, max_bandwidth_value)
        else:
            block_bandwidths[int(block)] = float(global_bandwidth)

    entity_bandwidths = np.empty(num_entities, dtype=float)
    for entity_idx in range(num_entities):
        gaps = entity_gaps[entity_idx]
        if bandwidth_mode == "auto_global_iqr":
            fallback = float(global_bandwidth)
        else:
            fallback = float(block_bandwidths.get(int(entity_blocks[entity_idx]), global_bandwidth))
        if len(gaps) >= 2:
            raw = robust_gap_scale(gaps, fallback / max(bandwidth_scale, 1e-12))
            entity_bandwidths[entity_idx] = clip_bandwidth(bandwidth_scale * raw, min_bandwidth, max_bandwidth_value)
        else:
            entity_bandwidths[entity_idx] = fallback

    return {
        "entity_bandwidths": entity_bandwidths,
        "block_bandwidths": block_bandwidths,
        "global_bandwidth": float(global_bandwidth),
        "diagnostics": diagnostics(entity_bandwidths, block_bandwidths, global_bandwidth, bandwidth_mode, bandwidth_scale, min_bandwidth, max_bandwidth_value),
    }


def inter_event_gaps(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.int32)
    if len(times) < 2:
        return np.asarray([], dtype=float)
    ordered = np.sort(times.astype(np.int64, copy=False))
    gaps = np.diff(ordered).astype(float)
    return gaps[np.isfinite(gaps)]


def concatenate_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(array, dtype=float) for array in arrays if len(array)]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=float)


def robust_gap_scale(gaps: np.ndarray, fallback: float) -> float:
    gaps = np.asarray(gaps, dtype=float)
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) == 0:
        return float(fallback)
    q25, q75 = np.percentile(gaps, [25.0, 75.0])
    iqr = float(q75 - q25)
    median = float(np.median(gaps))
    if np.isfinite(iqr) and iqr > 0.0:
        return iqr
    if np.isfinite(median) and median > 0.0:
        return median
    return float(fallback)


def clip_bandwidth(value: float, min_bandwidth: float, max_bandwidth: Optional[float]) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        value = float(min_bandwidth)
    value = max(value, float(min_bandwidth))
    if max_bandwidth is not None:
        value = min(value, float(max_bandwidth))
    return float(value)


def diagnostics(
    entity_bandwidths: np.ndarray,
    block_bandwidths: Dict[int, float],
    global_bandwidth: float,
    bandwidth_mode: str,
    bandwidth_scale: float,
    min_bandwidth: float,
    max_bandwidth: Optional[float],
) -> Dict[str, Any]:
    values = np.asarray(entity_bandwidths, dtype=float)
    if len(values):
        summary = {
            "min": float(np.min(values)),
            "p25": float(np.percentile(values, 25.0)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75.0)),
            "max": float(np.max(values)),
        }
    else:
        summary = {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    return {
        "bandwidth_mode": bandwidth_mode,
        "bandwidth_scale": float(bandwidth_scale),
        "min_bandwidth": float(min_bandwidth),
        "max_bandwidth": None if max_bandwidth is None else float(max_bandwidth),
        "global_bandwidth": float(global_bandwidth),
        "block_bandwidths": {str(int(block)): float(value) for block, value in sorted(block_bandwidths.items())},
        "entity_bandwidth_summary": summary,
        "bandwidth_selection_uses_synthetic_metrics": False,
    }
