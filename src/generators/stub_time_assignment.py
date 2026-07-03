"""Exact time-biased stub-to-slot assignment."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .time_biased_stub_sampler import (
    sample_desired_times_for_stubs_local_kernel,
    sample_desired_times_for_stubs_mixture_fast,
)


def assign_stubs_to_slots_by_time(
    entity_ids: Sequence[Any],
    entity_degrees: Mapping[Any, int],
    entity_blocks: Mapping[Any, int],
    slot_blocks: Sequence[int],
    slot_time_codes: Sequence[int],
    activity_model: Any,
    rng: np.random.Generator,
    jitter: float = 1e-3,
    log_label: str = "stubs",
    return_entity_indices: bool = False,
    desired_time_sampling_mode: str = "mixture_shrinkage",
    local_kernel_state: Dict[str, Any] | None = None,
    mixture_sampling_state: Dict[str, Any] | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Assign exact entity stubs to slots through integer time-biased sorting."""

    start = time.time()
    if desired_time_sampling_mode not in {"mixture_shrinkage", "empirical_bayes", "local_kernel", "empirical_exact"}:
        raise ValueError("unsupported desired_time_sampling_mode")
    slot_blocks = np.asarray(slot_blocks, dtype=np.int64)
    slot_time_codes = np.asarray(slot_time_codes, dtype=np.int32)
    state = mixture_sampling_state or activity_model.get_fast_sampling_state()
    model_entity_ids = np.asarray(state["entity_ids"], dtype=object)
    entity_to_index = state["entity_to_index"]
    requested_entity_ids = np.asarray(list(entity_ids), dtype=object)
    requested_indices = np.fromiter(
        (entity_to_index[entity] for entity in requested_entity_ids),
        dtype=np.int64,
        count=len(requested_entity_ids),
    )
    target_degrees = np.zeros(len(model_entity_ids), dtype=np.int64)
    for entity in requested_entity_ids:
        target_degrees[int(entity_to_index[entity])] = int(entity_degrees.get(entity, 0))

    entity_block_arr = np.asarray(state["entity_block"], dtype=np.int64)
    assigned_idx = np.empty(len(slot_blocks), dtype=np.int64)
    desired_actual_distances: list[float] = []
    max_mismatch = 0
    block_summaries = []
    active_entity_indices = requested_indices[target_degrees[requested_indices] > 0]
    blocks = sorted(set(slot_blocks.tolist()).union(entity_block_arr[active_entity_indices].astype(int).tolist()))

    for block in blocks:
        block_total_start = time.time()
        slot_idx = np.flatnonzero(slot_blocks == int(block))
        entity_idx_b = active_entity_indices[entity_block_arr[active_entity_indices] == int(block)]
        repeat_start = time.time()
        stubs = np.repeat(entity_idx_b.astype(np.int64), target_degrees[entity_idx_b])
        repeat_seconds = float(time.time() - repeat_start)
        mismatch = int(len(stubs) - len(slot_idx))
        max_mismatch = max(max_mismatch, abs(mismatch))
        if mismatch != 0:
            raise ValueError(
                f"Stub/slot mismatch for block {block}: {len(stubs)} stubs but {len(slot_idx)} slots"
            )
        if len(stubs) == 0:
            block_summary = {
                "block": int(block),
                "slots": 0,
                "stubs": 0,
                "repeat_seconds": repeat_seconds,
                "sample_desired_seconds": 0.0,
                "sort_seconds": 0.0,
                "assign_seconds": 0.0,
                "block_seconds": float(time.time() - block_total_start),
            }
            block_summaries.append(block_summary)
            print_block_timing(log_label, block_summary)
            continue

        sample_start = time.time()
        if desired_time_sampling_mode in {"local_kernel", "empirical_exact"}:
            kernel_state = local_kernel_state or {}
            desired_codes = sample_desired_times_for_stubs_local_kernel(
                stubs,
                state["empirical_offsets"],
                state["empirical_time_values"],
                entity_block_arr,
                state["block_time_values"],
                state["block_time_cdfs"],
                state["global_time_values"],
                state["global_time_cdf"],
                int(kernel_state.get("num_time_codes", len(activity_model.time_buckets))),
                rng,
                bandwidth_mode=kernel_state.get("bandwidth_mode", "auto_block_iqr"),
                bandwidth_scale=float(kernel_state.get("bandwidth_scale", 0.25)),
                min_bandwidth=float(kernel_state.get("min_bandwidth", 1.0)),
                max_bandwidth=kernel_state.get("max_bandwidth"),
                kernel="none" if desired_time_sampling_mode == "empirical_exact" else kernel_state.get("kernel", "discrete_laplace"),
                fallback_mode=kernel_state.get("fallback_mode", "block"),
                entity_bandwidths=kernel_state.get("entity_bandwidths"),
                block_bandwidths=kernel_state.get("block_bandwidths"),
                global_bandwidth=float(kernel_state.get("global_bandwidth", 7.0)),
            )
        else:
            desired_codes = sample_desired_times_for_stubs_mixture_fast(
                stubs,
                state["entity_mix_weight"],
                entity_block_arr,
                state["empirical_offsets"],
                state["empirical_time_values"],
                state["block_time_values"],
                state["block_time_cdfs"],
                state["global_time_values"],
                state["global_time_cdf"],
                rng,
            )
        sample_seconds = float(time.time() - sample_start)

        sort_start = time.time()
        slot_sort_key = slot_time_codes[slot_idx].astype(float, copy=False)
        slot_sort_key = slot_sort_key + float(jitter) * rng.random(len(slot_idx))
        stub_sort_key = desired_codes.astype(float, copy=False)
        stub_sort_key = stub_sort_key + float(jitter) * rng.random(len(stubs))
        slot_order = np.argsort(slot_sort_key, kind="mergesort")
        stub_order = np.argsort(stub_sort_key, kind="mergesort")
        sort_seconds = float(time.time() - sort_start)

        assign_start = time.time()
        ordered_slots = slot_idx[slot_order]
        ordered_stubs = stubs[stub_order]
        assigned_idx[ordered_slots] = ordered_stubs
        actual_codes = slot_time_codes[ordered_slots]
        desired_actual_distances.extend(np.abs(desired_codes[stub_order] - actual_codes).astype(float).tolist())
        assign_seconds = float(time.time() - assign_start)

        block_summary = {
            "block": int(block),
            "slots": int(len(slot_idx)),
            "stubs": int(len(stubs)),
            "repeat_seconds": repeat_seconds,
            "sample_desired_seconds": sample_seconds,
            "sort_seconds": sort_seconds,
            "assign_seconds": assign_seconds,
            "block_seconds": float(time.time() - block_total_start),
        }
        block_summaries.append(block_summary)
        print_block_timing(log_label, block_summary)

    summary = {
        "num_blocks": int(len(blocks)),
        "num_stubs": int(len(assigned_idx)),
        "exact_degree_preserved": True,
        "max_block_stub_slot_mismatch": int(max_mismatch),
        "mean_abs_desired_actual_time_distance": float(np.mean(desired_actual_distances)) if desired_actual_distances else 0.0,
        "median_abs_desired_actual_time_distance": float(np.median(desired_actual_distances)) if desired_actual_distances else 0.0,
        "stub_construction_seconds": float(sum(item["repeat_seconds"] for item in block_summaries)),
        "desired_time_sampling_seconds": float(sum(item["sample_desired_seconds"] for item in block_summaries)),
        "sorting_seconds": float(sum(item["sort_seconds"] for item in block_summaries)),
        "slot_assignment_seconds": float(sum(item["assign_seconds"] for item in block_summaries)),
        "assignment_seconds": float(time.time() - start),
        "block_timings": block_summaries,
        "returns_entity_indices": bool(return_entity_indices),
        "desired_time_sampling_mode": desired_time_sampling_mode,
    }
    if return_entity_indices:
        return assigned_idx, summary
    return model_entity_ids[assigned_idx], summary


def print_block_timing(label: str, summary: Mapping[str, Any]) -> None:
    print(
        f"[{label}] block={summary['block']} slots={summary['slots']:,} "
        f"stubs={summary['stubs']:,} repeat={summary['repeat_seconds']:.4f}s "
        f"sample_desired={summary['sample_desired_seconds']:.4f}s "
        f"sort={summary['sort_seconds']:.4f}s assign={summary['assign_seconds']:.4f}s",
        flush=True,
    )
