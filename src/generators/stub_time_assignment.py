"""Exact time-biased stub-to-slot assignment."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .time_biased_stub_sampler import sample_desired_times_for_stubs


def assign_stubs_to_slots_by_time(
    entity_ids: Sequence[Any],
    entity_degrees: Mapping[Any, int],
    entity_blocks: Mapping[Any, int],
    slot_blocks: Sequence[int],
    slot_time_codes: Sequence[int],
    activity_model: Any,
    rng: np.random.Generator,
    jitter: float = 1e-3,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Assign exact entity stubs to slots through time-biased sorting."""

    start = time.time()
    slot_blocks = np.asarray(slot_blocks, dtype=int)
    slot_time_codes = np.asarray(slot_time_codes, dtype=int)
    assigned = np.empty(len(slot_blocks), dtype=object)
    entity_ids = np.asarray(list(entity_ids), dtype=object)
    desired_actual_distances: list[float] = []
    max_mismatch = 0
    blocks = sorted(set(slot_blocks.tolist()).union(int(entity_blocks.get(entity, 0)) for entity in entity_ids))
    for block in blocks:
        slot_idx = np.where(slot_blocks == int(block))[0]
        block_entities = [entity for entity in entity_ids if int(entity_blocks.get(entity, 0)) == int(block)]
        degrees = np.asarray([int(entity_degrees.get(entity, 0)) for entity in block_entities], dtype=int)
        stubs = np.repeat(np.asarray(block_entities, dtype=object), degrees)
        mismatch = int(len(stubs) - len(slot_idx))
        max_mismatch = max(max_mismatch, abs(mismatch))
        if mismatch != 0:
            raise ValueError(
                f"Stub/slot mismatch for block {block}: {len(stubs)} stubs but {len(slot_idx)} slots"
            )
        if len(stubs) == 0:
            continue
        desired_codes = sample_desired_times_for_stubs(stubs, activity_model, rng)
        slot_order = np.argsort(slot_time_codes[slot_idx] + float(jitter) * rng.random(len(slot_idx)))
        stub_order = np.argsort(desired_codes + float(jitter) * rng.random(len(stubs)))
        ordered_slots = slot_idx[slot_order]
        ordered_stubs = stubs[stub_order]
        assigned[ordered_slots] = ordered_stubs
        actual_codes = slot_time_codes[ordered_slots]
        desired_actual_distances.extend(np.abs(desired_codes[stub_order] - actual_codes).astype(float).tolist())
    summary = {
        "num_blocks": int(len(blocks)),
        "num_stubs": int(len(assigned)),
        "exact_degree_preserved": True,
        "max_block_stub_slot_mismatch": int(max_mismatch),
        "mean_abs_desired_actual_time_distance": float(np.mean(desired_actual_distances)) if desired_actual_distances else 0.0,
        "median_abs_desired_actual_time_distance": float(np.median(desired_actual_distances)) if desired_actual_distances else 0.0,
        "assignment_seconds": float(time.time() - start),
    }
    return assigned, summary
