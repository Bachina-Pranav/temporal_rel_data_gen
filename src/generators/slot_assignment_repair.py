"""Degree repair for slot-based temporal event-spine assignment."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, Mapping, Sequence

import numpy as np


def repair_entity_degrees_by_replacement(
    slot_entity_ids: np.ndarray,
    slot_blocks: Sequence[int],
    slot_times: Sequence[Any],
    target_degrees: Mapping[Any, int],
    entity_blocks: Mapping[Any, int],
    activity_model: Any,
    rng: np.random.Generator,
    max_passes: int = 3,
    allow_degree_slack: bool = False,
    eps: float = 1e-12,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Repair assigned entity degrees by replacing overfull entities within blocks."""

    start = time.time()
    slot_entity_ids = np.asarray(slot_entity_ids, dtype=object)
    slot_blocks = np.asarray(slot_blocks, dtype=int)
    slot_times = np.asarray(slot_times, dtype=object)
    target = {entity: int(count) for entity, count in target_degrees.items()}
    current = Counter(slot_entity_ids.tolist())
    before = degree_error_summary(current, target)
    replacements = 0
    entities_by_block: Dict[int, list[Any]] = {}
    for entity, block in entity_blocks.items():
        if entity in target:
            entities_by_block.setdefault(int(block), []).append(entity)
    for _ in range(int(max_passes)):
        changed = False
        for block, entities in entities_by_block.items():
            block_slots = np.where(slot_blocks == int(block))[0]
            if len(block_slots) == 0:
                continue
            over = {entity for entity in entities if current.get(entity, 0) > target.get(entity, 0)}
            under = [entity for entity in entities if current.get(entity, 0) < target.get(entity, 0)]
            if not over or not under:
                continue
            for under_entity in under:
                deficit = target.get(under_entity, 0) - current.get(under_entity, 0)
                while deficit > 0:
                    over_slots = np.asarray(
                        [
                            idx
                            for idx in block_slots
                            if slot_entity_ids[idx] in over
                            and current.get(slot_entity_ids[idx], 0) > target.get(slot_entity_ids[idx], 0)
                        ],
                        dtype=int,
                    )
                    if len(over_slots) == 0:
                        break
                    chosen_slots = choose_replacement_slots(
                        over_slots,
                        slot_times,
                        under_entity,
                        min(deficit, len(over_slots)),
                        activity_model,
                        rng,
                        eps=eps,
                    )
                    made = 0
                    for slot_idx in chosen_slots:
                        old_entity = slot_entity_ids[slot_idx]
                        if current.get(old_entity, 0) <= target.get(old_entity, 0):
                            continue
                        slot_entity_ids[slot_idx] = under_entity
                        current[old_entity] -= 1
                        current[under_entity] += 1
                        replacements += 1
                        made += 1
                        changed = True
                        if current.get(under_entity, 0) >= target.get(under_entity, 0):
                            break
                    deficit = target.get(under_entity, 0) - current.get(under_entity, 0)
                    over = {entity for entity in entities if current.get(entity, 0) > target.get(entity, 0)}
                    if made == 0 or not over:
                        break
        if not changed or degree_error_summary(current, target)["l1_error"] == 0:
            break
    after = degree_error_summary(current, target)
    summary = {
        "l1_error_before": int(before["l1_error"]),
        "l1_error_after": int(after["l1_error"]),
        "max_abs_error_before": int(before["max_abs_error"]),
        "max_abs_error_after": int(after["max_abs_error"]),
        "num_replacements": int(replacements),
        "num_unresolved_entities": int(after["num_unresolved_entities"]),
        "repair_seconds": float(time.time() - start),
    }
    if after["l1_error"] != 0 and not allow_degree_slack:
        raise RuntimeError(
            "Degree repair could not satisfy exact targets: "
            f"L1 after={after['l1_error']}, max_abs after={after['max_abs_error']}"
        )
    return slot_entity_ids, summary


def choose_replacement_slots(
    candidate_slots: np.ndarray,
    slot_times: np.ndarray,
    under_entity: Any,
    count: int,
    activity_model: Any,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> np.ndarray:
    if int(count) <= 0 or len(candidate_slots) == 0:
        return np.asarray([], dtype=int)
    unique_times = np.unique(slot_times[candidate_slots])
    probs_by_time = {
        time_bucket: float(activity_model.probability(under_entity, time_bucket))
        for time_bucket in unique_times
    }
    weights = np.asarray([probs_by_time.get(slot_times[idx], eps) for idx in candidate_slots], dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    if float(weights.sum()) <= eps:
        weights = np.ones(len(candidate_slots), dtype=float)
    sample_size = min(int(count), len(candidate_slots))
    return rng.choice(candidate_slots, size=sample_size, replace=False, p=weights / weights.sum())


def degree_error_summary(current: Mapping[Any, int], target: Mapping[Any, int]) -> Dict[str, int]:
    keys = set(current).union(target)
    errors = [int(current.get(key, 0)) - int(target.get(key, 0)) for key in keys]
    abs_errors = [abs(error) for error in errors]
    return {
        "l1_error": int(sum(abs_errors)),
        "max_abs_error": int(max(abs_errors) if abs_errors else 0),
        "num_unresolved_entities": int(sum(error != 0 for error in errors)),
    }
