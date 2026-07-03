"""Desired-time sampling for exact temporal stubs."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np


def sample_desired_times_for_stubs(
    entity_ids_repeated: Iterable[Any],
    activity_model: Any,
    rng: np.random.Generator,
    time_buckets: Optional[list[str]] = None,
) -> np.ndarray:
    """Sample one desired time code per exact entity stub.

    The implementation loops over unique entities, not over events. Each
    entity's repeated stubs are sampled vectorially from its smoothed temporal
    activity distribution.
    """

    stubs = np.asarray(list(entity_ids_repeated), dtype=object)
    buckets = list(time_buckets if time_buckets is not None else activity_model.time_buckets)
    if len(stubs) == 0:
        return np.asarray([], dtype=int)
    if not buckets:
        return np.zeros(len(stubs), dtype=int)
    output = np.empty(len(stubs), dtype=int)
    unique_entities, inverse = np.unique(stubs, return_inverse=True)
    for entity_pos, entity in enumerate(unique_entities):
        positions = np.where(inverse == entity_pos)[0]
        probs = desired_time_probabilities(entity, activity_model, buckets)
        output[positions] = rng.choice(len(buckets), size=len(positions), replace=True, p=probs)
    return output


def desired_time_probabilities(entity_id: Any, activity_model: Any, time_buckets: list[str]) -> np.ndarray:
    probs = np.asarray([activity_model.probability(entity_id, bucket) for bucket in time_buckets], dtype=float)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    if float(probs.sum()) > 1e-12:
        return probs / probs.sum()
    block = int(activity_model.entity_block.get(entity_id, 0))
    block_probs = np.asarray(activity_model.block_time_probs.get(block, []), dtype=float)
    if len(block_probs) == len(time_buckets) and float(block_probs.sum()) > 1e-12:
        return block_probs / block_probs.sum()
    global_probs = np.asarray(activity_model.global_time_probs, dtype=float)
    if len(global_probs) == len(time_buckets) and float(global_probs.sum()) > 1e-12:
        return global_probs / global_probs.sum()
    return np.ones(len(time_buckets), dtype=float) / max(len(time_buckets), 1)
