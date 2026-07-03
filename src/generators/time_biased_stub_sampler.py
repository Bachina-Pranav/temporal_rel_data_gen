"""Desired-time sampling for exact temporal stubs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


def sample_desired_times_for_stubs(
    entity_ids_repeated: Iterable[Any],
    activity_model: Any,
    rng: np.random.Generator,
    time_buckets: Optional[list[str]] = None,
) -> np.ndarray:
    """Sample one desired time code per exact entity stub with mixture sampling.

    This compatibility wrapper maps raw entity IDs to compact integer entity
    indices, then samples directly from:

        w_e * empirical_times(e) + (1 - w_e) * block_time_distribution(block(e))

    It intentionally does not build a dense entity-by-time probability vector.
    """

    if time_buckets is not None and list(time_buckets) != list(activity_model.time_buckets):
        raise ValueError("custom time_buckets are not supported by the fast mixture sampler")
    stubs = np.asarray(list(entity_ids_repeated), dtype=object)
    if len(stubs) == 0:
        return np.asarray([], dtype=np.int32)
    state = activity_model.get_fast_sampling_state()
    entity_to_index = state["entity_to_index"]
    try:
        stub_entity_idx = np.fromiter((entity_to_index[entity] for entity in stubs), dtype=np.int64, count=len(stubs))
    except KeyError as exc:
        raise KeyError(f"unknown entity in desired-time sampler: {exc}") from exc
    return sample_desired_times_for_stubs_mixture_fast(
        stub_entity_idx,
        state["entity_mix_weight"],
        state["entity_block"],
        state["empirical_offsets"],
        state["empirical_time_values"],
        state["block_time_values"],
        state["block_time_cdfs"],
        state["global_time_values"],
        state["global_time_cdf"],
        rng,
    )


def sample_desired_times_for_stubs_mixture_fast(
    stub_entity_idx: Sequence[int],
    entity_mix_weight: np.ndarray,
    entity_block: np.ndarray,
    empirical_offsets: np.ndarray,
    empirical_time_values: np.ndarray,
    block_time_values: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    block_time_cdfs: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    global_time_values: np.ndarray,
    global_time_cdf: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample desired time codes without dense entity x time probabilities."""

    stub_entity_idx = np.asarray(stub_entity_idx, dtype=np.int64)
    n = int(len(stub_entity_idx))
    desired = np.empty(n, dtype=np.int32)
    if n == 0:
        return desired

    weights = np.nan_to_num(entity_mix_weight[stub_entity_idx], nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    use_empirical = rng.random(n) < weights
    filled = np.zeros(n, dtype=bool)

    empirical_positions = np.flatnonzero(use_empirical)
    if len(empirical_positions):
        empirical_entities = stub_entity_idx[empirical_positions]
        for entity_idx, positions in grouped_positions(empirical_entities, empirical_positions):
            lo = int(empirical_offsets[int(entity_idx)])
            hi = int(empirical_offsets[int(entity_idx) + 1])
            if hi > lo:
                sampled_offsets = rng.integers(lo, hi, size=len(positions))
                desired[positions] = empirical_time_values[sampled_offsets]
                filled[positions] = True

    fallback_positions = np.flatnonzero(~filled)
    if len(fallback_positions):
        fallback_blocks = entity_block[stub_entity_idx[fallback_positions]]
        for block, positions in grouped_positions(fallback_blocks, fallback_positions):
            values, cdf = distribution_for_block(
                int(block),
                block_time_values,
                block_time_cdfs,
                global_time_values,
                global_time_cdf,
            )
            desired[positions] = sample_from_cdf(values, cdf, len(positions), rng)
            filled[positions] = True

    return desired


def grouped_positions(keys: np.ndarray, positions: np.ndarray):
    keys = np.asarray(keys, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    if len(keys) == 0:
        return
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_positions = positions[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_keys)]
    for start, end in zip(starts, ends):
        yield int(sorted_keys[start]), sorted_positions[start:end]


def distribution_for_block(
    block: int,
    block_time_values: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    block_time_cdfs: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    global_time_values: np.ndarray,
    global_time_cdf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = lookup_distribution(block_time_values, block)
    cdf = lookup_distribution(block_time_cdfs, block)
    if values is None or cdf is None or len(values) == 0 or len(cdf) == 0:
        return global_time_values, global_time_cdf
    return values, cdf


def lookup_distribution(container: Mapping[int, np.ndarray] | Sequence[np.ndarray], key: int) -> Optional[np.ndarray]:
    if isinstance(container, Mapping):
        return container.get(int(key))
    try:
        return container[int(key)]
    except (IndexError, TypeError):
        return None


def sample_from_cdf(
    values: np.ndarray,
    cdf: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32)
    cdf = np.asarray(cdf, dtype=float)
    if len(values) == 0 or len(cdf) == 0:
        return np.zeros(int(size), dtype=np.int32)
    draws = rng.random(int(size))
    idx = np.searchsorted(cdf, draws, side="right")
    idx = np.minimum(idx, len(values) - 1)
    return values[idx].astype(np.int32, copy=False)
