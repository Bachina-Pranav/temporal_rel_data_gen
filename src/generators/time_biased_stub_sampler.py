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


def sample_desired_times_for_stubs_local_kernel(
    stub_entity_idx: Sequence[int],
    empirical_offsets: np.ndarray,
    empirical_time_values: np.ndarray,
    entity_block: np.ndarray,
    block_time_values: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    block_time_cdfs: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    global_time_values: np.ndarray,
    global_time_cdf: np.ndarray,
    num_time_codes: int,
    rng: np.random.Generator,
    bandwidth_mode: str = "auto_block_iqr",
    bandwidth_scale: float = 0.25,
    min_bandwidth: float = 1.0,
    max_bandwidth: Optional[float] = None,
    kernel: str = "discrete_laplace",
    fallback_mode: str = "block",
    entity_bandwidths: Optional[np.ndarray] = None,
    block_bandwidths: Optional[Mapping[int, float] | Sequence[float]] = None,
    global_bandwidth: float = 7.0,
) -> np.ndarray:
    """Sample desired time codes near each entity's observed event times.

    This sampler loops over unique entities appearing in the exact stubs. It
    never constructs dense entity-time probabilities and does not use pandas.
    """

    del bandwidth_mode, bandwidth_scale, min_bandwidth, max_bandwidth
    if kernel not in {"discrete_laplace", "discrete_gaussian", "none"}:
        raise ValueError("kernel must be discrete_laplace, discrete_gaussian, or none")
    if fallback_mode not in {"block", "global"}:
        raise ValueError("fallback_mode must be block or global")

    stub_entity_idx = np.asarray(stub_entity_idx, dtype=np.int64)
    empirical_offsets = np.asarray(empirical_offsets, dtype=np.int64)
    empirical_time_values = np.asarray(empirical_time_values, dtype=np.int32)
    entity_block = np.asarray(entity_block, dtype=np.int64)
    n = int(len(stub_entity_idx))
    desired = np.empty(n, dtype=np.int32)
    if n == 0:
        return desired

    positions = np.arange(n, dtype=np.int64)
    upper = max(int(num_time_codes) - 1, 0)
    for entity_idx, entity_positions in grouped_positions(stub_entity_idx, positions):
        lo = int(empirical_offsets[int(entity_idx)])
        hi = int(empirical_offsets[int(entity_idx) + 1])
        if hi > lo:
            sampled_offsets = rng.integers(lo, hi, size=len(entity_positions))
            base_times = empirical_time_values[sampled_offsets].astype(np.int32, copy=False)
            bandwidth = lookup_bandwidth(
                int(entity_idx),
                int(entity_block[int(entity_idx)]),
                entity_bandwidths,
                block_bandwidths,
                global_bandwidth,
            )
            noise = sample_discrete_noise(len(entity_positions), bandwidth, kernel, rng)
            desired[entity_positions] = np.clip(base_times.astype(np.int64) + noise.astype(np.int64), 0, upper).astype(np.int32)
            continue

        if fallback_mode == "block":
            values, cdf = distribution_for_block(
                int(entity_block[int(entity_idx)]),
                block_time_values,
                block_time_cdfs,
                global_time_values,
                global_time_cdf,
            )
        else:
            values, cdf = global_time_values, global_time_cdf
        desired[entity_positions] = np.clip(sample_from_cdf(values, cdf, len(entity_positions), rng), 0, upper).astype(np.int32)

    return desired


def sample_discrete_noise(
    size: int,
    bandwidth: float,
    kernel: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if kernel == "none" or float(bandwidth) <= 0.0:
        return np.zeros(int(size), dtype=np.int32)
    if kernel == "discrete_gaussian":
        noise = rng.normal(loc=0.0, scale=float(bandwidth), size=int(size))
    else:
        noise = rng.laplace(loc=0.0, scale=float(bandwidth), size=int(size))
    return np.rint(noise).astype(np.int32)


def lookup_bandwidth(
    entity_idx: int,
    block: int,
    entity_bandwidths: Optional[np.ndarray],
    block_bandwidths: Optional[Mapping[int, float] | Sequence[float]],
    global_bandwidth: float,
) -> float:
    if entity_bandwidths is not None and 0 <= int(entity_idx) < len(entity_bandwidths):
        value = float(entity_bandwidths[int(entity_idx)])
        if np.isfinite(value) and value > 0.0:
            return value
    if block_bandwidths is not None:
        if isinstance(block_bandwidths, Mapping):
            value = block_bandwidths.get(int(block))
        else:
            try:
                value = block_bandwidths[int(block)]
            except (IndexError, TypeError):
                value = None
        if value is not None and np.isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    return float(global_bandwidth)


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
