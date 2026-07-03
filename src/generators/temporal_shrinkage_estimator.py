"""Held-out temporal shrinkage estimation for time-biased event stubs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .fast_temporal_activity import canonical_time_bucket, resolve_auto_alpha


DEFAULT_CANDIDATE_ALPHAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def estimate_temporal_shrinkage_alpha(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    block_map: Mapping[Any, int],
    candidate_alphas: Optional[Iterable[float]] = None,
    min_degree_for_holdout: int = 2,
    holdout_per_entity: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Select shrinkage alpha by held-out timestamp likelihood.

    The model is fitted only on training timestamps:

        P_e(t) = w_e P_empirical_e(t) + (1 - w_e) P_block(e)(t)

    with w_e = degree_train_e / (degree_train_e + alpha). Evaluation touches
    only held-out timestamps, so complexity is linear in heldout events times
    the candidate grid, not entities times all time buckets.
    """

    if candidate_alphas is None:
        candidate_alphas = DEFAULT_CANDIDATE_ALPHAS
    candidates = [float(alpha) for alpha in candidate_alphas]
    if not candidates:
        raise ValueError("candidate_alphas must contain at least one value")
    if min(candidates) < 0.0:
        raise ValueError("candidate_alphas must be non-negative")
    if min_degree_for_holdout < 1:
        raise ValueError("min_degree_for_holdout must be >= 1")
    if holdout_per_entity < 0:
        raise ValueError("holdout_per_entity must be >= 0")

    frame = df[[entity_col, time_col]].copy()
    frame[time_col] = canonical_time_bucket(frame[time_col], granularity="day")
    entities = sorted(frame[entity_col].drop_duplicates().tolist(), key=stable_sort_key)
    entity_times: Dict[Any, list[str]] = {entity: [] for entity in entities}
    for entity, bucket in zip(frame[entity_col].to_numpy(dtype=object), frame[time_col].to_numpy(dtype=object)):
        entity_times.setdefault(entity, []).append(str(bucket))

    all_degrees = [len(times) for times in entity_times.values()]
    fallback_alpha = resolve_auto_alpha("auto", all_degrees)
    rng = np.random.default_rng(int(seed))

    train_counters: Dict[Any, Counter] = {}
    heldout_times: Dict[Any, list[str]] = {}
    block_train_counters: Dict[int, Counter] = {}
    block_train_totals: Counter = Counter()
    global_train_counter: Counter = Counter()
    global_train_total = 0

    for entity in entities:
        times = entity_times[entity]
        degree = len(times)
        if degree >= int(min_degree_for_holdout) and int(holdout_per_entity) > 0:
            holdout_count = min(int(holdout_per_entity), max(degree - 1, 0))
            holdout_indices = set(rng.choice(degree, size=holdout_count, replace=False).tolist())
        else:
            holdout_indices = set()

        train_times = [time for idx, time in enumerate(times) if idx not in holdout_indices]
        heldout = [time for idx, time in enumerate(times) if idx in holdout_indices]
        train_counter = Counter(train_times)
        train_counters[entity] = train_counter
        heldout_times[entity] = heldout

        block = int(block_map.get(entity, 0))
        block_counter = block_train_counters.setdefault(block, Counter())
        block_counter.update(train_counter)
        block_total = int(sum(train_counter.values()))
        block_train_totals[block] += block_total
        global_train_counter.update(train_counter)
        global_train_total += block_total

    num_holdout = int(sum(len(values) for values in heldout_times.values()))
    if num_holdout == 0:
        return {
            "best_alpha": float(fallback_alpha),
            "candidate_results": [],
            "num_entities": int(len(entities)),
            "num_holdout_events": 0,
            "fallback_used": True,
            "avg_log_likelihood": None,
            "alpha_candidate_grid": candidates,
            "fallback_alpha": float(fallback_alpha),
            "num_likelihood_evaluations": 0,
        }

    eps = 1e-12
    candidate_results = []
    best_alpha = candidates[0]
    best_avg_ll = -float("inf")
    likelihood_evaluations = 0

    for alpha in candidates:
        total_ll = 0.0
        for entity in entities:
            entity_heldout = heldout_times.get(entity, [])
            if not entity_heldout:
                continue
            train_counter = train_counters.get(entity, Counter())
            degree_train = int(sum(train_counter.values()))
            weight = float(degree_train / (degree_train + alpha)) if degree_train + alpha > 0.0 else 0.0
            block = int(block_map.get(entity, 0))
            block_counter = block_train_counters.get(block, Counter())
            block_total = int(block_train_totals.get(block, 0))
            for heldout_time in entity_heldout:
                empirical_prob = float(train_counter.get(heldout_time, 0) / degree_train) if degree_train > 0 else 0.0
                if block_total > 0:
                    block_prob = float(block_counter.get(heldout_time, 0) / block_total)
                elif global_train_total > 0:
                    block_prob = float(global_train_counter.get(heldout_time, 0) / global_train_total)
                else:
                    block_prob = 0.0
                probability = max(weight * empirical_prob + (1.0 - weight) * block_prob, eps)
                total_ll += float(np.log(probability))
                likelihood_evaluations += 1
        avg_ll = float(total_ll / max(num_holdout, 1))
        candidate_results.append(
            {
                "alpha": float(alpha),
                "avg_log_likelihood": avg_ll,
                "num_holdout_events": int(num_holdout),
            }
        )
        if avg_ll > best_avg_ll:
            best_avg_ll = avg_ll
            best_alpha = float(alpha)

    return {
        "best_alpha": float(best_alpha),
        "candidate_results": candidate_results,
        "num_entities": int(len(entities)),
        "num_holdout_events": int(num_holdout),
        "fallback_used": False,
        "avg_log_likelihood": float(best_avg_ll),
        "alpha_candidate_grid": candidates,
        "fallback_alpha": float(fallback_alpha),
        "num_likelihood_evaluations": int(likelihood_evaluations),
    }


def stable_sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))
