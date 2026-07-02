"""Fast quota sampling, low-rank batch pairing, and cheap repair helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence

import numpy as np


def sample_entities_with_quotas(
    entity_ids: Sequence[Any],
    remaining_degrees: Mapping[Any, int],
    activity_probs: Sequence[float],
    n: int,
    rng: np.random.Generator,
    eps: float = 1e-12,
    max_attempts: int = 5,
) -> tuple[np.ndarray, bool, Dict[str, Any]]:
    """Sample n entity stubs weighted by remaining quota and temporal activity."""

    entity_ids = np.asarray(entity_ids, dtype=object)
    activity_probs = np.asarray(activity_probs, dtype=float)
    if len(entity_ids) != len(activity_probs):
        raise ValueError("entity_ids and activity_probs must have the same length")
    n = int(n)
    if n == 0:
        return np.asarray([], dtype=object), True, {
            "requested_n": 0,
            "sampled_n": 0,
            "num_available_entities": 0,
            "total_available_quota": 0,
            "fallback_used": False,
        }
    remaining = np.asarray([int(remaining_degrees.get(entity, 0)) for entity in entity_ids], dtype=int)
    available_mask = remaining > 0
    total_quota = int(remaining[available_mask].sum())
    diagnostics = {
        "requested_n": n,
        "sampled_n": 0,
        "num_available_entities": int(available_mask.sum()),
        "total_available_quota": total_quota,
        "fallback_used": False,
    }
    if total_quota < n:
        diagnostics["reason"] = "insufficient_remaining_quota"
        return np.asarray([], dtype=object), False, diagnostics
    available_ids = entity_ids[available_mask]
    available_remaining = remaining[available_mask]
    probs = np.nan_to_num(activity_probs[available_mask], nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    weights = available_remaining.astype(float) * np.clip(probs, eps, None)
    if not np.isfinite(weights).all() or float(weights.sum()) <= eps:
        weights = available_remaining.astype(float)
        diagnostics["fallback_used"] = True
    samples = []
    local_counts = np.zeros(len(available_ids), dtype=int)
    needed = n
    for _ in range(max_attempts):
        if needed <= 0:
            break
        residual = available_remaining - local_counts
        valid = residual > 0
        if not bool(valid.any()):
            break
        draw_weights = weights.copy()
        draw_weights[~valid] = 0.0
        if draw_weights.sum() <= eps:
            draw_weights = residual.astype(float)
            draw_weights[~valid] = 0.0
            diagnostics["fallback_used"] = True
        draw_size = min(max(needed * 2, needed), int(residual.sum()))
        draw_probs = draw_weights / np.clip(draw_weights.sum(), eps, None)
        draws = rng.choice(len(available_ids), size=draw_size, replace=True, p=draw_probs)
        for draw in draws:
            if local_counts[draw] >= available_remaining[draw]:
                continue
            samples.append(available_ids[draw])
            local_counts[draw] += 1
            needed -= 1
            if needed <= 0:
                break
    if needed > 0:
        residual = available_remaining - local_counts
        order = np.argsort(-(weights + residual * eps))
        for idx in order:
            while residual[idx] > 0 and needed > 0:
                samples.append(available_ids[idx])
                residual[idx] -= 1
                needed -= 1
            if needed <= 0:
                break
    diagnostics["sampled_n"] = int(len(samples))
    success = len(samples) == n
    if not success:
        diagnostics["reason"] = "quota_sampling_exhausted_after_attempts"
    return np.asarray(samples, dtype=object), bool(success), diagnostics


def pair_stubs_by_dynamic_affinity(
    customers: Sequence[Any],
    products: Sequence[Any],
    time_bucket: Any,
    affinity_model: Any,
    rng: np.random.Generator,
    max_exact_cell_size: int = 512,
    large_cell_pairing: str = "projection_sort",
    nearest_neighbor_topk: int = 10,
) -> np.ndarray:
    """Return a product permutation paired to customers by low-rank dynamic affinity."""

    customers = np.asarray(customers, dtype=object)
    products = np.asarray(products, dtype=object)
    n = len(customers)
    if n != len(products):
        raise ValueError("customers and products must have the same length")
    if n <= 1:
        return products.copy()
    if n <= int(max_exact_cell_size):
        return _pair_exact_greedy(customers, products, time_bucket, affinity_model, rng)
    if large_cell_pairing == "nearest_neighbor":
        return _pair_nearest_neighbor(customers, products, time_bucket, affinity_model, rng, nearest_neighbor_topk)
    if large_cell_pairing != "projection_sort":
        raise ValueError("large_cell_pairing must be 'projection_sort' or 'nearest_neighbor'")
    return _pair_projection_sort(customers, products, time_bucket, affinity_model)


def _pair_exact_greedy(
    customers: np.ndarray,
    products: np.ndarray,
    time_bucket: Any,
    affinity_model: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    u_vectors = affinity_model.transformed_customer_vectors(customers, time_bucket)
    v_vectors = affinity_model.product_vectors(products)
    scores = u_vectors @ v_vectors.T
    n = len(customers)
    output = np.empty(n, dtype=object)
    used_customers = np.zeros(n, dtype=bool)
    used_products = np.zeros(n, dtype=bool)
    order = np.argsort(scores.ravel())[::-1]
    for flat in order:
        cidx = int(flat // n)
        pidx = int(flat % n)
        if used_customers[cidx] or used_products[pidx]:
            continue
        output[cidx] = products[pidx]
        used_customers[cidx] = True
        used_products[pidx] = True
        if bool(used_customers.all()):
            break
    missing_customers = np.where(~used_customers)[0]
    missing_products = np.where(~used_products)[0]
    if len(missing_customers):
        shuffled = missing_products.copy()
        rng.shuffle(shuffled)
        for cidx, pidx in zip(missing_customers, shuffled):
            output[cidx] = products[pidx]
    return output


def _pair_projection_sort(
    customers: np.ndarray,
    products: np.ndarray,
    time_bucket: Any,
    affinity_model: Any,
) -> np.ndarray:
    u_vectors = affinity_model.transformed_customer_vectors(customers, time_bucket)
    v_vectors = affinity_model.product_vectors(products)
    if len(u_vectors) == 0:
        return products.copy()
    q = np.mean(u_vectors, axis=0) + np.mean(v_vectors, axis=0)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12:
        q = np.ones(u_vectors.shape[1], dtype=float)
        norm = float(np.linalg.norm(q))
    q = q / max(norm, 1e-12)
    customer_order = np.argsort(u_vectors @ q)
    product_order = np.argsort(v_vectors @ q)
    output = np.empty(len(products), dtype=object)
    output[customer_order] = products[product_order]
    return output


def _pair_nearest_neighbor(
    customers: np.ndarray,
    products: np.ndarray,
    time_bucket: Any,
    affinity_model: Any,
    rng: np.random.Generator,
    topk: int,
) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors

        u_vectors = affinity_model.transformed_customer_vectors(customers, time_bucket)
        v_vectors = affinity_model.product_vectors(products)
        n_neighbors = max(1, min(int(topk), len(products)))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(v_vectors)
        _, indices = nn.kneighbors(u_vectors)
        output = np.empty(len(products), dtype=object)
        used_products = np.zeros(len(products), dtype=bool)
        customer_order = np.arange(len(customers))
        rng.shuffle(customer_order)
        for cidx in customer_order:
            chosen = None
            for pidx in indices[cidx]:
                if not used_products[pidx]:
                    chosen = int(pidx)
                    break
            if chosen is None:
                remaining = np.where(~used_products)[0]
                chosen = int(rng.choice(remaining))
            output[cidx] = products[chosen]
            used_products[chosen] = True
        return output
    except Exception:
        return _pair_projection_sort(customers, products, time_bucket, affinity_model)


def repair_bad_pairs_within_cell(
    customers: Sequence[Any],
    products: Sequence[Any],
    time_bucket: Any,
    real_event_set: set[tuple[Any, Any, str]],
    pair_counts: Mapping[tuple[Any, Any], int],
    rng: np.random.Generator,
    max_attempts: int = 10,
) -> np.ndarray:
    """Repair exact real-event overlaps and repeated pairs through cheap swaps."""

    customers = np.asarray(customers, dtype=object)
    products = np.asarray(products, dtype=object).copy()
    if len(products) <= 1:
        return products
    for idx in range(len(products)):
        current = pair_badness(customers[idx], products[idx], time_bucket, real_event_set, pair_counts)
        if current <= 0:
            continue
        for _ in range(int(max_attempts)):
            other = int(rng.integers(0, len(products)))
            if other == idx:
                continue
            before = current + pair_badness(customers[other], products[other], time_bucket, real_event_set, pair_counts)
            after = (
                pair_badness(customers[idx], products[other], time_bucket, real_event_set, pair_counts)
                + pair_badness(customers[other], products[idx], time_bucket, real_event_set, pair_counts)
            )
            if after < before:
                products[idx], products[other] = products[other], products[idx]
                current = pair_badness(customers[idx], products[idx], time_bucket, real_event_set, pair_counts)
                if current <= 0:
                    break
    return products


def pair_badness(
    customer: Any,
    product: Any,
    time_bucket: Any,
    real_event_set: set[tuple[Any, Any, str]],
    pair_counts: Mapping[tuple[Any, Any], int],
) -> int:
    badness = int((customer, product, time_bucket) in real_event_set)
    badness += int(pair_counts.get((customer, product), 0))
    return badness


def update_pair_counts(pair_counts: Counter, customers: Sequence[Any], products: Sequence[Any]) -> None:
    for customer, product in zip(customers, products):
        pair_counts[(customer, product)] += 1
