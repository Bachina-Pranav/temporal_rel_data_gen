"""Projection-based product reordering for ultrafast event-spine pairing."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def reorder_products_by_projection_affinity(
    customers: Sequence[Any],
    products: Sequence[Any],
    time_bucket: Any,
    affinity_model: Any,
    rng: np.random.Generator,
    dynamic: bool = True,
) -> np.ndarray:
    """Reorder products by low-rank projection affinity without an n x n score matrix."""

    customers = np.asarray(customers, dtype=object)
    products = np.asarray(products, dtype=object)
    if len(products) <= 1:
        return products.copy()
    u_vectors = (
        affinity_model.transformed_customer_vectors(customers, time_bucket)
        if dynamic
        else customer_vectors(affinity_model, customers)
    )
    v_vectors = affinity_model.product_vectors(products)
    if len(u_vectors) == 0:
        output = products.copy()
        rng.shuffle(output)
        return output
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


def reorder_products_by_exact_greedy_affinity(
    customers: Sequence[Any],
    products: Sequence[Any],
    time_bucket: Any,
    affinity_model: Any,
    rng: np.random.Generator,
    dynamic: bool = True,
) -> np.ndarray:
    """Greedy exact small-cell matching for optional ablations."""

    customers = np.asarray(customers, dtype=object)
    products = np.asarray(products, dtype=object)
    if len(products) <= 1:
        return products.copy()
    u_vectors = (
        affinity_model.transformed_customer_vectors(customers, time_bucket)
        if dynamic
        else customer_vectors(affinity_model, customers)
    )
    v_vectors = affinity_model.product_vectors(products)
    scores = u_vectors @ v_vectors.T
    n = len(products)
    output = np.empty(n, dtype=object)
    used_customers = np.zeros(n, dtype=bool)
    used_products = np.zeros(n, dtype=bool)
    for flat in np.argsort(scores.ravel())[::-1]:
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
        rng.shuffle(missing_products)
        for cidx, pidx in zip(missing_customers, missing_products):
            output[cidx] = products[pidx]
    return output


def reorder_products_for_cell(
    customers: Sequence[Any],
    products: Sequence[Any],
    time_bucket: Any,
    affinity_model: Any,
    rng: np.random.Generator,
    pairing_mode: str = "dynamic_projection",
    max_exact_affinity_cell_size: int = 128,
) -> np.ndarray:
    products = np.asarray(products, dtype=object)
    if len(products) <= 1:
        return products.copy()
    if pairing_mode == "random":
        output = products.copy()
        rng.shuffle(output)
        return output
    if pairing_mode == "static_projection":
        return reorder_products_by_projection_affinity(customers, products, time_bucket, affinity_model, rng, dynamic=False)
    if pairing_mode == "dynamic_projection":
        return reorder_products_by_projection_affinity(customers, products, time_bucket, affinity_model, rng, dynamic=True)
    if pairing_mode == "dynamic_exact_small":
        if len(products) <= int(max_exact_affinity_cell_size):
            return reorder_products_by_exact_greedy_affinity(customers, products, time_bucket, affinity_model, rng, dynamic=True)
        return reorder_products_by_projection_affinity(customers, products, time_bucket, affinity_model, rng, dynamic=True)
    raise ValueError("pairing_mode must be random, static_projection, dynamic_projection, or dynamic_exact_small")


def repair_cell_pairs_by_swaps(
    customers: Sequence[Any],
    products: Sequence[Any],
    time_bucket: Any,
    real_event_set: set[tuple[Any, Any, str]],
    pair_counts: Mapping[tuple[Any, Any], int],
    rng: np.random.Generator,
    max_attempts: int = 10,
) -> tuple[np.ndarray, dict[str, int]]:
    customers = np.asarray(customers, dtype=object)
    products = np.asarray(products, dtype=object).copy()
    before_bad = sum(pair_badness(c, p, time_bucket, real_event_set, pair_counts) for c, p in zip(customers, products))
    swaps = 0
    if len(products) <= 1:
        return products, {"badness_before": int(before_bad), "badness_after": int(before_bad), "num_swaps": 0}
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
                swaps += 1
                current = pair_badness(customers[idx], products[idx], time_bucket, real_event_set, pair_counts)
                if current <= 0:
                    break
    after_bad = sum(pair_badness(c, p, time_bucket, real_event_set, pair_counts) for c, p in zip(customers, products))
    return products, {"badness_before": int(before_bad), "badness_after": int(after_bad), "num_swaps": int(swaps)}


def pair_badness(
    customer: Any,
    product: Any,
    time_bucket: Any,
    real_event_set: set[tuple[Any, Any, str]],
    pair_counts: Mapping[tuple[Any, Any], int],
) -> int:
    return int((customer, product, time_bucket) in real_event_set) + int(pair_counts.get((customer, product), 0))


def customer_vectors(affinity_model: Any, customer_ids: Sequence[Any]) -> np.ndarray:
    ids = np.asarray(customer_ids, dtype=object)
    rank = int(getattr(affinity_model, "rank", 1))
    vectors = np.zeros((len(ids), rank), dtype=float)
    index = getattr(affinity_model, "customer_index", {})
    embeddings = getattr(affinity_model, "customer_embeddings", np.zeros((0, rank), dtype=float))
    for pos, customer in enumerate(ids):
        idx = index.get(customer)
        if idx is not None:
            vectors[pos] = embeddings[idx]
    return vectors


def update_pair_counts(pair_counts: Counter, customers: Sequence[Any], products: Sequence[Any]) -> None:
    for customer, product in zip(customers, products):
        pair_counts[(customer, product)] += 1
