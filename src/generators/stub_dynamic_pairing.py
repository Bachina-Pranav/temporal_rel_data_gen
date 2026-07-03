"""Dynamic product reordering for time-biased block stub matching."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .ultrafast_event_pairing import (
    reorder_products_by_exact_greedy_affinity,
    reorder_products_by_projection_affinity,
)


def reorder_products_within_cells_by_dynamic_affinity(
    slot_customer_ids: Sequence[Any],
    slot_product_ids: Sequence[Any],
    slot_customer_block: Sequence[int],
    slot_product_block: Sequence[int],
    slot_time_code: Sequence[int],
    slot_time_gate_code: Sequence[int],
    affinity_model: Any,
    pairing_mode: str,
    max_exact_affinity_cell_size: int,
    rng: np.random.Generator,
    large_cell_pairing: str = "projection_sort",
    code_to_time_gate: Optional[Sequence[str]] = None,
    code_to_time_bucket: Optional[Sequence[str]] = None,
    enable_fast_overlap_repair: bool = False,
    real_event_set: Optional[set[tuple[Any, Any, str]]] = None,
    slot_customer_idx: Optional[Sequence[int]] = None,
    slot_product_idx: Optional[Sequence[int]] = None,
    real_pair_keys: Optional[set[int]] = None,
    real_event_keys: Optional[set[int]] = None,
    num_products: Optional[int] = None,
    num_time_codes: Optional[int] = None,
    lambda_duplicate_pair: float = 1.0,
    lambda_real_pair_overlap: float = 1.0,
    lambda_exact_event_overlap: float = 3.0,
    max_overlap_attempts: int = 5,
    large_cell_local_swap_attempts: int = 2,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Reorder products inside exact cells while preserving all slot constraints."""

    if large_cell_pairing not in {"projection_sort", "exact_greedy"}:
        raise ValueError("large_cell_pairing must be projection_sort or exact_greedy")
    start = time.time()
    customers = np.asarray(slot_customer_ids, dtype=object)
    products = np.asarray(slot_product_ids, dtype=object).copy()
    customer_block = np.asarray(slot_customer_block, dtype=int)
    product_block = np.asarray(slot_product_block, dtype=int)
    time_code = np.asarray(slot_time_code, dtype=int)
    time_gate_code = np.asarray(slot_time_gate_code, dtype=int)
    customer_idx = (
        np.asarray(slot_customer_idx, dtype=np.int64)
        if slot_customer_idx is not None
        else ids_to_indices(customers, getattr(affinity_model, "customer_index", {}))
    )
    product_idx = (
        np.asarray(slot_product_idx, dtype=np.int64).copy()
        if slot_product_idx is not None
        else ids_to_indices(products, getattr(affinity_model, "product_index", {}))
    )
    if pairing_mode == "dynamic_exact_penalized":
        if num_products is None:
            num_products = int(max(product_idx.max() + 1, 1)) if len(product_idx) else 1
        if num_time_codes is None:
            num_time_codes = int(max(time_code.max() + 1, 1)) if len(time_code) else 1
        if real_pair_keys is None:
            real_pair_keys = set()
        if real_event_keys is None:
            real_event_keys = set()
        cell_code = encode_time_first_cell_codes(customer_block, product_block, time_code)
    else:
        cell_code = encode_cell_codes(customer_block, product_block, time_code)
    order = np.argsort(cell_code)
    sorted_codes = cell_code[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1] if len(sorted_codes) else np.asarray([], dtype=int)
    ends = np.r_[starts[1:], len(order)] if len(starts) else np.asarray([], dtype=int)
    exact_cells = 0
    projection_cells = 0
    penalized_cells = 0
    random_cells = 0
    random_events = 0
    exact_penalized_cells = 0
    projection_fallback_cells = 0
    exact_penalized_events = 0
    projection_fallback_events = 0
    pair_counts: Dict[int, int] = {}
    repair_summary = {"num_repaired_cells": 0, "num_swaps": 0, "overlaps_before": 0, "overlaps_after": 0}
    for start_idx, end_idx in zip(starts, ends):
        indices = order[start_idx:end_idx]
        cell_customers = customers[indices]
        cell_products = products[indices]
        cell_customer_idx = customer_idx[indices]
        cell_product_idx = product_idx[indices]
        if len(indices) <= 1:
            if pairing_mode == "dynamic_exact_penalized":
                exact_penalized_cells += int(len(indices) == 1)
                exact_penalized_events += int(len(indices))
                update_packed_pair_counts(pair_counts, cell_customer_idx, cell_product_idx, int(num_products))
            elif pairing_mode == "random":
                random_cells += int(len(indices) == 1)
                random_events += int(len(indices))
            continue
        gate_key = gate_lookup(time_gate_code[indices[0]], code_to_time_gate, time_code[indices[0]], code_to_time_bucket)
        if pairing_mode == "random":
            reordered = cell_products.copy()
            rng.shuffle(reordered)
            projection_cells += 1
            random_cells += 1
            random_events += int(len(indices))
        elif pairing_mode == "static_projection":
            reordered = reorder_products_by_projection_affinity(cell_customers, cell_products, gate_key, affinity_model, rng, dynamic=False)
            projection_cells += 1
        elif pairing_mode == "dynamic_projection":
            reordered = reorder_products_by_projection_affinity(cell_customers, cell_products, gate_key, affinity_model, rng, dynamic=True)
            projection_cells += 1
        elif pairing_mode == "dynamic_exact_small":
            if len(indices) <= int(max_exact_affinity_cell_size):
                reordered = reorder_products_by_exact_greedy_affinity(cell_customers, cell_products, gate_key, affinity_model, rng, dynamic=True)
                exact_cells += 1
            elif large_cell_pairing == "exact_greedy":
                reordered = reorder_products_by_exact_greedy_affinity(cell_customers, cell_products, gate_key, affinity_model, rng, dynamic=True)
                exact_cells += 1
            else:
                reordered = reorder_products_by_projection_affinity(cell_customers, cell_products, gate_key, affinity_model, rng, dynamic=True)
                projection_cells += 1
        elif pairing_mode == "dynamic_exact_penalized":
            if len(indices) <= int(max_exact_affinity_cell_size):
                reordered, reordered_idx = reorder_products_by_penalized_exact_greedy_affinity(
                    cell_customers,
                    cell_products,
                    cell_customer_idx,
                    cell_product_idx,
                    gate_key,
                    int(time_code[indices[0]]),
                    affinity_model,
                    rng,
                    pair_counts,
                    int(num_products),
                    int(num_time_codes),
                    real_pair_keys,
                    real_event_keys,
                    lambda_duplicate_pair,
                    lambda_real_pair_overlap,
                    lambda_exact_event_overlap,
                    max_score_matrix_size=int(max_exact_affinity_cell_size),
                )
                penalized_cells += 1
                exact_cells += 1
                exact_penalized_cells += 1
                exact_penalized_events += int(len(indices))
            else:
                reordered = reorder_products_by_projection_affinity(
                    cell_customers,
                    cell_products,
                    gate_key,
                    affinity_model,
                    rng,
                    dynamic=True,
                )
                projection_cells += 1
                projection_fallback_cells += 1
                projection_fallback_events += int(len(indices))
                if real_event_set is not None and int(large_cell_local_swap_attempts) > 0:
                    time_bucket = code_to_time_bucket[time_code[indices[0]]] if code_to_time_bucket is not None else str(time_code[indices[0]])
                    reordered, repair = repair_exact_overlaps_within_cell(
                        cell_customers,
                        reordered,
                        time_bucket,
                        real_event_set,
                        rng,
                        max_attempts=int(large_cell_local_swap_attempts),
                    )
                    if repair["num_swaps"] > 0:
                        repair_summary["num_repaired_cells"] += 1
                    repair_summary["num_swaps"] += int(repair["num_swaps"])
                    repair_summary["overlaps_before"] += int(repair["overlaps_before"])
                    repair_summary["overlaps_after"] += int(repair["overlaps_after"])
                product_lookup = getattr(affinity_model, "product_index", {})
                reordered_idx = ids_to_indices(reordered, product_lookup)
        else:
            raise ValueError("Unsupported pairing_mode")
        if enable_fast_overlap_repair and real_event_set is not None:
            time_bucket = code_to_time_bucket[time_code[indices[0]]] if code_to_time_bucket is not None else str(time_code[indices[0]])
            reordered, repair = repair_exact_overlaps_within_cell(
                cell_customers,
                reordered,
                time_bucket,
                real_event_set,
                rng,
                max_attempts=max_overlap_attempts,
            )
            if repair["num_swaps"] > 0:
                repair_summary["num_repaired_cells"] += 1
            repair_summary["num_swaps"] += int(repair["num_swaps"])
            repair_summary["overlaps_before"] += int(repair["overlaps_before"])
            repair_summary["overlaps_after"] += int(repair["overlaps_after"])
            if pairing_mode == "dynamic_exact_penalized":
                product_lookup = getattr(affinity_model, "product_index", {})
                reordered_idx = ids_to_indices(reordered, product_lookup)
        products[indices] = reordered
        if pairing_mode == "dynamic_exact_penalized":
            product_idx[indices] = reordered_idx
            update_packed_pair_counts(pair_counts, cell_customer_idx, reordered_idx, int(num_products))
    counts = np.diff(np.r_[starts, len(order)]) if len(starts) else np.asarray([], dtype=int)
    num_cells = int(len(starts))
    average_cell_size = float(np.mean(counts)) if len(counts) else 0.0
    largest_cell_size = int(np.max(counts)) if len(counts) else 0
    p95_cell_size = float(np.percentile(counts, 95.0)) if len(counts) else 0.0
    p99_cell_size = float(np.percentile(counts, 99.0)) if len(counts) else 0.0
    percent_projection_events = float(projection_fallback_events / max(len(products), 1))
    summary = {
        "num_cells": num_cells,
        "num_cells_processed": num_cells,
        "average_cell_size": average_cell_size,
        "max_cell_size": largest_cell_size,
        "largest_cell_size": largest_cell_size,
        "p95_cell_size": p95_cell_size,
        "p99_cell_size": p99_cell_size,
        "max_exact_affinity_cell_size": int(max_exact_affinity_cell_size),
        "pairing_mode": pairing_mode,
        "large_cell_pairing": large_cell_pairing,
        "num_exact_small_cells": int(exact_cells),
        "num_projection_cells": int(projection_cells),
        "num_penalized_cells": int(penalized_cells),
        "num_random_cells": int(random_cells),
        "num_exact_penalized_cells": int(exact_penalized_cells),
        "num_projection_fallback_cells": int(projection_fallback_cells),
        "num_events_exact_penalized": int(exact_penalized_events),
        "num_events_projection_fallback": int(projection_fallback_events),
        "num_events_random": int(random_events),
        "percent_cells_exact_penalized": float(exact_penalized_cells / max(num_cells, 1)),
        "percent_cells_projection_fallback": float(projection_fallback_cells / max(num_cells, 1)),
        "percent_cells_random": float(random_cells / max(num_cells, 1)),
        "percent_events_exact_penalized": float(exact_penalized_events / max(len(products), 1)),
        "percent_events_projection_fallback": percent_projection_events,
        "percent_events_random": float(random_events / max(len(products), 1)),
        "percent_large_cells_projection_sort": percent_projection_events,
        "large_cell_local_swap_attempts": int(large_cell_local_swap_attempts),
        "lambda_duplicate_pair": float(lambda_duplicate_pair),
        "lambda_real_pair_overlap": float(lambda_real_pair_overlap),
        "lambda_exact_event_overlap": float(lambda_exact_event_overlap),
        "dynamic_pairing_seconds": float(time.time() - start),
        **repair_summary,
    }
    return products, summary


def reorder_products_by_penalized_exact_greedy_affinity(
    customers: Sequence[Any],
    products: Sequence[Any],
    customer_idx: Sequence[int],
    product_idx: Sequence[int],
    time_bucket: Any,
    time_code: int,
    affinity_model: Any,
    rng: np.random.Generator,
    pair_counts: Dict[int, int],
    num_products: int,
    num_time_codes: int,
    real_pair_keys: set[int],
    real_event_keys: set[int],
    lambda_duplicate_pair: float,
    lambda_real_pair_overlap: float,
    lambda_exact_event_overlap: float,
    max_score_matrix_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    customers = np.asarray(customers, dtype=object)
    products = np.asarray(products, dtype=object)
    customer_idx = np.asarray(customer_idx, dtype=np.int64)
    product_idx = np.asarray(product_idx, dtype=np.int64)
    if len(products) <= 1:
        return products.copy(), product_idx.copy()
    if max_score_matrix_size is not None and len(products) > int(max_score_matrix_size):
        raise RuntimeError("Refusing to build exact penalized score matrix above max_score_matrix_size")
    u_vectors = affinity_model.transformed_customer_vectors(customers, time_bucket)
    v_vectors = affinity_model.product_vectors(products)
    scores = u_vectors @ v_vectors.T
    pair_keys = customer_idx[:, None] * int(num_products) + product_idx[None, :]
    if lambda_duplicate_pair != 0.0:
        duplicate_penalty = np.fromiter(
            (np.log1p(pair_counts.get(int(key), 0)) for key in pair_keys.ravel()),
            dtype=float,
            count=pair_keys.size,
        ).reshape(pair_keys.shape)
        scores = scores - float(lambda_duplicate_pair) * duplicate_penalty
    if lambda_real_pair_overlap != 0.0 and real_pair_keys:
        real_pair_penalty = np.fromiter(
            (int(int(key) in real_pair_keys) for key in pair_keys.ravel()),
            dtype=float,
            count=pair_keys.size,
        ).reshape(pair_keys.shape)
        scores = scores - float(lambda_real_pair_overlap) * real_pair_penalty
    if lambda_exact_event_overlap != 0.0 and real_event_keys:
        event_keys = pair_keys * int(num_time_codes) + int(time_code)
        exact_event_penalty = np.fromiter(
            (int(int(key) in real_event_keys) for key in event_keys.ravel()),
            dtype=float,
            count=event_keys.size,
        ).reshape(event_keys.shape)
        scores = scores - float(lambda_exact_event_overlap) * exact_event_penalty
    product_order = greedy_match_products(scores, rng)
    return products[product_order], product_idx[product_order]


def greedy_match_products(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    n_customers, n_products = scores.shape
    product_for_customer = np.full(n_customers, -1, dtype=np.int64)
    used_customers = np.zeros(n_customers, dtype=bool)
    used_products = np.zeros(n_products, dtype=bool)
    jitter = 1e-12 * rng.random(scores.size)
    for flat in np.argsort((scores.ravel() + jitter))[::-1]:
        cidx = int(flat // n_products)
        pidx = int(flat % n_products)
        if used_customers[cidx] or used_products[pidx]:
            continue
        product_for_customer[cidx] = pidx
        used_customers[cidx] = True
        used_products[pidx] = True
        if bool(used_customers.all()):
            break
    missing_customers = np.flatnonzero(~used_customers)
    missing_products = np.flatnonzero(~used_products)
    if len(missing_customers):
        rng.shuffle(missing_products)
        product_for_customer[missing_customers] = missing_products[: len(missing_customers)]
    return product_for_customer


def reorder_products_by_projection_affinity_for_test(
    customers: Sequence[Any],
    products: Sequence[Any],
    time_gate_code: Any,
    affinity_model: Any,
    dynamic: bool = True,
) -> np.ndarray:
    return reorder_products_by_projection_affinity(
        customers,
        products,
        time_gate_code,
        affinity_model,
        np.random.default_rng(0),
        dynamic=dynamic,
    )


def repair_exact_overlaps_within_cell(
    customers: Sequence[Any],
    products: Sequence[Any],
    time_bucket: str,
    real_event_set: set[tuple[Any, Any, str]],
    rng: np.random.Generator,
    max_attempts: int = 5,
) -> tuple[np.ndarray, Dict[str, int]]:
    customers = np.asarray(customers, dtype=object)
    products = np.asarray(products, dtype=object).copy()
    before = sum((customer, product, time_bucket) in real_event_set for customer, product in zip(customers, products))
    swaps = 0
    if len(products) <= 1 or before == 0:
        return products, {"overlaps_before": int(before), "overlaps_after": int(before), "num_swaps": 0}
    for idx in range(len(products)):
        if (customers[idx], products[idx], time_bucket) not in real_event_set:
            continue
        for _ in range(int(max_attempts)):
            other = int(rng.integers(0, len(products)))
            if other == idx:
                continue
            before_pair = int((customers[idx], products[idx], time_bucket) in real_event_set) + int(
                (customers[other], products[other], time_bucket) in real_event_set
            )
            after_pair = int((customers[idx], products[other], time_bucket) in real_event_set) + int(
                (customers[other], products[idx], time_bucket) in real_event_set
            )
            if after_pair < before_pair:
                products[idx], products[other] = products[other], products[idx]
                swaps += 1
                break
    after = sum((customer, product, time_bucket) in real_event_set for customer, product in zip(customers, products))
    return products, {"overlaps_before": int(before), "overlaps_after": int(after), "num_swaps": int(swaps)}


def encode_cell_codes(customer_block: np.ndarray, product_block: np.ndarray, time_code: np.ndarray) -> np.ndarray:
    max_product = int(product_block.max()) + 1 if len(product_block) else 1
    max_time = int(time_code.max()) + 1 if len(time_code) else 1
    return (customer_block.astype(np.int64) * max_product + product_block.astype(np.int64)) * max_time + time_code.astype(np.int64)


def encode_time_first_cell_codes(customer_block: np.ndarray, product_block: np.ndarray, time_code: np.ndarray) -> np.ndarray:
    max_customer = int(customer_block.max()) + 1 if len(customer_block) else 1
    max_product = int(product_block.max()) + 1 if len(product_block) else 1
    return (time_code.astype(np.int64) * max_customer + customer_block.astype(np.int64)) * max_product + product_block.astype(np.int64)


def ids_to_indices(ids: Sequence[Any], index: Dict[Any, int]) -> np.ndarray:
    return np.fromiter((int(index[item]) for item in ids), dtype=np.int64, count=len(ids))


def update_packed_pair_counts(
    pair_counts: Dict[int, int],
    customer_idx: Sequence[int],
    product_idx: Sequence[int],
    num_products: int,
) -> None:
    keys = np.asarray(customer_idx, dtype=np.int64) * int(num_products) + np.asarray(product_idx, dtype=np.int64)
    for key in keys:
        packed = int(key)
        pair_counts[packed] = pair_counts.get(packed, 0) + 1


def gate_lookup(
    gate_code: int,
    code_to_time_gate: Optional[Sequence[str]],
    time_code: int,
    code_to_time_bucket: Optional[Sequence[str]],
) -> Any:
    if code_to_time_gate is not None:
        return code_to_time_gate[int(gate_code)]
    if code_to_time_bucket is not None:
        return code_to_time_bucket[int(time_code)]
    return int(gate_code)
