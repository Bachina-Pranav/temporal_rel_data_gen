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
    max_overlap_attempts: int = 5,
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
    cell_code = encode_cell_codes(customer_block, product_block, time_code)
    order = np.argsort(cell_code)
    sorted_codes = cell_code[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1] if len(sorted_codes) else np.asarray([], dtype=int)
    ends = np.r_[starts[1:], len(order)] if len(starts) else np.asarray([], dtype=int)
    exact_cells = 0
    projection_cells = 0
    repair_summary = {"num_repaired_cells": 0, "num_swaps": 0, "overlaps_before": 0, "overlaps_after": 0}
    for start_idx, end_idx in zip(starts, ends):
        indices = order[start_idx:end_idx]
        cell_customers = customers[indices]
        cell_products = products[indices]
        if len(indices) <= 1:
            continue
        gate_key = gate_lookup(time_gate_code[indices[0]], code_to_time_gate, time_code[indices[0]], code_to_time_bucket)
        if pairing_mode == "random":
            reordered = cell_products.copy()
            rng.shuffle(reordered)
            projection_cells += 1
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
        products[indices] = reordered
    counts = np.diff(np.r_[starts, len(order)]) if len(starts) else np.asarray([], dtype=int)
    summary = {
        "num_cells": int(len(starts)),
        "average_cell_size": float(np.mean(counts)) if len(counts) else 0.0,
        "max_cell_size": int(np.max(counts)) if len(counts) else 0,
        "pairing_mode": pairing_mode,
        "large_cell_pairing": large_cell_pairing,
        "num_exact_small_cells": int(exact_cells),
        "num_projection_cells": int(projection_cells),
        "dynamic_pairing_seconds": float(time.time() - start),
        **repair_summary,
    }
    return products, summary


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
