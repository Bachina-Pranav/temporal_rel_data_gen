from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.stub_dynamic_pairing import (  # noqa: E402
    encode_cell_codes,
    reorder_products_by_projection_affinity_for_test,
    reorder_products_within_cells_by_dynamic_affinity,
)


class ToyAffinity:
    rank = 2
    customer_index = {"c0": 0, "c1": 1, "c2": 2}
    product_index = {"p0": 0, "p1": 1, "p2": 2}
    customer_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    product_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    def transformed_customer_vectors(self, customer_ids, time_bucket):
        gate = np.asarray([2.0, 0.5]) if str(time_bucket).startswith("2020-01") else np.asarray([0.5, 2.0])
        return np.vstack([self.customer_embeddings[self.customer_index[c]] * gate for c in customer_ids])

    def product_vectors(self, product_ids):
        return np.vstack([self.product_embeddings[self.product_index[p]] for p in product_ids])


def test_projection_affinity_for_test_returns_product_permutation():
    products = np.asarray(["p1", "p0", "p2"], dtype=object)

    reordered = reorder_products_by_projection_affinity_for_test(
        ["c0", "c1", "c2"],
        products,
        "2020-01",
        ToyAffinity(),
    )

    assert Counter(reordered.tolist()) == Counter(products.tolist())


def test_reorder_products_within_cells_preserves_each_cell_product_multiset():
    customers = np.asarray(["c0", "c1", "c2", "c0", "c1"], dtype=object)
    products = np.asarray(["p1", "p0", "p2", "p2", "p1"], dtype=object)
    customer_blocks = np.asarray([0, 0, 0, 1, 1])
    product_blocks = np.asarray([0, 0, 0, 1, 1])
    time_codes = np.asarray([0, 0, 0, 1, 1])
    time_gate_codes = np.asarray([0, 0, 0, 0, 0])
    cell_codes = encode_cell_codes(customer_blocks, product_blocks, time_codes)
    before = {
        cell: Counter(products[cell_codes == cell].tolist())
        for cell in sorted(set(cell_codes.tolist()))
    }

    reordered, summary = reorder_products_within_cells_by_dynamic_affinity(
        customers,
        products,
        customer_blocks,
        product_blocks,
        time_codes,
        time_gate_codes,
        ToyAffinity(),
        "dynamic_projection",
        128,
        np.random.default_rng(4),
        code_to_time_gate=["2020-01"],
        code_to_time_bucket=["2020-01-01", "2020-01-02"],
    )
    after = {
        cell: Counter(reordered[cell_codes == cell].tolist())
        for cell in sorted(set(cell_codes.tolist()))
    }

    assert before == after
    assert summary["num_cells"] == 2
    assert summary["pairing_mode"] == "dynamic_projection"


def test_penalized_pairing_can_avoid_real_exact_event_pair():
    customers = np.asarray(["c0", "c1"], dtype=object)
    products = np.asarray(["p0", "p1"], dtype=object)
    customer_blocks = np.asarray([0, 0])
    product_blocks = np.asarray([0, 0])
    time_codes = np.asarray([0, 0])
    time_gate_codes = np.asarray([0, 0])
    num_products = 3
    num_time_codes = 1
    real_pair_keys = {0 * num_products + 0, 1 * num_products + 1}
    real_event_keys = {key * num_time_codes + 0 for key in real_pair_keys}

    reordered, summary = reorder_products_within_cells_by_dynamic_affinity(
        customers,
        products,
        customer_blocks,
        product_blocks,
        time_codes,
        time_gate_codes,
        ToyAffinity(),
        "dynamic_exact_penalized",
        128,
        np.random.default_rng(5),
        code_to_time_gate=["2020-01"],
        code_to_time_bucket=["2020-01-01"],
        slot_customer_idx=np.asarray([0, 1]),
        slot_product_idx=np.asarray([0, 1]),
        real_pair_keys=real_pair_keys,
        real_event_keys=real_event_keys,
        num_products=num_products,
        num_time_codes=num_time_codes,
        lambda_duplicate_pair=0.0,
        lambda_real_pair_overlap=0.0,
        lambda_exact_event_overlap=10.0,
    )

    assert reordered.tolist() == ["p1", "p0"]
    assert summary["pairing_mode"] == "dynamic_exact_penalized"
    assert summary["num_penalized_cells"] == 1


def test_penalized_pairing_large_cells_use_projection_fallback():
    customers = np.asarray(["c0", "c1", "c2"], dtype=object)
    products = np.asarray(["p0", "p1", "p2"], dtype=object)
    customer_blocks = np.asarray([0, 0, 0])
    product_blocks = np.asarray([0, 0, 0])
    time_codes = np.asarray([0, 0, 0])
    time_gate_codes = np.asarray([0, 0, 0])

    reordered, summary = reorder_products_within_cells_by_dynamic_affinity(
        customers,
        products,
        customer_blocks,
        product_blocks,
        time_codes,
        time_gate_codes,
        ToyAffinity(),
        "dynamic_exact_penalized",
        2,
        np.random.default_rng(8),
        code_to_time_gate=["2020-01"],
        code_to_time_bucket=["2020-01-01"],
        slot_customer_idx=np.asarray([0, 1, 2]),
        slot_product_idx=np.asarray([0, 1, 2]),
        real_pair_keys=set(),
        real_event_keys=set(),
        num_products=3,
        num_time_codes=1,
    )

    assert Counter(reordered.tolist()) == Counter(products.tolist())
    assert summary["num_exact_penalized_cells"] == 0
    assert summary["num_projection_fallback_cells"] == 1
    assert summary["num_cells_processed"] == 1
    assert summary["num_events_projection_fallback"] == 3
    assert summary["percent_cells_projection_fallback"] == 1.0
    assert summary["percent_events_projection_fallback"] == 1.0
    assert summary["percent_large_cells_projection_sort"] == 1.0
    assert summary["largest_cell_size"] == 3
    assert summary["p95_cell_size"] == 3.0
    assert summary["p99_cell_size"] == 3.0
