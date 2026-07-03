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
