from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.ultrafast_event_pairing import reorder_products_by_projection_affinity  # noqa: E402


class ToyAffinity:
    rank = 2
    customer_index = {"c0": 0, "c1": 1}
    product_index = {"p0": 0, "p1": 1}
    customer_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    product_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    def transformed_customer_vectors(self, customer_ids, time_bucket):
        gate = np.asarray([1.0, 1.0]) if time_bucket == "2020-01-01" else np.asarray([2.0, 0.5])
        return np.vstack([self.customer_embeddings[self.customer_index[c]] * gate for c in customer_ids])

    def product_vectors(self, product_ids):
        return np.vstack([self.product_embeddings[self.product_index[p]] for p in product_ids])


def test_dynamic_projection_pairing_returns_product_permutation():
    customers = np.asarray(["c0", "c1", "c0", "c1"], dtype=object)
    products = np.asarray(["p1", "p0", "p1", "p0"], dtype=object)

    reordered = reorder_products_by_projection_affinity(
        customers,
        products,
        "2020-01-01",
        ToyAffinity(),
        np.random.default_rng(2),
        dynamic=True,
    )

    assert len(reordered) == len(products)
    assert sorted(reordered.tolist()) == sorted(products.tolist())
