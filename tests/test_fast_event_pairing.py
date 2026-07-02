from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from generators.fast_event_pairing import pair_stubs_by_dynamic_affinity, sample_entities_with_quotas  # noqa: E402


class ToyAffinity:
    def __init__(self):
        self.customer_vectors = {
            "c0": np.asarray([1.0, 0.0]),
            "c1": np.asarray([0.0, 1.0]),
        }
        self.product_vectors_map = {
            "p0": np.asarray([1.0, 0.0]),
            "p1": np.asarray([0.0, 1.0]),
        }

    def transformed_customer_vectors(self, customer_ids, time_bucket):
        return np.vstack([self.customer_vectors.get(customer, np.zeros(2)) for customer in customer_ids])

    def product_vectors(self, product_ids):
        return np.vstack([self.product_vectors_map.get(product, np.zeros(2)) for product in product_ids])


def test_quota_sampler_respects_remaining_degrees():
    rng = np.random.default_rng(4)
    entity_ids = np.asarray(["a", "b", "c"], dtype=object)
    remaining = {"a": 2, "b": 1, "c": 0}
    probs = np.asarray([0.1, 0.9, 1.0])

    sampled, success, diagnostics = sample_entities_with_quotas(entity_ids, remaining, probs, 3, rng)

    counts = pd.Series(sampled).value_counts().to_dict()
    assert success is True
    assert diagnostics["sampled_n"] == 3
    assert counts.get("a", 0) <= 2
    assert counts.get("b", 0) <= 1
    assert counts.get("c", 0) == 0


def test_exact_small_cell_pairing_chooses_high_affinity_matching():
    rng = np.random.default_rng(1)
    customers = np.asarray(["c0", "c1"], dtype=object)
    products = np.asarray(["p1", "p0"], dtype=object)

    paired = pair_stubs_by_dynamic_affinity(
        customers,
        products,
        "2020-01-01",
        ToyAffinity(),
        rng,
        max_exact_cell_size=10,
    )

    assert paired.tolist() == ["p0", "p1"]


def test_projection_large_cell_pairing_returns_product_permutation():
    rng = np.random.default_rng(2)
    customers = np.asarray(["c0", "c1"] * 10, dtype=object)
    products = np.asarray(["p0", "p1"] * 10, dtype=object)

    paired = pair_stubs_by_dynamic_affinity(
        customers,
        products,
        "2020-01-01",
        ToyAffinity(),
        rng,
        max_exact_cell_size=5,
        large_cell_pairing="projection_sort",
    )

    assert sorted(paired.tolist()) == sorted(products.tolist())
