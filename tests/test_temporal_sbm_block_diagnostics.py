from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_ct_2k_sbm_temporal_stubs import evaluate_temporal_stubs  # noqa: E402
from reldiff.generation import ContinuousTime2KSBMTemporalStubsGenerator  # noqa: E402
from reldiff.generation.block_diagnostics import (  # noqa: E402
    BLOCK_METADATA_WARNING,
    compute_all_block_diagnostics,
)


CUSTOMER_BLOCKS = {"c1": 0, "c2": 0, "c3": 1, "c4": 1}
PRODUCT_BLOCKS = {"p1": 0, "p2": 0, "p3": 1, "p4": 1}


def make_manual_block_reviews():
    rows = []
    specs = [
        ("c1", "p1", "2020-01-01"),
        ("c2", "p3", "2020-02-01"),
        ("c3", "p2", "2020-03-01"),
    ]
    for customer_id, product_id, start in specs:
        for offset, date in enumerate(pd.date_range(start, periods=5, freq="D")):
            rows.append(
                {
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "review_time": date.strftime("%Y-%m-%d"),
                    "rating": 1 + (offset % 5),
                    "verified": bool(offset % 2),
                    "summary": f"summary {customer_id} {product_id} {offset}",
                    "review_text": f"text {customer_id} {product_id} {offset}",
                }
            )
    real = pd.DataFrame(rows)
    synthetic = real[["customer_id", "product_id", "review_time"]].copy()
    return real, synthetic


def test_block_diagnostics_multiple_block_pairs_are_counted():
    real, synthetic = make_manual_block_reviews()
    diagnostics = compute_all_block_diagnostics(
        real,
        synthetic,
        CUSTOMER_BLOCKS,
        PRODUCT_BLOCKS,
        "customer_id",
        "product_id",
        "review_time",
        min_count=5,
    )
    assert diagnostics["num_customer_blocks"] == 2
    assert diagnostics["num_product_blocks"] == 2
    assert diagnostics["num_nonzero_block_pairs_real"] == 3
    assert diagnostics["block_pair_timestamp_ks_num_pairs"] == 3
    assert diagnostics["block_pair_count_exact_match_rate"] == 1.0
    assert diagnostics["block_pair_count_abs_error_sum"] == 0
    assert diagnostics["block_pair_timestamp_ks_mean"] == pytest.approx(0.0)


def test_evaluator_missing_block_metadata_warns_and_skips_block_metrics():
    real, synthetic = make_manual_block_reviews()
    with pytest.warns(UserWarning, match=BLOCK_METADATA_WARNING):
        results = evaluate_temporal_stubs(
            real,
            synthetic,
            customer_col="customer_id",
            product_col="product_id",
            timestamp_col="review_time",
        )
    assert results["additional"]["block_pair_timestamp_ks_num_pairs"] is None
    assert results["additional"]["block_pair_count_exact_match_rate"] is None


def test_single_true_block_pair_allows_one_timestamp_ks_pair():
    real, synthetic = make_manual_block_reviews()
    customer_blocks = {customer_id: 0 for customer_id in CUSTOMER_BLOCKS}
    product_blocks = {product_id: 0 for product_id in PRODUCT_BLOCKS}
    diagnostics = compute_all_block_diagnostics(
        real,
        synthetic,
        customer_blocks,
        product_blocks,
        "customer_id",
        "product_id",
        "review_time",
        min_count=5,
    )
    assert diagnostics["num_customer_blocks"] == 1
    assert diagnostics["num_product_blocks"] == 1
    assert diagnostics["block_pair_timestamp_ks_num_pairs"] == 1
    assert not any(
        "diagnostic bug" in warning
        for warning in diagnostics["block_diagnostic_warnings"]
    )


def test_temporal_stubs_generator_writes_canonical_block_debug_files(tmp_path):
    real, _ = make_manual_block_reviews()
    customers = pd.DataFrame({"customer_id": sorted(CUSTOMER_BLOCKS)})
    products = pd.DataFrame({"product_id": sorted(PRODUCT_BLOCKS)})
    debug_dir = tmp_path / "debug"
    generator = ContinuousTime2KSBMTemporalStubsGenerator(
        customers,
        products,
        real,
        seed=5,
        stub_pairing="temporal_sorted",
    )
    generator.fit()
    synthetic = generator.generate(debug_dir=debug_dir)
    assert len(synthetic) == len(real)
    for filename in [
        "customer_blocks.csv",
        "product_blocks.csv",
        "block_pair_counts.csv",
        "ct_2k_sbm_temporal_stubs_summary.json",
    ]:
        assert (debug_dir / filename).exists()

    with (debug_dir / "ct_2k_sbm_temporal_stubs_summary.json").open() as handle:
        summary = json.load(handle)
    for key in [
        "num_customer_blocks",
        "num_product_blocks",
        "num_total_blocks",
        "num_nonzero_block_pairs_real",
        "num_nonzero_block_pairs_synthetic",
        "num_possible_block_pairs",
        "block_pair_count_exact_match_rate",
        "block_pair_count_abs_error_sum",
        "block_pair_count_max_abs_error",
        "sbm_backend",
        "used_existing_reldiff_sbm",
        "seed",
    ]:
        assert key in summary
