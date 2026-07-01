from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from reldiff.generation import (  # noqa: E402
    ContinuousTime2KSBMPlusGenerator,
    ContinuousTimeTemporalSBMGenerator,
)
from evaluate_ct_2k_sbm_plus import evaluate_plus  # noqa: E402


def make_midnight_amazon():
    customers = pd.DataFrame({"customer_id": [f"c{i}" for i in range(5)]})
    products = pd.DataFrame({"product_id": [f"p{i}" for i in range(4)]})
    rows = [
        ("c0", "p0", "2020-01-01"),
        ("c0", "p0", "2020-01-02"),
        ("c0", "p1", "2020-01-03"),
        ("c0", "p2", "2020-01-04"),
        ("c0", "p3", "2020-01-05"),
        ("c1", "p0", "2020-01-06"),
        ("c1", "p1", "2020-01-07"),
        ("c1", "p1", "2020-01-08"),
        ("c1", "p2", "2020-01-09"),
        ("c1", "p3", "2020-01-10"),
        ("c2", "p0", "2020-01-11"),
        ("c2", "p1", "2020-01-12"),
        ("c2", "p2", "2020-01-13"),
        ("c2", "p2", "2020-01-14"),
        ("c2", "p3", "2020-01-15"),
        ("c3", "p0", "2020-01-16"),
        ("c3", "p0", "2020-01-17"),
        ("c3", "p1", "2020-01-18"),
        ("c3", "p2", "2020-01-19"),
        ("c3", "p3", "2020-01-20"),
        ("c4", "p0", "2020-01-21"),
        ("c4", "p1", "2020-01-22"),
        ("c4", "p2", "2020-01-23"),
        ("c4", "p3", "2020-01-24"),
        ("c4", "p3", "2020-01-25"),
        ("c0", "p1", "2020-01-26"),
        ("c1", "p2", "2020-01-27"),
        ("c2", "p3", "2020-01-28"),
        ("c3", "p0", "2020-01-29"),
        ("c4", "p3", "2020-01-30"),
    ]
    reviews = pd.DataFrame(rows, columns=["customer_id", "product_id", "review_time"])
    return customers, products, reviews


def degree_counts(df, column):
    return df[column].value_counts().sort_index()


def block_pair_counts(df, customer_blocks, product_blocks):
    annotated = df.copy()
    annotated["customer_block"] = annotated["customer_id"].map(customer_blocks)
    annotated["product_block"] = annotated["product_id"].map(product_blocks)
    return annotated.groupby(["customer_block", "product_block"]).size().sort_index()


def test_old_continuous_time_temporal_sbm_still_runs():
    customers, products, reviews = make_midnight_amazon()
    generator = ContinuousTimeTemporalSBMGenerator(customers, products, reviews, seed=4)
    synthetic = generator.generate()
    assert list(synthetic.columns) == ["customer_id", "product_id", "review_time"]
    assert len(synthetic) == len(reviews)


def test_ct_2k_sbm_plus_preserves_stubs_and_midnight_timestamps(tmp_path):
    customers, products, reviews = make_midnight_amazon()
    output_path = tmp_path / "synthetic_review.csv"
    debug_dir = tmp_path / "debug"

    generator = ContinuousTime2KSBMPlusGenerator(
        customers, products, reviews, seed=8, stub_pairing="time_sorted"
    )
    generator.fit()
    synthetic = generator.generate(output_path=output_path, debug_dir=debug_dir)

    assert output_path.exists()
    assert list(synthetic.columns) == ["customer_id", "product_id", "review_time"]
    assert len(synthetic) == len(reviews)
    assert set(synthetic["customer_id"]).issubset(set(customers["customer_id"]))
    assert set(synthetic["product_id"]).issubset(set(products["product_id"]))

    real_times = pd.to_datetime(reviews["review_time"])
    synthetic_times = pd.to_datetime(synthetic["review_time"])
    assert synthetic_times.min() >= real_times.min()
    assert synthetic_times.max() <= real_times.max()
    assert (synthetic_times.dt.hour == 0).all()
    assert (synthetic_times.dt.minute == 0).all()
    assert (synthetic_times.dt.second == 0).all()

    assert degree_counts(synthetic, "product_id").equals(
        degree_counts(reviews, "product_id")
    )
    assert degree_counts(synthetic, "customer_id").equals(
        degree_counts(reviews, "customer_id")
    )
    assert block_pair_counts(
        synthetic,
        generator.sbm_result.customer_blocks,
        generator.sbm_result.product_blocks,
    ).equals(
        block_pair_counts(
            reviews,
            generator.sbm_result.customer_blocks,
            generator.sbm_result.product_blocks,
        )
    )

    expected_debug_files = [
        "customer_blocks.csv",
        "product_blocks.csv",
        "block_pair_counts.csv",
        "ct_2k_sbm_plus_summary.json",
        "ct_2k_sbm_plus_block_pairs.csv",
        "ct_2k_sbm_plus_customer_degree_check.csv",
        "ct_2k_sbm_plus_product_degree_check.csv",
        "ct_2k_sbm_plus_timestamp_diagnostics.json",
        "ct_2k_sbm_plus_pair_multiplicity.json",
    ]
    for filename in expected_debug_files:
        assert (debug_dir / filename).exists()

    with (debug_dir / "ct_2k_sbm_plus_summary.json").open() as handle:
        summary = json.load(handle)
    assert summary["generator"] == "ct_2k_sbm_plus"
    assert summary["timestamp_granularity_mode"] == "date_only"
    assert summary["sbm_block_level_requested"] == "auto"
    assert "sbm_block_level_resolved" in summary
    assert "num_nonzero_block_pairs_real" in summary
    assert "num_nonzero_block_pairs_synthetic" in summary


def test_ct_2k_sbm_plus_evaluation_metrics_json():
    customers, products, reviews = make_midnight_amazon()
    generator = ContinuousTime2KSBMPlusGenerator(customers, products, reviews, seed=9)
    generator.fit()
    synthetic = generator.generate()
    results = evaluate_plus(
        pd.DataFrame(reviews).assign(review_time=pd.to_datetime(reviews["review_time"])),
        synthetic,
        customer_col="customer_id",
        product_col="product_id",
        timestamp_col="review_time",
        customer_blocks=generator.sbm_result.customer_blocks,
        product_blocks=generator.sbm_result.product_blocks,
    )
    assert "structural" in results
    assert "temporal" in results
    assert results["additional"]["product_degree_exact_match_rate"] == 1.0
    assert results["additional"]["customer_degree_exact_match_rate"] == 1.0
    assert results["additional"]["block_pair_count_exact_match_rate"] == 1.0
