from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from reldiff.generation import ContinuousTime2KSBMTemporalKDEStubsGenerator  # noqa: E402


def make_date_only_amazon():
    customers = pd.DataFrame({"customer_id": [f"c{i}" for i in range(5)]})
    products = pd.DataFrame({"product_id": [f"p{i}" for i in range(4)]})
    pair_multiplicities = [
        ("c0", "p0", 3),
        ("c0", "p1", 2),
        ("c0", "p2", 1),
        ("c1", "p0", 3),
        ("c1", "p1", 2),
        ("c1", "p2", 1),
        ("c1", "p3", 1),
        ("c2", "p0", 4),
        ("c2", "p1", 2),
        ("c3", "p1", 3),
        ("c3", "p2", 2),
        ("c3", "p3", 1),
        ("c4", "p0", 2),
        ("c4", "p1", 1),
        ("c4", "p2", 2),
    ]
    rows = []
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    date_index = 0
    for customer_id, product_id, multiplicity in pair_multiplicities:
        for _ in range(multiplicity):
            rows.append((customer_id, product_id, dates[date_index].strftime("%Y-%m-%d")))
            date_index += 1
    reviews = pd.DataFrame(rows, columns=["customer_id", "product_id", "review_time"])
    return customers, products, reviews


def make_datetime_amazon():
    customers, products, reviews = make_date_only_amazon()
    times = pd.date_range("2020-01-01 08:00:00", periods=len(reviews), freq="6H")
    reviews = reviews.copy()
    reviews["review_time"] = times.strftime("%Y-%m-%d %H:%M:%S")
    return customers, products, reviews


def degree_counts(df, column):
    return df[column].value_counts().sort_index()


def block_pair_counts(df, customer_blocks, product_blocks):
    annotated = df.copy()
    annotated["customer_block"] = annotated["customer_id"].map(customer_blocks)
    annotated["product_block"] = annotated["product_id"].map(product_blocks)
    return annotated.groupby(["customer_block", "product_block"]).size().sort_index()


def timestamp_multiset(df):
    return pd.to_datetime(df["review_time"]).value_counts().sort_index()


def test_temporal_kde_stubs_preserves_degrees_and_block_pairs_date_only(tmp_path):
    customers, products, reviews = make_date_only_amazon()
    output_path = tmp_path / "synthetic_review.csv"
    debug_dir = tmp_path / "debug"
    generator = ContinuousTime2KSBMTemporalKDEStubsGenerator(
        customers,
        products,
        reviews,
        seed=23,
        timestamp_model="auto",
        timestamp_min_block_count=3,
        avoid_real_edge_prob=0.0,
    )
    generator.fit()
    synthetic = generator.generate(output_path=output_path, debug_dir=debug_dir)

    assert output_path.exists()
    assert list(synthetic.columns) == ["customer_id", "product_id", "review_time"]
    assert len(synthetic) == len(reviews)
    assert degree_counts(synthetic, "customer_id").equals(
        degree_counts(reviews, "customer_id")
    )
    assert degree_counts(synthetic, "product_id").equals(
        degree_counts(reviews, "product_id")
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

    synthetic_times = pd.to_datetime(synthetic["review_time"])
    assert (synthetic_times.dt.hour == 0).all()
    assert (synthetic_times.dt.minute == 0).all()
    assert not timestamp_multiset(synthetic).equals(timestamp_multiset(reviews))

    expected_debug_files = [
        "customer_blocks.csv",
        "product_blocks.csv",
        "block_pair_counts.csv",
        "block_pair_timestamp_diagnostics.csv",
        "timestamp_model_summary.json",
        "summary.json",
    ]
    for filename in expected_debug_files:
        assert (debug_dir / filename).exists()

    with (debug_dir / "summary.json").open() as handle:
        summary = json.load(handle)
    assert summary["generator"] == "ct_2k_sbm_temporal_kde_stubs"
    assert summary["timestamp_model"] == "smoothed_date_pmf"
    assert summary["timestamp_granularity_mode"] == "date_only"
    assert summary["timestamp_multiset_preserved_exactly"] is False
    assert summary["reuses_exact_timestamp_stubs"] is False
    assert summary["block_pair_count_exact_match_rate"] == 1.0


def test_temporal_kde_stubs_datetime_outputs_stay_in_range():
    customers, products, reviews = make_datetime_amazon()
    generator = ContinuousTime2KSBMTemporalKDEStubsGenerator(
        customers,
        products,
        reviews,
        seed=29,
        timestamp_model="block_pair_kde",
        timestamp_min_block_count=3,
        avoid_real_edge_prob=0.0,
    )
    generator.fit()
    synthetic = generator.generate()

    real_times = pd.to_datetime(reviews["review_time"])
    synthetic_times = pd.to_datetime(synthetic["review_time"])
    assert synthetic_times.min() >= real_times.min()
    assert synthetic_times.max() <= real_times.max()
    assert not set(synthetic_times).issubset(set(real_times))


def test_evaluate_all_structure_methods_writes_json_and_csv(tmp_path):
    _, _, reviews = make_date_only_amazon()
    real_path = tmp_path / "review.csv"
    reviews.to_csv(real_path, index=False)
    outputs_root = tmp_path / "outputs"

    for method in [
        "continuous_time_temporal_sbm",
        "ct_2k_sbm_plus",
        "ct_2k_sbm_temporal_stubs",
        "ct_2k_sbm_temporal_kde_stubs",
    ]:
        method_dir = outputs_root / method
        debug_dir = method_dir / "debug"
        debug_dir.mkdir(parents=True)
        reviews.to_csv(method_dir / "synthetic_review.csv", index=False)
        pd.DataFrame(
            {
                "customer_id": sorted(reviews["customer_id"].unique()),
                "customer_block": 0,
            }
        ).to_csv(debug_dir / "customer_blocks.csv", index=False)
        pd.DataFrame(
            {
                "product_id": sorted(reviews["product_id"].unique()),
                "product_block": 0,
            }
        ).to_csv(debug_dir / "product_blocks.csv", index=False)

    output_json = tmp_path / "all_structure_metrics.json"
    output_csv = tmp_path / "all_structure_metrics.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "scripts" / "evaluate_all_structure_methods.py"),
            "--real-reviews",
            str(real_path),
            "--outputs-root",
            str(outputs_root),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
    )

    assert output_json.exists()
    assert output_csv.exists()
    rows = json.loads(output_json.read_text())
    assert [row["method"] for row in rows] == [
        "continuous_time_temporal_sbm",
        "ct_2k_sbm_plus",
        "ct_2k_sbm_temporal_stubs",
        "ct_2k_sbm_temporal_kde_stubs",
    ]
    assert pd.read_csv(output_csv)["method"].tolist() == [row["method"] for row in rows]
