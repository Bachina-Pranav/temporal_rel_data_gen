from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from audit_all_structure_methods import audit_method  # noqa: E402
from reldiff.generation.sbm_hierarchy import (  # noqa: E402
    select_block_level,
    summarize_raw_assignments,
)


def make_reviews():
    rows = []
    pairs = [
        ("c1", "p1"),
        ("c2", "p2"),
        ("c3", "p3"),
        ("c4", "p4"),
        ("c1", "p3"),
        ("c3", "p1"),
    ]
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    for index in range(12):
        customer_id, product_id = pairs[index % len(pairs)]
        rows.append(
            {
                "customer_id": customer_id,
                "product_id": product_id,
                "review_time": dates[index].strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def test_fake_hierarchy_level_selection_bottom_top_auto():
    level0 = {
        ("customer", "c1"): 0,
        ("customer", "c2"): 1,
        ("customer", "c3"): 2,
        ("product", "p1"): 3,
        ("product", "p2"): 4,
        ("product", "p3"): 5,
        ("product", "p4"): 6,
    }
    level1 = {
        ("customer", "c1"): 0,
        ("customer", "c2"): 0,
        ("customer", "c3"): 1,
        ("product", "p1"): 2,
        ("product", "p2"): 2,
        ("product", "p3"): 3,
        ("product", "p4"): 3,
    }
    level2 = {
        ("customer", "c1"): 0,
        ("customer", "c2"): 0,
        ("customer", "c3"): 0,
        ("product", "p1"): 1,
        ("product", "p2"): 1,
        ("product", "p3"): 1,
        ("product", "p4"): 1,
    }
    summaries = pd.DataFrame(
        [
            summarize_raw_assignments(level0, 0),
            summarize_raw_assignments(level1, 1),
            summarize_raw_assignments(level2, 2),
        ]
    )
    assert summaries.loc[0, "num_customer_blocks"] == 3
    assert summaries.loc[0, "num_product_blocks"] == 4
    assert select_block_level(summaries, "bottom")[0] == 0
    assert select_block_level(summaries, "top")[0] == 2
    assert select_block_level(summaries, "auto")[0] == 0


def test_fake_hierarchy_detects_mixed_type_blocks():
    mixed = {
        ("customer", "c1"): 0,
        ("customer", "c2"): 1,
        ("product", "p1"): 0,
        ("product", "p2"): 2,
    }
    summary = summarize_raw_assignments(mixed, 0)
    assert summary["num_mixed_blocks"] == 1
    assert "Mixed customer/product blocks detected" in summary["warnings"][0]


def test_method_audit_handles_missing_block_metadata(tmp_path):
    real = make_reviews()
    method_dir = tmp_path / "unknown_method"
    method_dir.mkdir()
    real.to_csv(method_dir / "synthetic_review.csv", index=False)
    result = audit_method(
        method_dir,
        real,
        "customer_id",
        "product_id",
        "review_time",
        min_count=2,
    )
    assert result["interpretation"] == "no_block_metadata"
    assert result["num_customer_blocks"] is None
    assert result["block_pair_timestamp_ks_num_pairs"] is None


def test_method_audit_handles_degenerate_blocks(tmp_path):
    real = make_reviews()
    method_dir = tmp_path / "ct_2k_sbm_temporal_stubs"
    debug_dir = method_dir / "debug"
    debug_dir.mkdir(parents=True)
    real.to_csv(method_dir / "synthetic_review.csv", index=False)
    pd.DataFrame(
        {"customer_id": sorted(real["customer_id"].unique()), "customer_block": 0}
    ).to_csv(debug_dir / "customer_blocks.csv", index=False)
    pd.DataFrame(
        {"product_id": sorted(real["product_id"].unique()), "product_block": 0}
    ).to_csv(debug_dir / "product_blocks.csv", index=False)

    result = audit_method(
        method_dir,
        real,
        "customer_id",
        "product_id",
        "review_time",
        min_count=2,
    )
    assert result["learned_vs_preserved_summary"]["has_nontrivial_block_structure"] is False
    assert result["interpretation"] == "global_stub_rewiring"
    assert result["num_customer_blocks"] == 1
    assert result["num_product_blocks"] == 1


def test_method_audit_handles_nontrivial_blocks(tmp_path):
    real = make_reviews()
    method_dir = tmp_path / "ct_2k_sbm_temporal_stubs"
    debug_dir = method_dir / "debug"
    debug_dir.mkdir(parents=True)
    real.to_csv(method_dir / "synthetic_review.csv", index=False)
    pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "customer_block": [0, 0, 1, 1],
        }
    ).to_csv(debug_dir / "customer_blocks.csv", index=False)
    pd.DataFrame(
        {"product_id": ["p1", "p2", "p3", "p4"], "product_block": [0, 0, 1, 1]}
    ).to_csv(debug_dir / "product_blocks.csv", index=False)

    result = audit_method(
        method_dir,
        real,
        "customer_id",
        "product_id",
        "review_time",
        min_count=2,
    )
    assert result["learned_vs_preserved_summary"]["has_nontrivial_block_structure"] is True
    assert result["interpretation"] == "nontrivial_sbm_block_model"
    assert result["num_customer_blocks"] == 2
    assert result["num_product_blocks"] == 2
    assert result["num_nonzero_block_pairs_real"] >= 3
