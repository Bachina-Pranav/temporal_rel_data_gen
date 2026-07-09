from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from audit_rel_amazon_single_event_table import audit_inputs  # noqa: E402


def test_rel_amazon_audit_reports_counts_and_fk_coverage(tmp_path):
    real, customer, product, spine = write_tiny_rel_amazon_tables(tmp_path, spine_rows=4)
    args = argparse.Namespace(
        real_table=str(real),
        customer_table=str(customer),
        product_table=str(product),
        synthetic_spine=str(spine),
        output=str(tmp_path / "audit.json"),
        allow_spine_row_mismatch=False,
    )

    report = audit_inputs(args)

    assert report["real_review"]["row_count"] == 4
    assert report["synthetic_spine"]["row_count"] == 4
    assert report["spine_row_count_matches_real"] is True
    assert report["fk_parent_coverage"]["customer_id"]["invalid_count"] == 0
    assert report["fatal_errors"] == []


def test_rel_amazon_audit_fails_when_spine_row_count_mismatches(tmp_path):
    real, customer, product, spine = write_tiny_rel_amazon_tables(tmp_path, spine_rows=2)
    args = argparse.Namespace(
        real_table=str(real),
        customer_table=str(customer),
        product_table=str(product),
        synthetic_spine=str(spine),
        output=str(tmp_path / "audit.json"),
        allow_spine_row_mismatch=False,
    )

    report = audit_inputs(args)

    assert report["spine_row_count_matches_real"] is False
    assert report["fatal_errors"]


def write_tiny_rel_amazon_tables(tmp_path: Path, spine_rows: int = 4):
    customer = pd.DataFrame({"customer_id": ["c1", "c2"]})
    product = pd.DataFrame({"product_id": ["p1", "p2"]})
    real = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c1", "c2"],
            "product_id": ["p1", "p2", "p1", "p2"],
            "review_time": pd.date_range("2020-01-01", periods=4, freq="D"),
            "rating": [5, 4, 5, 4],
            "verified": [1, 0, 1, 0],
            "summary": ["good", "ok", "great", "fine"],
            "review_text": ["good product", "ok item", "great product", "fine item"],
        }
    )
    spine = real[["customer_id", "product_id", "review_time"]].head(spine_rows)
    paths = []
    for name, frame in [("review.csv", real), ("customer.csv", customer), ("product.csv", product), ("synthetic_review.csv", spine)]:
        path = tmp_path / name
        frame.to_csv(path, index=False)
        paths.append(path)
    return tuple(paths)
