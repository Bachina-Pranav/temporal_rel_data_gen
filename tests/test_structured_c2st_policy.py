from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.paper_metrics.c2st import (  # noqa: E402
    structured_c2st_feature_manifest,
    structured_c2st_metrics,
)


def test_structured_c2st_excludes_fixed_text_and_length_features():
    table = {
        "primary_key": "event_id",
        "columns": {
            "event_id": {"type": "categorical"},
            "user_id": {"type": "foreign_key"},
            "item_id": {"type": "foreign_key"},
            "event_time": {"type": "datetime"},
            "rating": {"type": "categorical"},
            "verified": {"type": "boolean"},
            "price": {"type": "numerical"},
            "review_text": {"type": "text"},
            "review_text_token_length": {"type": "numerical"},
            "fixed_score": {"type": "numerical", "generated": False},
        },
    }

    manifest = structured_c2st_feature_manifest(table)

    assert manifest["included_columns"] == ["rating", "verified", "price"]
    reasons = {
        row["column"]: row["reason"] for row in manifest["excluded_columns"]
    }
    assert "identifier" in reasons["event_id"]
    assert "foreign key" in reasons["user_id"]
    assert "timestamp" in reasons["event_time"]
    assert "text evaluated separately" in reasons["review_text"]
    assert "length" in reasons["review_text_token_length"]
    assert "fixed/non-generated" in reasons["fixed_score"]


def test_structured_c2st_feature_names_ignore_large_text_and_spine_shift():
    real = pd.DataFrame(
        {
            "user_id": [f"real-{index}" for index in range(40)],
            "event_time": pd.date_range("2020-01-01", periods=40, freq="D"),
            "rating": [1, 2, 3, 4, 5] * 8,
            "review_text": ["short"] * 40,
            "review_text_length": [1] * 40,
        }
    )
    synthetic = real.copy()
    synthetic["user_id"] = [f"synthetic-{index}" for index in range(40)]
    synthetic["event_time"] = pd.date_range("2040-01-01", periods=40, freq="D")
    synthetic["review_text"] = ["very long synthetic text " * 100] * 40
    synthetic["review_text_length"] = [400] * 40
    config = {
        "table": {
            "columns": {
                "user_id": {"type": "foreign_key"},
                "event_time": {"type": "datetime"},
                "rating": {"type": "categorical"},
                "review_text": {"type": "text"},
                "review_text_length": {"type": "numerical"},
            }
        },
        "evaluation": {
            "random_seed": 42,
            "c2st": {
                "enabled": True,
                "classifiers": ["logistic_regression"],
                "max_rows": 40,
            },
        },
    }

    metrics, _ = structured_c2st_metrics(real, synthetic, config)

    assert metrics["feature_manifest"]["included_columns"] == ["rating"]
    assert all(name.startswith("rating_") for name in metrics["feature_names"])


def test_expected_benchmark_structured_feature_sets_resolve_from_schema():
    expected = {
        "single_event_table_paper_metrics_amazon_toy.yaml": ["rating", "verified"],
        "single_event_table_paper_metrics_movielens_100k.yaml": ["rating"],
        "single_event_table_paper_metrics_hm_10k_customers.yaml": [
            "price",
            "sales_channel_id",
        ],
        "single_event_table_paper_metrics_yelp_100k.yaml": [
            "stars",
            "useful",
            "funny",
            "cool",
        ],
        "single_event_table_paper_metrics_retailrocket_100k.yaml": ["event_type"],
    }
    for filename, columns in expected.items():
        config = yaml.safe_load(
            (ROOT / "configs/evaluation" / filename).read_text(encoding="utf-8")
        )
        manifest = structured_c2st_feature_manifest(config["table"])
        assert manifest["included_columns"] == columns
