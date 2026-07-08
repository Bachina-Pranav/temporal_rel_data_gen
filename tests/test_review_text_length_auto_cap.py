from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import resolve_auto_review_text_config  # noqa: E402


def base_raw(path: Path, feasible: int) -> dict:
    return {
        "paths": {"train_data_path": str(path)},
        "columns": {
            "condition": {"foreign_keys": ["customer_id", "product_id"], "datetimes": ["review_time"]},
            "target": {"categorical": ["rating", "verified"], "numerical": [], "text": ["summary", "review_text"]},
        },
        "auxiliary_targets": {"categorical": ["summary_length_bucket", "review_text_length_bucket"]},
        "text": {"max_length": {"summary": 8, "review_text": "auto"}},
        "review_text": {
            "max_tokens": "auto",
            "max_tokens_strategy": "max_if_feasible_else_p99",
            "max_feasible_tokens": feasible,
            "min_coverage_rate": 0.99,
        },
    }


def write_lengths(tmp_path: Path) -> Path:
    path = tmp_path / "reviews.csv"
    rows = []
    for idx, length in enumerate([2, 5, 10, 20, 100]):
        rows.append({"review_text": " ".join(f"w{j}" for j in range(length))})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_review_text_auto_cap_uses_max_when_feasible(tmp_path):
    raw = resolve_auto_review_text_config(base_raw(write_lengths(tmp_path), feasible=128))
    assert raw["text"]["max_length"]["review_text"] == 102
    assert raw["review_text"]["length_cap_source"] == "max"
    assert raw["review_text"]["coverage_rate_train"] == 1.0
    assert raw["review_text"]["truncation_rate_train"] == 0.0


def test_review_text_auto_cap_records_truncation_when_feasible_cap_binds(tmp_path):
    raw = resolve_auto_review_text_config(base_raw(write_lengths(tmp_path), feasible=50))
    assert raw["text"]["max_length"]["review_text"] == 50
    assert raw["review_text"]["length_cap_source"] == "max_feasible_under_p99"
    assert raw["review_text"]["coverage_rate_train"] < 1.0
    assert raw["review_text"]["truncation_rate_train"] > 0.0

