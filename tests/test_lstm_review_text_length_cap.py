from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import resolve_auto_review_text_config  # noqa: E402


def test_lstm_review_text_cap_is_data_derived_not_short_hardcode(tmp_path):
    path = tmp_path / "reviews.csv"
    pd.DataFrame({"review_text": [" ".join(f"w{i}" for i in range(150))]}).to_csv(path, index=False)
    raw = {
        "paths": {"train_data_path": str(path)},
        "columns": {
            "target": {"categorical": ["rating", "verified"], "numerical": [], "text": ["summary", "review_text"]},
        },
        "auxiliary_targets": {"categorical": ["summary_length_bucket", "review_text_length_bucket"]},
        "text": {"max_length": {"summary": 32, "review_text": "auto"}},
        "review_text": {"max_tokens": "auto", "max_feasible_tokens": 768, "min_coverage_rate": 0.99},
    }
    resolved = resolve_auto_review_text_config(raw)
    cap = resolved["text"]["max_length"]["review_text"]
    assert cap == 152
    assert cap not in {64, 128}
    assert resolved["review_text"]["length_cap_source"] == "max"

