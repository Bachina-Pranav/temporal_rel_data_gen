from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.evaluate import evaluate_frames  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402


def test_text_consistency_scores_normalized_rating_labels():
    pytest.importorskip("sklearn")
    rows = []
    for idx in range(30):
        good = idx % 2 == 0
        rows.append(
            {
                "customer_id": f"c{idx}",
                "product_id": f"p{idx}",
                "review_time": "2020-01-01",
                "rating": "5.0" if good else "1.0",
                "verified": "true" if good else "false",
                "summary": "excellent wonderful perfect five stars" if good else "terrible broken awful one star",
            }
        )
    real = pd.DataFrame(rows)
    synthetic = real.copy()
    synthetic["rating"] = [5 if value == "5.0" else 1 for value in synthetic["rating"]]
    synthetic["verified"] = [1 if value == "true" else 0 for value in synthetic["verified"]]

    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        text_targets=("summary",),
        text_max_lengths={"summary": 16},
    )
    config = ConditionalTABDLMConfig(
        raw={
            "paths": {
                "train_data_path": "unused.csv",
                "synthetic_spine_path": "unused.csv",
                "output_dir": "unused",
            }
        },
        schema=schema,
    )

    metrics = evaluate_frames(real, synthetic, config)
    consistency = metrics["text_consistency"]

    assert consistency["rating_text_consistency_accuracy"] > 0.5
    assert consistency["predicted_rating_distribution"]
