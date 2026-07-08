from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.evaluate import evaluate_frames  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402


def test_lstm_evaluator_reports_review_text_and_real_synthetic_similarity():
    real = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "product_id": ["p1", "p2", "p3", "p4"],
            "review_time": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
            "rating": [5, 4, 1, 2],
            "verified": [1, 1, 0, 0],
            "summary": ["great", "nice", "bad", "poor"],
            "review_text": ["great item works well", "nice product", "bad fit", "poor quality"],
        }
    )
    synthetic = real.copy()
    synthetic["summary"] = ["great", "nice", "bad", "poor"]
    synthetic["review_text"] = ["great item", "nice product works", "bad fit issue", "poor quality item"]
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket", "review_text_length_bucket"),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 8, "review_text": 16},
        summary_length_enabled=True,
        summary_length_buckets={"len_1_2": (1, 2), "len_3_5": (3, 5)},
        review_text_length_buckets={"q0_q20": (1, 2), "q20_q40": (3, 5)},
    )
    config = ConditionalTABDLMConfig(raw={"paths": {"output_dir": "unused"}, "evaluation": {"review_text_privacy_sample_size": 2}}, schema=schema)
    metrics = evaluate_frames(real, synthetic, config)
    assert "review_text_length_ks" in metrics["length_diagnostics"]
    assert "review_text_distinct_1" in metrics["text"]
    assert "review_text_exact_train_overlap_rate" in metrics["text_privacy"]
    assert "real_summary_review_text_rougeL_mean" in metrics["text_consistency"]
    assert "synthetic_summary_review_text_token_jaccard_mean" in metrics["text_consistency"]
