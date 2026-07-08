from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402


def test_full_review_text_joint_target_layout():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket", "review_text_length_bucket"),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 8, "review_text": 64},
        summary_length_enabled=True,
        summary_length_buckets={"len_1_2": (1, 2)},
        review_text_length_buckets={"q0_q20": (0, 5), "q20_q40": (6, 20)},
    )

    assert schema.model_target_columns == (
        "rating",
        "verified",
        "summary_length_bucket",
        "review_text_length_bucket",
        "summary",
        "review_text",
    )
    assert schema.text_column_for_length_bucket("review_text_length_bucket") == "review_text"
    assert "review_text" in schema.target_columns

