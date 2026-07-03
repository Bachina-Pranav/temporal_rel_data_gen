from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMSchema,
)


def test_schema_rejects_engineered_features():
    with pytest.raises(ValueError, match="Engineered"):
        ConditionalTABDLMSchema(
            foreign_key_columns=("customer_id", "customer_block"),
            datetime_columns=("review_time",),
            categorical_targets=("rating",),
            text_targets=("summary",),
            text_max_lengths={"summary": 8},
        ).validate()


def test_schema_keeps_condition_and_target_roles_separate():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        numerical_targets=(),
        text_targets=("summary",),
        text_max_lengths={"summary": 8},
    )
    schema.validate()
    assert schema.condition_columns == ("customer_id", "product_id", "review_time")
    assert schema.target_columns == ("rating", "verified", "summary")

