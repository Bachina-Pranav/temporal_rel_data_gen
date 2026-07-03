from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMSchema,
    FORBIDDEN_ENGINEERED_FEATURES,
    load_config,
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


def test_v1_2_config_does_not_require_engineered_features():
    config = load_config(ROOT / "configs/attribute_generation/conditional_tabdlm_rel_amazon_exp1_2.yaml")
    serialized_inputs = set(config.schema.condition_columns)
    serialized_inputs.update(config.schema.target_columns)
    serialized_inputs.update(config.schema.auxiliary_categorical_targets)

    assert not FORBIDDEN_ENGINEERED_FEATURES.intersection(serialized_inputs)
