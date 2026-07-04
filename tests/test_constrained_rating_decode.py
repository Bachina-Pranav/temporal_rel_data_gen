from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.constrained import (  # noqa: E402
    categorical_validity_mask,
    validate_output_categoricals,
)
from attribute_generation.conditional_tabdlm.evaluate import validity_metrics  # noqa: E402
from attribute_generation.conditional_tabdlm.sample import sample_categorical_logits  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab  # noqa: E402


def test_rating_logits_sample_only_valid_rating_values():
    vocab = CategoryVocab.from_values("rating", ["1", "2", "3", "4", "5"])
    logits = torch.full((128, vocab.size), -10.0)
    logits[:, vocab.token_to_id["<missing>"]] = 100.0
    sampled = sample_categorical_logits(logits, "rating", vocab)
    decoded = [int(vocab.decode(idx)) for idx in sampled.tolist()]
    assert set(decoded).issubset({1, 2, 3, 4, 5})


def test_output_categorical_validator_rejects_invalid_rating():
    vocab = CategoryVocab.from_values("rating", ["1", "2", "3", "4", "5"])
    frame = pd.DataFrame({"rating": [1, 5, 6]})
    with pytest.raises(ValueError):
        validate_output_categoricals(frame, {"rating": vocab})


def test_output_categorical_validator_normalizes_integer_like_rating():
    vocab = CategoryVocab.from_values("rating", ["1", "2", "3", "4", "5"])
    frame = pd.DataFrame({"rating": ["1.0", "5.0"]})
    out = validate_output_categoricals(frame, {"rating": vocab})
    assert out["rating"].tolist() == [1, 5]


def test_evaluator_catches_nan_empty_and_out_of_range_ratings():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        text_targets=("summary",),
        text_max_lengths={"summary": 8},
    )
    real = pd.DataFrame(
        {
            "customer_id": ["u1"] * 5,
            "product_id": ["i1"] * 5,
            "review_time": ["2020-01-01"] * 5,
            "rating": [1, 2, 3, 4, 5],
            "verified": ["True"] * 5,
            "summary": ["ok"] * 5,
        }
    )
    synthetic = real.copy()
    synthetic["rating"] = ["5.0", "", None, 0, 6]
    metrics = validity_metrics(real, synthetic, schema)
    assert metrics["invalid_rating_rate"] == 0.8
    assert categorical_validity_mask(pd.Series(["5.0"]), "rating", [1, 2, 3, 4, 5]).tolist() == [True]
