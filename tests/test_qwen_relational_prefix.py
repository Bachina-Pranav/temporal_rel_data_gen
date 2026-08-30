from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from attribute_generation.qwen_text_decoder.relational_prefix import (
    EncodedRows,
    RelationalSoftPrefix,
    assert_only_prefix_trainable,
    classify_relational_support,
    combined_query_indices,
    deterministic_training_subset,
    freeze_language_model,
)


class TinyTokenizer:
    eos_token = "<eos>"

    def __init__(self):
        self.seen = []

    def __call__(self, values, add_special_tokens=False):
        if isinstance(values, str):
            values = [values]
        self.seen.extend(values)
        return {"input_ids": [[ord(char) % 31 for char in value] for value in values]}


def test_soft_prefix_shape_and_frozen_language_model():
    language = torch.nn.Linear(3, 5)
    freeze_language_model(language)
    prefix = RelationalSoftPrefix(7, 11, num_tokens=4, hidden_dim=13)
    output = prefix(torch.randn(6, 7))
    audit = assert_only_prefix_trainable(language, prefix)
    assert output.shape == (6, 4, 11)
    assert audit["language_model_trainable_parameters"] == 0
    assert audit["prefix_trainable_parameters"] > 0


def test_deterministic_subset_and_combined_query_offsets():
    frame = pd.DataFrame({"value": range(20)})
    first, first_indices = deterministic_training_subset(frame, 5, seed=42)
    second, second_indices = deterministic_training_subset(frame, 5, seed=42)
    assert np.array_equal(first_indices, second_indices)
    assert first.equals(second)
    combined, queries = combined_query_indices(
        [pd.DataFrame({"x": range(3)}), pd.DataFrame({"x": range(4)})],
        [0, 2],
    )
    assert len(combined) == 7
    assert queries.tolist() == [3, 5]


def test_encoded_rows_canonicalizes_missing_text():
    frame = pd.DataFrame(
        [{"rating": 5, "verified": True, "summary": np.nan, "review_text": None}]
    )
    tokenizer = TinyTokenizer()
    encoded = EncodedRows(frame, torch.zeros(1, 4), tokenizer, max_length=200)
    assert len(encoded) == 1
    # Missing values must never be serialized as the literal token sequence "nan".
    assert all("nan" not in value.lower() for value in tokenizer.seen)


def test_relational_support_requires_correct_context_to_beat_both_controls():
    rows = [
        {"mode": "R0_no_prefix", "macro_c2st": 0.60},
        {"mode": "R1_correct_context", "macro_c2st": 0.54},
        {"mode": "R2_shuffled_context", "macro_c2st": 0.59},
    ]
    decision = classify_relational_support(
        rows, {"strong_improvement": 0.03, "moderate_improvement": 0.015}
    )
    assert decision["classification"] == "strongly_supported"
    rows[2]["macro_c2st"] = 0.53
    decision = classify_relational_support(
        rows, {"strong_improvement": 0.03, "moderate_improvement": 0.015}
    )
    assert decision["classification"] == "unresolved"
