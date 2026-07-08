from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_joint import lstm_joint_loss  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import SimpleTextTokenizer  # noqa: E402


def test_lstm_review_text_loss_is_mean_normalized():
    tokenizer = SimpleTextTokenizer().fit(["alpha beta gamma"])
    content_id = tokenizer.vocab["alpha"]
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=(),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 9, "review_text": 301},
    )
    summary = torch.full((1, 9), content_id, dtype=torch.long)
    review = torch.full((1, 301), content_id, dtype=torch.long)
    batch = {
        "foreign_key_ids": torch.zeros((1, 2), dtype=torch.long),
        "categorical_ids": torch.empty((1, 0), dtype=torch.long),
        "text_ids": {"summary": summary, "review_text": review},
    }
    vocab_size = tokenizer.vocab_size
    logits = {
        "categorical": {},
        "text": {
            "summary": torch.zeros((1, 8, vocab_size), dtype=torch.float32),
            "review_text": torch.zeros((1, 300, vocab_size), dtype=torch.float32),
        },
    }
    _, components = lstm_joint_loss(logits, batch, schema, {"summary_text": 1.0, "review_text": 1.0}, tokenizer)
    summary_loss = components["summary_text"]["loss_sum"] / components["summary_text"]["count"]
    review_loss = components["review_text"]["loss_sum"] / components["review_text"]["count"]
    assert math.isclose(summary_loss, review_loss, rel_tol=1e-6)

