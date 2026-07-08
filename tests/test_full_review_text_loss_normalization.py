from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import SimpleTextTokenizer  # noqa: E402
from attribute_generation.conditional_tabdlm.train import denoising_loss  # noqa: E402


def test_review_text_token_loss_is_field_normalized():
    tokenizer = SimpleTextTokenizer().fit(["alpha beta gamma"])
    content_id = tokenizer.vocab["alpha"]
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=(),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 8, "review_text": 200},
    )
    batch = {
        "foreign_key_ids": torch.zeros((1, 2), dtype=torch.long),
        "text_labels": {
            "summary": torch.full((1, 8), content_id, dtype=torch.long),
            "review_text": torch.full((1, 200), content_id, dtype=torch.long),
        },
    }
    vocab_size = tokenizer.vocab_size
    logits = {
        "categorical": {},
        "text": {
            "summary": torch.zeros((1, 8, vocab_size), dtype=torch.float32),
            "review_text": torch.zeros((1, 200, vocab_size), dtype=torch.float32),
        },
    }
    _, components = denoising_loss(
        logits,
        batch,
        schema,
        loss_weights={"summary": 1.0, "review_text": 1.0},
        text_tokenizer=tokenizer,
        text_token_loss_weights={"summary": {"content": 1.0}, "review_text": {"content": 1.0}},
    )
    summary_loss = components["summary"]["loss_sum"] / components["summary"]["count"]
    review_loss = components["review_text"]["loss_sum"] / components["review_text"]["count"]
    assert math.isclose(summary_loss, review_loss, rel_tol=1e-6)

