from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_joint import lstm_joint_loss  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_lstm_text_label_smoothing_changes_text_loss():
    frame, config, _, tokenizer, _ = make_lstm_fast_fixture()
    batch = {
        "foreign_key_ids": torch.zeros(len(frame), 2, dtype=torch.long),
        "categorical_ids": torch.zeros(len(frame), len(config.schema.model_categorical_targets), dtype=torch.long),
        "text_ids": {
            "summary": torch.tensor([tokenizer.encode("great item", 8)[0] for _ in range(len(frame))], dtype=torch.long),
            "review_text": torch.tensor([tokenizer.encode("great item works", 16)[0] for _ in range(len(frame))], dtype=torch.long),
        },
    }
    vocab = tokenizer.vocab_size
    logits = {
        "categorical": {column: torch.randn(len(frame), 2) for column in config.schema.model_categorical_targets},
        "text": {
            "summary": torch.randn(len(frame), 7, vocab),
            "review_text": torch.randn(len(frame), 15, vocab),
        },
    }
    raw = copy.deepcopy(config.raw)
    raw["loss"] = {"text_label_smoothing": {"enabled": True, "summary": 0.2, "review_text": 0.2}}
    smooth_config = type(config)(raw=raw, schema=config.schema, config_path=None)

    loss_plain, _ = lstm_joint_loss(logits, batch, config.schema, {"summary_text": 1.0, "review_text": 1.0}, tokenizer)
    loss_smooth, _ = lstm_joint_loss(logits, batch, config.schema, {"summary_text": 1.0, "review_text": 1.0}, tokenizer, config=smooth_config)

    assert not torch.allclose(loss_plain, loss_smooth)
