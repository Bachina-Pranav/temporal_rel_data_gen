from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.tokenization import SimpleTextTokenizer  # noqa: E402
from attribute_generation.conditional_tabdlm.train import summary_token_weights, weighted_summary_token_loss  # noqa: E402


def test_summary_token_loss_weights_assign_special_token_weights():
    tokenizer = SimpleTextTokenizer().fit(["great product"])
    labels = torch.tensor(
        [[
            tokenizer.bos_id,
            tokenizer.vocab["great"],
            tokenizer.eos_id,
            tokenizer.pad_id,
            tokenizer.unk_id,
            -100,
        ]]
    )

    weights = summary_token_weights(
        labels,
        tokenizer,
        {"pad": 0.15, "eos": 2.0, "bos": 0.0, "content": 1.0, "unk": 1.0},
    )

    assert weights[0, 0].item() == 0.0
    assert weights[0, 1].item() == 1.0
    assert weights[0, 2].item() == 2.0
    assert weights[0, 3].item() == pytest.approx(0.15)
    assert weights[0, 4].item() == 1.0
    assert weights[0, 5].item() == 0.0


def test_pad_is_downweighted_but_not_ignored():
    tokenizer = SimpleTextTokenizer().fit(["great product"])
    labels = torch.tensor([[tokenizer.pad_id, tokenizer.eos_id]])
    logits = torch.zeros((1, 2, tokenizer.vocab_size), dtype=torch.float32)

    loss, denom, subcomponents = weighted_summary_token_loss(
        logits,
        labels,
        tokenizer,
        {"pad": 0.15, "eos": 2.0, "bos": 0.0, "content": 1.0, "unk": 1.0},
    )

    assert loss.item() > 0.0
    assert denom.item() > 2.0
    assert subcomponents["pad"]["loss_sum"] > 0.0
