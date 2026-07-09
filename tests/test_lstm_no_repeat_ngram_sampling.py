from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import apply_no_repeat_ngram_blocking  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_no_repeat_ngram_blocks_repeated_next_token():
    _, _, _, tokenizer, _ = make_lstm_fast_fixture()
    great = tokenizer.vocab["great"]
    item = tokenizer.vocab["item"]
    works = tokenizer.vocab["works"]
    previous = torch.tensor([[tokenizer.bos_id, great, item, works, great, item]], dtype=torch.long)
    logits = torch.zeros(1, tokenizer.vocab_size)

    apply_no_repeat_ngram_blocking(logits, previous, 3, tokenizer)

    assert torch.isneginf(logits[0, works])
