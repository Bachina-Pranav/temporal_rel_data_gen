from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    cached_generated_candidate,
    normalize_privacy_text,
    remember_generated_candidate,
)
from attribute_generation.conditional_tabdlm.tokenization import SimpleTextTokenizer  # noqa: E402


def test_generated_candidate_cache_never_stores_training_text():
    tokenizer = SimpleTextTokenizer().fit(["copied train text", "fresh generated text"])
    train_set = {normalize_privacy_text("copied train text")}
    options = FastSamplerOptions(generated_candidate_cache_enabled=True)
    copied_ids, _ = tokenizer.encode("copied train text", max_length=8)
    fresh_ids, _ = tokenizer.encode("fresh generated text", max_length=8)

    remember_generated_candidate(options, "title", "short", "copied train text", torch.tensor(copied_ids), train_set, tokenizer)
    assert cached_generated_candidate(options, "title", "short") is None

    remember_generated_candidate(options, "title", "short", "fresh generated text", torch.tensor(fresh_ids), train_set, tokenizer)
    cached = cached_generated_candidate(options, "title", "short")

    assert cached is not None
    assert cached["normalized"] == normalize_privacy_text("fresh generated text")
