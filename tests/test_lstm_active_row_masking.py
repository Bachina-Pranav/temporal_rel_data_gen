from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import attribute_generation.conditional_tabdlm.lstm_fast_sampler as fast_sampler  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_lstm_active_row_masking_stops_finished_rows(monkeypatch):
    _, _, _, tokenizer, model = make_lstm_fast_fixture()
    context_dim = model.row_hidden_dim + len(model.schema.model_categorical_targets) * model.categorical_context_dim
    context = torch.randn(2, context_dim)

    def fake_step(logits, tokenizer_arg, *, step, lows, highs, previous_ids, temperature, top_p, repetition_penalty):
        values = torch.full((logits.shape[0],), tokenizer.vocab["great"], dtype=torch.long, device=logits.device)
        values[0] = tokenizer.eos_id
        return values

    monkeypatch.setattr(fast_sampler, "sample_text_step_fast", fake_step)
    ids = fast_sampler.generate_text_group_fast(
        model,
        "review_text",
        context,
        lows=[0, 0],
        highs=[4, 4],
        tokenizer=tokenizer,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        active_row_masking=True,
    )

    assert int(ids[0, 1]) == tokenizer.eos_id
    assert ids[0, 2:].eq(tokenizer.pad_id).all()
    assert int(ids[1, 1]) == tokenizer.vocab["great"]
