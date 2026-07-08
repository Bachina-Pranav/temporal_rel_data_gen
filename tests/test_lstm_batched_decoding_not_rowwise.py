from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import generate_text_group_fast  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


class CountingDecoder(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


def test_lstm_batched_decoding_calls_decoder_by_timestep_not_row():
    _, _, _, tokenizer, model = make_lstm_fast_fixture()
    decoder = CountingDecoder(model.text_decoders["review_text"])
    model.text_decoders["review_text"] = decoder
    context_dim = model.row_hidden_dim + len(model.schema.model_categorical_targets) * model.categorical_context_dim
    context = torch.randn(4, context_dim)

    generate_text_group_fast(
        model,
        "review_text",
        context,
        lows=[0, 0, 0, 0],
        highs=[4, 4, 4, 4],
        tokenizer=tokenizer,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        active_row_masking=True,
    )

    assert decoder.calls <= 5
