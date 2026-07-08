from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import generate_text_column_fast  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_lstm_length_bucketed_decoding_respects_short_bucket_cap():
    _, _, _, tokenizer, model = make_lstm_fast_fixture()
    context_dim = model.row_hidden_dim + len(model.schema.model_categorical_targets) * model.categorical_context_dim
    context = torch.randn(2, context_dim)

    ids = generate_text_column_fast(
        model,
        "review_text",
        context,
        bucket_names=["short", "long"],
        tokenizer=tokenizer,
        temperature=1.0,
        top_p=1.0,
        min_content_tokens=0,
        repetition_penalty=1.0,
        active_row_masking=True,
        length_bucketed=True,
    )

    assert tokenizer.eos_id in ids[0, :5].tolist()
    assert ids[0, 5:].eq(tokenizer.pad_id).all()
