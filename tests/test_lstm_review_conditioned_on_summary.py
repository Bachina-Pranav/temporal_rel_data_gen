from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_joint import build_lstm_model  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_review_text_initial_state_changes_with_summary_representation():
    _, config, vocabs, tokenizer, _ = make_lstm_fast_fixture()
    raw = copy.deepcopy(config.raw)
    raw["review_text_decoder"] = {
        "condition_on_summary": True,
        "summary_condition_type": "final_hidden_plus_mean_pool",
        "summary_condition_dim": 8,
        "summary_condition_dropout": 0.0,
    }
    config = type(config)(raw=raw, schema=config.schema, config_path=None)
    model = build_lstm_model(config, vocabs, tokenizer)
    context = torch.randn(2, model.decoder_context_dim)
    summary_a = torch.zeros(2, model.summary_condition_dim)
    summary_b = torch.ones(2, model.summary_condition_dim)

    state_a = model.initial_state("review_text", context, summary_repr=summary_a)
    state_b = model.initial_state("review_text", context, summary_repr=summary_b)

    assert not torch.allclose(state_a[0], state_b[0])
