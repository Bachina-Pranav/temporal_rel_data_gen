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


def test_decoder_input_token_dropout_train_only():
    _, config, vocabs, tokenizer, _ = make_lstm_fast_fixture()
    raw = copy.deepcopy(config.raw)
    raw["training_regularization"] = {
        "decoder_input_token_dropout": {
            "enabled": True,
            "summary": 1.0,
            "review_text": 0.0,
            "replacement": "UNK",
        }
    }
    config = type(config)(raw=raw, schema=config.schema, config_path=None)
    model = build_lstm_model(config, vocabs, tokenizer)
    teacher = torch.tensor([tokenizer.encode("great item", 8)[0][:-1]], dtype=torch.long)

    model.train()
    dropped = model.apply_decoder_input_token_dropout("summary", teacher)
    model.eval()
    unchanged = model.apply_decoder_input_token_dropout("summary", teacher)

    content_mask = ~torch.isin(teacher, torch.tensor([tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.mask_id, tokenizer.unk_id]))
    assert torch.all(dropped[content_mask] == tokenizer.unk_id)
    assert torch.equal(unchanged, teacher)
