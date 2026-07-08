from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import ConditionalTABDLMDataset  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_joint import build_lstm_model, make_lstm_collate_fn, scatter_state  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def make_fixture():
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "product_id": ["p1", "p2"],
            "review_time": ["2020-01-01", "2020-01-02"],
            "rating": ["5", "1"],
            "verified": ["True", "False"],
            "summary": ["great product", "bad fit"],
            "review_text": ["great product works well", "bad fit for me"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket", "review_text_length_bucket"),
        text_targets=("summary", "review_text"),
        text_max_lengths={"summary": 6, "review_text": 10},
        summary_length_enabled=True,
        summary_length_buckets={"len_1_2": (1, 2), "len_3_5": (3, 5)},
        review_text_length_buckets={"q0_q20": (1, 3), "q20_q40": (4, 8)},
    )
    vocabs = {
        "rating": CategoryVocab.from_values("rating", frame["rating"]),
        "verified": CategoryVocab.from_values("verified", frame["verified"]),
        "summary_length_bucket": CategoryVocab.from_values("summary_length_bucket", ["len_1_2", "len_3_5"]),
        "review_text_length_bucket": CategoryVocab.from_values("review_text_length_bucket", ["q0_q20", "q20_q40"]),
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"].tolist() + frame["review_text"].tolist())
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "model": {"row_hidden_dim": 32, "latent_noise_dim": 8, "categorical_context_dim": 4, "dropout": 0.0, "use_graph_context": False},
        "text_decoder": {"embedding_dim": 16, "hidden_dim": 24, "num_layers": 1, "dropout": 0.0, "type": "lstm"},
        "id_encoding": {"num_buckets": 64, "embedding_dim": 8},
        "datetime_encoding": {"embedding_dim": 8},
    }
    return frame, ConditionalTABDLMConfig(raw=raw, schema=schema), vocabs, tokenizer


def test_lstm_joint_forward_outputs_all_heads():
    frame, config, vocabs, tokenizer = make_fixture()
    dataset = ConditionalTABDLMDataset(frame, config.schema, vocabs, tokenizer, num_hash_buckets=64)
    batch = make_lstm_collate_fn([dataset[0], dataset[1]])
    model = build_lstm_model(config, vocabs, tokenizer)

    output = model(
        batch["foreign_key_ids"],
        batch["datetime_values"],
        batch["categorical_ids"],
        batch["text_ids"],
    )

    assert output["categorical"]["rating"].shape == (2, vocabs["rating"].size)
    assert output["categorical"]["verified"].shape == (2, vocabs["verified"].size)
    assert output["categorical"]["summary_length_bucket"].shape == (2, vocabs["summary_length_bucket"].size)
    assert output["categorical"]["review_text_length_bucket"].shape == (2, vocabs["review_text_length_bucket"].size)
    assert output["text"]["summary"].shape[:2] == (2, 5)
    assert output["text"]["review_text"].shape[:2] == (2, 9)


def test_lstm_joint_generate_returns_all_outputs():
    frame, config, vocabs, tokenizer = make_fixture()
    dataset = ConditionalTABDLMDataset(frame, config.schema, vocabs, tokenizer, num_hash_buckets=64)
    batch = make_lstm_collate_fn([dataset[0], dataset[1]])
    model = build_lstm_model(config, vocabs, tokenizer)
    with torch.no_grad():
        generated = model.generate(
            batch["foreign_key_ids"],
            batch["datetime_values"],
            vocabs,
            tokenizer,
            temperature=1.0,
            top_p=1.0,
            min_tokens={"summary": 1, "review_text": 1},
        )
    assert set(generated["categorical"]) >= {"rating", "verified", "summary_length_bucket", "review_text_length_bucket"}
    assert len(generated["text"]["summary"]) == 2
    assert len(generated["text"]["review_text"]) == 2


def test_lstm_scatter_state_handles_autocast_dtype_mismatch():
    hidden = torch.zeros(1, 3, 2, dtype=torch.bfloat16)
    cell = torch.zeros(1, 3, 2, dtype=torch.bfloat16)
    new_hidden = torch.ones(1, 2, 2, dtype=torch.float16)
    new_cell = torch.ones(1, 2, 2, dtype=torch.float16)
    index = torch.tensor([0, 2], dtype=torch.long)

    out_hidden, out_cell = scatter_state((hidden, cell), (new_hidden, new_cell), index, "lstm")

    assert out_hidden.dtype == torch.bfloat16
    assert out_cell.dtype == torch.bfloat16
    assert torch.allclose(out_hidden[:, index, :].float(), torch.ones(1, 2, 2))
    assert torch.allclose(out_cell[:, index, :].float(), torch.ones(1, 2, 2))
