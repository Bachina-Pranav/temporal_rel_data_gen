from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.dataset import ConditionalTABDLMDataset  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_joint import (  # noqa: E402
    build_lstm_model,
    make_lstm_collate_fn,
    run_lstm_fixed_steps,
)
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402


def test_fixed_step_training_loop_runs_exact_optimizer_steps(tmp_path):
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "product_id": ["p1", "p2", "p3", "p4"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "rating": ["5", "4", "2", "1"],
            "summary": ["great", "good", "bad", "awful"],
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating",),
        text_targets=("summary",),
        text_max_lengths={"summary": 5},
    )
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": str(tmp_path)},
        "training": {"mixed_precision": False, "amp_dtype": "fp16"},
        "model": {"row_hidden_dim": 16, "latent_noise_dim": 4, "categorical_context_dim": 4, "dropout": 0.0, "use_graph_context": False},
        "text_decoder": {"embedding_dim": 8, "hidden_dim": 12, "num_layers": 1, "dropout": 0.0, "type": "lstm"},
        "id_encoding": {"num_buckets": 32, "embedding_dim": 4},
        "datetime_encoding": {"embedding_dim": 4},
    }
    config = ConditionalTABDLMConfig(raw=raw, schema=schema)
    vocabs = {"rating": CategoryVocab.from_values("rating", frame["rating"])}
    tokenizer = SimpleTextTokenizer().fit(frame["summary"])
    dataset = ConditionalTABDLMDataset(frame, schema, vocabs, tokenizer, num_hash_buckets=32)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=make_lstm_collate_fn)
    valid_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=make_lstm_collate_fn)
    model = build_lstm_model(config, vocabs, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    summary = run_lstm_fixed_steps(
        model,
        loader,
        valid_loader,
        optimizer,
        scaler,
        "cpu",
        False,
        {},
        tokenizer,
        {},
        max_steps=3,
        gradient_accumulation_steps=2,
        steps_per_eval=1,
        steps_per_checkpoint=1,
        checkpoint_dir=checkpoint_dir,
        log_path=tmp_path / "train_log.jsonl",
        metrics_log_path=tmp_path / "metrics.jsonl",
        categorical_vocabs=vocabs,
        config=config,
    )

    rows = [json.loads(line) for line in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert summary["best_step"] in {1, 2, 3}
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "last.pt").exists()
