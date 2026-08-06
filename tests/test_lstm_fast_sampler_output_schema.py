from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    BatchSample,
    FastSamplerOptions,
    materialize_batch_output,
)
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.runtime_profiler import RuntimeProfiler  # noqa: E402
from lstm_fast_sampler_test_utils import make_lstm_fast_fixture  # noqa: E402


def test_lstm_fast_sampler_output_schema():
    frame, config, vocabs, tokenizer, _ = make_lstm_fast_fixture()
    summary_ids = torch.tensor([tokenizer.encode("great item", 8)[0] for _ in range(2)], dtype=torch.long)
    review_ids = torch.tensor([tokenizer.encode("great item works", 16)[0] for _ in range(2)], dtype=torch.long)
    batch = BatchSample(
        frame=frame.head(2),
        categorical={
            "rating": [5, 4],
            "verified": ["True", "False"],
            "summary_length_bucket": ["short", "short"],
            "review_text_length_bucket": ["short", "short"],
        },
        text_ids={"summary": summary_ids, "review_text": review_ids},
        text={},
        text_lengths={"summary": [2, 2], "review_text": [3, 3]},
    )

    output = materialize_batch_output(batch, config.schema, vocabs, tokenizer, RuntimeProfiler(), FastSamplerOptions())

    assert list(output.columns) == ["customer_id", "product_id", "review_time", "rating", "verified", "summary", "review_text"]


def test_lstm_fast_sampler_preserves_event_id_and_materializes_numerical_targets():
    frame, _, vocabs, tokenizer, _ = make_lstm_fast_fixture()
    frame.insert(0, "event_id", ["e1", "e2", "e3", "e4"])
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        numerical_targets=("price",),
    )
    batch = BatchSample(
        frame=frame.head(2),
        categorical={"rating": [5, 4], "verified": ["True", "False"]},
        numerical={"price": [0.10, 0.20]},
        text_ids={},
        text={},
        text_lengths={},
    )

    output = materialize_batch_output(
        batch,
        schema,
        vocabs,
        tokenizer,
        RuntimeProfiler(),
        FastSamplerOptions(),
    )

    assert list(output.columns) == [
        "event_id",
        "customer_id",
        "product_id",
        "review_time",
        "rating",
        "verified",
        "price",
    ]
    assert output["event_id"].tolist() == ["e1", "e2"]
    assert output["price"].tolist() == [0.10, 0.20]
