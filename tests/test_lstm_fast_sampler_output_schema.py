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
