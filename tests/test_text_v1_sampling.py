from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from textgen.masked_summary_dataset import SimpleSummaryTokenizer  # noqa: E402
from textgen.masked_text_diffusion import TemporalSummaryMaskedDiffusionV1  # noqa: E402
from textgen.text_sampling import sample_summaries  # noqa: E402


def test_text_v1_sampling_outputs_clean_summaries():
    tokenizer = SimpleSummaryTokenizer().fit(["great product", "bad item", "works well"])
    model = TemporalSummaryMaskedDiffusionV1(
        vocab_size=tokenizer.vocab_size,
        condition_dim=3,
        max_summary_tokens=4,
        num_condition_tokens=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
    )
    features = np.zeros((3, 3), dtype=np.float32)

    summaries = sample_summaries(
        model,
        tokenizer,
        features,
        max_summary_tokens=4,
        num_denoising_steps=2,
        top_k=5,
        batch_size=2,
        seed=7,
    )

    assert len(summaries) == 3
    for summary in summaries:
        assert summary.strip()
        assert "[MASK]" not in summary
        assert "[PAD]" not in summary
        assert "[CLS]" not in summary
        assert "[SEP]" not in summary
