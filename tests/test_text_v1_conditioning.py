from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from textgen.masked_text_diffusion import TemporalSummaryMaskedDiffusionV1  # noqa: E402
from textgen.text_conditioning import ConditionFeatureNormalizer, build_text_condition_features  # noqa: E402


def test_text_conditioning_and_soft_prompt_shape(tmp_path):
    reviews = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c1"],
            "product_id": ["p1", "p1", "p2"],
            "review_time": ["2020-01-01", "2020-01-02", "2020-02-01"],
            "rating": [5, 4, 2],
            "verified": [True, False, True],
            "summary": ["Great", "Good", "Bad"],
        }
    )
    debug = tmp_path / "debug"
    debug.mkdir()
    pd.DataFrame({"customer_id": ["c1", "c2"], "customer_block": [0, 1]}).to_csv(debug / "customer_blocks.csv", index=False)
    pd.DataFrame({"product_id": ["p1", "p2"], "product_block": [1, 0]}).to_csv(debug / "product_blocks.csv", index=False)

    result = build_text_condition_features(reviews, structure_debug_dir=debug)
    normalizer = ConditionFeatureNormalizer().fit(result.features)
    matrix = normalizer.transform(result.features)
    model = TemporalSummaryMaskedDiffusionV1(
        vocab_size=20,
        condition_dim=matrix.shape[1],
        max_summary_tokens=6,
        num_condition_tokens=4,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
    )
    prompt = model.soft_prompt(torch.tensor(matrix, dtype=torch.float32))

    assert matrix.shape[0] == len(reviews)
    assert "customer_past_review_count" in result.features.columns
    assert "block_pair_past_avg_rating" in result.features.columns
    assert prompt.shape == (len(reviews), 4, 16)
    assert np.isfinite(matrix).all()
