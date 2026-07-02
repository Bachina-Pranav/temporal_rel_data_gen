from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from textgen.masked_summary_dataset import MaskedSummaryDataset, SimpleSummaryTokenizer  # noqa: E402


def test_masked_summary_dataset_masks_only_content_tokens():
    frame = pd.DataFrame({"summary": ["Great product", "Not bad"]})
    tokenizer = SimpleSummaryTokenizer().fit(frame["summary"])
    features = np.zeros((len(frame), 3), dtype=np.float32)
    dataset = MaskedSummaryDataset(
        frame,
        tokenizer,
        features,
        max_summary_tokens=5,
        min_mask_prob=1.0,
        max_mask_prob=1.0,
        seed=5,
    )

    item = dataset[0]
    content_positions = item["attention_mask"] == 1
    pad_positions = item["attention_mask"] == 0

    assert (item["input_ids"][content_positions] == tokenizer.mask_token_id).all()
    assert (item["labels"][content_positions] != -100).all()
    assert (item["labels"][pad_positions] == -100).all()
    assert (item["input_ids"][pad_positions] == tokenizer.pad_token_id).all()
