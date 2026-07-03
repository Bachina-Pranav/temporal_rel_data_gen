from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.sample import calibrated_length_probs  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema  # noqa: E402
from attribute_generation.conditional_tabdlm.tokenization import CategoryVocab, SimpleTextTokenizer  # noqa: E402
from attribute_generation.conditional_tabdlm.train import compute_summary_length_class_weights  # noqa: E402


def test_length_bucket_class_weights_boost_rare_buckets():
    frame = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(10)],
            "product_id": [f"p{i}" for i in range(10)],
            "review_time": ["2020-01-01"] * 10,
            "rating": ["5"] * 10,
            "verified": ["True"] * 10,
            "summary": ["good"] * 8 + ["this product is really very excellent"] * 2,
        }
    )
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket",),
        text_targets=("summary",),
        text_max_lengths={"summary": 12},
        summary_length_enabled=True,
        summary_length_buckets={"len_1_2": (1, 2), "len_6_10": (6, 10)},
    )
    raw = {
        "paths": {"train_data_path": "unused.csv", "synthetic_spine_path": "unused.csv", "output_dir": "unused"},
        "summary_length_loss": {
            "class_balanced": True,
            "class_weight_power": 0.5,
            "min_class_weight": 0.5,
            "max_class_weight": 5.0,
        },
    }
    tokenizer = SimpleTextTokenizer().fit(frame["summary"])
    vocab = CategoryVocab.from_values("summary_length_bucket", ["len_1_2"] * 8 + ["len_6_10"] * 2)
    weights = compute_summary_length_class_weights(
        frame,
        ConditionalTABDLMConfig(raw=raw, schema=schema),
        {"summary_length_bucket": vocab},
        tokenizer,
    )
    assert weights is not None
    payload = weights["json"]["weights"]
    assert payload["len_6_10"] > payload["len_1_2"]
    assert 0.5 <= payload["len_6_10"] <= 5.0


def test_calibration_decreases_overpredicted_short_bucket():
    schema = ConditionalTABDLMSchema(
        foreign_key_columns=("customer_id", "product_id"),
        datetime_columns=("review_time",),
        categorical_targets=("rating", "verified"),
        auxiliary_categorical_targets=("summary_length_bucket",),
        text_targets=("summary",),
        text_max_lengths={"summary": 8},
        summary_length_enabled=True,
        summary_length_buckets={"len_1_2": (1, 2), "len_3_5": (3, 5), "len_6_10": (6, 10)},
    )
    vocab = CategoryVocab.from_values("summary_length_bucket", ["len_1_2", "len_3_5", "len_6_10"])
    logits = torch.full((1, vocab.size), -20.0, dtype=torch.float32)
    logits[0, vocab.encode("len_1_2")] = torch.log(torch.tensor(0.8))
    logits[0, vocab.encode("len_3_5")] = torch.log(torch.tensor(0.15))
    logits[0, vocab.encode("len_6_10")] = torch.log(torch.tensor(0.05))
    calibration = {
        "calibration_ratio": {
            "len_1_2": 0.4 / 0.8,
            "len_3_5": 0.4 / 0.15,
            "len_6_10": 0.2 / 0.05,
        },
        "calibration_strength": 1.0,
    }

    calibrated = calibrated_length_probs(logits, vocab, calibration, schema)[0]

    assert calibrated[vocab.encode("len_1_2")].item() < 0.8
    assert calibrated[vocab.encode("len_3_5")].item() > 0.15
    assert calibrated[vocab.encode("len_6_10")].item() > 0.05
