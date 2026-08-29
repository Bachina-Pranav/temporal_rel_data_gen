from __future__ import annotations

import pandas as pd
import json
import yaml

from attribute_generation.qwen_text_decoder.experiment import (
    alignment_audit,
    conditioning_prefix,
    encode_training_example,
    parse_generated_text,
    serialize_example,
    QwenTextExperiment,
    REQUIRED_RUNTIME_VERSIONS,
)


def test_serialization_excludes_ids_time_and_lengths():
    row = {"customer_id": 9, "product_id": 8, "review_time": "2020-01-01", "rating": 2.0, "verified": True, "summary": "not good", "review_text": "stopped working", "review_text_length": 99}
    value = serialize_example(row, "<eos>")
    assert value == "Rating: 2\nVerified: true\nSummary: not good\nReview: stopped working<eos>"
    assert "customer" not in value and "2020" not in value and "99" not in value


def test_parser_tracks_missing_review_without_filling_real_text():
    summary, review, flags = parse_generated_text("Summary: Fine")
    assert summary == "Fine" and review == ""
    assert flags["parse_failure"] and flags["missing_review_marker"]


def test_exact_ordered_alignment_is_required():
    real = pd.DataFrame({"customer_id": [1, 2], "product_id": [3, 4], "review_time": ["2020-01-01", "2020-01-02"]})
    assert alignment_audit(real, real.copy())["aligned"]
    assert not alignment_audit(real, real.iloc[::-1].reset_index(drop=True))["aligned"]


def test_conditioning_contains_only_semantic_attributes():
    assert conditioning_prefix(5, False) == "Rating: 5\nVerified: false\n"


def test_prefix_tokens_are_all_masked_from_causal_loss():
    class Encoded:
        def __init__(self, ids): self.input_ids = ids
    class Tokenizer:
        eos_token = "<eos>"
        def __call__(self, text, **_): return Encoded(list(range(1, len(text.split()) + 1)))
    row = {"rating": 3, "verified": True, "summary": "works well", "review_text": "still works"}
    tokenizer = Tokenizer()
    encoded = encode_training_example(row, tokenizer, 100)
    prefix_length = len(tokenizer(conditioning_prefix(3, True)).input_ids)
    assert encoded["labels"][:prefix_length] == [-100] * prefix_length
    assert all(value != -100 for value in encoded["labels"][prefix_length:])


def test_offline_preflight_records_exact_frozen_split_hashes(tmp_path):
    benchmark = tmp_path / "benchmark"; benchmark.mkdir()
    frame = pd.DataFrame({"customer_id": [1], "product_id": [2], "review_time": ["2020-01-01"], "rating": [5], "verified": [True], "summary": ["good"], "review_text": ["works"]})
    for split in ("train", "validation", "test"):
        frame.to_csv(benchmark / f"{split}_real.csv", index=False)
    (benchmark / "benchmark_manifest.json").write_text(json.dumps({"row_counts": {"evaluation_real": 1}}))
    output = tmp_path / "output"
    config = {"seed": 42, "output_dir": str(output), "model": {"model_id": "Qwen/Qwen3-0.6B-Base"}, "data": {"benchmark_dir": str(benchmark), "structured_candidates": [], "lstm_candidates": [], "diffusion_text_candidates": []}}
    config_path = tmp_path / "config.yaml"; config_path.write_text(yaml.safe_dump(config))
    report = QwenTextExperiment(config_path).preflight(resolve_model=False)
    assert report["row_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert all(len(item["sha256"]) == 64 for item in report["splits"].values())
    assert (output / "preflight.md").is_file()


def test_huggingface_runtime_is_fully_pinned():
    assert REQUIRED_RUNTIME_VERSIONS == {
        "torch": "2.2.2",
        "torchvision": "0.17.2",
        "torchaudio": "2.2.2",
        "transformers": "4.51.3",
        "tokenizers": "0.21.1",
        "peft": "0.15.2",
        "accelerate": "1.6.0",
        "huggingface-hub": "0.30.2",
    }
