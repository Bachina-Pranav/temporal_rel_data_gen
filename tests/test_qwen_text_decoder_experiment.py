from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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
from attribute_generation.qwen_text_decoder.followup import (
    build_prompt,
    discover_diffusion_artifacts,
    parse_policy_continuation,
    select_fixed_subset,
    trim_generated_ids,
)
from attribute_generation.qwen_text_decoder.phase1 import (
    align_exact_population,
    matched_memorization_metrics,
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


def test_followup_prompts_exclude_ids_time_and_lengths():
    row = pd.Series(
        {
            "customer_id": 99,
            "product_id": 44,
            "review_time": "2020-01-01",
            "rating": 2,
            "verified": True,
            "summary": "short summary",
            "review_text_length": 800,
        }
    )
    oracle = build_prompt(row, "rating_verified", "oracle")
    assert oracle == "Rating: 2\nVerified: true\nSummary: short summary\nReview:"
    assert "99" not in oracle and "2020" not in oracle and "800" not in oracle
    assert build_prompt(row, "none", "generated") == "Summary:"
    assert build_prompt(row, "rating", "generated") == "Rating: 2\nSummary:"
    assert build_prompt(row, "verified", "generated") == "Verified: true\nSummary:"


def test_followup_parser_supports_generated_oracle_and_omitted_summary():
    summary, review, status = parse_policy_continuation(
        "A useful item. Review: It worked well.", summary_mode="generated"
    )
    assert (summary, review) == ("A useful item.", "It worked well.")
    assert not status["parse_failure"]
    summary, review, status = parse_policy_continuation(
        "It worked well.", summary_mode="oracle", oracle_summary="Useful"
    )
    assert (summary, review) == ("Useful", "It worked well.")
    summary, review, _ = parse_policy_continuation(
        "It worked well.", summary_mode="omitted"
    )
    assert (summary, review) == ("", "It worked well.")
    assert trim_generated_ids([10, 11, 2, 2, 2], 2) == [10, 11, 2]
    assert trim_generated_ids([10, 11], 2) == [10, 11]


def test_followup_subset_uses_complete_small_validation_and_seeded_large_sample():
    small = pd.DataFrame({"value": range(3982)})
    subset, metadata = select_fixed_subset(
        small, target_rows=5000, maximum_rows=10000, seed=42
    )
    assert len(subset) == 3982
    assert subset["_source_row_index"].tolist() == list(range(3982))
    assert "complete frozen validation split" in metadata["selection_reason"]
    large = pd.DataFrame({"value": range(6000)})
    first, _ = select_fixed_subset(large, target_rows=5000, maximum_rows=10000, seed=42)
    second, _ = select_fixed_subset(large, target_rows=5000, maximum_rows=10000, seed=42)
    assert first["_source_row_index"].tolist() == second["_source_row_index"].tolist()
    assert first["_source_row_index"].is_monotonic_increasing


def test_followup_diffusion_discovery_requires_complete_latest_root(tmp_path):
    older = tmp_path / "diagnostics" / "older"
    newer = tmp_path / "diagnostics" / "newer"
    for root in (older, newer):
        for index, label in enumerate(("O1", "O2", "O3", "O4"), start=1):
            run = root / f"{index:03d}_{label}_seed42"
            run.mkdir(parents=True)
            (run / "synthetic_table.csv").write_text("summary,review_text\na,b\n")
            (run / "run_manifest.json").write_text(
                json.dumps({"status": "completed", "label": label, "seed": 42})
            )
    newer.touch()
    selected, artifacts = discover_diffusion_artifacts(
        [tmp_path / "diagnostics"], ["O1", "O2", "O3", "O4"]
    )
    assert selected == newer
    assert {item["label"] for item in artifacts} == {"O1", "O2", "O3", "O4"}
    assert all("description" in item["conditioning"] for item in artifacts)


def test_followup_does_not_require_unavailable_generated_structured_mode(tmp_path):
    config_path = tmp_path / "base.yaml"
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "output_dir": str(tmp_path / "base"),
                "data": {"benchmark_dir": str(benchmark)},
            }
        )
    )
    followup_path = tmp_path / "followup.yaml"
    followup_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "base_experiment_config": str(config_path),
                "base_output_dir": str(tmp_path / "base"),
                "output_dir": str(tmp_path / "base" / "followup"),
            }
        )
    )
    from attribute_generation.qwen_text_decoder.followup import QwenFollowupExperiment

    required = QwenFollowupExperiment(followup_path)._required_base_artifacts()
    assert not any("generated_structured" in str(path) for path in required)


def test_phase1_alignment_reorders_exact_heldout_multiset_without_positional_slice():
    real = pd.DataFrame(
        {
            "customer_id": [1, 2, 1],
            "product_id": [9, 8, 9],
            "review_time": ["2020-01-01", "2020-01-02", "2020-01-01"],
        }
    )
    candidate = real.assign(
        summary=["first", "middle", "second"],
        review_text=["a", "b", "c"],
    ).iloc[[2, 0, 1]].reset_index(drop=True)
    aligned, audit = align_exact_population(real, candidate)
    assert audit["aligned"]
    assert not audit["positional_slice_used"]
    assert audit["reference_duplicate_key_groups"] == 1
    assert aligned.summary.tolist() == ["second", "middle", "first"]


def test_phase1_full_table_alignment_locates_test_rows_from_verified_real_identity():
    full_real = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "product_id": [9, 9, 8],
            "review_time": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "rating": [1, 5, 3],
            "verified": [False, True, True],
            "summary": ["bad", "good", "fine"],
            "review_text": ["failed", "worked", "okay"],
        }
    )
    reference = full_real.iloc[[1, 2]].reset_index(drop=True)
    candidate = full_real[["customer_id", "product_id", "review_time"]].copy()
    candidate["summary"] = ["synthetic row zero", "synthetic row one", "synthetic row two"]
    candidate["review_text"] = ["zero", "one", "two"]
    aligned, audit = align_exact_population(
        reference, candidate, full_reference=full_real
    )
    assert audit["aligned"]
    assert audit["full_table_ordered_key_match"]
    assert audit["real_target_content_used_only_for_source_row_location"]
    assert aligned.summary.tolist() == ["synthetic row one", "synthetic row two"]


def test_phase1_memorization_control_uses_matched_metrics_and_requested_quantiles():
    train = pd.DataFrame(
        {
            "summary": ["same", "different title"],
            "review_text": ["works very well", "failed quickly"],
        }
    )
    real = pd.DataFrame(
        {"summary": ["same", "new"], "review_text": ["works well", "brand new"]}
    )
    qwen = pd.DataFrame(
        {
            "summary": ["same", "same"],
            "review_text": ["works very well", "works well"],
        }
    )
    result = matched_memorization_metrics(train, real, qwen, training_rows=20)
    assert result["summary"]["real_heldout"]["exact_train_overlap_rate"] == 0.5
    assert result["summary"]["qwen"]["exact_train_overlap_rate"] == 1.0
    for field in ("summary", "review_text"):
        for side in ("real_heldout", "qwen"):
            assert {"mean", "median", "p90", "p95", "max"}.issubset(
                result[field][side]["nearest_neighbor"]
            )


def test_phase1_config_has_exactly_three_fresh_qwen_generation_policies():
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1]
         / "configs/experiments/qwen_text_decoder_06b_phase1.yaml").read_text()
    )
    assert set(config["generation"]["policies"]) == {
        "oracle_summary",
        "no_summary",
        "generated_structured",
    }
    assert config["compute_budget"]["fresh_qwen_generations"] == 3
    projected = (
        config["evaluation_population"]["expected_rows"]
        * config["compute_budget"]["fresh_qwen_generations"]
        / config["compute_budget"]["measured_qwen_rows_per_second"]
        / 3600
        + config["compute_budget"]["estimated_non_generation_gpu_hours"]
    )
    assert projected < config["compute_budget"]["hard_gpu_hours"]
    assert config["evaluation"]["embedding_revision"] == (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
