from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from compare_lstm_length_preserving_privacy_ablation import compare_runs  # noqa: E402


def test_length_preserving_privacy_comparison_writes_outputs(tmp_path):
    run_root = tmp_path / "runs"
    write_fake_run(run_root, "good", seconds=100.0, summary_ks=0.05, summary_overlap=0.0)
    write_fake_run(run_root, "bad", seconds=120.0, summary_ks=0.2, summary_overlap=0.3)

    payload = compare_runs(["good", "bad"], run_root, tmp_path / "comparison")

    assert (tmp_path / "comparison" / "comparison.json").exists()
    assert (tmp_path / "comparison" / "comparison.md").exists()
    assert payload["verdicts"]["recommended_default"] == "good"
    assert payload["verdicts"]["best_length_preservation"] == "good"


def write_fake_run(run_root: Path, variant: str, *, seconds: float, summary_ks: float, summary_overlap: float) -> None:
    run_dir = run_root / variant
    metadata = run_dir / "metadata"
    evaluation = run_dir / "evaluation"
    metadata.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    write_json(metadata / "sampling_config.json", {"variant": variant, "checkpoint_source": "v5.1"})
    write_json(
        metadata / "runtime_sampling_fast.json",
        {
            "total_sampling_seconds": seconds,
            "projected_hours_for_10m_rows": seconds / 50_000 * 10_000_000 / 3600,
            "rows_per_second": 50_000 / seconds,
            "length_preserving_exact_blocking_enabled": True,
            "dependency_aware_text_decoding_enabled": True,
            "text_fields_with_privacy_blocking": ["summary", "review_text"],
            "no_repeat_ngram_enabled": False,
            "review_text_no_repeat_ngram_enabled": False,
            "summary_length_bucket_preserved_count": 5000,
            "summary_length_bucket_changed_count": 0,
            "review_text_length_bucket_preserved_count": 5000,
            "review_text_length_bucket_changed_count": 0,
            "summary_resample_attempts_total": 5,
            "summary_resample_attempts_mean": 1.0,
            "review_text_resample_attempts_total": 2,
            "review_text_resample_attempts_mean": 1.0,
        },
    )
    write_json(
        evaluation / "eval_metrics_fast_sampler_fixed_decode_normalized.json",
        {
            "validity": {
                "invalid_rating_rate": 0.0,
                "invalid_verified_rate": 0.0,
                "empty_summary_rate": 0.0,
                "empty_review_text_rate": 0.0,
            },
            "marginal_categorical": {"rating_distribution_l1": 0.1, "verified_distribution_l1": 0.05},
            "joint": {"rating_verified_joint_l1": 0.15},
            "length_diagnostics": {
                "summary_length_ks": summary_ks,
                "summary_length_bucket_l1": 0.04,
                "review_text_length_ks": 0.1,
                "review_text_length_bucket_l1": 0.08,
            },
            "text_privacy": {
                "summary_exact_train_overlap_rate": summary_overlap,
                "review_text_exact_train_overlap_rate": 0.0,
            },
            "text": {"summary_unique_rate": 0.8, "review_text_unique_rate": 0.95},
            "text_consistency": {
                "synthetic_summary_review_text_rougeL_mean": 0.03,
                "synthetic_summary_review_text_token_jaccard_mean": 0.02,
                "rating_text_consistency_accuracy": 0.5,
                "rating_review_text_consistency_accuracy": 0.5,
                "verified_review_text_predictor_auc": 0.6,
            },
        },
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
