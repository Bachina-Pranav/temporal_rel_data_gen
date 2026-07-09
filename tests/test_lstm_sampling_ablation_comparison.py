from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from compare_lstm_sampling_privacy_ablation import compare_runs  # noqa: E402


def test_sampling_ablation_comparison_writes_json_markdown_and_html(tmp_path):
    run_root = tmp_path / "runs"
    write_fake_run(run_root, "v5_exact_block_only", checkpoint_source="v5", seconds=100.0, summary_overlap=0.0)
    write_fake_run(run_root, "v51_no_block_no_ngram", checkpoint_source="v5.1", seconds=200.0, summary_overlap=0.2)

    payload = compare_runs(
        ["v5_exact_block_only", "v51_no_block_no_ngram"],
        run_root,
        tmp_path / "comparison",
    )

    assert (tmp_path / "comparison" / "comparison.json").exists()
    assert (tmp_path / "comparison" / "comparison.md").exists()
    assert (tmp_path / "comparison" / "comparison.html").exists()
    assert payload["verdicts"]["best_speed"] == "v5_exact_block_only"
    assert payload["verdicts"]["recommended_default"] == "v5_exact_block_only"


def write_fake_run(run_root: Path, variant: str, *, checkpoint_source: str, seconds: float, summary_overlap: float) -> None:
    run_dir = run_root / variant
    metadata = run_dir / "metadata"
    evaluation = run_dir / "evaluation"
    metadata.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    write_json(
        metadata / "sampling_config.json",
        {
            "variant": variant,
            "checkpoint_source": checkpoint_source,
            "checkpoint_path": f"{checkpoint_source}/best.pt",
            "config_path": f"{checkpoint_source}.yaml",
        },
    )
    write_json(
        metadata / "runtime_sampling_fast.json",
        {
            "total_sampling_seconds": seconds,
            "projected_hours_for_10m_rows": seconds / 50_000 * 10_000_000 / 3600,
            "rows_per_second": 50_000 / seconds,
            "seconds_per_1000_rows": seconds / 50,
            "review_text_decoding_seconds": seconds * 0.8,
            "summary_decoding_seconds": seconds * 0.1,
            "exact_train_overlap_blocking_enabled": True,
            "summary_exact_blocking_enabled": True,
            "review_text_exact_blocking_enabled": True,
            "no_repeat_ngram_enabled": False,
            "summary_no_repeat_ngram_enabled": False,
            "review_text_no_repeat_ngram_enabled": False,
            "summary_temperature": 0.9,
            "review_text_temperature": 0.9,
            "summary_top_p": 0.95,
            "review_text_top_p": 0.95,
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
            "marginal_categorical": {
                "rating_distribution_l1": 0.1,
                "verified_distribution_l1": 0.05,
            },
            "joint": {
                "rating_verified_joint_l1": 0.15,
                "rating_distribution_given_verified_l1": 0.1,
                "verified_rate_by_rating_mae": 0.02,
            },
            "conditional_fidelity": {
                "customer_rating_top_1000_mae": 0.1,
                "customer_verified_top_1000_mae": 0.1,
                "product_rating_top_1000_mae": 0.1,
                "product_verified_top_1000_mae": 0.1,
            },
            "length_diagnostics": {
                "summary_length_ks": 0.05,
                "summary_length_bucket_l1": 0.1,
                "review_text_length_ks": 0.1,
                "review_text_length_bucket_l1": 0.1,
                "review_text_length_mean_synthetic": 100.0,
                "review_text_length_p95_synthetic": 300.0,
                "review_text_length_p99_synthetic": 500.0,
            },
            "text": {
                "summary_unique_rate": 0.8,
                "review_text_unique_rate": 0.95,
                "summary_distinct_1": 0.1,
                "summary_distinct_2": 0.2,
                "review_text_distinct_1": 0.3,
                "review_text_distinct_2": 0.4,
                "summary_top_100_overlap_rate": 0.2,
                "review_text_top_100_overlap_rate": 0.1,
            },
            "text_privacy": {
                "summary_exact_train_overlap_rate": summary_overlap,
                "review_text_exact_train_overlap_rate": 0.0,
                "summary_nearest_neighbor_rougeL_mean": 0.1,
                "summary_nearest_neighbor_token_jaccard_mean": 0.1,
                "review_text_nearest_neighbor_rougeL_mean": 0.1,
                "review_text_nearest_neighbor_token_jaccard_mean": 0.1,
            },
            "text_consistency": {
                "rating_text_consistency_accuracy": 0.5,
                "rating_review_text_consistency_accuracy": 0.5,
                "verified_text_predictor_auc": 0.6,
                "verified_review_text_predictor_auc": 0.6,
                "real_summary_review_text_rougeL_mean": 0.03,
                "synthetic_summary_review_text_rougeL_mean": 0.03,
                "real_summary_review_text_token_jaccard_mean": 0.02,
                "synthetic_summary_review_text_token_jaccard_mean": 0.02,
                "real_summary_review_text_exact_match_rate": 0.0,
                "synthetic_summary_review_text_exact_match_rate": 0.0,
            },
        },
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
