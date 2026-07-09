#!/usr/bin/env python3
"""Compare v5.3 length-preserving LSTM privacy ablation variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TABLE_COLUMNS = [
    "variant",
    "checkpoint_source",
    "total_sampling_seconds",
    "projected_hours_for_10m_rows",
    "rows_per_second",
    "rating_distribution_l1",
    "verified_distribution_l1",
    "rating_verified_joint_l1",
    "summary_length_ks",
    "summary_length_bucket_l1",
    "review_text_length_ks",
    "review_text_length_bucket_l1",
    "summary_exact_train_overlap_rate",
    "review_text_exact_train_overlap_rate",
    "summary_unique_rate",
    "review_text_unique_rate",
    "summary_review_text_rougeL_mean",
    "summary_review_text_token_jaccard_mean",
    "rating_text_consistency_accuracy",
    "rating_review_text_consistency_accuracy",
    "verified_review_text_predictor_auc",
    "summary_length_bucket_preserved_count",
    "summary_length_bucket_changed_count",
    "review_text_length_bucket_preserved_count",
    "review_text_length_bucket_changed_count",
]

METRIC_PATHS = {
    "invalid_rating_rate": ["validity.invalid_rating_rate"],
    "invalid_verified_rate": ["validity.invalid_verified_rate"],
    "empty_summary_rate": ["validity.empty_summary_rate"],
    "empty_review_text_rate": ["validity.empty_review_text_rate"],
    "rating_distribution_l1": ["marginal_categorical.rating_distribution_l1"],
    "verified_distribution_l1": ["marginal_categorical.verified_distribution_l1"],
    "rating_verified_joint_l1": ["joint.rating_verified_joint_l1"],
    "summary_length_ks": ["length_diagnostics.summary_length_ks", "text.summary_length_ks", "validity.summary_length_ks"],
    "summary_length_bucket_l1": ["length_diagnostics.summary_length_bucket_l1"],
    "summary_length_mean_real": ["length_diagnostics.summary_length_mean_real", "validity.summary_length_mean_real"],
    "summary_length_mean_synthetic": ["length_diagnostics.summary_length_mean_synthetic", "validity.summary_length_mean_synthetic"],
    "summary_length_p95_synthetic": ["length_diagnostics.summary_length_p95_synthetic"],
    "summary_length_p99_synthetic": ["length_diagnostics.summary_length_p99_synthetic"],
    "review_text_length_ks": [
        "length_diagnostics.review_text_length_ks",
        "text.review_text_length_ks",
        "validity.review_text_length_ks",
    ],
    "review_text_length_bucket_l1": ["length_diagnostics.review_text_length_bucket_l1"],
    "review_text_length_mean_real": ["length_diagnostics.review_text_length_mean_real", "validity.review_text_length_mean_real"],
    "review_text_length_mean_synthetic": [
        "length_diagnostics.review_text_length_mean_synthetic",
        "validity.review_text_length_mean_synthetic",
    ],
    "review_text_length_p95_synthetic": ["length_diagnostics.review_text_length_p95_synthetic"],
    "review_text_length_p99_synthetic": ["length_diagnostics.review_text_length_p99_synthetic"],
    "summary_exact_train_overlap_rate": ["text_privacy.summary_exact_train_overlap_rate"],
    "review_text_exact_train_overlap_rate": ["text_privacy.review_text_exact_train_overlap_rate"],
    "summary_nearest_neighbor_rougeL_mean": ["text_privacy.summary_nearest_neighbor_rougeL_mean"],
    "summary_nearest_neighbor_token_jaccard_mean": ["text_privacy.summary_nearest_neighbor_token_jaccard_mean"],
    "review_text_nearest_neighbor_rougeL_mean": ["text_privacy.review_text_nearest_neighbor_rougeL_mean"],
    "review_text_nearest_neighbor_token_jaccard_mean": ["text_privacy.review_text_nearest_neighbor_token_jaccard_mean"],
    "summary_unique_rate": ["text.summary_unique_rate", "text.unique_summary_rate"],
    "review_text_unique_rate": ["text.review_text_unique_rate"],
    "summary_review_text_rougeL_mean": [
        "text_consistency.synthetic_summary_review_text_rougeL_mean",
        "text_consistency.summary_review_text_rougeL_mean",
    ],
    "summary_review_text_token_jaccard_mean": [
        "text_consistency.synthetic_summary_review_text_token_jaccard_mean",
        "text_consistency.summary_review_text_token_jaccard_mean",
    ],
    "rating_text_consistency_accuracy": ["text_consistency.rating_text_consistency_accuracy"],
    "rating_review_text_consistency_accuracy": ["text_consistency.rating_review_text_consistency_accuracy"],
    "verified_review_text_predictor_auc": ["text_consistency.verified_review_text_predictor_auc"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v5.3 length-preserving privacy ablation runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_runs(args.variants, args.run_root, args.output_dir)


def compare_runs(variants: list[str], run_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    if (run_root / "runs").is_dir():
        run_root = run_root / "runs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [build_row(variant, run_root / variant) for variant in variants]
    payload = {"rows": rows, "verdicts": verdicts(rows), "strong_targets": strong_targets(rows)}
    write_json(payload, output_dir / "comparison.json")
    write_markdown(payload, output_dir / "comparison.md")
    print(output_dir / "comparison.json")
    print(output_dir / "comparison.md")
    return payload


def build_row(variant: str, run_dir: Path) -> dict[str, Any]:
    eval_path = run_dir / "evaluation" / "eval_metrics_fast_sampler_fixed_decode_normalized.json"
    runtime_path = run_dir / "metadata" / "runtime_sampling_fast.json"
    config_path = run_dir / "metadata" / "sampling_config.json"
    metrics = load_json(eval_path)
    runtime = load_json(runtime_path)
    sampling_config = load_json(config_path) if config_path.exists() else {}
    row: dict[str, Any] = {
        "variant": variant,
        "checkpoint_source": sampling_config.get("checkpoint_source"),
        "eval_metrics_path": str(eval_path),
        "runtime_metrics_path": str(runtime_path),
        "sampling_config_path": str(config_path),
    }
    for key in [
        "total_sampling_seconds",
        "projected_hours_for_10m_rows",
        "rows_per_second",
        "review_text_decoding_seconds",
        "summary_decoding_seconds",
        "length_preserving_exact_blocking_enabled",
        "dependency_aware_text_decoding_enabled",
        "text_fields_with_privacy_blocking",
        "no_repeat_ngram_enabled",
        "review_text_no_repeat_ngram_enabled",
        "summary_length_bucket_preserved_count",
        "summary_length_bucket_changed_count",
        "review_text_length_bucket_preserved_count",
        "review_text_length_bucket_changed_count",
        "summary_resample_attempts_total",
        "summary_resample_attempts_mean",
        "review_text_resample_attempts_total",
        "review_text_resample_attempts_mean",
    ]:
        row[key] = runtime.get(key)
    for key, paths in METRIC_PATHS.items():
        row[key] = first_path(metrics, paths)
    return row


def verdicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "best_speed": min(rows, key=lambda row: value_or_inf(row.get("projected_hours_for_10m_rows")))["variant"],
        "best_privacy": min(rows, key=privacy_sort_key)["variant"],
        "best_length_preservation": min(rows, key=length_preservation_sort_key)["variant"],
        "best_balanced": min(rows, key=balanced_sort_key)["variant"],
        "recommended_default": recommended_default(rows),
    }


def recommended_default(rows: list[dict[str, Any]]) -> str | None:
    eligible = [row for row in rows if is_recommended_eligible(row)]
    if not eligible:
        return min(rows, key=balanced_sort_key)["variant"] if rows else None
    return min(eligible, key=lambda row: value_or_inf(row.get("projected_hours_for_10m_rows")))["variant"]


def is_recommended_eligible(row: dict[str, Any]) -> bool:
    return (
        eq(row.get("invalid_rating_rate"), 0.0)
        and eq(row.get("invalid_verified_rate"), 0.0)
        and le(row.get("empty_summary_rate"), 0.01)
        and le(row.get("empty_review_text_rate"), 0.01)
        and le(row.get("projected_hours_for_10m_rows"), 35.0)
        and le(row.get("rating_distribution_l1"), 0.18)
        and le(row.get("rating_verified_joint_l1"), 0.22)
        and le(row.get("summary_length_ks"), 0.08)
        and le(row.get("review_text_length_ks"), 0.16)
        and le(row.get("summary_exact_train_overlap_rate"), 0.02)
        and le(row.get("review_text_exact_train_overlap_rate"), 0.002)
    )


def strong_targets(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        row["variant"]: (
            eq(row.get("summary_exact_train_overlap_rate"), 0.0)
            and eq(row.get("review_text_exact_train_overlap_rate"), 0.0)
            and le(row.get("summary_length_ks"), 0.06)
            and le(row.get("review_text_length_ks"), 0.13)
            and le(row.get("projected_hours_for_10m_rows"), 30.0)
        )
        for row in rows
    }


def privacy_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        value_or_inf(row.get("summary_exact_train_overlap_rate")) + value_or_inf(row.get("review_text_exact_train_overlap_rate")),
        value_or_inf(row.get("projected_hours_for_10m_rows")),
    )


def length_preservation_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        value_or_inf(row.get("summary_length_ks")) + value_or_inf(row.get("review_text_length_ks")),
        value_or_inf(row.get("summary_length_bucket_l1")) + value_or_inf(row.get("review_text_length_bucket_l1")),
    )


def balanced_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    quality = sum(
        value_or_inf(row.get(key))
        for key in [
            "rating_distribution_l1",
            "verified_distribution_l1",
            "rating_verified_joint_l1",
            "summary_length_ks",
            "review_text_length_ks",
            "summary_exact_train_overlap_rate",
        ]
    )
    quality += 5.0 * value_or_inf(row.get("review_text_exact_train_overlap_rate"))
    return (quality, value_or_inf(row.get("projected_hours_for_10m_rows")))


def first_path(data: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        value = get_path(data, path)
        if value is not None:
            return value
    return None


def get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def le(value: Any, threshold: float) -> bool:
    try:
        return float(value) <= float(threshold)
    except (TypeError, ValueError):
        return False


def eq(value: Any, expected: float) -> bool:
    try:
        return abs(float(value) - float(expected)) <= 1e-12
    except (TypeError, ValueError):
        return False


def value_or_inf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = payload["rows"]
    lines = [
        "# v5.3 Length-Preserving Privacy Ablation",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in TABLE_COLUMNS) + " |")
    lines.extend(["", "## Verdicts", ""])
    for key, value in payload.get("verdicts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Strong Target Pass", ""])
    for variant, passed in payload.get("strong_targets", {}).items():
        lines.append(f"- {variant}: {str(passed).lower()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


if __name__ == "__main__":
    main()
