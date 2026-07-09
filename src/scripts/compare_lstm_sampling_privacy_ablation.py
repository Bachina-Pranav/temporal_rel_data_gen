#!/usr/bin/env python3
"""Compare v5.2 LSTM sampling privacy ablation variants."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


V5_BASELINE = {
    "total_sampling_seconds": 474.97,
    "projected_hours_for_10m_rows": 26.39,
    "rating_distribution_l1": 0.1383,
    "verified_distribution_l1": 0.0736,
    "rating_verified_joint_l1": 0.1623,
    "summary_length_ks": 0.0482,
    "review_text_length_ks": 0.1135,
    "summary_exact_train_overlap_rate": 0.29124,
    "review_text_exact_train_overlap_rate": 0.00852,
    "summary_review_text_rougeL_mean": 0.02933,
    "summary_review_text_token_jaccard_mean": 0.02093,
}

V51_BASELINE = {
    "total_sampling_seconds": 1358.81,
    "projected_hours_for_10m_rows": 75.49,
    "rating_distribution_l1": 0.1611,
    "verified_distribution_l1": 0.0593,
    "rating_verified_joint_l1": 0.1819,
    "summary_length_ks": 0.1257,
    "review_text_length_ks": 0.1145,
    "summary_exact_train_overlap_rate": 0.0,
    "review_text_exact_train_overlap_rate": 0.0,
    "summary_review_text_rougeL_mean": 0.02288,
    "summary_review_text_token_jaccard_mean": 0.01531,
}

TABLE_COLUMNS = [
    "variant",
    "checkpoint_source",
    "total_sampling_seconds",
    "projected_hours_for_10m_rows",
    "rows_per_second",
    "review_text_decoding_seconds",
    "rating_distribution_l1",
    "verified_distribution_l1",
    "rating_verified_joint_l1",
    "summary_length_ks",
    "review_text_length_ks",
    "summary_exact_train_overlap_rate",
    "review_text_exact_train_overlap_rate",
    "summary_unique_rate",
    "review_text_unique_rate",
    "summary_review_text_rougeL_mean",
    "summary_review_text_token_jaccard_mean",
    "rating_text_consistency_accuracy",
    "rating_review_text_consistency_accuracy",
    "verified_review_text_predictor_auc",
]

METRIC_PATHS = {
    "invalid_rating_rate": ["validity.invalid_rating_rate"],
    "invalid_verified_rate": ["validity.invalid_verified_rate"],
    "empty_summary_rate": ["validity.empty_summary_rate"],
    "empty_review_text_rate": ["validity.empty_review_text_rate"],
    "rating_distribution_l1": ["marginal_categorical.rating_distribution_l1"],
    "verified_distribution_l1": ["marginal_categorical.verified_distribution_l1"],
    "rating_verified_joint_l1": ["joint.rating_verified_joint_l1"],
    "rating_distribution_given_verified_l1": ["joint.rating_distribution_given_verified_l1"],
    "verified_rate_by_rating_mae": ["joint.verified_rate_by_rating_mae"],
    "customer_rating_top_1000_mae": ["conditional_fidelity.customer_rating_top_1000_mae"],
    "customer_verified_top_1000_mae": ["conditional_fidelity.customer_verified_top_1000_mae"],
    "product_rating_top_1000_mae": ["conditional_fidelity.product_rating_top_1000_mae"],
    "product_verified_top_1000_mae": ["conditional_fidelity.product_verified_top_1000_mae"],
    "summary_length_ks": ["length_diagnostics.summary_length_ks", "text.summary_length_ks", "validity.summary_length_ks"],
    "summary_length_bucket_l1": ["length_diagnostics.summary_length_bucket_l1"],
    "review_text_length_ks": [
        "length_diagnostics.review_text_length_ks",
        "text.review_text_length_ks",
        "validity.review_text_length_ks",
    ],
    "review_text_length_bucket_l1": ["length_diagnostics.review_text_length_bucket_l1"],
    "review_text_length_mean_synthetic": [
        "length_diagnostics.review_text_length_mean_synthetic",
        "validity.review_text_length_mean_synthetic",
    ],
    "review_text_length_p95_synthetic": ["length_diagnostics.review_text_length_p95_synthetic"],
    "review_text_length_p99_synthetic": ["length_diagnostics.review_text_length_p99_synthetic"],
    "summary_unique_rate": ["text.summary_unique_rate", "text.unique_summary_rate"],
    "review_text_unique_rate": ["text.review_text_unique_rate"],
    "summary_distinct_1": ["text.summary_distinct_1", "text.distinct_1"],
    "summary_distinct_2": ["text.summary_distinct_2", "text.distinct_2"],
    "review_text_distinct_1": ["text.review_text_distinct_1"],
    "review_text_distinct_2": ["text.review_text_distinct_2"],
    "summary_top_100_overlap_rate": ["text.summary_top_100_overlap_rate", "text.top_100_summary_overlap_rate"],
    "review_text_top_100_overlap_rate": ["text.review_text_top_100_overlap_rate"],
    "summary_exact_train_overlap_rate": ["text_privacy.summary_exact_train_overlap_rate"],
    "review_text_exact_train_overlap_rate": ["text_privacy.review_text_exact_train_overlap_rate"],
    "summary_nearest_neighbor_rougeL_mean": ["text_privacy.summary_nearest_neighbor_rougeL_mean"],
    "summary_nearest_neighbor_token_jaccard_mean": ["text_privacy.summary_nearest_neighbor_token_jaccard_mean"],
    "review_text_nearest_neighbor_rougeL_mean": ["text_privacy.review_text_nearest_neighbor_rougeL_mean"],
    "review_text_nearest_neighbor_token_jaccard_mean": [
        "text_privacy.review_text_nearest_neighbor_token_jaccard_mean",
    ],
    "rating_text_consistency_accuracy": ["text_consistency.rating_text_consistency_accuracy"],
    "rating_review_text_consistency_accuracy": ["text_consistency.rating_review_text_consistency_accuracy"],
    "verified_text_predictor_auc": ["text_consistency.verified_text_predictor_auc"],
    "verified_review_text_predictor_auc": ["text_consistency.verified_review_text_predictor_auc"],
    "real_summary_review_text_rougeL_mean": ["text_consistency.real_summary_review_text_rougeL_mean"],
    "synthetic_summary_review_text_rougeL_mean": ["text_consistency.synthetic_summary_review_text_rougeL_mean"],
    "summary_review_text_rougeL_mean": [
        "text_consistency.synthetic_summary_review_text_rougeL_mean",
        "text_consistency.summary_review_text_rougeL_mean",
    ],
    "real_summary_review_text_token_jaccard_mean": [
        "text_consistency.real_summary_review_text_token_jaccard_mean",
    ],
    "synthetic_summary_review_text_token_jaccard_mean": [
        "text_consistency.synthetic_summary_review_text_token_jaccard_mean",
    ],
    "summary_review_text_token_jaccard_mean": [
        "text_consistency.synthetic_summary_review_text_token_jaccard_mean",
        "text_consistency.summary_review_text_token_jaccard_mean",
    ],
    "real_summary_review_text_exact_match_rate": ["text_consistency.real_summary_review_text_exact_match_rate"],
    "synthetic_summary_review_text_exact_match_rate": [
        "text_consistency.synthetic_summary_review_text_exact_match_rate",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v5.2 LSTM sampling privacy ablation runs.")
    parser.add_argument("--run-root", required=True, help="Directory containing runs/<variant> style outputs, or the runs directory itself.")
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
    payload = {
        "baselines": {"v5_fast": V5_BASELINE, "v5_1_privacy_alignment": V51_BASELINE},
        "rows": rows,
        "deltas_vs_v5": {row["variant"]: deltas(row, V5_BASELINE) for row in rows},
        "deltas_vs_v51": {row["variant"]: deltas(row, V51_BASELINE) for row in rows},
        "verdicts": verdicts(rows),
    }
    write_json(payload, output_dir / "comparison.json")
    write_markdown(payload, output_dir / "comparison.md")
    write_html(payload, output_dir / "comparison.html")
    print(output_dir / "comparison.json")
    print(output_dir / "comparison.md")
    print(output_dir / "comparison.html")
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
        "checkpoint_path": sampling_config.get("checkpoint_path"),
        "config_path": sampling_config.get("config_path"),
        "eval_metrics_path": str(eval_path),
        "runtime_metrics_path": str(runtime_path),
        "sampling_config_path": str(config_path),
    }
    for key in [
        "total_sampling_seconds",
        "projected_hours_for_10m_rows",
        "projected_seconds_for_10m_rows",
        "rows_per_second",
        "seconds_per_1000_rows",
        "review_text_decoding_seconds",
        "summary_decoding_seconds",
        "graph_context_total_seconds",
        "detokenization_seconds",
        "csv_writing_seconds",
        "misc_overhead_seconds",
        "exact_train_overlap_blocking_enabled",
        "summary_exact_blocking_enabled",
        "review_text_exact_blocking_enabled",
        "no_repeat_ngram_enabled",
        "summary_no_repeat_ngram_enabled",
        "review_text_no_repeat_ngram_enabled",
        "summary_temperature",
        "review_text_temperature",
        "summary_top_p",
        "review_text_top_p",
    ]:
        row[key] = runtime.get(key)
    for key, paths in METRIC_PATHS.items():
        row[key] = first_path(metrics, paths)
    return row


def deltas(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    return {
        key: numeric_delta(row.get(key), value)
        for key, value in baseline.items()
    }


def verdicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "best_speed": min(rows, key=lambda row: value_or_inf(row.get("projected_hours_for_10m_rows")))["variant"],
        "best_privacy": min(rows, key=privacy_sort_key)["variant"],
        "best_alignment": max(rows, key=alignment_score)["variant"],
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
        le(row.get("total_sampling_seconds"), 750.0)
        and le(row.get("projected_hours_for_10m_rows"), 42.0)
        and eq(row.get("invalid_rating_rate"), 0.0)
        and eq(row.get("invalid_verified_rate"), 0.0)
        and le(row.get("empty_summary_rate"), 0.01)
        and le(row.get("empty_review_text_rate"), 0.01)
        and le(row.get("rating_distribution_l1"), 0.18)
        and le(row.get("rating_verified_joint_l1"), 0.22)
        and le(row.get("summary_length_ks"), 0.10)
        and le(row.get("review_text_length_ks"), 0.16)
        and le(row.get("summary_exact_train_overlap_rate"), 0.05)
        and le(row.get("review_text_exact_train_overlap_rate"), 0.002)
    )


def privacy_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        value_or_inf(row.get("summary_exact_train_overlap_rate")) + value_or_inf(row.get("review_text_exact_train_overlap_rate")),
        value_or_inf(row.get("projected_hours_for_10m_rows")),
    )


def alignment_score(row: dict[str, Any]) -> float:
    return value_or_neg_inf(row.get("summary_review_text_rougeL_mean")) + value_or_neg_inf(row.get("summary_review_text_token_jaccard_mean"))


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


def value_or_neg_inf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -float("inf")


def numeric_delta(value: Any, baseline: Any) -> float | None:
    try:
        return float(value) - float(baseline)
    except (TypeError, ValueError):
        return None


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
        "# v5.2 LSTM Sampling Privacy Ablation",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in TABLE_COLUMNS) + " |")
    lines.extend(["", "## Verdicts", ""])
    for key, value in payload.get("verdicts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Deltas Vs v5 Fast Baseline", ""])
    delta_columns = [
        "total_sampling_seconds",
        "projected_hours_for_10m_rows",
        "rating_distribution_l1",
        "rating_verified_joint_l1",
        "summary_exact_train_overlap_rate",
        "review_text_exact_train_overlap_rate",
    ]
    lines.append("| variant | " + " | ".join(delta_columns) + " |")
    lines.append("|---|" + "|".join("---:" for _ in delta_columns) + "|")
    for row in rows:
        deltas_for_row = payload["deltas_vs_v5"].get(row["variant"], {})
        lines.append("| " + row["variant"] + " | " + " | ".join(fmt(deltas_for_row.get(column)) for column in delta_columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(payload: dict[str, Any], path: Path) -> None:
    rows = payload["rows"]
    lines = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>v5.2 LSTM Sampling Privacy Ablation</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px 8px;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#f5f5f5}</style>",
        "</head><body>",
        "<h1>v5.2 LSTM Sampling Privacy Ablation</h1>",
        "<table>",
        "<thead><tr>" + "".join(f"<th>{html.escape(column)}</th>" for column in TABLE_COLUMNS) + "</tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{html.escape(fmt(row.get(column)))}</td>" for column in TABLE_COLUMNS) + "</tr>")
    lines.extend(["</tbody></table>", "<h2>Verdicts</h2>", "<ul>"])
    for key, value in payload.get("verdicts", {}).items():
        lines.append(f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>")
    lines.extend(["</ul>", "</body></html>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
