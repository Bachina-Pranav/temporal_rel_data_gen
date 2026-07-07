#!/usr/bin/env python3
"""Compare Conditional TABDLM v2, v3, and v3.1 ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MODELS = ["v2", "v3", "v31a", "v31b", "v31c"]

METRICS = [
    ("marginal_categorical.rating_distribution_l1", "lower", "global_categorical"),
    ("marginal_categorical.verified_distribution_l1", "lower", "global_categorical"),
    ("joint.rating_verified_joint_l1", "lower", "global_categorical"),
    ("conditional_fidelity.customer_rating_top_1000_mae", "lower", "conditional_fidelity"),
    ("conditional_fidelity.customer_verified_top_1000_mae", "lower", "conditional_fidelity"),
    ("conditional_fidelity.product_rating_top_1000_mae", "lower", "conditional_fidelity"),
    ("conditional_fidelity.product_verified_top_1000_mae", "lower", "conditional_fidelity"),
    ("temporal.monthly_rating_mean_corr", "higher", "temporal"),
    ("temporal.monthly_rating_mean_mae", "lower", "temporal"),
    ("temporal.monthly_verified_rate_corr", "higher", "temporal"),
    ("temporal.monthly_verified_rate_mae", "lower", "temporal"),
    ("temporal.monthly_summary_length_corr", "higher", "summary_length"),
    ("temporal.monthly_summary_length_mae", "lower", "summary_length"),
    ("length_diagnostics.summary_length_mean_synthetic", "reference", "summary_length"),
    ("length_diagnostics.summary_length_ks", "lower", "summary_length"),
    ("length_diagnostics.summary_length_bucket_l1", "lower", "summary_length"),
    ("text.distinct_1", "higher", "text_privacy"),
    ("text.distinct_2", "higher", "text_privacy"),
    ("text.unique_summary_rate", "higher", "text_privacy"),
    ("text_privacy.exact_summary_train_overlap_rate", "lower", "text_privacy"),
    ("text_privacy.nearest_neighbor_rougeL_mean", "lower", "text_privacy"),
    ("text_consistency.rating_text_consistency_accuracy", "higher", "text_consistency"),
    ("text_consistency.verified_text_predictor_auc", "higher", "text_consistency"),
]

CATEGORY_GROUPS = {
    "best_global_categorical": ["global_categorical"],
    "best_conditional_fidelity": ["conditional_fidelity"],
    "best_summary_length": ["summary_length"],
    "best_text_privacy": ["text_privacy", "text_consistency"],
    "best_overall_balanced": ["global_categorical", "conditional_fidelity", "summary_length", "text_privacy", "text_consistency", "temporal"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Conditional TABDLM v2/v3/v3.1 ablation metrics.")
    parser.add_argument("--v2", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--v31a", required=True)
    parser.add_argument("--v31b", required=True)
    parser.add_argument("--v31c", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: getattr(args, name) for name in MODELS}
    metrics = {name: load_json(path) for name, path in paths.items()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(metrics)
    safety = {name: safety_metadata(metrics[name], model_name=name) for name in MODELS}
    winners = select_winners(rows, safety)
    payload = {
        "metrics_paths": paths,
        "metric_comparison": rows,
        "safety_metadata": safety,
        "category_winners": winners,
    }
    with (output_dir / "comparison.json").open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown(payload, output_dir / "comparison.md")
    print(output_dir / "comparison.json")
    invalid = [name for name, meta in safety.items() if not meta["valid_experiment"]]
    if invalid:
        print("WARNING: invalid experiments: " + ", ".join(invalid), file=sys.stderr)


def build_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path, direction, category in METRICS:
        values = {name: get_path(payload, path) for name, payload in metrics.items()}
        row = {
            "metric": path,
            "category": category,
            "preferred_direction": direction,
            **values,
        }
        base = values.get("v2")
        for name in ["v3", "v31a", "v31b", "v31c"]:
            value = values.get(name)
            row[f"delta_{name}_minus_v2"] = None if value is None or base is None else float(value) - float(base)
        row["best_model"] = best_model(values, direction)
        rows.append(row)
    return rows


def select_winners(rows: list[dict[str, Any]], safety: dict[str, dict[str, Any]]) -> dict[str, str | None]:
    valid_models = [name for name in MODELS if safety.get(name, {}).get("valid_experiment", False)]
    winners: dict[str, str | None] = {}
    for output_key, groups in CATEGORY_GROUPS.items():
        candidate_rows = [row for row in rows if row["category"] in groups and row["preferred_direction"] != "reference"]
        scores = {name: 0.0 for name in valid_models}
        counts = {name: 0 for name in valid_models}
        for row in candidate_rows:
            ranked = ranked_models({name: row.get(name) for name in valid_models}, row["preferred_direction"])
            for rank, name in enumerate(ranked, start=1):
                scores[name] += rank
                counts[name] += 1
        averaged = {name: scores[name] / max(counts[name], 1) for name in valid_models if counts[name] > 0}
        winners[output_key] = min(averaged, key=averaged.get) if averaged else None
    return winners


def ranked_models(values: dict[str, Any], direction: str) -> list[str]:
    present = [(name, float(value)) for name, value in values.items() if value is not None]
    reverse = direction == "higher"
    return [name for name, _ in sorted(present, key=lambda item: item[1], reverse=reverse)]


def best_model(values: dict[str, Any], direction: str) -> str | None:
    ranked = ranked_models(values, direction)
    return ranked[0] if ranked else None


def safety_metadata(payload: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    meta = dict(payload.get("graph_conditioning", {}) or {})
    mode = meta.get("graph_conditioning_mode")
    common_checks = {
        "temporal_filter_enabled": True,
        "temporal_filter_mode": "past_only",
        "graph_uses_future_events": False,
        "real_graph_used_at_sampling": False,
    }
    attr_checks = {
        "graph_uses_clean_target_attributes": False,
        "graph_uses_clean_future_attributes": False,
        "history_source_sampling": "generated_past_synthetic_attributes",
        "sampling_chronological": True,
    }
    structure_checks = {
        "graph_uses_target_attributes": False,
    }
    checks = dict(common_checks)
    checks.update(attr_checks if mode == "temporal_attribute_denoising" else structure_checks)
    invalid = []
    for key, expected in checks.items():
        if meta.get(key) != expected:
            invalid.append(f"{key} expected {expected!r}, got {meta.get(key)!r}")
    validity = payload.get("validity", {})
    if validity.get("invalid_rating_rate") not in (0, 0.0):
        invalid.append(f"invalid_rating_rate expected 0.0, got {validity.get('invalid_rating_rate')!r}")
    if validity.get("invalid_verified_rate") not in (0, 0.0):
        invalid.append(f"invalid_verified_rate expected 0.0, got {validity.get('invalid_verified_rate')!r}")
    return {
        "valid_experiment": not invalid,
        "invalid_reasons": invalid,
        "graph_attr_inputs": meta.get("graph_attr_inputs"),
        "include_summary_tokens_in_graph": meta.get("include_summary_tokens_in_graph"),
        "include_summary_length_in_graph": meta.get("include_summary_length_in_graph"),
        "auxiliary_neighbor_denoising_weight": meta.get("auxiliary_neighbor_denoising_weight"),
        "history_attr_mask_prob": meta.get("history_attr_mask_prob"),
        "summary_attr_gate": meta.get("summary_attr_gate"),
        "graph_conditioning_mode": mode,
        **{key: meta.get(key) for key in checks},
    }


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def get_path(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def write_markdown(payload: dict[str, Any], path: str | Path) -> None:
    lines = ["# Conditional TABDLM v2/v3/v3.1 Comparison", ""]
    lines.append("## Category Winners")
    for key, value in payload["category_winners"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Safety Metadata"])
    for name, meta in payload["safety_metadata"].items():
        lines.append(f"- {name}: valid=`{meta['valid_experiment']}`, graph_attr_inputs=`{meta.get('graph_attr_inputs')}`")
        for reason in meta.get("invalid_reasons", []):
            lines.append(f"  - {reason}")
    header = "| Metric | Category | Direction | v2 | v3 | v31a | v31b | v31c | Best |"
    lines.extend(["", "## Metrics", "", header, "|---|---|---|---:|---:|---:|---:|---:|---|"])
    for row in payload["metric_comparison"]:
        lines.append(
            "| {metric} | {category} | {direction} | {v2} | {v3} | {v31a} | {v31b} | {v31c} | {best} |".format(
                metric=row["metric"],
                category=row["category"],
                direction=row["preferred_direction"],
                v2=format_value(row.get("v2")),
                v3=format_value(row.get("v3")),
                v31a=format_value(row.get("v31a")),
                v31b=format_value(row.get("v31b")),
                v31c=format_value(row.get("v31c")),
                best=row.get("best_model"),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
