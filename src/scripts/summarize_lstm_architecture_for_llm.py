#!/usr/bin/env python3
"""Create a compact, GPT-ready summary of architecture-finalization results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DECISION = Path(
    "outputs/architecture_finalization/final_architecture_decision.json"
)
DEFAULT_MARKDOWN_NAME = "gpt_analysis_summary.md"
DEFAULT_JSON_NAME = "gpt_analysis_summary.json"

HEADLINE_METRICS = (
    "full_row_c2st",
    "numerical_only_c2st",
    "categorical_only_c2st",
    "shape_error",
    "trend_error",
    "support_tv",
    "support_js",
    "categorical_tv",
    "conditional_numerical_error",
    "conditional_categorical_error",
    "text_embedding_c2st",
    "constraint_violation",
    "fk_similarity",
    "rows_per_second",
    "training_seconds",
    "sampling_seconds",
)

PAIRED_METRICS = (
    "full_row_c2st",
    "numerical_only_c2st",
    "categorical_only_c2st",
    "shape_error",
    "trend_error",
    "support_tv",
    "rows_per_second",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        type=Path,
        default=DEFAULT_DECISION,
        help="Path to final_architecture_decision.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the decision file's parent directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.decision.is_file():
        raise FileNotFoundError(f"Missing final decision: {args.decision}")
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    summary = build_summary(decision, args.decision)
    output_dir = args.output_dir or args.decision.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / DEFAULT_MARKDOWN_NAME
    json_path = output_dir / DEFAULT_JSON_NAME
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(markdown_path)
    print(json_path)


def build_summary(decision: dict[str, Any], source: Path) -> dict[str, Any]:
    final_model = str(decision.get("final_model_name", "unknown"))
    selection = decision.get("selection_policy") or {}
    validation = selection.get("validation_selection") or {}
    selected_metrics = validation.get("selected_metrics") or {}
    checks = compact_checks(decision.get("acceptance_checks") or {})
    aggregate = [
        compact_aggregate_row(row)
        for row in decision.get("aggregate_metrics") or []
    ]
    aggregate = [row for row in aggregate if row]

    validation_candidates = []
    for candidate in validation.get("eligible_candidates") or []:
        validation_candidates.append(
            clean_mapping(
                candidate,
                keys=(
                    "model",
                    "full_row_c2st",
                    "numerical_only_c2st",
                    "support_tv",
                    "seed_std",
                    "complexity_rank",
                ),
            )
        )

    rel_hm_test = [
        row
        for row in aggregate
        if row.get("dataset") == "rel_hm"
        and row.get("split") == "test"
        and row.get("model")
        in {"M0_original_lstm_v53", "M2_global_support", final_model}
    ]
    transfer = [
        row
        for row in aggregate
        if row.get("dataset") in {"movielens_100k", "amazon_toy"}
        and row.get("split") == "test"
        and row.get("model") in {"M2_global_support", "final"}
    ]

    evaluator = decision.get("evaluator_audit") or {}
    temperature = selection.get("temperature_selection") or {}
    categorical = selection.get("categorical_selection") or {}
    failed = [name for name, value in checks.items() if not value["passed"]]
    passed = [name for name, value in checks.items() if value["passed"]]

    return {
        "purpose": (
            "Compact evidence packet for reviewing the final LSTM attribute "
            "generator architecture. Lower fidelity errors are better; C2ST "
            "error 0 is chance-level discrimination."
        ),
        "source": str(source),
        "decision": {
            "recommendation": decision.get("freeze_recommendation"),
            "freeze_architecture": bool(decision.get("freeze_architecture")),
            "final_model": final_model,
            "selected_on": selection.get("architecture_selected_on"),
            "test_data_used_for_selection": selection.get(
                "test_data_used_for_selection"
            ),
            "acceptance_checks_passed": len(passed),
            "acceptance_checks_total": len(checks),
            "failed_checks": failed,
        },
        "architecture": sanitize(
            {
                **(decision.get("final_architecture") or {}),
                "chosen_hyperparameters": decision.get(
                    "chosen_hyperparameters"
                ),
            }
        ),
        "validation_selection": {
            "selected_model": validation.get("selected_model"),
            "selected_metrics": clean_mapping(
                selected_metrics,
                keys=(
                    "full_row_c2st",
                    "numerical_only_c2st",
                    "support_tv",
                    "seed_std",
                    "complexity_rank",
                ),
            ),
            "candidate_comparison": validation_candidates,
            "numerical_temperature": temperature.get(
                "selected_temperature"
            ),
            "categorical_prior_adopted": categorical.get("adopted"),
            "selection_split": validation.get("selection_split"),
            "test_metrics_consulted": validation.get(
                "test_metrics_consulted"
            ),
        },
        "rel_hm_three_seed_test": rel_hm_test,
        "rel_hm_paired_deltas": compact_paired_deltas(
            decision.get("paired_deltas") or []
        ),
        "single_seed_transfer": transfer,
        "transfer_deltas_final_minus_m2": transfer_deltas(transfer),
        "acceptance_checks": checks,
        "evaluator_audit": sanitize(
            {
                "status": evaluator.get("status"),
                "fixed_evaluator_seed": evaluator.get(
                    "fixed_evaluator_seed"
                ),
                "generator_seed_decoupled_from_evaluator_seed": evaluator.get(
                    "generator_seed_decoupled_from_evaluator_seed"
                ),
                "hash_mismatches": evaluator.get("hash_mismatches"),
                "resolved_seed_mismatches": evaluator.get(
                    "resolved_seed_mismatches"
                ),
                "missing_metrics": evaluator.get("missing_metrics"),
            }
        ),
        "remaining_weaknesses": decision.get("remaining_weaknesses") or [],
        "interpretation_notes": [
            "NaN-heavy raw fields were intentionally removed. Most NaNs mean a metric is not applicable to that dataset schema.",
            "Rel-HM test results use three generator seeds; uncertainty remains limited by n=3.",
            "MovieLens and Amazon-Toy transfer comparisons use one generator seed each and are supporting evidence, not a stability study.",
            "C2ST error is 2 * abs(AUC - 0.5): 0 is ideal/chance and 1 is perfectly distinguishable.",
            "A COMPLETE experiment only means all artifacts were produced; the freeze recommendation and failed acceptance checks determine whether the architecture passed the predefined decision rule.",
        ],
        "suggested_questions_for_gpt": [
            "Do the predefined acceptance checks support freezing this architecture?",
            "Which improvements over M0 and M2 are consistent across Rel-HM seeds?",
            "Do the transfer results reveal a material regression or evaluator/data-type issue?",
            "Which remaining weakness should be prioritized before making a paper claim?",
            "Which claims are justified by three-seed evidence, and which must be qualified as single-seed observations?",
        ],
    }


def compact_aggregate_row(row: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("dataset", "split", "model", "num_seeds"):
        if key in row:
            compact[key] = sanitize(row[key])
    metrics: dict[str, Any] = {}
    for metric in HEADLINE_METRICS:
        mean = finite_or_none(row.get(f"{metric}_mean"))
        std = finite_or_none(row.get(f"{metric}_std"))
        if mean is not None:
            metrics[metric] = {"mean": mean, "std": std}
    if metrics:
        compact["metrics"] = metrics
    return compact


def compact_paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metric = str(row.get("metric", ""))
        if metric not in PAIRED_METRICS:
            continue
        baseline = str(row.get("baseline", ""))
        delta = finite_or_none(row.get("candidate_minus_baseline"))
        if not baseline or delta is None:
            continue
        grouped[(baseline, metric)].append(
            {"seed": sanitize(row.get("seed")), "delta": delta}
        )

    output = []
    for (baseline, metric), values in sorted(grouped.items()):
        deltas = [item["delta"] for item in values]
        output.append(
            {
                "baseline": baseline,
                "metric": metric,
                "mean_delta": sum(deltas) / len(deltas),
                "seed_deltas": values,
                "lower_is_better": metric != "rows_per_second",
            }
        )
    return output


def transfer_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_dataset[str(row.get("dataset"))][str(row.get("model"))] = row
    for dataset, models in sorted(by_dataset.items()):
        baseline = models.get("M2_global_support", {}).get("metrics", {})
        final = models.get("final", {}).get("metrics", {})
        for metric in HEADLINE_METRICS:
            before = (baseline.get(metric) or {}).get("mean")
            after = (final.get(metric) or {}).get("mean")
            if before is None or after is None:
                continue
            output.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "m2": before,
                    "final": after,
                    "final_minus_m2": after - before,
                    "lower_is_better": metric != "rows_per_second",
                }
            )
    return output


def compact_checks(checks: dict[str, Any]) -> dict[str, Any]:
    return {
        str(name): {
            "passed": bool(value.get("passed")),
            "observed": sanitize(value.get("observed")),
        }
        for name, value in checks.items()
    }


def clean_mapping(
    value: dict[str, Any], *, keys: tuple[str, ...]
) -> dict[str, Any]:
    output = {}
    for key in keys:
        if key not in value:
            continue
        clean = sanitize(value[key])
        if clean is not None:
            output[key] = clean
    return output


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            clean = sanitize(item)
            if clean is not None:
                output[str(key)] = clean
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            clean = sanitize(item)
            if clean is not None:
                output.append(clean)
        return output
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return finite_or_none(value)
    return str(value)


def finite_or_none(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if isinstance(value, int):
        return value
    return numeric


def render_markdown(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    architecture = summary["architecture"]
    validation = summary["validation_selection"]
    evaluator = summary["evaluator_audit"]
    lines = [
        "# GPT Analysis Brief: Final LSTM Architecture",
        "",
        "## Bottom Line",
        "",
        f"- Recommendation: **{decision.get('recommendation')}**",
        f"- Final model: `{decision.get('final_model')}`",
        f"- Acceptance checks: **{decision.get('acceptance_checks_passed')}/{decision.get('acceptance_checks_total')} passed**",
        f"- Failed checks: {comma_list(decision.get('failed_checks'))}",
        f"- Selection data: {decision.get('selected_on')}",
        f"- Test used for selection: `{decision.get('test_data_used_for_selection')}`",
        f"- Evaluator audit: **{evaluator.get('status')}** with fixed evaluator seed `{evaluator.get('fixed_evaluator_seed')}`",
        "",
        "## Final Architecture",
        "",
        f"- Numerical routing: {architecture.get('numerical_routing')}",
        f"- Continuous head: {architecture.get('continuous_head')}",
        f"- Support head: `{architecture.get('support_head_equation')}`",
        f"- Temporal-relational context: `{architecture.get('temporal_relational_context')}`",
        f"- Text architecture changed: `{architecture.get('text_architecture_changed')}`",
        f"- Hyperparameters: `{short_json(architecture.get('chosen_hyperparameters'))}`",
        "",
        "## Validation-Only Selection",
        "",
        f"- Selected model: `{validation.get('selected_model')}`",
        f"- Selected metrics: `{short_json(validation.get('selected_metrics'))}`",
        f"- Numerical sampling temperature: `{validation.get('numerical_temperature')}`",
        f"- Categorical prior adopted: `{validation.get('categorical_prior_adopted')}`",
        f"- Test metrics consulted: `{validation.get('test_metrics_consulted')}`",
        "",
        compact_result_table(
            "Rel-HM validation candidates",
            validation.get("candidate_comparison") or [],
            validation_table_rows,
            "Model | Full C2ST | Numerical C2ST | Support TV | Seed std | Complexity",
        ),
        "",
        compact_result_table(
            "Rel-HM three-seed test",
            summary.get("rel_hm_three_seed_test") or [],
            aggregate_table_rows,
            "Model | Seeds | Full C2ST | Numerical C2ST | Categorical C2ST | Shape | Trend | Support TV | Rows/s",
        ),
        "",
        compact_result_table(
            "Rel-HM paired final-minus-baseline deltas",
            summary.get("rel_hm_paired_deltas") or [],
            paired_table_rows,
            "Baseline | Metric | Mean delta | Per-seed deltas | Direction",
        ),
        "",
        compact_result_table(
            "Single-seed transfer results",
            summary.get("single_seed_transfer") or [],
            aggregate_table_rows,
            "Dataset/model | Seeds | Full C2ST | Numerical C2ST | Categorical C2ST | Shape | Trend | Text C2ST | Rows/s",
        ),
        "",
        compact_result_table(
            "Transfer deltas: final minus M2",
            summary.get("transfer_deltas_final_minus_m2") or [],
            transfer_delta_table_rows,
            "Dataset | Metric | M2 | Final | Delta | Direction",
        ),
        "",
        "## Acceptance Checks",
        "",
        "Check | Result | Observed",
        "--- | --- | ---",
    ]
    for name, value in summary["acceptance_checks"].items():
        lines.append(
            f"{name} | {'PASS' if value['passed'] else '**FAIL**'} | {short_json(value.get('observed'), 180)}"
        )
    lines.extend(
        [
            "",
            "## Remaining Weaknesses",
            "",
            *[f"- {item}" for item in summary["remaining_weaknesses"]],
            "",
            "## Interpretation Notes",
            "",
            *[f"- {item}" for item in summary["interpretation_notes"]],
            "",
            "## Questions For Joint Analysis",
            "",
            *[
                f"{index}. {item}"
                for index, item in enumerate(
                    summary["suggested_questions_for_gpt"], start=1
                )
            ],
            "",
        ]
    )
    return "\n".join(lines)


def compact_result_table(
    title: str,
    rows: list[dict[str, Any]],
    renderer: Any,
    header: str,
) -> str:
    lines = [f"## {title}", "", header, " | ".join("---" for _ in header.split(" | "))]
    rendered = renderer(rows)
    lines.extend(rendered or ["_No applicable results._"])
    return "\n".join(lines)


def validation_table_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [
        " | ".join(
            (
                str(row.get("model")),
                fmt(row.get("full_row_c2st")),
                fmt(row.get("numerical_only_c2st")),
                fmt(row.get("support_tv")),
                fmt(row.get("seed_std")),
                fmt(row.get("complexity_rank")),
            )
        )
        for row in rows
    ]


def aggregate_table_rows(rows: list[dict[str, Any]]) -> list[str]:
    rendered = []
    for row in rows:
        metrics = row.get("metrics") or {}
        name = (
            f"{row.get('dataset')}/{row.get('model')}"
            if row.get("dataset") != "rel_hm"
            else str(row.get("model"))
        )
        rendered.append(
            " | ".join(
                (
                    name,
                    fmt(row.get("num_seeds")),
                    mean_std(metrics.get("full_row_c2st")),
                    mean_std(metrics.get("numerical_only_c2st")),
                    mean_std(metrics.get("categorical_only_c2st")),
                    mean_std(metrics.get("shape_error")),
                    mean_std(metrics.get("trend_error")),
                    mean_std(
                        metrics.get("text_embedding_c2st")
                        if row.get("dataset") != "rel_hm"
                        else metrics.get("support_tv")
                    ),
                    mean_std(metrics.get("rows_per_second")),
                )
            )
        )
    return rendered


def paired_table_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [
        " | ".join(
            (
                str(row["baseline"]),
                str(row["metric"]),
                fmt(row["mean_delta"]),
                ", ".join(
                    f"s{item['seed']}={fmt(item['delta'])}"
                    for item in row["seed_deltas"]
                ),
                "lower" if row["lower_is_better"] else "higher",
            )
        )
        for row in rows
    ]


def transfer_delta_table_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [
        " | ".join(
            (
                str(row["dataset"]),
                str(row["metric"]),
                fmt(row["m2"]),
                fmt(row["final"]),
                fmt(row["final_minus_m2"]),
                "lower" if row["lower_is_better"] else "higher",
            )
        )
        for row in rows
    ]


def mean_std(value: Any) -> str:
    if not isinstance(value, dict):
        return "NA"
    mean = value.get("mean")
    std = value.get("std")
    if mean is None:
        return "NA"
    return fmt(mean) if std is None else f"{fmt(mean)} +/- {fmt(std)}"


def short_json(value: Any, limit: int = 260) -> str:
    if value is None:
        return "NA"
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def comma_list(values: Any) -> str:
    return ", ".join(str(value) for value in (values or [])) or "none"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
