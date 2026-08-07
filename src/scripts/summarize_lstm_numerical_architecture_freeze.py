#!/usr/bin/env python3
"""Consolidate numerical-head experiments into a model-freeze decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default=(
            "configs/experiments/"
            "lstm_numerical_heads_hm_10k.yaml"
        ),
    )
    parser.add_argument(
        "--router-regression",
        default=(
            "outputs/lstm_numerical_router_regressions/"
            "regression_audit.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/hm-10k-customers/lstm_numerical_heads/"
            "architecture_freeze"
        ),
    )
    parser.add_argument(
        "--calibration-interpretation",
        default=(
            "outputs/hm-10k-customers/lstm_numerical_heads/"
            "calibration_q0_q4/calibration_interpretation.json"
        ),
    )
    parser.add_argument(
        "--m2c-root",
        default=(
            "outputs/hm-10k-customers/lstm_numerical_heads/"
            "M2C_posthoc_calibrated_support/seed_42"
        ),
    )
    parser.add_argument("--tests-passed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(Path(args.experiment_config))
    comparison = (
        Path(matrix["output_root"])
        / "results_comparison"
        / "per_model_per_seed_metrics.csv"
    )
    frame = (
        pd.read_csv(comparison)
        if comparison.exists()
        else pd.DataFrame()
    )
    router = load_json_optional(Path(args.router_regression))
    calibration = load_json_optional(
        Path(args.calibration_interpretation)
    )
    m2c_root = Path(args.m2c_root)
    m2c_metrics = load_json_optional(
        m2c_root / "test" / "evaluation" / "paper_grade" / "metrics.json"
    )
    m2c_selection = load_json_optional(
        m2c_root / "selection.json"
    )
    seeds = sorted(int(seed) for seed in matrix["seeds"])
    stability = paired_stability(
        frame,
        "M2_global_support",
        "M0_original_lstm_v53",
        seeds,
    )
    prior_models = [
        "M2P_R0_global_prior",
        "M2P_R1_weak_residual",
        "M2P_R2_moderate_residual",
        "M2P_R3_full_residual",
    ]
    prior = best_model(frame, prior_models, seeds)
    m2 = model_means(frame, "M2_global_support")
    prior_improves = (
        prior["status"] == "completed"
        and m2 is not None
        and prior["metrics"]["full_row_c2st"]
        <= float(m2["full_row_c2st"])
        and prior["metrics"]["numerical_only_c2st"]
        <= float(m2["numerical_only_c2st"])
    )
    checks = {
        "m2_stable_across_seeds": state(stability["stable"]),
        "support_prior_improves_or_matches_m2": state(
            prior_improves
            if prior["status"] == "completed"
            else None
        ),
        "movielens_preserved": state(
            regression_dataset_passed(router, "movielens_100k")
        ),
        "amazon_preserved": state(
            regression_dataset_passed(router, "amazon_toy")
        ),
        "auto_router_sensible": state(
            router.get("auto_router_sensible")
        ),
        "legacy_and_tests_pass": state(
            True if args.tests_passed else None
        ),
    }
    auto_status = checks["auto_router_sensible"]["status"]
    non_auto_checks = {
        name: value
        for name, value in checks.items()
        if name != "auto_router_sensible"
    }
    if any(
        value["status"] == "failed"
        for value in non_auto_checks.values()
    ):
        decision = "do_not_freeze"
    elif all(value["status"] == "passed" for value in checks.values()):
        decision = "freeze"
    elif (
        auto_status == "failed"
        and all(
            value["status"] == "passed"
            for value in non_auto_checks.values()
        )
    ):
        decision = "freeze_with_explicit_routing_only"
    else:
        decision = "not_evaluable"
    concise_answers = build_concise_answers(
        decision=decision,
        stability=stability,
        m2=m2,
        prior=prior,
        calibration=calibration,
        m2c_metrics=m2c_metrics,
        m2c_selection=m2c_selection,
        router=router,
    )
    result = {
        "decision": decision,
        "checks": checks,
        "m2_stability": stability,
        "best_support_prior": prior,
        "router_regression": router,
        "m2c_selection": m2c_selection,
        "concise_answers": concise_answers,
        "dataset_specific_changes": False,
        "supporting_artifacts": {
            "comparability": str(
                Path(matrix["output_root"])
                / "comparability_report.json"
            ),
            "mean_std_and_runtime": str(
                Path(matrix["output_root"])
                / "results_comparison"
                / "aggregate_mean_std_wide.csv"
            ),
            "support_calibration": str(
                Path(matrix["output_root"])
                / "m2_support_calibration"
                / "m2_support_calibration_report.md"
            ),
            "m2c_validation_grid": str(
                m2c_root / "validation_grid.csv"
            ),
        },
        "final_architecture_if_frozen": {
            "legacy_override": "continuous",
            "structured_numerical": "support_prior",
            "continuous_numerical": "continuous",
            "router": "training-only auto with explicit override",
            "prior": "smoothed empirical training marginal",
            "residual": (
                prior.get("model")
                if prior.get("status") == "completed"
                else "pending residual-strength result"
            ),
            "categorical_and_text_heads": "unchanged",
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_freeze_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    table = cross_dataset_table(frame, router)
    table.to_csv(
        output_dir / "cross_dataset_architecture_table.csv",
        index=False,
    )
    (output_dir / "model_freeze_report.md").write_text(
        markdown_report(result, table),
        encoding="utf-8",
    )
    print(output_dir / "model_freeze_decision.json")
    print(output_dir / "cross_dataset_architecture_table.csv")
    print(output_dir / "model_freeze_report.md")
    print(json.dumps(concise_answers, indent=2, sort_keys=True))


def paired_stability(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    seeds: list[int],
) -> dict[str, Any]:
    if frame.empty:
        return {"stable": None, "reason": "missing_comparison_results"}
    left = frame[frame["model"] == candidate].set_index("seed")
    right = frame[frame["model"] == baseline].set_index("seed")
    common = sorted(set(left.index.astype(int)) & set(right.index.astype(int)))
    if common != seeds:
        return {
            "stable": None,
            "reason": "missing_paired_seeds",
            "paired_seeds": common,
            "required_seeds": seeds,
        }
    rows = []
    for seed in common:
        rows.append(
            {
                "seed": int(seed),
                "full_row_c2st_improved": bool(
                    left.loc[seed, "full_row_c2st"]
                    < right.loc[seed, "full_row_c2st"]
                ),
                "numerical_only_c2st_improved": bool(
                    left.loc[seed, "numerical_only_c2st"]
                    < right.loc[seed, "numerical_only_c2st"]
                ),
            }
        )
    return {
        "stable": bool(
            all(
                row["full_row_c2st_improved"]
                and row["numerical_only_c2st_improved"]
                for row in rows
            )
        ),
        "paired_seeds": rows,
    }


def best_model(
    frame: pd.DataFrame,
    models: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    if frame.empty or "model" not in frame:
        return {"status": "not_evaluable", "models": models}
    candidates = []
    for model in models:
        group = frame[frame["model"] == model]
        if sorted(group["seed"].astype(int).tolist()) != seeds:
            continue
        means = group.mean(numeric_only=True)
        candidates.append(
            {
                "model": model,
                "metrics": {
                    "full_row_c2st": float(means["full_row_c2st"]),
                    "numerical_only_c2st": float(
                        means["numerical_only_c2st"]
                    ),
                    "shape_error": float(means["shape_error"]),
                    "trend_error": float(means["trend_error"]),
                },
            }
        )
    if not candidates:
        return {"status": "not_evaluable", "models": models}
    best = min(
        candidates,
        key=lambda item: (
            item["metrics"]["full_row_c2st"]
            + item["metrics"]["numerical_only_c2st"]
        ),
    )
    return {"status": "completed", **best}


def model_means(
    frame: pd.DataFrame,
    model: str,
) -> pd.Series | None:
    if frame.empty or "model" not in frame:
        return None
    selected = frame[frame["model"] == model]
    return (
        selected.mean(numeric_only=True)
        if len(selected)
        else None
    )


def regression_dataset_passed(
    report: dict[str, Any],
    dataset: str,
) -> bool | None:
    item = (report.get("dataset_reports") or {}).get(dataset)
    if not item:
        return None
    return item.get("non_regression_status") in {
        "passed",
        "passed_by_exact_architecture_identity",
    }


def state(value: bool | None) -> dict[str, Any]:
    return {
        "status": (
            "not_evaluable"
            if value is None
            else "passed"
            if value
            else "failed"
        )
    }


def build_concise_answers(
    *,
    decision: str,
    stability: dict[str, Any],
    m2: pd.Series | None,
    prior: dict[str, Any],
    calibration: dict[str, Any],
    m2c_metrics: dict[str, Any],
    m2c_selection: dict[str, Any],
    router: dict[str, Any],
) -> dict[str, Any]:
    q1 = calibration.get("q1_global_full_row_c2st")
    m2_full = (
        float(m2["full_row_c2st"])
        if m2 is not None and pd.notna(m2.get("full_row_c2st"))
        else None
    )
    m2c_summary = m2c_metrics.get("paper_metrics_summary") or {}
    m2c_full = m2c_summary.get("single_table_c2st_error")
    calibration_fraction = None
    if (
        m2_full is not None
        and m2c_full is not None
        and q1 is not None
        and m2_full > float(q1)
    ):
        calibration_fraction = (
            m2_full - float(m2c_full)
        ) / (m2_full - float(q1))
    best_prior = prior.get("model")
    residual_interpretation = (
        "pure_prior"
        if best_prior == "M2P_R0_global_prior"
        else "weak"
        if best_prior == "M2P_R1_weak_residual"
        else "moderate"
        if best_prior == "M2P_R2_moderate_residual"
        else "full"
        if best_prior == "M2P_R3_full_residual"
        else None
    )
    return {
        "1_m2_stable_across_seeds": answer(
            stability.get("stable")
        ),
        "2_m2_to_q1_gap_probability_calibration_fraction": (
            calibration_fraction
        ),
        "2_selected_m2c_lambda": m2c_selection.get("best_lambda"),
        "3_marginal_prior_outperforms_or_matches_m2": answer(
            (
                prior["metrics"]["full_row_c2st"] <= m2_full
                and prior["metrics"]["numerical_only_c2st"]
                <= float(m2["numerical_only_c2st"])
            )
            if prior.get("status") == "completed"
            and m2 is not None
            and m2_full is not None
            else None
        ),
        "4_useful_neural_residual_strength": residual_interpretation,
        "5_movielens_preserved": answer(
            regression_dataset_passed(router, "movielens_100k")
        ),
        "6_amazon_preserved": answer(
            regression_dataset_passed(router, "amazon_toy")
        ),
        "7_dataset_specific_changes": "no",
        "8_architecture_freeze_decision": decision,
        "9_exact_final_architecture": (
            {
                "routing": (
                    "auto"
                    if decision == "freeze"
                    else "explicit"
                    if decision
                    == "freeze_with_explicit_routing_only"
                    else "pending"
                ),
                "continuous_columns": "continuous head",
                "structured_numerical_columns": "support_prior head",
                "prior": "smoothed empirical training marginal",
                "residual_variant": best_prior,
                "categorical_and_text_heads": "unchanged",
            }
        ),
    }


def answer(value: bool | None) -> str:
    return (
        "not_evaluable"
        if value is None
        else "yes"
        if value
        else "no"
    )


def cross_dataset_table(
    frame: pd.DataFrame,
    router: dict[str, Any],
) -> pd.DataFrame:
    m0 = model_means(frame, "M0_original_lstm_v53")
    m2p = best_model(
        frame,
        [
            "M2P_R0_global_prior",
            "M2P_R1_weak_residual",
            "M2P_R2_moderate_residual",
            "M2P_R3_full_residual",
        ],
        sorted(frame["seed"].dropna().astype(int).unique().tolist())
        if len(frame)
        else [],
    )
    hm_report = (
        (router.get("dataset_reports") or {}).get("rel_hm") or {}
    )
    hm_routing = (
        (hm_report.get("auto_routing") or {}).get("price") or {}
    )
    rows = [
        {
            "dataset": "Rel-HM",
            "numerical_attribute": "price",
            "inferred_type": hm_routing.get(
                "label",
                "repeated_or_quantized",
            ),
            "legacy_head": "continuous",
            "auto_head": hm_routing.get(
                "recommended_head",
                "support_prior",
            ),
            "legacy_c2st": (
                float(m0["full_row_c2st"]) if m0 is not None else None
            ),
            "new_c2st": (
                m2p.get("metrics", {}).get("full_row_c2st")
            ),
            "decision": m2p["status"],
            "modalities": ", ".join(
                hm_report.get(
                    "modalities",
                    ["numerical", "categorical", "temporal", "relational"],
                )
            ),
        }
    ]
    for dataset, report in (
        router.get("dataset_reports") or {}
    ).items():
        for column in report.get("categorical_targets", []):
            if column != "rating":
                continue
            legacy = report.get("legacy_metrics_summary") or {}
            auto = report.get("auto_metrics_summary") or {}
            rows.append(
                {
                    "dataset": dataset,
                    "numerical_attribute": column,
                    "inferred_type": "schema_categorical",
                    "legacy_head": "categorical_ordinal",
                    "auto_head": "not_applicable",
                    "legacy_c2st": legacy.get(
                        "single_table_c2st_error"
                    ),
                    "new_c2st": auto.get(
                        "single_table_c2st_error"
                    ),
                    "decision": "unchanged",
                    "modalities": ", ".join(
                        report.get("modalities") or []
                    ),
                }
            )
    return pd.DataFrame(rows)


def markdown_report(
    result: dict[str, Any],
    table: pd.DataFrame,
) -> str:
    lines = [
        "# LSTM Numerical Architecture Freeze Report",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {name}: {value['status']}"
        for name, value in result["checks"].items()
    )
    lines.extend(["", "## Cross-Dataset Architecture", ""])
    columns = [str(column) for column in table.columns]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in table.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(str(value) for value in row) + " |"
        )
    lines.extend(["", "## Concise Answers", ""])
    lines.extend(
        f"- {key}: {value}"
        for key, value in result["concise_answers"].items()
    )
    return "\n".join(lines) + "\n"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def load_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
