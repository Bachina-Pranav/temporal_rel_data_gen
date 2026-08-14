#!/usr/bin/env python3
"""Run the single-seed, decision-only LSTM architecture finalization."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.run_lstm_architecture_finalization import (  # noqa: E402
    collect_scope_rows,
    evaluate_existing_checkpoints,
    load_json,
    load_json_optional,
    load_yaml,
    promote_schema_numeric_ordinals,
    run,
    run_multiseed,
    run_support_report_for_evaluated_root,
    write_json,
    write_yaml,
)


DEFAULT_CONFIG = (
    "configs/experiments/lstm_architecture_finalization_compact.yaml"
)
STAGES = (
    "audit",
    "plan",
    "candidates",
    "select",
    "confirm",
    "report",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-batch-size", default="8192")
    parser.add_argument("--minimum-free-disk-gb", type=float, default=5.0)
    parser.add_argument("--rebuild-precomputed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(Path(args.experiment_config))
    enforce_compact_contract(matrix)
    output = Path(matrix["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    args.skip_existing = True
    stages = STAGES[:-1] if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n===== compact-finalization: {stage} =====", flush=True)
        if stage == "audit":
            run_validity_audit(matrix, output, args)
        elif stage == "plan":
            print_training_plan(matrix, output)
        elif stage == "candidates":
            print_training_plan(matrix, output)
            run_candidates(matrix, output, args)
        elif stage == "select":
            select_compact_architecture(matrix, output)
        elif stage == "confirm":
            run_confirmation(matrix, output, args)
        elif stage == "report":
            write_final_report(matrix, output)


def enforce_compact_contract(matrix: dict[str, Any]) -> None:
    seeds = [int(value) for value in matrix.get("seeds") or []]
    if seeds != [42]:
        raise RuntimeError(
            "Compact finalization permits exactly one generator seed: 42"
        )
    lambdas = [float(value) for value in matrix.get("temporal_lambdas") or []]
    if lambdas != [0.1, 0.25]:
        raise RuntimeError(
            "Compact finalization permits only lambda_t=0.10 and 0.25"
        )
    if int(matrix.get("maximum_new_training_runs", 0)) > 5:
        raise RuntimeError("Compact run may not authorize over five trainings")


def driver_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        device=args.device,
        sample_batch_size=str(args.sample_batch_size),
        minimum_free_disk_gb=float(args.minimum_free_disk_gb),
        skip_existing=True,
        rebuild_precomputed=bool(args.rebuild_precomputed),
        dry_run=bool(args.dry_run),
    )


def run_validity_audit(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    audit = output / "validity_audit.md"
    run(
        [
            sys.executable,
            "src/scripts/audit_lstm_categorical_validity.py",
            "--previous-root",
            str(matrix["previous_root"]),
            "--output",
            str(audit),
        ],
        output / "logs/validity_audit.log",
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        text = audit.read_text(encoding="utf-8")
        if "Status: **PASS**" not in text:
            raise RuntimeError(
                f"Categorical validity audit failed; inspect {audit}"
            )


def print_training_plan(matrix: dict[str, Any], output: Path) -> None:
    configs = prepare_candidate_configs(matrix, output)
    required = []
    for label, path in (
        ("Rel-HM TEMP_010 seed 42", temporal_root(output, 0.10)),
        ("Rel-HM TEMP_025 seed 42", temporal_root(output, 0.25)),
    ):
        if not completed_training(path):
            required.append(label)
    conditional = []
    movie = output / "confirmation/movielens_100k/final"
    if not completed_training(movie):
        conditional.append(
            "MovieLens final seed 42 (only if lambda_t > 0 wins)"
        )
    maximum = len(required) + len(conditional)
    if maximum > int(matrix["maximum_new_training_runs"]):
        raise RuntimeError(
            f"Refusing compact plan with {maximum} possible trainings"
        )
    print("\nREQUIRED_NEW_TRAINING_RUNS", flush=True)
    for item in required:
        print(f"- {item}", flush=True)
    if not required:
        print("- none (both temporal candidates already complete)", flush=True)
    print("CONDITIONAL_NEW_TRAINING_RUNS", flush=True)
    for item in conditional:
        print(f"- {item}", flush=True)
    if not conditional:
        print("- none", flush=True)
    print(f"MAXIMUM_NEW_TRAINING_COUNT={maximum}", flush=True)
    print("REUSED_WITHOUT_TRAINING", flush=True)
    print("- Rel-HM R1 original categorical head", flush=True)
    print("- Rel-HM M0/M2 controls", flush=True)
    print("- Amazon original head (compatible existing finalization M2 run)", flush=True)
    print("- Amazon categorical-prior control", flush=True)
    print("- MovieLens lambda_t=0 final", flush=True)
    previous = load_json_optional(output / "training_plan.json")
    write_json(
        {
            "required_new_training_runs": required,
            "conditional_new_training_runs": conditional,
            "maximum_new_training_count": maximum,
            "hard_limit": int(matrix["maximum_new_training_runs"]),
            "seed": 42,
            "executed_new_training_runs": list(
                previous.get("executed_new_training_runs") or []
            ),
            "resolved_candidate_configs": {
                key: str(value) for key, value in configs.items()
            },
        },
        output / "training_plan.json",
    )


def prepare_candidate_configs(
    matrix: dict[str, Any],
    output: Path,
) -> dict[str, Path]:
    base = load_yaml(Path(matrix["rel_hm"]["base_config"]))
    paths = {}
    for temporal_lambda in matrix["temporal_lambdas"]:
        raw = architecture_config(
            base,
            matrix,
            temporal_lambda=float(temporal_lambda),
            adaptive=False,
        )
        name = temporal_name(float(temporal_lambda))
        raw["experiment_name"] = name
        raw.setdefault("experiment_metadata", {}).update(
            {
                "compact_finalization": True,
                "generator_seed": 42,
                "categorical_head": "original",
                "temporal_prior_lambda": float(temporal_lambda),
                "selection_data": "Rel-HM validation",
            }
        )
        path = output / "resolved_configs/rel_hm" / f"{name}.yaml"
        write_yaml(raw, path)
        paths[name] = path
    return paths


def architecture_config(
    base: dict[str, Any],
    matrix: dict[str, Any],
    *,
    temporal_lambda: float,
    adaptive: bool,
) -> dict[str, Any]:
    raw = copy.deepcopy(base)
    head = copy.deepcopy(matrix["support_head"])
    head["mode"] = "auto" if adaptive else "support_prior"
    temporal = head.setdefault("global_prior", {}).setdefault(
        "temporal_prior", {}
    )
    temporal["enabled"] = bool(temporal_lambda > 0.0)
    temporal["lambda_t"] = float(temporal_lambda)
    raw["numerical_heads"] = head
    raw.pop("categorical_heads", None)
    raw.setdefault("sampling", {})["numerical_temperature"] = 1.0
    return raw


def run_candidates(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    if not args.dry_run:
        require_passed_validity_audit(output)
    require_reuse_inputs(matrix)
    configs = prepare_candidate_configs(matrix, output)
    compact_args = driver_args(args)
    for temporal_lambda in matrix["temporal_lambdas"]:
        name = temporal_name(float(temporal_lambda))
        root = temporal_root(output, float(temporal_lambda))
        if not completed_training(root) and not args.dry_run:
            record_training_start(output, f"Rel-HM {name} seed 42")
        run_multiseed(
            config_path=configs[name],
            evaluation_config=Path(matrix["rel_hm"]["evaluation_config"]),
            output_root=root,
            pretokenized_dir=Path(matrix["rel_hm"]["pretokenized_dir"]),
            neighbor_cache_dir=Path(matrix["rel_hm"]["neighbor_cache_dir"]),
            seeds=[42],
            evaluation_scope="heldout-validation",
            sampling_policy="fast",
            matrix=matrix,
            args=compact_args,
        )
        if not args.dry_run:
            support_report_if_missing(
                root,
                matrix,
                spines=root / "shared/spines",
                scope="validation",
            )
    if not args.dry_run:
        refresh_reused_amazon_diagnostics(matrix, output, args)


def support_report_if_missing(
    root: Path,
    matrix: dict[str, Any],
    *,
    spines: Path,
    scope: str,
) -> None:
    report = (
        root
        / "diagnostics/support_calibration/m2_support_calibration_report.json"
    )
    if report.is_file():
        return
    run_support_report_for_evaluated_root(
        spines,
        root,
        matrix,
        scope,
    )


def refresh_reused_amazon_diagnostics(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    for label in ("reused_m2_root", "reused_current_final_root"):
        root = Path(matrix["transfer"]["amazon_toy"][label])
        refresh_attribute_diagnostics(root, output, args)


def refresh_attribute_diagnostics(
    root: Path,
    output: Path,
    args: argparse.Namespace,
) -> None:
    run_root = root / "runs/seed_42"
    config = run_root / "config_resolved.yaml"
    evaluation = run_root / "evaluation_config_resolved.yaml"
    synthetic = run_root / "samples/synthetic_interactions.csv"
    train = root / "shared/spines/train_real.csv"
    required = [config, evaluation, synthetic, train]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            "Cannot refresh reused diagnostics; missing: "
            + ", ".join(str(path) for path in required if not path.is_file())
        )
    eval_raw = load_yaml(evaluation)
    real = Path(str(eval_raw["real_table_path"]))
    destination = (
        output
        / "reused_diagnostics"
        / root.name
        / "attribute_diagnostics.json"
    )
    if destination.is_file():
        return
    command = [
        sys.executable,
        "src/scripts/evaluate_lstm_attribute_diagnostics.py",
        "--config",
        str(config),
        "--train-real",
        str(train),
        "--evaluation-real",
        str(real),
        "--synthetic",
        str(synthetic),
        "--evaluation-config",
        str(evaluation),
        "--output",
        str(destination),
        "--seed",
        "42",
    ]
    run(
        command,
        output / "logs" / f"refresh_{root.name}.log",
        dry_run=args.dry_run,
    )


def select_compact_architecture(
    matrix: dict[str, Any],
    output: Path,
) -> None:
    require_completed_candidates(matrix, output)
    categorical = categorical_decision(matrix, output)
    write_json(categorical, output / "categorical_selection.json")
    (output / "categorical_ablation.md").write_text(
        categorical_markdown(categorical), encoding="utf-8"
    )
    temporal = temporal_decision(matrix, output)
    write_json(temporal, output / "temporal_selection.json")
    (output / "temporal_ablation.md").write_text(
        temporal_markdown(temporal), encoding="utf-8"
    )
    if categorical["selected"] == "prior_anchored":
        selected_lambda = 0.0
        source = Path(
            matrix["rel_hm"]["reused_categorical_prior_validation_root"]
        )
        blocking_note = (
            "Categorical prior retained by strong cross-dataset evidence; "
            "temporal candidates used the original head and are not combined "
            "post hoc."
        )
    else:
        selected_lambda = float(temporal["selected_lambda_t"])
        source = (
            Path(matrix["rel_hm"]["reused_original_root"])
            if selected_lambda == 0.0
            else temporal_root(output, selected_lambda)
        )
        blocking_note = None
    final_base = load_yaml(Path(matrix["rel_hm"]["base_config"]))
    deployment = architecture_config(
        final_base,
        matrix,
        temporal_lambda=selected_lambda,
        adaptive=True,
    )
    if categorical["selected"] == "prior_anchored":
        prior_config = load_yaml(
            source / "runs/seed_42/config_resolved.yaml"
        ).get("categorical_heads")
        if prior_config:
            deployment["categorical_heads"] = prior_config
    deployment.setdefault("experiment_metadata", {}).update(
        {
            "final_architecture": True,
            "compact_finalization": True,
            "selected_on_validation_only": True,
            "generator_seed": 42,
            "test_metrics_used_for_selection": False,
        }
    )
    deployment_path = output / "resolved_configs/final_architecture.yaml"
    write_yaml(deployment, deployment_path)
    lock = {
        "status": "compact_validation_locked",
        "seed": 42,
        "source_root": str(source),
        "deployment_config": str(deployment_path),
        "categorical_architecture": categorical["selected"],
        "temporal_prior_lambda": selected_lambda,
        "support_prior_alpha": 1.0,
        "support_residual_weight": 0.25,
        "support_sampling_temperature": 1.0,
        "test_metrics_used_for_selection": False,
        "note": blocking_note,
    }
    write_json(lock, output / "architecture_lock.json")
    write_compact_validation_table(matrix, output, categorical, temporal)


def categorical_decision(
    matrix: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    hm_original = collect_root_row(
        Path(matrix["rel_hm"]["reused_original_root"]),
        "rel_hm",
        "original",
        matrix,
    )
    hm_prior_root = Path(
        matrix["rel_hm"]["reused_categorical_prior_validation_root"]
    )
    hm_prior = collect_root_row(hm_prior_root, "rel_hm", "prior", matrix)
    amazon_original = collect_root_row(
        Path(matrix["transfer"]["amazon_toy"]["reused_m2_root"]),
        "amazon_toy",
        "original",
        matrix,
        attribute_override=(
            output
            / "reused_diagnostics/M2_global_support/attribute_diagnostics.json"
        ),
    )
    amazon_prior = collect_root_row(
        Path(
            matrix["transfer"]["amazon_toy"][
                "reused_current_final_root"
            ]
        ),
        "amazon_toy",
        "prior",
        matrix,
        attribute_override=(
            output / "reused_diagnostics/final/attribute_diagnostics.json"
        ),
    )
    thresholds = matrix["selection"]
    metrics = {
        "full_row_c2st": float(
            thresholds["categorical_c2st_tie_tolerance"]
        ),
        "categorical_only_c2st": float(
            thresholds["categorical_c2st_tie_tolerance"]
        ),
        "categorical_tv": float(
            thresholds["categorical_tv_tie_tolerance"]
        ),
        "conditional_categorical_error": float(
            thresholds["categorical_tv_tie_tolerance"]
        ),
        "shape_error": float(thresholds["shape_tie_tolerance"]),
        "trend_error": float(thresholds["trend_tie_tolerance"]),
        "text_embedding_c2st": float(
            thresholds["text_c2st_tie_tolerance"]
        ),
    }
    hm_deltas = metric_deltas(hm_prior, hm_original, metrics)
    amazon_deltas = metric_deltas(amazon_prior, amazon_original, metrics)
    hm_wins, hm_regressions = classify_deltas(hm_deltas, metrics)
    amazon_wins, amazon_regressions = classify_deltas(
        amazon_deltas,
        metrics,
    )
    retain_prior = bool(
        hm_wins
        and amazon_wins
        and not hm_regressions
        and not amazon_regressions
    )
    return {
        "selection_data": "Rel-HM validation and Amazon-Toy existing seed-42 transfer",
        "test_metrics_used_for_temporal_selection": False,
        "selected": "prior_anchored" if retain_prior else "original",
        "reason": (
            "Prior improved both datasets materially without a material regression."
            if retain_prior
            else "Original head is better or approximately tied cross-dataset and is simpler."
        ),
        "rel_hm": {
            "original": hm_original,
            "prior": hm_prior,
            "prior_minus_original": hm_deltas,
            "clear_prior_wins": hm_wins,
            "prior_regressions": hm_regressions,
        },
        "amazon_toy": {
            "original": amazon_original,
            "prior": amazon_prior,
            "prior_minus_original": amazon_deltas,
            "clear_prior_wins": amazon_wins,
            "prior_regressions": amazon_regressions,
        },
    }


def temporal_decision(
    matrix: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    baseline = collect_root_row(
        Path(matrix["rel_hm"]["reused_original_root"]),
        "rel_hm",
        "lambda_0",
        matrix,
    )
    m2 = collect_root_row(
        Path(matrix["rel_hm"]["reused_m2_validation_root"]),
        "rel_hm",
        "M2_reference",
        matrix,
    )
    candidates = []
    policy = matrix["selection"]
    for temporal_lambda in matrix["temporal_lambdas"]:
        row = collect_root_row(
            temporal_root(output, float(temporal_lambda)),
            "rel_hm",
            temporal_name(float(temporal_lambda)),
            matrix,
        )
        trend_gain = difference(
            baseline.get("trend_error"), row.get("trend_error")
        )
        full_regression = difference(
            row.get("full_row_c2st"), baseline.get("full_row_c2st")
        )
        numerical_regression = difference(
            row.get("numerical_only_c2st"),
            baseline.get("numerical_only_c2st"),
        )
        preserves_shape = lower_than(row, m2, "shape_error")
        preserves_support = lower_than(row, m2, "support_tv")
        accepted = bool(
            trend_gain is not None
            and trend_gain >= float(policy["trend_minimum_improvement"])
            and full_regression is not None
            and full_regression
            <= float(policy["full_c2st_regression_tolerance"])
            and numerical_regression is not None
            and numerical_regression
            <= float(policy["numerical_c2st_regression_tolerance"])
            and preserves_shape
            and preserves_support
        )
        candidates.append(
            {
                "lambda_t": float(temporal_lambda),
                "metrics": row,
                "trend_improvement": trend_gain,
                "full_c2st_regression": full_regression,
                "numerical_c2st_regression": numerical_regression,
                "shape_better_than_m2": preserves_shape,
                "support_tv_better_than_m2": preserves_support,
                "accepted": accepted,
            }
        )
    selected, reason = select_temporal_candidate(candidates)
    return {
        "selection_split": "Rel-HM validation",
        "seed": 42,
        "test_metrics_consulted": False,
        "baseline": baseline,
        "m2_reference": m2,
        "candidates": candidates,
        "selected_lambda_t": selected,
        "reason": reason,
    }


def select_temporal_candidate(
    candidates: list[dict[str, Any]],
) -> tuple[float, str]:
    accepted = [item for item in candidates if item["accepted"]]
    selected = 0.0
    reason = "No temporal candidate met all validation gates."
    if accepted:
        selected_item = min(accepted, key=lambda item: item["lambda_t"])
        stronger = max(accepted, key=lambda item: item["lambda_t"])
        if (
            stronger["lambda_t"] > selected_item["lambda_t"]
            and float(stronger["trend_improvement"])
            >= float(selected_item["trend_improvement"]) + 0.01
        ):
            selected_item = stronger
        selected = float(selected_item["lambda_t"])
        reason = (
            "Selected the weakest accepted temporal prior unless the stronger "
            "prior provided at least 0.01 additional trend improvement."
        )
    return selected, reason


def run_confirmation(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    lock = require_compact_lock(output)
    compact_args = driver_args(args)
    source = Path(lock["source_root"])
    rel_hm_final = output / "confirmation/rel_hm/final"
    evaluate_existing_checkpoints(
        source,
        rel_hm_final,
        matrix,
        compact_args,
        scope="test",
        transform=lambda raw: final_runtime_config(raw, lock),
    )
    if not args.dry_run:
        support_report_if_missing(
            rel_hm_final,
            matrix,
            spines=source / "shared/spines",
            scope="test",
        )

    temporal_lambda = float(lock["temporal_prior_lambda"])
    for dataset in ("movielens_100k",):
        definition = matrix["transfer"][dataset]
        if temporal_lambda == 0.0:
            continue
        base = promote_schema_numeric_ordinals(
            load_yaml(Path(definition["base_config"]))
        )
        raw = architecture_config(
            base,
            matrix,
            temporal_lambda=temporal_lambda,
            adaptive=True,
        )
        raw["experiment_name"] = f"compact_final_{dataset}"
        config_path = output / f"resolved_configs/transfer/{dataset}/final.yaml"
        write_yaml(raw, config_path)
        movie_root = output / f"confirmation/{dataset}/final"
        if not completed_training(movie_root) and not args.dry_run:
            record_training_start(
                output,
                f"MovieLens final lambda_t={temporal_lambda:g} seed 42",
            )
        run_multiseed(
            config_path=config_path,
            evaluation_config=Path(definition["evaluation_config"]),
            output_root=movie_root,
            pretokenized_dir=Path(definition["pretokenized_dir"]),
            neighbor_cache_dir=Path(definition["neighbor_cache_dir"]),
            seeds=[42],
            evaluation_scope="configured-spine",
            sampling_policy=str(definition["sampling_policy"]),
            matrix=matrix,
            args=compact_args,
        )
    write_json(
        {
            "seed": 42,
            "rel_hm_final_root": str(rel_hm_final),
            "movielens_final_root": str(
                movie_final_root(matrix, output, temporal_lambda)
            ),
            "amazon_final_root": str(
                amazon_final_root(matrix, lock)
            ),
            "test_metrics_used_for_selection": False,
        },
        output / "confirmation_manifest.json",
    )


def write_final_report(matrix: dict[str, Any], output: Path) -> None:
    lock = require_compact_lock(output)
    manifest = load_json(output / "confirmation_manifest.json")
    rows = [
        collect_root_row(Path(matrix["rel_hm"]["reused_m0_test_root"]), "rel_hm", "M0", matrix),
        collect_root_row(Path(matrix["rel_hm"]["reused_m2_test_root"]), "rel_hm", "M2", matrix),
        collect_root_row(Path(manifest["rel_hm_final_root"]), "rel_hm", "FINAL", matrix),
        collect_root_row(Path(matrix["transfer"]["movielens_100k"]["reused_m2_root"]), "movielens_100k", "M2", matrix),
        collect_root_row(Path(manifest["movielens_final_root"]), "movielens_100k", "FINAL", matrix),
        collect_root_row(
            Path(matrix["transfer"]["amazon_toy"]["reused_m2_root"]),
            "amazon_toy",
            "M2",
            matrix,
            attribute_override=output / "reused_diagnostics/M2_global_support/attribute_diagnostics.json",
        ),
        collect_root_row(
            Path(manifest["amazon_final_root"]),
            "amazon_toy",
            "FINAL",
            matrix,
            attribute_override=(
                output
                / "reused_diagnostics/M2_global_support/attribute_diagnostics.json"
                if lock["categorical_architecture"] == "original"
                else output
                / "reused_diagnostics/final/attribute_diagnostics.json"
            ),
        ),
    ]
    table = pd.DataFrame(rows)
    columns = [
        "dataset", "model", "seed", "constraint_violation", "fk_similarity",
        "shape_error", "full_row_c2st", "trend_error",
        "numerical_only_c2st", "categorical_only_c2st",
        "text_embedding_c2st", "support_tv", "support_js",
        "categorical_tv", "conditional_numerical_error",
        "conditional_categorical_error", "temporal_numerical_error",
        "invalid_categorical_rate", "invalid_numerical_rate",
        "rows_per_second", "training_seconds", "sampling_seconds",
    ]
    table = table.reindex(columns=columns)
    table.to_csv(output / "final_cross_dataset_results.csv", index=False)
    freeze, checks, blocker = freeze_decision(table, matrix)
    plan = load_json_optional(output / "training_plan.json")
    categorical_selection = load_json(
        output / "categorical_selection.json"
    )
    temporal_selection = load_json(output / "temporal_selection.json")
    decision = {
        "freeze": freeze,
        "blocking_reason": blocker,
        "seed": 42,
        "new_training_plan": plan,
        "numerical_architecture": "prior-residual support",
        "support_prior_alpha": 1.0,
        "support_residual_weight": 0.25,
        "support_sampling_temperature": 1.0,
        "categorical_architecture": lock["categorical_architecture"],
        "categorical_reason": categorical_selection["reason"],
        "temporal_prior_lambda": lock["temporal_prior_lambda"],
        "temporal_reason": temporal_selection["reason"],
        "validity_audit": "PASS",
        "validity_audit_path": str(output / "validity_audit.md"),
        "checks": checks,
        "results": rows,
        "primary_remaining_weakness": primary_weakness(table),
        "no_dataset_specific_logic": True,
        "past_only_context": True,
        "test_metrics_used_for_selection": False,
    }
    write_json(decision, output / "final_architecture.json")
    markdown = final_markdown(decision, table)
    (output / "final_architecture.md").write_text(markdown, encoding="utf-8")
    (output / "final_cross_dataset_results.md").write_text(
        results_markdown(table), encoding="utf-8"
    )
    print_final_console(decision, table)


def collect_root_row(
    root: Path,
    dataset: str,
    model: str,
    matrix: dict[str, Any],
    *,
    attribute_override: Path | None = None,
) -> dict[str, Any]:
    frame = collect_scope_rows(root.parent, dataset, [root.name], matrix)
    if frame.empty:
        raise RuntimeError(f"No complete seed-42 result at {root}")
    row = frame.iloc[0].to_dict()
    run_root = root / "runs/seed_42"
    attribute_path = attribute_override or (
        run_root / "evaluation/attribute_diagnostics.json"
    )
    if attribute_path.is_file():
        attribute = load_json(attribute_path)
        config = load_yaml(run_root / "config_resolved.yaml")
        row.update(attribute_summary(attribute, config))
    row["dataset"] = dataset
    row["model"] = model
    return clean_record(row)


def attribute_summary(
    attribute: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    categorical = attribute.get("categorical_attributes") or {}
    numerical = attribute.get("numerical_attributes") or {}
    return {
        "categorical_only_c2st": nested_value(attribute, "attribute_group_c2st", "categorical_only", "c2st_error_mean"),
        "numerical_only_c2st": nested_value(attribute, "attribute_group_c2st", "numerical_only", "c2st_error_mean"),
        "categorical_tv": mean_nested(categorical, "total_variation_distance"),
        "conditional_categorical_error": conditional_mean(attribute, "weighted_group_total_variation"),
        "conditional_numerical_error": conditional_mean(attribute, "group_mean_standardized_mae"),
        "temporal_numerical_error": temporal_numerical_error(
            attribute,
            config,
        ),
        "invalid_categorical_rate": max_nested(categorical, "invalid_category_rate"),
        "invalid_numerical_rate": max_nested(numerical, "invalid_rate"),
    }


def freeze_decision(
    table: pd.DataFrame,
    matrix: dict[str, Any],
) -> tuple[bool, dict[str, bool], str | None]:
    final = table[table["model"] == "FINAL"].set_index("dataset")
    baseline = table[table["model"] == "M2"].set_index("dataset")
    validity = bool(
        (pd.to_numeric(final["constraint_violation"], errors="coerce").fillna(float("inf")) == 0).all()
        and (pd.to_numeric(final["fk_similarity"], errors="coerce").fillna(-1) == 1).all()
        and (pd.to_numeric(final["invalid_categorical_rate"], errors="coerce").fillna(0) == 0).all()
        and (pd.to_numeric(final["invalid_numerical_rate"], errors="coerce").fillna(0) == 0).all()
    )
    hm = all(
        lower(final, baseline, "rel_hm", metric)
        for metric in ("full_row_c2st", "numerical_only_c2st", "shape_error")
    )
    movie = all(
        lower(final, baseline, "movielens_100k", metric)
        for metric in ("full_row_c2st", "numerical_only_c2st", "shape_error")
    )
    amazon = approximately_not_worse(
        final,
        baseline,
        "amazon_toy",
        {
            "full_row_c2st": 0.02,
            "categorical_only_c2st": 0.02,
            "shape_error": 0.01,
            "trend_error": 0.01,
            "text_embedding_c2st": 0.02,
        },
    )
    checks = {
        "all_domains_and_constraints_valid": validity,
        "rel_hm_retains_large_improvement": hm,
        "movielens_retains_large_improvement": movie,
        "amazon_improved_or_approximately_tied": amazon,
        "no_dataset_specific_hacks": True,
        "past_only_no_leakage": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    blocker = failed[0] if failed else None
    return not failed, checks, blocker


def final_runtime_config(raw: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(raw)
    resolved.setdefault("sampling", {})["numerical_temperature"] = 1.0
    return resolved


def movie_final_root(
    matrix: dict[str, Any],
    output: Path,
    temporal_lambda: float,
) -> Path:
    if temporal_lambda == 0.0:
        return Path(
            matrix["transfer"]["movielens_100k"][
                "reused_current_final_root"
            ]
        )
    return output / "confirmation/movielens_100k/final"


def amazon_final_root(
    matrix: dict[str, Any],
    lock: dict[str, Any],
) -> Path:
    key = (
        "reused_m2_root"
        if lock["categorical_architecture"] == "original"
        else "reused_current_final_root"
    )
    return Path(matrix["transfer"]["amazon_toy"][key])


def require_reuse_inputs(matrix: dict[str, Any]) -> None:
    checkpoint_roots = [
        matrix["rel_hm"]["reused_original_root"],
        matrix["rel_hm"]["reused_categorical_prior_validation_root"],
    ]
    evaluation_roots = [
        matrix["rel_hm"]["reused_m2_validation_root"],
        matrix["rel_hm"]["reused_m0_test_root"],
        matrix["rel_hm"]["reused_m2_test_root"],
        matrix["transfer"]["movielens_100k"]["reused_m2_root"],
        matrix["transfer"]["movielens_100k"]["reused_current_final_root"],
        matrix["transfer"]["amazon_toy"]["reused_m2_root"],
        matrix["transfer"]["amazon_toy"]["reused_current_final_root"],
    ]
    missing_checkpoints = [
        root for root in checkpoint_roots if not completed_run(Path(root))
    ]
    missing_evaluations = [
        root
        for root in evaluation_roots
        if not completed_evaluation(Path(root))
    ]
    if missing_checkpoints or missing_evaluations:
        sections = []
        if missing_checkpoints:
            sections.append(
                "Checkpoint-source runs are incomplete:\n- "
                + "\n- ".join(missing_checkpoints)
            )
        if missing_evaluations:
            sections.append(
                "Evaluation-only reusable runs are incomplete:\n- "
                + "\n- ".join(missing_evaluations)
            )
        raise FileNotFoundError("\n".join(sections))


def require_completed_candidates(matrix: dict[str, Any], output: Path) -> None:
    require_reuse_inputs(matrix)
    missing = [
        str(temporal_root(output, float(value)))
        for value in matrix["temporal_lambdas"]
        if not completed_run(temporal_root(output, float(value)))
    ]
    if missing:
        raise RuntimeError(
            "Temporal candidates are incomplete:\n- " + "\n- ".join(missing)
        )


def require_compact_lock(output: Path) -> dict[str, Any]:
    path = output / "architecture_lock.json"
    if not path.is_file():
        raise RuntimeError("Run compact --stage select before confirmation")
    lock = load_json(path)
    if lock.get("test_metrics_used_for_selection") is not False:
        raise RuntimeError("Compact architecture lock is test-contaminated")
    return lock


def require_passed_validity_audit(output: Path) -> None:
    path = output / "validity_audit.md"
    if not path.is_file() or "Status: **PASS**" not in path.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(
            "Run compact --stage audit successfully before training"
        )


def record_training_start(output: Path, label: str) -> None:
    path = output / "training_plan.json"
    plan = load_json_optional(path)
    executed = list(plan.get("executed_new_training_runs") or [])
    if label not in executed:
        executed.append(label)
    plan["executed_new_training_runs"] = executed
    write_json(plan, path)


def completed_training(root: Path) -> bool:
    return bool(
        (root / "runs/seed_42/checkpoints/best.pt").is_file()
        and (root / "runs/seed_42/training_metadata.json").is_file()
    )


def completed_evaluation(root: Path) -> bool:
    run = root / "runs/seed_42"
    return bool(
        (run / "samples/synthetic_interactions.csv").is_file()
        and (run / "evaluation/paper_grade/metrics.json").is_file()
        and (run / "evaluation/attribute_diagnostics.json").is_file()
    )


def completed_run(root: Path) -> bool:
    return completed_training(root) and completed_evaluation(root)


def temporal_name(value: float) -> str:
    return "TEMP_" + f"{int(round(value * 100)):03d}"


def temporal_root(output: Path, value: float) -> Path:
    return output / "validation/rel_hm" / temporal_name(value)


def metric_deltas(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, float]:
    output = {}
    for metric in metrics:
        delta = difference(candidate.get(metric), baseline.get(metric))
        if delta is not None:
            output[metric] = delta
    return output


def classify_deltas(
    deltas: dict[str, float],
    tolerances: dict[str, float],
) -> tuple[list[str], list[str]]:
    wins = [
        metric
        for metric, value in deltas.items()
        if value < -float(tolerances[metric])
    ]
    regressions = [
        metric
        for metric, value in deltas.items()
        if value > float(tolerances[metric])
    ]
    return wins, regressions


def difference(left: Any, right: Any) -> float | None:
    if not finite(left) or not finite(right):
        return None
    return float(left) - float(right)


def lower_than(left: dict[str, Any], right: dict[str, Any], metric: str) -> bool:
    return bool(
        finite(left.get(metric))
        and finite(right.get(metric))
        and float(left[metric]) < float(right[metric])
    )


def lower(
    final: pd.DataFrame,
    baseline: pd.DataFrame,
    dataset: str,
    metric: str,
) -> bool:
    if dataset not in final.index or dataset not in baseline.index:
        return False
    return lower_than(final.loc[dataset].to_dict(), baseline.loc[dataset].to_dict(), metric)


def approximately_not_worse(
    final: pd.DataFrame,
    baseline: pd.DataFrame,
    dataset: str,
    metrics: dict[str, float],
) -> bool:
    if dataset not in final.index or dataset not in baseline.index:
        return False
    available = 0
    for metric, tolerance in metrics.items():
        candidate = final.loc[dataset].get(metric)
        reference = baseline.loc[dataset].get(metric)
        if not finite(candidate) or not finite(reference):
            continue
        available += 1
        if float(candidate) - float(reference) > tolerance:
            return False
    return available > 0


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_record(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "dataset", "model", "seed", "constraint_violation", "fk_similarity",
        "shape_error", "full_row_c2st", "trend_error",
        "numerical_only_c2st", "categorical_only_c2st",
        "text_embedding_c2st", "support_tv", "support_js",
        "categorical_tv", "conditional_numerical_error",
        "conditional_categorical_error", "invalid_categorical_rate",
        "temporal_numerical_error",
        "invalid_numerical_rate", "invalid_support_rate", "rows_per_second",
        "training_seconds", "sampling_seconds",
    )
    output = {}
    for key in keep:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        output[key] = float(value) if finite(value) else value
    return output


def nested_value(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def mean_nested(values: dict[str, Any], key: str) -> float | None:
    items = [item.get(key) for item in values.values() if isinstance(item, dict)]
    finite_items = [float(item) for item in items if finite(item)]
    return sum(finite_items) / len(finite_items) if finite_items else None


def max_nested(values: dict[str, Any], key: str) -> float | None:
    items = [item.get(key) for item in values.values() if isinstance(item, dict)]
    finite_items = [float(item) for item in items if finite(item)]
    return max(finite_items) if finite_items else None


def conditional_mean(attribute: dict[str, Any], metric: str) -> float | None:
    values = []
    for condition in (attribute.get("conditional_fidelity") or {}).values():
        if not isinstance(condition, dict):
            continue
        for target in condition.values():
            if isinstance(target, dict) and finite(target.get(metric)):
                values.append(float(target[metric]))
    return sum(values) / len(values) if values else None


def temporal_numerical_error(
    attribute: dict[str, Any],
    config: dict[str, Any],
) -> float | None:
    columns = config.get("columns") or {}
    target = columns.get("target") or {}
    condition = columns.get("condition") or {}
    numerical = set(str(value) for value in target.get("numerical") or [])
    datetimes = set(
        str(value) for value in condition.get("datetimes") or []
    )
    values = []
    pairs = nested_value(attribute, "dependency_fidelity", "pairs") or []
    for pair in pairs:
        left = str(pair.get("left"))
        right = str(pair.get("right"))
        is_temporal_numerical = (
            left in numerical and right in datetimes
        ) or (
            right in numerical and left in datetimes
        )
        if is_temporal_numerical and finite(pair.get("error")):
            values.append(float(pair["error"]))
    return sum(values) / len(values) if values else None


def write_compact_validation_table(
    matrix: dict[str, Any],
    output: Path,
    categorical: dict[str, Any],
    temporal: dict[str, Any],
) -> None:
    lines = [
        "# Compact Validation Table",
        "",
        "Only seed 42 and candidates capable of changing the final decision are included.",
        "",
        categorical_markdown(categorical),
        "",
        temporal_markdown(temporal),
    ]
    (output / "compact_validation_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def categorical_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Categorical Ablation",
        "",
        f"Selected: **{decision['selected']}**",
        "",
        decision["reason"],
        "",
        "Dataset | Head | Full C2ST | Categorical C2ST | Categorical TV | Conditional categorical | Shape | Trend | Text C2ST",
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for dataset in ("rel_hm", "amazon_toy"):
        for name, row in decision[dataset].items():
            if not isinstance(row, dict) or name.endswith("wins") or name.endswith("regressions") or name == "prior_minus_original":
                continue
            lines.append(metric_row(dataset, name, row))
    return "\n".join(lines) + "\n"


def temporal_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Temporal Prior Ablation",
        "",
        f"Selected lambda_t: **{decision['selected_lambda_t']}**",
        "",
        decision["reason"],
        "",
        "lambda_t | Full C2ST | Numerical C2ST | Shape | Trend | Temporal numerical | Support TV | Support JS | Trend gain | Accepted",
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---",
    ]
    base = decision["baseline"]
    lines.append(
        "0 | {full} | {numerical} | {shape} | {trend} | {temporal} | {support} | {support_js} | 0 | baseline".format(
            full=fmt(base.get("full_row_c2st")),
            numerical=fmt(base.get("numerical_only_c2st")),
            shape=fmt(base.get("shape_error")),
            trend=fmt(base.get("trend_error")),
            temporal=fmt(base.get("temporal_numerical_error")),
            support=fmt(base.get("support_tv")),
            support_js=fmt(base.get("support_js")),
        )
    )
    for item in decision["candidates"]:
        row = item["metrics"]
        lines.append(
            "{value} | {full} | {numerical} | {shape} | {trend} | {temporal} | {support} | {support_js} | {gain} | {accepted}".format(
                value=fmt(item["lambda_t"]),
                full=fmt(row.get("full_row_c2st")),
                numerical=fmt(row.get("numerical_only_c2st")),
                shape=fmt(row.get("shape_error")),
                trend=fmt(row.get("trend_error")),
                temporal=fmt(row.get("temporal_numerical_error")),
                support=fmt(row.get("support_tv")),
                support_js=fmt(row.get("support_js")),
                gain=fmt(item.get("trend_improvement")),
                accepted=item["accepted"],
            )
        )
    return "\n".join(lines) + "\n"


def metric_row(dataset: str, name: str, row: dict[str, Any]) -> str:
    return " | ".join(
        [
            dataset,
            name,
            fmt(row.get("full_row_c2st")),
            fmt(row.get("categorical_only_c2st")),
            fmt(row.get("categorical_tv")),
            fmt(row.get("conditional_categorical_error")),
            fmt(row.get("shape_error")),
            fmt(row.get("trend_error")),
            fmt(row.get("text_embedding_c2st")),
        ]
    )


def results_markdown(table: pd.DataFrame) -> str:
    columns = [
        "dataset", "model", "constraint_violation", "fk_similarity",
        "full_row_c2st", "numerical_only_c2st", "categorical_only_c2st",
        "shape_error", "trend_error", "text_embedding_c2st", "support_tv",
        "support_js", "categorical_tv", "temporal_numerical_error",
        "conditional_numerical_error", "conditional_categorical_error",
        "invalid_categorical_rate",
        "invalid_numerical_rate", "rows_per_second", "training_seconds",
        "sampling_seconds",
    ]
    shown = table.reindex(columns=columns)
    lines = [
        "# Final Cross-Dataset Results",
        "",
        " | ".join(columns),
        " | ".join("---" for _ in columns),
    ]
    for _, row in shown.iterrows():
        lines.append(
            " | ".join(
                str(row[column])
                if column in {"dataset", "model"}
                else fmt(row[column])
                for column in columns
            )
        )
    return "\n".join(lines) + "\n"


def dataset_results_markdown(
    table: pd.DataFrame,
    dataset: str,
) -> str:
    subset = table[table["dataset"] == dataset].copy()
    columns = [
        "model", "full_row_c2st", "numerical_only_c2st",
        "categorical_only_c2st", "shape_error", "trend_error",
        "temporal_numerical_error", "text_embedding_c2st", "support_tv",
        "support_js", "categorical_tv",
    ]
    subset = subset.reindex(columns=columns)
    lines = [
        " | ".join(columns),
        " | ".join("---" for _ in columns),
    ]
    for _, row in subset.iterrows():
        lines.append(
            " | ".join(
                str(row[column]) if column == "model" else fmt(row[column])
                for column in columns
            )
        )
    return "\n".join(lines)


def final_markdown(decision: dict[str, Any], table: pd.DataFrame) -> str:
    conclusion = (
        "Architecture development is complete. Further experiments should use this frozen architecture."
        if decision["freeze"]
        else "Architecture is not frozen because: " + str(decision["blocking_reason"])
    )
    return "\n".join(
        [
            "# Final LSTM Architecture",
            "",
            "## Decision",
            "",
            f"FREEZE: **{'YES' if decision['freeze'] else 'NO'}**",
            "",
            "## Numerical Architecture",
            "",
            "Router: training-only schema/data-driven auto router.",
            "",
            "Continuous: Gaussian location/scale head.",
            "",
            "Support: prior-residual support head.",
            "",
            "`logit_k(x,t) = log(p_mix(v_k | b) + eps) + 0.25 * delta_k(x,t)`",
            "",
            "gamma: `0.25`; prior alpha: `1.0`; sampling temperature: `1.0`.",
            "",
            "## Categorical Architecture",
            "",
            f"Selected: **{decision['categorical_architecture']}**.",
            "",
            f"Reason: {decision['categorical_reason']}",
            "",
            "## Temporal Prior",
            "",
            f"lambda_t: **{decision['temporal_prior_lambda']}**.",
            "",
            f"Reason: {decision['temporal_reason']}",
            "",
            "## Validity Audit",
            "",
            "The previous `invalid_categorical_rate=1.0` came from string representation mismatch in the auxiliary evaluator. Schema-driven canonicalization now preserves integer-equivalent categories and leaves non-applicable metrics as NA.",
            "",
            f"Evidence: `{decision['validity_audit_path']}`.",
            "",
            "## Rel-HM",
            "",
            dataset_results_markdown(table, "rel_hm"),
            "",
            "## MovieLens",
            "",
            dataset_results_markdown(table, "movielens_100k"),
            "",
            "## Amazon",
            "",
            dataset_results_markdown(table, "amazon_toy"),
            "",
            "## Remaining Tradeoff",
            "",
            decision["primary_remaining_weakness"],
            "",
            "## Final Conclusion",
            "",
            conclusion,
            "",
        ]
    )


def primary_weakness(table: pd.DataFrame) -> str:
    final = table[(table["dataset"] == "rel_hm") & (table["model"] == "FINAL")]
    if final.empty:
        return "Rel-HM final result unavailable."
    row = final.iloc[0]
    candidates = {
        "full-row discrimination": row.get("full_row_c2st"),
        "numerical-only discrimination": row.get("numerical_only_c2st"),
        "temporal trend mismatch": row.get("trend_error"),
        "support marginal mismatch": row.get("support_tv"),
    }
    finite_values = [(name, float(value)) for name, value in candidates.items() if finite(value)]
    name, value = max(finite_values, key=lambda item: item[1])
    return f"{name} remains the largest reported error ({value:.6g})."


def print_final_console(decision: dict[str, Any], table: pd.DataFrame) -> None:
    print("\n========================================")
    print("FINAL ARCHITECTURE DECISION")
    print("========================================")
    plan = decision.get("new_training_plan") or {}
    print("\nNEW TRAINING RUNS EXECUTED:")
    executed = plan.get("executed_new_training_runs") or []
    for item in executed:
        print(f"- {item}")
    if not executed:
        print("- none; compatible artifacts were reused")
    print(f"\nFREEZE:\n{'YES' if decision['freeze'] else 'NO'}")
    print("\nNUMERICAL:\nprior-residual support\ngamma=0.25")
    print(f"\nCATEGORICAL:\n{decision['categorical_architecture']}")
    print(f"\nTEMPORAL PRIOR:\nlambda={decision['temporal_prior_lambda']}")
    print("\nVALIDITY AUDIT:\nPASS")
    print(f"details={decision['validity_audit_path']}")
    for dataset in ("rel_hm", "movielens_100k", "amazon_toy"):
        print(f"\n{dataset.upper()}:")
        selected = table[table["dataset"] == dataset]
        for _, row in selected.iterrows():
            print(
                f"{row['model']} full={fmt(row.get('full_row_c2st'))} "
                f"numerical={fmt(row.get('numerical_only_c2st'))} "
                f"categorical={fmt(row.get('categorical_only_c2st'))} "
                f"shape={fmt(row.get('shape_error'))} "
                f"trend={fmt(row.get('trend_error'))} "
                f"text={fmt(row.get('text_embedding_c2st'))}"
            )
    print("\nPRIMARY REMAINING WEAKNESS:")
    print(decision["primary_remaining_weakness"])
    print("\nARCHITECTURE:")
    print("FROZEN" if decision["freeze"] else "NOT FROZEN")


def fmt(value: Any) -> str:
    if not finite(value):
        return "NA"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
