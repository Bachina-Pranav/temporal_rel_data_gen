#!/usr/bin/env python3
"""Run the validation-locked LSTM structured-head finalization study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

try:
    from importlib import metadata as package_metadata
except ImportError:  # pragma: no cover - Python 3.7 compatibility
    import importlib_metadata as package_metadata


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    "configs/experiments/lstm_architecture_finalization.yaml"
)
STAGES = (
    "prepare",
    "validation",
    "select",
    "temperature",
    "categorical",
    "test",
    "transfer",
    "report",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-batch-size", default="8192")
    parser.add_argument("--minimum-free-disk-gb", type=float, default=5.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebuild-precomputed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(Path(args.experiment_config))
    output = Path(matrix["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    stages = STAGES[:-1] if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n===== architecture-finalization: {stage} =====", flush=True)
        if stage == "prepare":
            prepare(matrix, output)
        elif stage == "validation":
            run_validation(matrix, output, args)
        elif stage == "select":
            select_numerical_architecture(matrix, output)
        elif stage == "temperature":
            run_temperature_sweep(matrix, output, args)
        elif stage == "categorical":
            run_categorical_sanity(matrix, output, args)
        elif stage == "test":
            run_rel_hm_test(matrix, output, args)
        elif stage == "transfer":
            run_transfer(matrix, output, args)
        elif stage == "report":
            run_report(matrix, output, args)


def prepare(matrix: dict[str, Any], output: Path) -> None:
    definition_sha256 = architecture_definition_sha256(matrix)
    existing_manifest = load_json_optional(
        output / "experiment_manifest.json"
    )
    existing_definition = existing_manifest.get("definition_sha256")
    has_run_artifacts = any(output.rglob("checkpoints/best.pt")) or any(
        output.rglob("evaluation/paper_grade/metrics.json")
    )
    if (
        existing_manifest
        and has_run_artifacts
        and existing_definition != definition_sha256
    ):
        raise RuntimeError(
            f"Existing experiment at {output} was created from a different "
            "architecture definition. Refusing to overwrite it; choose a "
            "new output_root."
        )
    config_paths = write_rel_hm_variant_configs(matrix, output)
    evaluator = evaluator_fingerprint(
        Path(matrix["rel_hm"]["evaluation_config"])
    )
    write_json(evaluator, output / "evaluator_fingerprint.json")
    write_json(
        {
            "experiment_name": matrix["experiment_name"],
            "definition_sha256": definition_sha256,
            "git_commit": git_revision(),
            "python": sys.version,
            "platform": platform.platform(),
            "runtime_environment": runtime_environment(),
            "seeds": matrix["seeds"],
            "evaluator_seed": matrix["evaluator_seed"],
            "variant_configs": {
                name: str(path) for name, path in config_paths.items()
            },
            "validation_only_selection": True,
            "test_data_used_for_selection": False,
        },
        output / "experiment_manifest.json",
    )
    (output / "evaluator_audit.md").write_text(
        evaluator_audit_markdown(evaluator), encoding="utf-8"
    )
    loss_audit = support_loss_audit(matrix)
    (output / "loss_audit.md").write_text(loss_audit, encoding="utf-8")
    (output / "support_head_loss_audit.md").write_text(
        loss_audit,
        encoding="utf-8",
    )
    print(f"Prepared {len(config_paths)} immutable candidate configs.")


def architecture_definition_sha256(matrix: dict[str, Any]) -> str:
    referenced = [
        Path(matrix["rel_hm"]["base_config"]),
        Path(matrix["rel_hm"]["evaluation_config"]),
    ]
    for definition in matrix["transfer"]["datasets"].values():
        referenced.extend(
            [
                Path(definition["base_config"]),
                Path(definition["evaluation_config"]),
            ]
        )
    controlled = [
        ROOT / "src/attribute_generation/conditional_tabdlm/lstm_joint.py",
        ROOT / "src/attribute_generation/conditional_tabdlm/numerical_head.py",
        ROOT / "src/attribute_generation/conditional_tabdlm/numerical_type.py",
        ROOT / "src/attribute_generation/conditional_tabdlm/categorical_head.py",
        ROOT / "src/scripts/run_lstm_multiseed_experiment.py",
        ROOT / "src/scripts/run_lstm_architecture_finalization.py",
    ]
    return object_sha256(
        {
            "matrix": matrix,
            "referenced_configs": {
                str(path): file_sha256(path) for path in referenced
            },
            "controlled_sources": {
                str(path): file_sha256(path) for path in controlled
            },
        }
    )


def write_rel_hm_variant_configs(
    matrix: dict[str, Any],
    output: Path,
) -> dict[str, Path]:
    base = load_yaml(Path(matrix["rel_hm"]["base_config"]))
    common = matrix.get("numerical_head") or {}
    directory = output / "resolved_configs" / "rel_hm"
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, definition in matrix["variants"].items():
        resolved = copy.deepcopy(base)
        head = copy.deepcopy(common)
        deep_update(head, definition.get("numerical_heads") or {})
        resolved["numerical_heads"] = head
        resolved["experiment_name"] = name
        resolved.setdefault("experiment_metadata", {}).update(
            {
                "architecture_finalization": True,
                "variant": name,
                "training_only_routing_and_prior": True,
                "test_data_used_for_selection": False,
            }
        )
        path = directory / f"{name}.yaml"
        write_yaml(resolved, path)
        paths[name] = path
    return paths


def run_validation(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    config_paths = write_rel_hm_variant_configs(matrix, output)
    selected = list(args.variants or config_paths)
    unknown = sorted(set(selected).difference(config_paths))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    for name in selected:
        root = output / "rel_hm" / "validation" / name
        run_multiseed(
            config_path=config_paths[name],
            evaluation_config=Path(matrix["rel_hm"]["evaluation_config"]),
            output_root=root,
            pretokenized_dir=Path(matrix["rel_hm"]["pretokenized_dir"]),
            neighbor_cache_dir=Path(matrix["rel_hm"]["neighbor_cache_dir"]),
            seeds=[int(value) for value in matrix["seeds"]],
            evaluation_scope="heldout-validation",
            sampling_policy="fast",
            matrix=matrix,
            args=args,
        )
        if not args.dry_run:
            run_support_and_context_diagnostics(
                root,
                name,
                config_paths[name],
                matrix,
                args,
                scope="validation",
            )


def run_multiseed(
    *,
    config_path: Path,
    evaluation_config: Path,
    output_root: Path,
    pretokenized_dir: Path,
    neighbor_cache_dir: Path,
    seeds: list[int],
    evaluation_scope: str,
    sampling_policy: str,
    matrix: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    command = [
        python(),
        "src/scripts/run_lstm_multiseed_experiment.py",
        "--config", str(config_path),
        "--evaluation-config", str(evaluation_config),
        "--output-root", str(output_root),
        "--pretokenized-dir", str(pretokenized_dir),
        "--neighbor-cache-dir", str(neighbor_cache_dir),
        "--seeds", *[str(seed) for seed in seeds],
        "--device", args.device,
        "--sample-batch-size", str(args.sample_batch_size),
        "--minimum-free-disk-gb", str(args.minimum_free_disk_gb),
        "--evaluation-scope", evaluation_scope,
        "--evaluation-seed", str(matrix["evaluator_seed"]),
        "--sampling-policy", sampling_policy,
        "--skip-smoke",
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    if args.rebuild_precomputed:
        command.append("--rebuild-precomputed")
    if args.dry_run:
        command.append("--dry-run")
    run(
        command,
        output_root / "logs" / "architecture_driver.log",
        dry_run=args.dry_run,
    )


def run_support_and_context_diagnostics(
    root: Path,
    model_name: str,
    config_path: Path,
    matrix: dict[str, Any],
    args: argparse.Namespace,
    *,
    scope: str,
) -> None:
    raw = load_yaml(config_path)
    numerical = list(
        ((raw.get("columns") or {}).get("target") or {}).get(
            "numerical", []
        )
    )
    spines = root / "shared" / "spines"
    real_name = "validation_real.csv" if scope == "validation" else "test_real.csv"
    spine_name = "validation_spine.csv" if scope == "validation" else "test_spine.csv"
    prefix_name = "train_spine.csv" if scope == "validation" else "history_prefix_spine.csv"
    if numerical:
        synthetic_entries = [
            f"seed_{seed}={root / 'runs' / f'seed_{seed}' / 'samples/synthetic_interactions.csv'}"
            for seed in matrix["seeds"]
        ]
        run(
            [
                python(),
                "src/scripts/analyze_m2_support_calibration.py",
                "--train-real", str(spines / "train_real.csv"),
                "--validation-real", str(spines / real_name),
                "--synthetic", *synthetic_entries,
                "--numerical-columns", *[str(value) for value in numerical],
                "--output-dir", str(root / "diagnostics/support_calibration"),
            ],
            root / "logs" / f"support_calibration_{scope}.log",
        )
    for seed in matrix["seeds"]:
        run_root = root / "runs" / f"seed_{seed}"
        run(
            [
                python(),
                "src/scripts/diagnose_lstm_numerical_context_usage.py",
                "--checkpoint", str(run_root / "checkpoints/best.pt"),
                "--synthetic-spine", str(spines / spine_name),
                "--graph-history-prefix", str(spines / prefix_name),
                "--evaluation-real", str(spines / real_name),
                "--output", str(run_root / "evaluation/numerical_context_usage.json"),
                "--num-rows", "2048",
                "--device", args.device,
                "--seed", str(seed),
            ],
            run_root / "logs" / f"context_diagnostic_{scope}.log",
        )


def select_numerical_architecture(
    matrix: dict[str, Any],
    output: Path,
) -> None:
    comparability = validation_comparability(matrix, output)
    write_json(
        comparability,
        output / "validation_comparability_audit.json",
    )
    if not comparability["comparable"]:
        raise RuntimeError(
            "Validation candidates have incompatible evaluator, split, or "
            "precomputation fingerprints; selection is blocked. See "
            f"{output / 'validation_comparability_audit.json'}"
        )
    rows = collect_scope_rows(
        output / "rel_hm" / "validation",
        "rel_hm",
        list(matrix["variants"]),
        matrix,
    )
    if rows.empty:
        raise RuntimeError("No completed validation results were found")
    required = {int(seed) for seed in matrix["seeds"]}
    eligible: list[dict[str, Any]] = []
    for model, group in rows.groupby("model"):
        seeds = set(group["seed"].astype(int))
        validity = hard_validity(group, matrix)
        if seeds == required and validity["all_passed"]:
            eligible.append(
                {
                    "model": model,
                    "full_row_c2st": finite_mean(group["full_row_c2st"]),
                    "numerical_only_c2st": finite_mean(group["numerical_only_c2st"]),
                    "support_tv": finite_mean(group["support_tv"]),
                    "seed_std": finite_std(group["full_row_c2st"]),
                    "complexity_rank": int(
                        matrix["variants"][model].get("complexity_rank", 99)
                    ),
                    "validity": validity,
                }
            )
    if not eligible:
        raise RuntimeError(
            "No candidate passed validation validity checks; test evaluation remains locked"
        )
    winner, selection_trace = select_validation_winner(
        eligible,
        matrix["selection"],
    )
    selection = {
        "selection_split": "validation",
        "test_metrics_consulted": False,
        "required_seeds": sorted(required),
        "evaluator_seed": int(matrix["evaluator_seed"]),
        "evaluator_hash": evaluator_fingerprint(
            Path(matrix["rel_hm"]["evaluation_config"])
        )["evaluator_hash"],
        "eligible_candidates": eligible,
        "selected_model": winner["model"],
        "selected_metrics": winner,
        "equivalence_policy": selection_trace,
        "selection_order": [
            "full_row_c2st",
            "numerical_only_c2st",
            "support_tv",
            "complexity_rank",
            "seed_std",
        ],
    }
    write_json(selection, output / "validation_architecture_selection.json")
    lock = {
        "status": "validation_locked",
        "selected_model": winner["model"],
        "selected_config": str(
            output / "resolved_configs/rel_hm" / f"{winner['model']}.yaml"
        ),
        "numerical_temperature": 1.0,
        "categorical_prior": {"enabled": False},
        "selection_sha256": object_sha256(selection),
        "test_metrics_consulted": False,
    }
    write_json(lock, output / "architecture_lock.json")
    rows.to_csv(output / "validation_all_runs.csv", index=False)
    print(f"Validation-selected numerical architecture: {winner['model']}")


def select_validation_winner(
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply fidelity equivalence bands before rewarding complexity."""

    stages = [
        (
            "full_row_c2st",
            float(policy["full_c2st_equivalence_tolerance"]),
        ),
        (
            "numerical_only_c2st",
            float(policy["numerical_c2st_equivalence_tolerance"]),
        ),
        (
            "support_tv",
            float(policy["support_tv_equivalence_tolerance"]),
        ),
    ]
    remaining = list(candidates)
    trace = []
    for metric, tolerance in stages:
        finite = [
            null_last(candidate.get(metric))
            for candidate in remaining
            if np.isfinite(null_last(candidate.get(metric)))
        ]
        if not finite:
            trace.append(
                {
                    "metric": metric,
                    "status": "not_evaluable",
                    "remaining": [item["model"] for item in remaining],
                }
            )
            continue
        best = min(finite)
        remaining = [
            candidate
            for candidate in remaining
            if null_last(candidate.get(metric)) <= best + tolerance
        ]
        trace.append(
            {
                "metric": metric,
                "best": best,
                "equivalence_tolerance": tolerance,
                "remaining": [item["model"] for item in remaining],
            }
        )
    winner = min(
        remaining,
        key=lambda item: (
            item["complexity_rank"],
            null_last(item["seed_std"]),
            null_last(item["full_row_c2st"]),
            null_last(item["numerical_only_c2st"]),
        ),
    )
    return winner, {
        "principle": (
            "Within recorded fidelity-equivalence bands, prefer the "
            "simpler and more stable architecture."
        ),
        "stages": trace,
    }


def validation_comparability(
    matrix: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Require all validation candidates to share evaluation/data lineage."""

    fields = (
        "evaluation_config_sha256",
        "c2st_source_sha256",
        "split_fingerprints",
        "precomputed_split_fingerprints",
        "pretokenized_metadata_sha256",
        "neighbor_cache_metadata_sha256",
    )
    manifests = {}
    schema_fingerprints = {}
    missing = []
    for model in matrix["variants"]:
        path = (
            output
            / "rel_hm"
            / "validation"
            / model
            / "shared"
            / "comparability_manifest.json"
        )
        if not path.is_file():
            missing.append(str(path))
            continue
        manifests[model] = load_json(path)
        config = load_yaml(
            output / "resolved_configs" / "rel_hm" / f"{model}.yaml"
        )
        schema_fingerprints[model] = object_sha256(
            {
                key: config.get(key)
                for key in (
                    "dataset",
                    "event_spine",
                    "generated_attributes",
                    "columns",
                    "schema",
                    "graph_conditioning",
                    "text",
                    "text_decoder",
                    "review_text_decoder",
                    "summary_decoder",
                    "text_length_prediction",
                )
            }
        )
    mismatches = {}
    if manifests:
        reference_name = next(iter(manifests))
        reference = manifests[reference_name]
        for model, manifest in manifests.items():
            changed = [
                field
                for field in fields
                if manifest.get(field) != reference.get(field)
            ]
            if changed:
                mismatches[model] = changed
    return {
        "comparable": bool(
            not missing
            and not mismatches
            and len(manifests) == len(matrix["variants"])
            and len(set(schema_fingerprints.values())) == 1
        ),
        "required_fields": list(fields),
        "models": sorted(manifests),
        "missing_manifests": missing,
        "mismatches": mismatches,
        "schema_fingerprints": schema_fingerprints,
        "schema_mismatch": len(set(schema_fingerprints.values())) != 1,
    }


def run_temperature_sweep(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    lock = require_lock(output)
    model = str(lock["selected_model"])
    source = output / "rel_hm" / "validation" / model
    roots: dict[float, Path] = {}
    for temperature in matrix["temperature_grid"]:
        label = temperature_label(float(temperature))
        destination = output / "rel_hm" / "temperature" / f"tau_{label}"
        evaluate_existing_checkpoints(
            source,
            destination,
            matrix,
            args,
            scope="validation",
            transform=lambda raw, tau=float(temperature): set_numerical_temperature(raw, tau),
        )
        roots[float(temperature)] = destination
    rows = pd.concat(
        [
            collect_scope_rows(root.parent, "rel_hm", [root.name], matrix)
            .assign(temperature=temperature, model=model)
            for temperature, root in roots.items()
        ],
        ignore_index=True,
    )
    means = rows.groupby("temperature", as_index=False).mean(numeric_only=True)
    best = min(
        means.to_dict(orient="records"),
        key=lambda row: (
            null_last(row.get("full_row_c2st")),
            null_last(row.get("numerical_only_c2st")),
            null_last(row.get("support_tv")),
        ),
    )
    write_json(
        {
            "selection_split": "validation",
            "test_metrics_consulted": False,
            "grid": matrix["temperature_grid"],
            "selected_temperature": float(best["temperature"]),
            "aggregate": means.to_dict(orient="records"),
        },
        output / "temperature_selection.json",
    )
    lock["numerical_temperature"] = float(best["temperature"])
    lock["temperature_selection_sha256"] = object_sha256(best)
    write_json(lock, output / "architecture_lock.json")
    rows.to_csv(output / "temperature_sweep.csv", index=False)


def run_categorical_sanity(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    lock = require_lock(output)
    base_path = Path(lock["selected_config"])
    base = load_yaml(base_path)
    base.setdefault("sampling", {})["numerical_temperature"] = float(
        lock["numerical_temperature"]
    )
    candidates: dict[str, Path] = {}
    for gamma in matrix["categorical_residual_grid"]:
        name = f"{lock['selected_model']}__cat_prior_{temperature_label(float(gamma))}"
        raw = copy.deepcopy(base)
        raw["experiment_name"] = name
        raw["categorical_heads"] = {
            "prior": {
                "enabled": True,
                "alpha": 1.0,
                "residual_weight": float(gamma),
                "residual_init_scale": 0.001,
                "smoothing": 0.0,
                "epsilon": 1.0e-8,
            }
        }
        path = output / "resolved_configs/rel_hm" / f"{name}.yaml"
        write_yaml(raw, path)
        candidates[name] = path
        root = output / "rel_hm" / "categorical" / name
        run_multiseed(
            config_path=path,
            evaluation_config=Path(matrix["rel_hm"]["evaluation_config"]),
            output_root=root,
            pretokenized_dir=Path(matrix["rel_hm"]["pretokenized_dir"]),
            neighbor_cache_dir=Path(matrix["rel_hm"]["neighbor_cache_dir"]),
            seeds=[int(value) for value in matrix["seeds"]],
            evaluation_scope="heldout-validation",
            sampling_policy="fast",
            matrix=matrix,
            args=args,
        )
    if args.dry_run:
        return
    baseline_root = temperature_root(output, float(lock["numerical_temperature"]))
    baseline = collect_scope_rows(
        baseline_root.parent, "rel_hm", [baseline_root.name], matrix
    ).assign(model=str(lock["selected_model"]), categorical_gamma=np.nan)
    frames = [baseline]
    for name in candidates:
        frame = collect_scope_rows(
            output / "rel_hm" / "categorical", "rel_hm", [name], matrix
        )
        frame["categorical_gamma"] = float(name.rsplit("_", 1)[-1].replace("p", "."))
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    aggregates = rows.groupby("model", as_index=False).mean(numeric_only=True)
    baseline_mean = aggregates[aggregates["model"] == lock["selected_model"]].iloc[0]
    tolerance_c2st = float(matrix["selection"]["categorical_c2st_regression_tolerance"])
    tolerance_tv = float(matrix["selection"]["categorical_tv_regression_tolerance"])
    accepted = []
    for row in aggregates.to_dict(orient="records"):
        if row["model"] == lock["selected_model"]:
            continue
        if (
            row.get("categorical_tv", float("inf"))
            < baseline_mean.get("categorical_tv", float("inf"))
            and row.get("full_row_c2st", float("inf"))
            <= baseline_mean.get("full_row_c2st", float("inf")) + tolerance_c2st
            and row.get("conditional_categorical_error", float("inf"))
            <= baseline_mean.get("conditional_categorical_error", float("inf")) + tolerance_tv
        ):
            accepted.append(row)
    selected = min(
        accepted,
        key=lambda row: (
            null_last(row.get("categorical_tv")),
            null_last(row.get("full_row_c2st")),
        ),
    ) if accepted else None
    decision = {
        "selection_split": "validation",
        "test_metrics_consulted": False,
        "baseline": baseline_mean.to_dict(),
        "candidates": aggregates.to_dict(orient="records"),
        "adopted": selected is not None,
        "selected": selected,
    }
    write_json(decision, output / "categorical_selection.json")
    rows.to_csv(output / "categorical_sweep.csv", index=False)
    if selected is not None:
        name = str(selected["model"])
        lock["selected_model"] = name
        lock["selected_config"] = str(candidates[name])
        lock["categorical_prior"] = {
            "enabled": True,
            "residual_weight": float(selected["categorical_gamma"]),
        }
        lock["validation_source_root"] = str(
            output / "rel_hm" / "categorical" / name
        )
    else:
        lock["validation_source_root"] = str(
            output / "rel_hm" / "validation" / lock["selected_model"]
        )
    deployment = load_yaml(Path(lock["selected_config"]))
    selected_mode = str(
        (deployment.get("numerical_heads") or {}).get("mode", "")
    ).lower()
    adaptive_router_enabled = selected_mode not in {
        "continuous",
        "continuous_baseline",
    }
    if adaptive_router_enabled:
        deployment.setdefault("numerical_heads", {})["mode"] = "auto"
    deployment.setdefault("sampling", {})["numerical_temperature"] = float(
        lock["numerical_temperature"]
    )
    deployment.setdefault("experiment_metadata", {}).update(
        {
            "final_architecture": True,
            "universal_training_only_numerical_router": (
                adaptive_router_enabled
            ),
            "selected_on_validation_only": True,
        }
    )
    deployment_path = output / "resolved_configs/final_architecture.yaml"
    write_yaml(deployment, deployment_path)
    lock["deployment_config"] = str(deployment_path)
    lock["adaptive_router_enabled"] = adaptive_router_enabled
    lock["status"] = "fully_frozen_on_validation"
    write_json(lock, output / "architecture_lock.json")


def run_rel_hm_test(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    lock = require_lock(output, require_fully_frozen=True)
    final_model = str(lock["selected_model"])
    models = list(
        dict.fromkeys(
            [
                "M0_original_lstm_v53",
                "M2_global_support",
                final_model,
            ]
        )
    )
    for model in models:
        source = validation_source_for_model(output, model, lock)
        if not source.exists():
            raise FileNotFoundError(
                f"Validation-trained checkpoints missing for {model}: {source}"
            )
        destination = output / "rel_hm" / "test" / model
        temperature = (
            float(lock["numerical_temperature"])
            if model == final_model
            else 1.0
        )
        evaluate_existing_checkpoints(
            source,
            destination,
            matrix,
            args,
            scope="test",
            transform=lambda raw, tau=temperature: set_numerical_temperature(raw, tau),
        )
    write_json(
        {
            "architecture_lock_sha256": file_sha256(output / "architecture_lock.json"),
            "selection_was_validation_only": True,
            "test_evaluation_started_after_lock": True,
            "models": models,
        },
        output / "test_evaluation_manifest.json",
    )


def evaluate_existing_checkpoints(
    source: Path,
    destination: Path,
    matrix: dict[str, Any],
    args: argparse.Namespace,
    *,
    scope: str,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    source_spines = source / "shared" / "spines"
    destination.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in matrix["seeds"]]
    real_name = "validation_real.csv" if scope == "validation" else "test_real.csv"
    spine_name = "validation_spine.csv" if scope == "validation" else "test_spine.csv"
    prefix_name = "train_spine.csv" if scope == "validation" else "history_prefix_spine.csv"
    for seed in seeds:
        source_run = source / "runs" / f"seed_{seed}"
        run_root = destination / "runs" / f"seed_{seed}"
        synthetic = run_root / "samples/synthetic_interactions.csv"
        metrics = run_root / "evaluation/paper_grade/metrics.json"
        attributes = run_root / "evaluation/attribute_diagnostics.json"
        for relative in (
            "training_metadata.json",
            "metadata/numerical_head_routing.json",
            "metadata/numerical_type_inference.json",
            "metadata/categorical_head_metadata.json",
        ):
            source_artifact = source_run / relative
            if source_artifact.is_file():
                write_json(
                    load_json(source_artifact),
                    run_root / relative,
                )
        if args.skip_existing and all(path.is_file() for path in (synthetic, metrics, attributes)):
            continue
        raw = transform(load_yaml(source_run / "config_resolved.yaml"))
        raw.setdefault("paths", {})["synthetic_spine_path"] = str(
            source_spines / spine_name
        )
        config_path = run_root / "config_resolved.yaml"
        write_yaml(raw, config_path)
        eval_raw = load_yaml(source_run / "evaluation_config_resolved.yaml")
        eval_raw["real_table_path"] = str(source_spines / real_name)
        eval_raw["synthetic_table_path"] = str(synthetic)
        eval_raw.setdefault("evaluation", {})["random_seed"] = int(
            matrix["evaluator_seed"]
        )
        eval_path = run_root / "evaluation_config_resolved.yaml"
        write_yaml(eval_raw, eval_path)
        run(
            [
                python(),
                "src/scripts/sample_lstm_joint_full_review_text_fast.py",
                "--config", str(config_path),
                "--checkpoint", str(source_run / "checkpoints/best.pt"),
                "--synthetic-spine", str(source_spines / spine_name),
                "--graph-history-prefix", str(source_spines / prefix_name),
                "--output", str(synthetic),
                "--num-rows", "all",
                "--batch-size", str(args.sample_batch_size),
                "--device", args.device,
                "--seed", str(seed),
                "--mixed-precision",
                "--profile",
            ],
            run_root / "logs/sample.log",
            dry_run=args.dry_run,
        )
        numerical = list(
            (((raw.get("columns") or {}).get("target") or {}).get("numerical") or [])
        )
        if numerical:
            run(
                [
                    python(),
                    "src/scripts/diagnose_lstm_numerical_context_usage.py",
                    "--checkpoint", str(source_run / "checkpoints/best.pt"),
                    "--synthetic-spine", str(source_spines / spine_name),
                    "--graph-history-prefix", str(source_spines / prefix_name),
                    "--evaluation-real", str(source_spines / real_name),
                    "--output", str(run_root / "evaluation/numerical_context_usage.json"),
                    "--num-rows", "2048",
                    "--device", args.device,
                    "--seed", str(seed),
                ],
                run_root / "logs/context_diagnostic.log",
                dry_run=args.dry_run,
            )
        run(
            [
                python(),
                "src/scripts/evaluate_single_event_table_paper_metrics.py",
                "--config", str(eval_path),
                "--real-table", str(source_spines / real_name),
                "--synthetic-table", str(synthetic),
                "--output-dir", str(run_root / "evaluation/paper_grade"),
                "--seed", str(matrix["evaluator_seed"]),
            ],
            run_root / "logs/evaluate.log",
            dry_run=args.dry_run,
        )
        run(
            [
                python(),
                "src/scripts/evaluate_lstm_attribute_diagnostics.py",
                "--config", str(config_path),
                "--train-real", str(source_spines / "train_real.csv"),
                "--evaluation-real", str(source_spines / real_name),
                "--synthetic", str(synthetic),
                "--graph-history-prefix", str(source_spines / prefix_name),
                "--evaluation-config", str(eval_path),
                "--output", str(attributes),
                "--seed", str(seed),
            ],
            run_root / "logs/attribute_diagnostics.log",
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        copy_shared_manifest(source, destination)
        run_support_report_for_evaluated_root(
            source_spines, destination, matrix, scope
        )


def run_transfer(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    lock = require_lock(output, require_fully_frozen=True)
    selected = set(args.datasets or matrix["transfer"]["datasets"])
    unknown = sorted(selected.difference(matrix["transfer"]["datasets"]))
    if unknown:
        raise ValueError(f"Unknown transfer datasets: {unknown}")
    final_hm = load_yaml(
        Path(lock.get("deployment_config", lock["selected_config"]))
    )
    final_head = copy.deepcopy(final_hm.get("numerical_heads") or {})
    if lock.get("adaptive_router_enabled", True):
        final_head["mode"] = "auto"
    categorical = copy.deepcopy(final_hm.get("categorical_heads"))
    for dataset in selected:
        definition = matrix["transfer"]["datasets"][dataset]
        base = load_yaml(Path(definition["base_config"]))
        base = promote_schema_numeric_ordinals(base)
        variants = {
            "M2_global_support": transfer_config(
                base,
                {
                    **copy.deepcopy(matrix["numerical_head"]),
                    "mode": "support",
                    "class_frequency_weighting": "inverse_sqrt",
                    "label_smoothing": 0.01,
                },
                None,
                1.0,
            ),
            "final": transfer_config(
                base,
                final_head,
                categorical,
                float(lock["numerical_temperature"]),
            ),
        }
        for name, raw in variants.items():
            config_path = output / "resolved_configs/transfer" / dataset / f"{name}.yaml"
            raw["experiment_name"] = f"architecture_finalization_{dataset}_{name}"
            write_yaml(raw, config_path)
            run_multiseed(
                config_path=config_path,
                evaluation_config=Path(definition["evaluation_config"]),
                output_root=output / "transfer" / dataset / name,
                pretokenized_dir=Path(definition["pretokenized_dir"]),
                neighbor_cache_dir=Path(definition["neighbor_cache_dir"]),
                seeds=[42],
                evaluation_scope="configured-spine",
                sampling_policy=str(definition["sampling_policy"]),
                matrix=matrix,
                args=args,
            )


def run_report(
    matrix: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> None:
    require_lock(output, require_fully_frozen=True)
    run(
        [
            python(),
            "src/scripts/summarize_lstm_architecture_finalization.py",
            "--experiment-config", str(args.experiment_config),
        ],
        output / "logs/final_report.log",
        dry_run=args.dry_run,
    )


def collect_scope_rows(
    parent: Path,
    dataset: str,
    models: list[str],
    matrix: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        root = parent / model
        support = load_json_optional(
            root / "diagnostics/support_calibration/m2_support_calibration_report.json"
        )
        for seed in matrix["seeds"]:
            run_root = root / "runs" / f"seed_{seed}"
            paper = load_json_optional(run_root / "evaluation/paper_grade/metrics.json")
            attribute = load_json_optional(run_root / "evaluation/attribute_diagnostics.json")
            if not paper or not attribute:
                continue
            summary = paper.get("paper_metrics_summary") or {}
            runtime = load_json_optional(
                run_root / "samples/metadata/runtime_sampling_fast.json"
            )
            training = load_json_optional(run_root / "training_metadata.json")
            sampling_validation = load_json_optional(run_root / "sampling_validation.json")
            routing = load_json_optional(
                run_root / "metadata/numerical_head_routing.json"
            )
            uses_support_head = any(
                str(column.get("implementation_mode", ""))
                in {"discrete_support", "hierarchical_support"}
                for column in (routing.get("columns") or [])
            )
            support_seed = (support.get("runs") or {}).get(f"seed_{seed}") or {}
            support_metrics = list(support_seed.values())
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "seed": int(seed),
                    "constraint_violation": summary.get("constraint_violation_rate"),
                    "fk_similarity": summary.get("fk_cardinality_similarity"),
                    "shape_error": summary.get("shape_error"),
                    "full_row_c2st": summary.get("single_table_c2st_error"),
                    "temporal_event_distance": summary.get("temporal_event_distance"),
                    "text_embedding_c2st": summary.get("text_embedding_c2st_error"),
                    "trend_error": summary.get("trend_error"),
                    "numerical_only_c2st": nested(attribute, "attribute_group_c2st", "numerical_only", "c2st_error_mean"),
                    "categorical_only_c2st": nested(attribute, "attribute_group_c2st", "categorical_only", "c2st_error_mean"),
                    "categorical_tv": average_nested(attribute.get("categorical_attributes") or {}, "total_variation_distance"),
                    "conditional_categorical_error": conditional_average(attribute, "weighted_group_total_variation"),
                    "conditional_numerical_error": conditional_average(attribute, "group_mean_standardized_mae"),
                    "invalid_categorical_rate": maximum_nested(attribute.get("categorical_attributes") or {}, "invalid_category_rate"),
                    "invalid_numerical_rate": maximum_nested(attribute.get("numerical_attributes") or {}, "invalid_rate"),
                    "support_tv": finite_mean([item.get("total_variation_train_vs_generated") for item in support_metrics]),
                    "support_js": finite_mean([item.get("jensen_shannon_train_vs_generated") for item in support_metrics]),
                    "support_entropy_error": finite_mean([abs(float(item.get("entropy_difference_generated_minus_train"))) for item in support_metrics if item.get("entropy_difference_generated_minus_train") is not None]),
                    "invalid_support_rate": (
                        finite_mean(
                            [
                                item.get("invalid_support_rate")
                                for item in support_metrics
                            ]
                        )
                        if uses_support_head
                        else None
                    ),
                    "uses_support_head": uses_support_head,
                    "training_seconds": training.get("total_training_seconds", training.get("train_time_seconds")),
                    "sampling_seconds": runtime.get("total_sampling_seconds"),
                    "rows_per_second": runtime.get("rows_per_second"),
                    "parameter_count": training.get("parameter_count"),
                    "peak_training_gpu_memory_mb": training.get("peak_gpu_memory_mb"),
                    "peak_sampling_gpu_memory_mb": runtime.get("peak_gpu_memory_mb"),
                    "sample_valid": sampling_validation.get("valid"),
                    "metrics_path": str(run_root / "evaluation/paper_grade/metrics.json"),
                    **flatten_numeric_scalars(attribute, "attribute"),
                }
            )
    return pd.DataFrame(rows)


def hard_validity(
    group: pd.DataFrame,
    matrix: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "constraint_zero": bool((group["constraint_violation"].fillna(np.inf).abs() <= 1e-12).all()),
        "fk_similarity_one": bool((group["fk_similarity"].fillna(-np.inf) >= 1.0 - 1e-12).all()),
        "categorical_domain_valid": bool((group["invalid_categorical_rate"].fillna(0.0) <= 1e-12).all()),
        "numerical_domain_valid": bool((group["invalid_numerical_rate"].fillna(0.0) <= 1e-12).all()),
        "support_domain_valid": bool((group["invalid_support_rate"].fillna(0.0) <= 1e-12).all()),
        "sampling_speed": bool((group["rows_per_second"].fillna(-np.inf) > float(matrix["selection"]["minimum_rows_per_second"])).all()),
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def promote_schema_numeric_ordinals(raw: dict[str, Any]) -> dict[str, Any]:
    """Route numeric ordinal schema fields without dataset/column names."""

    resolved = copy.deepcopy(raw)
    targets = resolved.setdefault("columns", {}).setdefault("target", {})
    categorical = [str(value) for value in targets.get("categorical", [])]
    numerical = [str(value) for value in targets.get("numerical", [])]
    generated = resolved.get("generated_attributes") or {}
    fields = (resolved.get("schema") or {}).get("fields") or {}
    promoted = []
    for column in categorical:
        metadata = {**dict(fields.get(column) or {}), **dict(generated.get(column) or {})}
        semantic = str(metadata.get("semantic_type", "")).lower()
        domain = metadata.get("valid_domain")
        numeric_domain = bool(
            isinstance(domain, list)
            and domain
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in domain)
        )
        if semantic in {"ordinal_numerical", "ordinal_categorical", "quantized_numerical"} and numeric_domain:
            promoted.append(column)
    targets["categorical"] = [value for value in categorical if value not in promoted]
    targets["numerical"] = list(dict.fromkeys([*numerical, *promoted]))
    resolved.setdefault("experiment_metadata", {})["schema_promoted_numeric_ordinals"] = promoted
    return resolved


def transfer_config(
    base: dict[str, Any],
    numerical_head: dict[str, Any],
    categorical_head: dict[str, Any] | None,
    numerical_temperature: float,
) -> dict[str, Any]:
    resolved = copy.deepcopy(base)
    if ((resolved.get("columns") or {}).get("target") or {}).get("numerical"):
        resolved["numerical_heads"] = copy.deepcopy(numerical_head)
    if categorical_head:
        resolved["categorical_heads"] = copy.deepcopy(categorical_head)
    resolved.setdefault("sampling", {})["numerical_temperature"] = float(numerical_temperature)
    resolved.setdefault("experiment_metadata", {}).update(
        {
            "architecture_frozen_before_transfer": True,
            "dataset_specific_architecture_logic": False,
        }
    )
    return resolved


def evaluator_fingerprint(path: Path) -> dict[str, Any]:
    controlled = [
        ROOT / "src/evaluation/paper_metrics/c2st.py",
        ROOT / "src/evaluation/paper_metrics/utils.py",
        ROOT / "src/scripts/evaluate_single_event_table_paper_metrics.py",
        ROOT / "src/scripts/evaluate_lstm_attribute_diagnostics.py",
    ]
    hashes = {str(item): file_sha256(item) for item in controlled}
    policy = load_yaml(path)
    policy.pop("real_table_path", None)
    policy.pop("synthetic_table_path", None)
    policy_hash = object_sha256(policy)
    return {
        "evaluator_hash": object_sha256(
            {"policy_hash": policy_hash, "controlled_files": hashes}
        ),
        "policy_hash": policy_hash,
        "config_path": str(path),
        "controlled_files": hashes,
        "fixed_classifier_seed": 42,
        "preprocessing_inside_cv": source_contains(
            ROOT / "src/evaluation/paper_metrics/c2st.py", "make_pipeline("
        ),
        "primary_key_excluded": source_contains(
            ROOT / "src/evaluation/paper_metrics/c2st.py", "primary_key"
        ),
    }


def evaluator_audit_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Evaluator Audit",
            "",
            f"- Evaluator hash: `{report['evaluator_hash']}`",
            f"- Fixed classifier/evaluator seed: `{report['fixed_classifier_seed']}`",
            f"- Fold-local preprocessing pipeline detected: `{report['preprocessing_inside_cv']}`",
            f"- Primary-key exclusion logic detected: `{report['primary_key_excluded']}`",
            "- Every candidate is evaluated with the same config, feature code, hashing policy, and classifier seed.",
            "- Generator seeds vary; evaluator randomness does not.",
            "",
            "Models with a different evaluator hash are excluded from the final comparison until reevaluated.",
            "",
        ]
    )


def support_loss_audit(matrix: dict[str, Any]) -> str:
    rows = []
    for name, definition in matrix["variants"].items():
        head = definition.get("numerical_heads") or {}
        rows.append(
            (
                name,
                head.get("class_frequency_weighting", "inherited/none"),
                head.get("label_smoothing", "inherited/0"),
                nested(head, "global_prior", "residual_weight"),
            )
        )
    lines = [
        "# Support-Head Loss Audit",
        "",
        "The ordinary M2/M3/M4 setup uses inverse-square-root class weights and label smoothing. Inverse-frequency-derived weights increase the loss contribution of rare support values, while label smoothing assigns nonzero target mass to every support class. On a strongly skewed support, both mechanisms mathematically encourage flatter predictions and can explain rare-value overproduction.",
        "",
        "The M2P sweep uses standard unweighted cross-entropy (`class_frequency_weighting: none`) and zero label smoothing. No entropy regularizer, balancing sampler, support truncation, or test-derived calibration is enabled. The empirical prior is fitted only from exact training counts (`smoothing: 0`; epsilon is used only in the logarithm). Sampling temperature is selected later on validation only.",
        "",
        "| Variant | Class weighting | Label smoothing | Residual weight |",
        "| --- | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {name} | {weight} | {smoothing} | {residual} |"
        for name, weight, smoothing, residual in rows
    )
    lines.extend(
        [
            "",
            "`M2U_standard_ce` isolates the loss change (standard CE) without adding the empirical-prior residual. Comparing M2, M2U, and M2P therefore separates weighting/smoothing effects from prior anchoring.",
            "",
        ]
    )
    return "\n".join(lines)


def run_support_report_for_evaluated_root(
    spines: Path,
    root: Path,
    matrix: dict[str, Any],
    scope: str,
) -> None:
    first_config = root / "runs" / f"seed_{matrix['seeds'][0]}" / "config_resolved.yaml"
    raw = load_yaml(first_config)
    numerical = list((((raw.get("columns") or {}).get("target") or {}).get("numerical") or []))
    if not numerical:
        return
    real_name = "validation_real.csv" if scope == "validation" else "test_real.csv"
    run(
        [
            python(), "src/scripts/analyze_m2_support_calibration.py",
            "--train-real", str(spines / "train_real.csv"),
            "--validation-real", str(spines / real_name),
            "--synthetic", *[
                f"seed_{seed}={root / 'runs' / f'seed_{seed}' / 'samples/synthetic_interactions.csv'}"
                for seed in matrix["seeds"]
            ],
            "--numerical-columns", *numerical,
            "--output-dir", str(root / "diagnostics/support_calibration"),
        ],
        root / "logs/support_calibration.log",
    )


def validation_source_for_model(
    output: Path,
    model: str,
    lock: dict[str, Any],
) -> Path:
    if model == lock["selected_model"] and lock.get("validation_source_root"):
        return Path(lock["validation_source_root"])
    categorical = output / "rel_hm" / "categorical" / model
    return categorical if categorical.exists() else output / "rel_hm" / "validation" / model


def temperature_root(output: Path, temperature: float) -> Path:
    return output / "rel_hm" / "temperature" / f"tau_{temperature_label(temperature)}"


def set_numerical_temperature(
    raw: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    resolved = copy.deepcopy(raw)
    resolved.setdefault("sampling", {})["numerical_temperature"] = float(temperature)
    return resolved


def copy_shared_manifest(source: Path, destination: Path) -> None:
    shared = destination / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "source_validation_root": str(source),
            "checkpoints_reused_without_retraining": True,
            "source_comparability_manifest_sha256": (
                file_sha256(source / "shared/comparability_manifest.json")
                if (source / "shared/comparability_manifest.json").is_file()
                else None
            ),
        },
        shared / "evaluation_reuse_manifest.json",
    )


def require_lock(
    output: Path,
    *,
    require_fully_frozen: bool = False,
) -> dict[str, Any]:
    path = output / "architecture_lock.json"
    if not path.is_file():
        raise RuntimeError(
            "Architecture is not validation-locked. Run --stage select first."
        )
    lock = load_json(path)
    if lock.get("test_metrics_consulted") is not False:
        raise RuntimeError("Invalid architecture lock: test selection flag is not false")
    if require_fully_frozen and lock.get("status") != "fully_frozen_on_validation":
        raise RuntimeError(
            "Temperature/categorical validation decisions are not frozen. "
            "Run --stage temperature and --stage categorical before test."
        )
    return lock


def run(
    command: list[str],
    log_path: Path,
    *,
    dry_run: bool = False,
) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def conditional_average(report: dict[str, Any], metric: str) -> float | None:
    values = []
    for condition in (report.get("conditional_fidelity") or {}).values():
        if not isinstance(condition, dict):
            continue
        for target in condition.values():
            if isinstance(target, dict) and target.get(metric) is not None:
                values.append(target[metric])
    return finite_mean(values)


def average_nested(mapping: dict[str, Any], key: str) -> float | None:
    return finite_mean(
        [value.get(key) for value in mapping.values() if isinstance(value, dict)]
    )


def maximum_nested(mapping: dict[str, Any], key: str) -> float | None:
    values = finite_values(
        [value.get(key) for value in mapping.values() if isinstance(value, dict)]
    )
    return max(values) if values else None


def flatten_numeric_scalars(
    value: Any,
    prefix: str,
) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            output.update(
                flatten_numeric_scalars(
                    child,
                    f"{prefix}.{key}" if prefix else str(key),
                )
            )
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if np.isfinite(number):
            output[prefix] = (
                int(value)
                if isinstance(value, (int, np.integer))
                else number
            )
    return output


def finite_values(values: Any) -> list[float]:
    output = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            output.append(number)
    return output


def finite_mean(values: Any) -> float | None:
    numeric = finite_values(values)
    return float(np.mean(numeric)) if numeric else None


def finite_std(values: Any) -> float | None:
    numeric = finite_values(values)
    return float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0 if numeric else None


def null_last(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if np.isfinite(number) else float("inf")


def temperature_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_yaml(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def load_json_optional(path: Path) -> dict[str, Any]:
    return load_json(path) if path.is_file() else {}


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def object_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=json_default).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_environment() -> dict[str, Any]:
    """Record lightweight software and hardware provenance for the sweep."""

    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "pyyaml"):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    hardware: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch

        hardware.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None
                ),
                "gpu_count": int(torch.cuda.device_count()),
                "gpus": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except (ImportError, RuntimeError) as exc:
        hardware["torch_hardware_probe_error"] = str(exc)
    return {"packages": packages, "hardware": hardware}


def python() -> str:
    return sys.executable


if __name__ == "__main__":
    main()
