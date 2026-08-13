#!/usr/bin/env python3
"""Run a reproducible multi-seed LSTM attribute-generation experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMConfig,
    ConditionalTABDLMSchema,
    resolve_auto_review_text_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pretokenized-dir", required=True)
    parser.add_argument("--neighbor-cache-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-batch-size", default="8192")
    parser.add_argument(
        "--sampling-policy",
        choices=["fast", "v53-length-preserving"],
        default="fast",
        help=(
            "Use the ordinary optimized sampler, or the established v5.3 "
            "length-preserving exact-overlap policy for text datasets."
        ),
    )
    parser.add_argument(
        "--evaluation-scope",
        choices=[
            "heldout-validation",
            "heldout-test",
            "configured-spine",
        ],
        default="heldout-test",
        help=(
            "Evaluate on the held-out real test spine (default), or on "
            "the config's fixed full evaluation spine and full real table."
        ),
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help=(
            "Fixed evaluator/classifier seed. When omitted, retain the "
            "legacy behavior of using each generator seed. Architecture "
            "selection runs should pass one fixed value such as 42."
        ),
    )
    parser.add_argument("--smoke-rows", type=int, default=64)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebuild-precomputed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--comparison-metrics", nargs="*", default=[])
    parser.add_argument(
        "--minimum-free-disk-gb",
        type=float,
        default=2.0,
        help=(
            "Refuse to start a non-reused seed below this free-space "
            "threshold."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.config)
    eval_config_path = Path(args.evaluation_config)
    output_root = Path(args.output_root)
    shared = output_root / "shared"
    logs = output_root / "logs"
    for path in [output_root, shared, logs]:
        path.mkdir(parents=True, exist_ok=True)

    base = resolve_auto_review_text_config(load_yaml(base_config_path))
    schema = ConditionalTABDLMSchema.from_config_dict(base)
    config = ConditionalTABDLMConfig(
        raw=base,
        schema=schema,
        config_path=base_config_path,
    )
    require_file(config.train_data_path, "prepared interaction table")
    require_file(eval_config_path, "paper-metrics config")
    inventory = experiment_inventory(config, base_config_path, eval_config_path)
    write_json(inventory, shared / "repository_inventory.json")
    print(json.dumps(inventory, indent=2, sort_keys=True), flush=True)

    run_stage(
        [
            python(),
            "src/scripts/audit_lstm_interaction_experiment.py",
            "--config",
            str(base_config_path),
            "--evaluation-config",
            str(eval_config_path),
            "--output-dir",
            str(shared / "audit"),
        ],
        logs / "audit.log",
        args,
    )
    run_stage(
        [
            python(),
            "src/scripts/materialize_interaction_lstm_splits.py",
            "--config",
            str(base_config_path),
            "--output-dir",
            str(shared / "spines"),
        ],
        logs / "materialize_spines.log",
        args,
    )
    if args.dry_run:
        expected_splits = {}
    else:
        expected_splits = load_json(
            shared / "spines" / "split_spines_summary.json"
        )["splits"]

    ensure_pretokenized(
        base_config_path,
        config.train_data_path,
        Path(args.pretokenized_dir),
        expected_splits,
        args,
        logs,
    )
    ensure_neighbor_cache(
        base_config_path,
        config.train_data_path,
        Path(args.neighbor_cache_dir),
        expected_materialized_rows(expected_splits)
        if not args.dry_run else None,
        args,
        logs,
    )
    if not args.dry_run:
        write_comparability_manifest(
            output_root=output_root,
            config_path=base_config_path,
            evaluation_config_path=eval_config_path,
            spines=shared / "spines",
            pretokenized_dir=Path(args.pretokenized_dir),
            neighbor_cache_dir=Path(args.neighbor_cache_dir),
            expected_splits=expected_splits,
        )
    run_stage(
        [
            python(),
            "src/scripts/audit_c2st_integrity.py",
            "--config",
            str(eval_config_path),
            "--real-table",
            str(shared / "spines" / "test_real.csv"),
            "--output",
            str(shared / "c2st_integrity_audit.json"),
            "--max-rows-per-side",
            "5000",
            "--seed",
            "42",
        ],
        logs / "c2st_integrity.log",
        args,
    )

    if not args.skip_smoke:
        run_seed(
            seed=int(args.seeds[0]),
            base=base,
            eval_template=load_yaml(eval_config_path),
            output_dir=output_root / "smoke",
            shared=shared,
            pretokenized_dir=Path(args.pretokenized_dir),
            neighbor_cache_dir=Path(args.neighbor_cache_dir),
            evaluation_scope=resolve_evaluation_scope(
                config,
                shared,
                args.evaluation_scope,
            ),
            args=args,
            smoke=True,
        )
    if args.smoke_only or args.dry_run:
        print(
            "Smoke/dry-run phase complete; full seed training was not launched.",
            flush=True,
        )
        return

    per_seed = []
    for seed in args.seeds:
        per_seed.append(
            run_seed(
                seed=int(seed),
                base=base,
                eval_template=load_yaml(eval_config_path),
                output_dir=output_root / "runs" / f"seed_{int(seed)}",
                shared=shared,
                pretokenized_dir=Path(args.pretokenized_dir),
                neighbor_cache_dir=Path(args.neighbor_cache_dir),
                evaluation_scope=resolve_evaluation_scope(
                    config,
                    shared,
                    args.evaluation_scope,
                ),
                args=args,
                smoke=False,
            )
        )
    write_aggregate_outputs(
        per_seed,
        output_root,
        args.comparison_metrics,
    )
    print(f"Completed all seeds. Results: {output_root / 'results'}", flush=True)


def run_seed(
    *,
    seed: int,
    base: dict[str, Any],
    eval_template: dict[str, Any],
    output_dir: Path,
    shared: Path,
    pretokenized_dir: Path,
    neighbor_cache_dir: Path,
    evaluation_scope: dict[str, Path | str],
    args: argparse.Namespace,
    smoke: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    real_for_eval = prepare_evaluation_real(
        Path(evaluation_scope["real_table"]),
        output_dir,
        smoke_rows=int(args.smoke_rows) if smoke else None,
        dry_run=bool(args.dry_run),
    )
    resolved = resolve_seed_config(
        base,
        seed,
        output_dir,
        Path(evaluation_scope["spine"]),
        pretokenized_dir,
        neighbor_cache_dir,
        numerical_head_training_table=(
            shared / "spines" / "train_real.csv"
        ),
        smoke=smoke,
    )
    config_path = output_dir / "config_resolved.yaml"
    eval_resolved = resolve_evaluation_config(
        eval_template,
        shared / "spines" / "train_real.csv",
        real_for_eval,
        output_dir / "samples" / "synthetic_interactions.csv",
        ConditionalTABDLMSchema.from_config_dict(resolved),
        seed if args.evaluation_seed is None else args.evaluation_seed,
    )
    eval_path = output_dir / "evaluation_config_resolved.yaml"
    existing_compatible = existing_run_request_is_compatible(
        config_path,
        eval_path,
        resolved,
        eval_resolved,
    )
    existing_artifacts = any(
        path.exists()
        for path in (
            output_dir / "checkpoints" / "best.pt",
            output_dir / "training_metadata.json",
            output_dir / "samples" / "synthetic_interactions.csv",
            output_dir / "evaluation" / "paper_grade" / "metrics.json",
        )
    )
    if args.skip_existing and existing_artifacts and not existing_compatible:
        raise RuntimeError(
            f"Refusing to reuse incompatible artifacts at {output_dir}. "
            "Use a new output root or intentionally rerun without "
            "--skip-existing."
        )
    persisted_resolved = (
        load_yaml(config_path)
        if existing_compatible and existing_artifacts
        else resolved
    )
    write_yaml(persisted_resolved, config_path)
    write_yaml(eval_resolved, eval_path)
    write_json(
        {
            "version": 1,
            "generator_seed": int(seed),
            "evaluator_seed": int(
                seed
                if args.evaluation_seed is None
                else args.evaluation_seed
            ),
            "evaluation_scope": str(evaluation_scope["mode"]),
            "sampling_policy": str(args.sampling_policy),
            "model_request_sha256": object_sha256(
                comparable_model_config(resolved)
            ),
            "evaluation_request_sha256": object_sha256(eval_resolved),
        },
        output_dir / "run_request.json",
    )
    checkpoint = output_dir / "checkpoints" / "best.pt"
    training_metadata = output_dir / "training_metadata.json"
    synthetic = output_dir / "samples" / "synthetic_interactions.csv"
    sampling_runtime = (
        output_dir / "samples" / "metadata" / "runtime_sampling_fast.json"
    )
    sampling_validation = output_dir / "sampling_validation.json"
    paper_metrics = output_dir / "evaluation" / "paper_grade" / "metrics.json"
    attribute_diagnostics = output_dir / "evaluation" / "attribute_diagnostics.json"
    if (
        args.skip_existing
        and checkpoint.is_file()
        and training_metadata.is_file()
        and synthetic.is_file()
        and sampling_runtime.is_file()
        and sampling_validation.is_file()
        and paper_metrics.is_file()
        and attribute_diagnostics.is_file()
    ):
        print(f"[seed {seed}] reusing completed run at {output_dir}", flush=True)
        return collect_seed_result(seed, output_dir)

    train_command = [
        python(),
        "src/scripts/train_lstm_joint_full_review_text.py",
        "--config",
        str(config_path),
        "--pretokenized-dir",
        str(pretokenized_dir),
        "--neighbor-cache-dir",
        str(neighbor_cache_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--mixed-precision",
        "--seed",
        str(seed),
    ]
    training_reused = completed_stage(
        args.skip_existing,
        checkpoint,
        training_metadata,
    )
    if training_reused:
        print(f"[seed {seed}] reusing completed training", flush=True)
    else:
        if not args.dry_run:
            require_free_disk_space(
                output_dir,
                minimum_gb=float(args.minimum_free_disk_gb),
                context=f"seed {seed} training",
            )
        run_stage(
            train_command,
            output_dir / "logs" / "train.log",
            args,
        )
    if not args.dry_run:
        require_file(checkpoint, f"best checkpoint for seed {seed}")

    sample_rows = str(args.smoke_rows) if smoke else "all"
    sample_command = [
        python(),
        "src/scripts/sample_lstm_joint_full_review_text_fast.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--synthetic-spine",
        str(evaluation_scope["spine"]),
        "--output",
        str(synthetic),
        "--num-rows",
        sample_rows,
        "--batch-size",
        str(args.sample_batch_size),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--mixed-precision",
        "--profile",
    ]
    graph_history_prefix = evaluation_scope.get("graph_history_prefix")
    if graph_history_prefix is not None:
        sample_command.extend(
            ["--graph-history-prefix", str(graph_history_prefix)]
        )
    if args.sampling_policy == "v53-length-preserving":
        sample_command[1] = (
            "src/scripts/"
            "sample_lstm_joint_length_preserving_privacy_fast.py"
        )
    sampling_reused = completed_stage(
        args.skip_existing,
        synthetic,
        sampling_runtime,
    )
    if sampling_reused:
        print(f"[seed {seed}] reusing completed sampling", flush=True)
    else:
        run_stage(
            sample_command,
            output_dir / "logs" / "sample.log",
            args,
        )
    if not args.dry_run:
        if sampling_reused and sampling_validation.is_file():
            print(f"[seed {seed}] reusing sampling validation", flush=True)
            validation = load_json(sampling_validation)
        else:
            validation = validate_sampled_table(
                Path(evaluation_scope["spine"]),
                synthetic,
                shared / "spines" / "train_real.csv",
                ConditionalTABDLMSchema.from_config_dict(resolved),
                num_rows=int(args.smoke_rows) if smoke else None,
            )
            write_json(validation, sampling_validation)
        if not validation["valid"]:
            raise RuntimeError(
                f"Sample validation failed for seed {seed}: {validation['errors']}"
            )

    eval_command = [
        python(),
        "src/scripts/evaluate_single_event_table_paper_metrics.py",
        "--config",
        str(eval_path),
        "--real-table",
        str(real_for_eval),
        "--synthetic-table",
        str(synthetic),
        "--output-dir",
        str(output_dir / "evaluation" / "paper_grade"),
        "--seed",
        str(
            seed
            if args.evaluation_seed is None
            else args.evaluation_seed
        ),
    ]
    if smoke:
        eval_command.extend(["--sample-size", str(args.smoke_rows)])
    evaluation_reused = (
        sampling_reused
        and completed_stage(args.skip_existing, paper_metrics)
    )
    if evaluation_reused:
        print(f"[seed {seed}] reusing paper-grade evaluation", flush=True)
    else:
        run_stage(
            eval_command,
            output_dir / "logs" / "evaluate.log",
            args,
        )
    if completed_stage(args.skip_existing, attribute_diagnostics):
        print(f"[seed {seed}] reusing attribute diagnostics", flush=True)
    else:
        run_stage(
            attribute_diagnostics_command(
                config_path=config_path,
                evaluation_config_path=eval_path,
                train_real_path=shared / "spines" / "train_real.csv",
                evaluation_real_path=real_for_eval,
                synthetic_path=synthetic,
                graph_history_prefix_path=(
                    Path(graph_history_prefix)
                    if graph_history_prefix is not None
                    else None
                ),
                output_path=attribute_diagnostics,
                seed=seed,
            ),
            output_dir / "logs" / "attribute_diagnostics.log",
            args,
        )
    if args.dry_run:
        return {"seed": int(seed), "dry_run": True}
    return collect_seed_result(seed, output_dir)


def completed_stage(skip_existing: bool, *artifacts: Path) -> bool:
    return bool(skip_existing and all(path.is_file() for path in artifacts))


def resolve_evaluation_scope(
    config: ConditionalTABDLMConfig,
    shared: Path,
    mode: str,
) -> dict[str, Path | str]:
    """Resolve fixed real/spine inputs without exposing target attributes."""

    if mode == "heldout-validation":
        return {
            "mode": mode,
            "real_table": shared / "spines" / "validation_real.csv",
            "spine": shared / "spines" / "validation_spine.csv",
            "graph_history_prefix": shared / "spines" / "train_spine.csv",
        }
    if mode == "heldout-test":
        return {
            "mode": mode,
            "real_table": shared / "spines" / "test_real.csv",
            "spine": shared / "spines" / "test_spine.csv",
            "graph_history_prefix": (
                shared / "spines" / "history_prefix_spine.csv"
            ),
        }
    if mode == "configured-spine":
        return {
            "mode": mode,
            "real_table": config.train_data_path,
            "spine": config.synthetic_spine_path,
            "graph_history_prefix": None,
        }
    raise ValueError(f"Unknown evaluation scope: {mode!r}")


def attribute_diagnostics_command(
    *,
    config_path: Path,
    evaluation_config_path: Path,
    train_real_path: Path,
    evaluation_real_path: Path,
    synthetic_path: Path,
    graph_history_prefix_path: Path | None,
    output_path: Path,
    seed: int,
) -> list[str]:
    command = [
        python(),
        "src/scripts/evaluate_lstm_attribute_diagnostics.py",
        "--config",
        str(config_path),
        "--train-real",
        str(train_real_path),
        "--evaluation-real",
        str(evaluation_real_path),
        "--synthetic",
        str(synthetic_path),
        "--evaluation-config",
        str(evaluation_config_path),
        "--output",
        str(output_path),
        "--seed",
        str(seed),
    ]
    if graph_history_prefix_path is not None:
        command.extend(
            ["--graph-history-prefix", str(graph_history_prefix_path)]
        )
    return command


def prepare_evaluation_real(
    test_real_path: Path,
    output_dir: Path,
    *,
    smoke_rows: int | None,
    dry_run: bool,
) -> Path:
    if smoke_rows is None:
        return test_real_path
    smoke_path = output_dir / "evaluation_real_smoke.csv"
    if not dry_run:
        real = pd.read_csv(
            test_real_path,
            nrows=int(smoke_rows),
            low_memory=False,
        )
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        real.to_csv(smoke_path, index=False)
    return smoke_path


def resolve_seed_config(
    base: dict[str, Any],
    seed: int,
    output_dir: Path,
    test_spine: Path,
    pretokenized_dir: Path,
    neighbor_cache_dir: Path,
    *,
    numerical_head_training_table: Path | None = None,
    smoke: bool,
) -> dict[str, Any]:
    resolved = copy.deepcopy(base)
    paths = resolved.setdefault("paths", {})
    paths["output_dir"] = str(output_dir)
    paths["synthetic_spine_path"] = str(test_spine)
    paths["pretokenized_dir"] = str(pretokenized_dir)
    paths["neighbor_cache_dir"] = str(neighbor_cache_dir)
    if numerical_head_training_table is not None:
        paths["numerical_head_training_table_path"] = str(
            numerical_head_training_table
        )
    training = resolved.setdefault("training", {})
    training["seed"] = int(seed)
    sampling = resolved.setdefault("sampling", {})
    sampling["seed"] = int(seed)
    sampling["num_rows"] = "all"
    if smoke:
        training["max_steps"] = 2
        if "epochs" in training:
            training["epochs"] = min(int(training["epochs"]), 2)
        training["steps_per_eval"] = 1
        training["steps_per_checkpoint"] = 1
        training["validation_max_batches"] = 2
        training["early_stopping_patience"] = 2
    resolved.setdefault("experiment_metadata", {})["seed"] = int(seed)
    resolved["experiment_metadata"].setdefault(
        "baseline_architecture_changed",
        False,
    )
    return resolved


def resolve_evaluation_config(
    template: dict[str, Any],
    train_real_path: Path,
    test_real_path: Path,
    synthetic_path: Path,
    schema: ConditionalTABDLMSchema,
    evaluator_seed: int,
) -> dict[str, Any]:
    resolved = copy.deepcopy(template)
    resolved["real_table_path"] = str(test_real_path)
    resolved["synthetic_table_path"] = str(synthetic_path)
    resolved.setdefault("evaluation", {})["random_seed"] = int(
        evaluator_seed
    )
    train = pd.read_csv(train_real_path, low_memory=False)
    columns = resolved.setdefault("table", {}).setdefault("columns", {})
    for column in schema.categorical_targets:
        if column in train and column in columns:
            values = train[column].dropna().drop_duplicates().tolist()
            columns[column]["valid_values"] = [
                value.item() if isinstance(value, np.generic) else value
                for value in values
            ]
    for column in schema.numerical_targets:
        if column in train and column in columns:
            values = pd.to_numeric(train[column], errors="coerce").dropna()
            if len(values):
                columns[column]["support"] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
    return resolved


def validate_sampled_table(
    spine_path: Path,
    synthetic_path: Path,
    train_path: Path,
    schema: ConditionalTABDLMSchema,
    num_rows: int | None,
) -> dict[str, Any]:
    spine = pd.read_csv(spine_path, low_memory=False)
    if num_rows is not None:
        spine = spine.head(int(num_rows)).copy()
    synthetic = pd.read_csv(synthetic_path, low_memory=False)
    train = pd.read_csv(train_path, low_memory=False)
    errors = []
    expected_columns = [
        *(
            ["event_id"]
            if "event_id" in spine.columns
            else []
        ),
        *schema.condition_columns,
        *schema.target_columns,
    ]
    if list(synthetic.columns) != list(expected_columns):
        errors.append(
            f"Output columns differ: expected={expected_columns}, got={list(synthetic.columns)}"
        )
    if len(synthetic) != len(spine):
        errors.append(
            f"Row-count mismatch: spine={len(spine)}, synthetic={len(synthetic)}"
        )
    for column in [
        *(["event_id"] if "event_id" in spine.columns else []),
        *schema.foreign_key_columns,
    ]:
        if column not in synthetic:
            continue
        if not np.array_equal(
            spine[column].astype(str).to_numpy(),
            synthetic[column].astype(str).to_numpy(),
        ):
            errors.append(f"Pass-through column changed or was reordered: {column}")
    for column in schema.datetime_columns:
        if column not in synthetic:
            continue
        real_time = pd.to_datetime(spine[column], errors="coerce", utc=True)
        syn_time = pd.to_datetime(synthetic[column], errors="coerce", utc=True)
        if not real_time.equals(syn_time):
            errors.append(f"Timestamp column changed or was reordered: {column}")
    categorical = {}
    for column in schema.categorical_targets:
        domain = set(train[column].dropna().astype(str))
        invalid = ~synthetic[column].astype(str).isin(domain)
        categorical[column] = {
            "train_domain": sorted(domain),
            "invalid_count": int(invalid.sum()),
        }
        if invalid.any():
            errors.append(
                f"Categorical target {column!r} has {int(invalid.sum())} invalid values"
            )
    numerical = {}
    for column in schema.numerical_targets:
        train_values = pd.to_numeric(train[column], errors="coerce").dropna()
        values = pd.to_numeric(synthetic[column], errors="coerce")
        invalid = (
            values.isna()
            | ~np.isfinite(values)
            | (values < train_values.min())
            | (values > train_values.max())
        )
        numerical[column] = {
            "train_min": float(train_values.min()),
            "train_max": float(train_values.max()),
            "invalid_or_out_of_range_count": int(invalid.sum()),
        }
        if invalid.any():
            errors.append(
                f"Numerical target {column!r} has {int(invalid.sum())} invalid/out-of-range values"
            )
    return {
        "valid": not errors,
        "errors": errors,
        "row_count": int(len(synthetic)),
        "expected_row_count": int(len(spine)),
        "event_spine_preserved": not any(
            "changed or was reordered" in error for error in errors
        ),
        "categorical_targets": categorical,
        "numerical_targets": numerical,
        "synthetic_sha256": file_sha256(synthetic_path),
    }


def ensure_pretokenized(
    config_path: Path,
    table_path: Path,
    output_dir: Path,
    expected_splits: dict[str, Any],
    args: argparse.Namespace,
    logs: Path,
) -> None:
    metadata_path = output_dir / "metadata.json"
    valid = False
    expected = {
        "train_rows": int(expected_splits["train"]["rows"]),
        "valid_rows": int(expected_splits["validation"]["rows"]),
        "test_rows": int(expected_splits["test"]["rows"]),
    }
    cache_counts: dict[str, int] | None = None
    if metadata_path.exists() and not args.dry_run:
        metadata = load_json(metadata_path)
        metadata_counts = {
            key: int(metadata.get(key, -1))
            for key in expected
        }
        cache_counts = pretokenized_split_counts(output_dir)
        valid = (
            cache_counts == expected
            if cache_counts is not None
            else metadata_counts == expected
        )
        if valid:
            source = (
                "split index arrays"
                if cache_counts == expected
                else "metadata"
            )
            print(
                f"[pretokenized] reusing {output_dir}; "
                f"{source} match {expected}",
                flush=True,
            )
    if valid:
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.rebuild_precomputed:
            raise RuntimeError(
                f"Pretokenized cache at {output_dir} does not match "
                f"the explicit split. expected={expected}, "
                f"index_counts={cache_counts}. "
                "Rerun with --rebuild-precomputed."
            )
        shutil.rmtree(output_dir)
    run_stage(
        [
            python(),
            "src/scripts/pretokenize_single_event_table_text_fields.py",
            "--config",
            str(config_path),
            "--real-table",
            str(table_path),
            "--output-dir",
            str(output_dir),
            "--chunk-size",
            "500000",
        ],
        logs / "pretokenize.log",
        args,
    )


def pretokenized_split_counts(
    output_dir: Path,
) -> dict[str, int] | None:
    paths = {
        "train_rows": output_dir / "train_indices.npy",
        "valid_rows": output_dir / "valid_indices.npy",
        "test_rows": output_dir / "test_indices.npy",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        return {
            key: int(len(np.load(path, mmap_mode="r")))
            for key, path in paths.items()
        }
    except (OSError, ValueError):
        return None


def ensure_neighbor_cache(
    config_path: Path,
    table_path: Path,
    output_dir: Path,
    expected_rows: int | None,
    args: argparse.Namespace,
    logs: Path,
) -> None:
    metadata_path = output_dir / "metadata.json"
    valid = False
    if metadata_path.exists() and expected_rows is not None:
        metadata = load_json(metadata_path)
        safety = metadata.get("temporal_safety_sample") or {}
        valid = (
            int(metadata.get("num_rows", -1)) == int(expected_rows)
            and int(safety.get("future_or_same_time_violations", 1)) == 0
        )
        if valid:
            print(f"[neighbor-cache] reusing verified cache at {output_dir}", flush=True)
    if valid:
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.rebuild_precomputed:
            raise RuntimeError(
                f"Neighbor cache at {output_dir} is stale or unsafe. "
                "Rerun with --rebuild-precomputed."
            )
        shutil.rmtree(output_dir)
    run_stage(
        [
            python(),
            "src/scripts/precompute_temporal_neighbor_cache.py",
            "--config",
            str(config_path),
            "--real-table",
            str(table_path),
            "--output-dir",
            str(output_dir),
            "--chunk-size",
            "500000",
        ],
        logs / "neighbor_cache.log",
        args,
    )


def expected_materialized_rows(
    expected_splits: dict[str, Any],
) -> int:
    return int(
        sum(
            int((expected_splits.get(name) or {}).get("rows", 0))
            for name in ("train", "validation", "test")
        )
    )


def collect_seed_result(seed: int, output_dir: Path) -> dict[str, Any]:
    metrics = load_json(output_dir / "evaluation" / "paper_grade" / "metrics.json")
    attribute_diagnostics_path = (
        output_dir / "evaluation" / "attribute_diagnostics.json"
    )
    attribute_diagnostics = load_json(attribute_diagnostics_path)
    training = load_json(output_dir / "training_metadata.json")
    sampling = load_json(
        output_dir / "samples" / "metadata" / "runtime_sampling_fast.json"
    )
    summary = metrics.get("paper_metrics_summary") or {}
    result = {
        "dataset": metrics.get("dataset", {}).get("dataset_name"),
        "model": "lstm_v53",
        "seed": int(seed),
        "constraint_violation": summary.get("constraint_violation_rate"),
        "fk_similarity": summary.get("fk_cardinality_similarity"),
        "shape_error": summary.get("shape_error"),
        "single_table_c2st": summary.get("single_table_c2st_error"),
        "temporal_event_distance": summary.get("temporal_event_distance"),
        "trend_error": summary.get("trend_error"),
        "text_embedding_c2st": summary.get("text_embedding_c2st_error"),
        "training_time_seconds": training.get(
            "total_training_seconds",
            training.get("train_time_seconds"),
        ),
        "sampling_time_seconds": sampling.get("total_sampling_seconds"),
        "rows_per_second": sampling.get("rows_per_second"),
        "peak_training_gpu_memory_mb": training.get("peak_gpu_memory_mb"),
        "peak_sampling_gpu_memory_mb": sampling.get("peak_gpu_memory_mb"),
        "best_checkpoint": training.get("best_checkpoint_path"),
        "synthetic_table": str(
            output_dir / "samples" / "synthetic_interactions.csv"
        ),
        "metrics_path": str(
            output_dir / "evaluation" / "paper_grade" / "metrics.json"
        ),
        "attribute_diagnostics_path": str(attribute_diagnostics_path),
        "resolved_config": str(output_dir / "config_resolved.yaml"),
    }
    result.update(
        flatten_numeric_scalars(
            attribute_diagnostics,
            prefix="attribute",
        )
    )
    return result


def flatten_numeric_scalars(
    value: Any,
    *,
    prefix: str,
) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(
                flatten_numeric_scalars(child, prefix=child_prefix)
            )
    elif isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric):
            flattened[prefix] = (
                int(value)
                if isinstance(value, (int, np.integer))
                and not isinstance(value, bool)
                else numeric
            )
    return flattened


def write_aggregate_outputs(
    rows: list[dict[str, Any]],
    output_root: Path,
    comparison_metrics: list[str],
) -> None:
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "per_seed_metrics.csv", index=False)
    numeric = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column != "seed"
    ]
    aggregate: dict[str, Any] = {
        "Dataset": frame["dataset"].iloc[0] if len(frame) else None,
        "Model": frame["model"].iloc[0] if len(frame) else None,
        "Seeds": ",".join(str(seed) for seed in frame["seed"].tolist()),
        "num_seeds": int(frame["seed"].nunique()),
    }
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        aggregate[f"{column}_mean"] = float(values.mean()) if len(values) else None
        aggregate[f"{column}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        )
    aggregate_frame = pd.DataFrame([aggregate])
    aggregate_frame.to_csv(results_dir / "aggregate_mean_std.csv", index=False)
    summary = {
        "per_seed": rows,
        "aggregate": aggregate,
        "metric_direction": {
            "constraint_violation": "lower is better; zero is ideal",
            "fk_similarity": "higher is better",
            "shape_error": "lower is better",
            "single_table_c2st": "lower is better; zero corresponds to chance AUC",
            "temporal_event_distance": "lower is better",
            "trend_error": "lower is better",
        },
    }
    write_json(summary, results_dir / "summary.json")
    (results_dir / "report.md").write_text(
        aggregate_markdown(frame, aggregate),
        encoding="utf-8",
    )
    if comparison_metrics:
        write_comparison(comparison_metrics, frame, results_dir)


def aggregate_markdown(
    per_seed: pd.DataFrame,
    aggregate: dict[str, Any],
) -> str:
    lines = [
        "# Multi-seed LSTM Attribute Experiment",
        "",
        f"- Dataset: {aggregate.get('Dataset')}",
        f"- Model: {aggregate.get('Model')}",
        f"- Seeds: {aggregate.get('Seeds')}",
        "",
        "## Aggregate",
        "",
        "| Metric | Mean | Std |",
        "| --- | ---: | ---: |",
    ]
    for metric in [
        "constraint_violation",
        "fk_similarity",
        "shape_error",
        "single_table_c2st",
        "temporal_event_distance",
        "trend_error",
        "training_time_seconds",
        "sampling_time_seconds",
        "rows_per_second",
    ]:
        lines.append(
            f"| {metric} | {format_value(aggregate.get(metric + '_mean'))} | "
            f"{format_value(aggregate.get(metric + '_std'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Structural FK and timestamp metrics are evaluated on a fixed held-out event spine; they verify preservation rather than event-spine generation quality.",
            "- Rel-HM is a text-free mixed-attribute task: `price` is numerical and `sales_channel_id` is categorical.",
            "- Compare stability and validity across datasets, but do not rank datasets solely by raw C2ST because their schemas differ.",
            "",
            "## Per-seed artifacts",
            "",
        ]
    )
    for _, row in per_seed.iterrows():
        lines.append(
            f"- Seed {int(row['seed'])}: checkpoint `{row['best_checkpoint']}`, "
            f"synthetic table `{row['synthetic_table']}`, metrics `{row['metrics_path']}`"
        )
    return "\n".join(lines) + "\n"


def write_comparison(
    paths: list[str],
    rel_hm: pd.DataFrame,
    results_dir: Path,
) -> None:
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        metrics = load_json(path)
        summary = metrics.get("paper_metrics_summary") or {}
        rows.append(
            {
                "dataset": metrics.get("dataset", {}).get("dataset_name"),
                "metrics_path": str(path),
                **summary,
            }
        )
    rows.append(
        {
            "dataset": rel_hm["dataset"].iloc[0],
            "metrics_path": "three-seed aggregate; see aggregate_mean_std.csv",
            "constraint_violation_rate": rel_hm["constraint_violation"].mean(),
            "fk_cardinality_similarity": rel_hm["fk_similarity"].mean(),
            "shape_error": rel_hm["shape_error"].mean(),
            "single_table_c2st_error": rel_hm["single_table_c2st"].mean(),
            "temporal_event_distance": rel_hm["temporal_event_distance"].mean(),
            "trend_error": rel_hm["trend_error"].mean(),
        }
    )
    pd.DataFrame(rows).to_csv(
        results_dir / "cross_dataset_context.csv",
        index=False,
    )


def experiment_inventory(
    config: ConditionalTABDLMConfig,
    config_path: Path,
    eval_config_path: Path,
) -> dict[str, Any]:
    table = config.train_data_path
    header = pd.read_csv(table, nrows=0)
    return {
        "model_config": str(config_path),
        "evaluation_config": str(eval_config_path),
        "interaction_table": str(table),
        "interaction_columns": list(header.columns),
        "event_spine_columns": list(config.schema.condition_columns),
        "generated_attributes": list(config.schema.target_columns),
        "configured_output_dir": str(config.output_dir),
        "existing_checkpoints": sorted(
            str(path)
            for path in config.output_dir.rglob("*.pt")
        )
        if config.output_dir.exists()
        else [],
        "existing_synthetic_tables": sorted(
            str(path)
            for path in config.output_dir.rglob("*synthetic*.csv")
        )
        if config.output_dir.exists()
        else [],
        "git_commit": git_revision(),
        "config_sha256": file_sha256(config_path),
        "table_sha256": file_sha256(table),
    }


def write_comparability_manifest(
    *,
    output_root: Path,
    config_path: Path,
    evaluation_config_path: Path,
    spines: Path,
    pretokenized_dir: Path,
    neighbor_cache_dir: Path,
    expected_splits: dict[str, Any],
) -> None:
    split_files = (
        "train_real.csv",
        "validation_real.csv",
        "test_real.csv",
        "test_spine.csv",
        "history_prefix_spine.csv",
    )
    index_files = (
        "train_indices.npy",
        "valid_indices.npy",
        "test_indices.npy",
    )
    manifest = {
        "version": 1,
        "git_commit": git_revision(),
        "config_sha256": file_sha256(config_path),
        "evaluation_config_sha256": file_sha256(
            evaluation_config_path
        ),
        "c2st_source_sha256": file_sha256(
            ROOT / "src/evaluation/paper_metrics/c2st.py"
        ),
        "controlled_source_fingerprints": {
            path: file_sha256(ROOT / path)
            for path in (
                "src/attribute_generation/conditional_tabdlm/lstm_joint.py",
                "src/attribute_generation/conditional_tabdlm/numerical.py",
                "src/attribute_generation/conditional_tabdlm/numerical_head.py",
                "src/attribute_generation/conditional_tabdlm/schema.py",
                "src/evaluation/paper_metrics/c2st.py",
                "src/scripts/evaluate_lstm_attribute_diagnostics.py",
            )
        },
        "expected_splits": expected_splits,
        "split_fingerprints": {
            name: file_sha256(spines / name)
            for name in split_files
            if (spines / name).exists()
        },
        "precomputed_split_fingerprints": {
            name: file_sha256(pretokenized_dir / name)
            for name in index_files
            if (pretokenized_dir / name).exists()
        },
        "pretokenized_metadata_sha256": (
            file_sha256(pretokenized_dir / "metadata.json")
            if (pretokenized_dir / "metadata.json").exists()
            else None
        ),
        "neighbor_cache_metadata_sha256": (
            file_sha256(neighbor_cache_dir / "metadata.json")
            if (neighbor_cache_dir / "metadata.json").exists()
            else None
        ),
    }
    write_json(
        manifest,
        output_root / "shared" / "comparability_manifest.json",
    )


def run_stage(
    command: list[str],
    log_path: Path,
    args: argparse.Namespace,
) -> None:
    print("$ " + " ".join(command), flush=True)
    if args.dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def require_free_disk_space(
    path: Path,
    *,
    minimum_gb: float,
    context: str,
) -> None:
    usage = shutil.disk_usage(path)
    free_gb = float(usage.free / (1024**3))
    free_inodes = int(os.statvfs(path).f_favail)
    if free_gb < float(minimum_gb) or free_inodes <= 0:
        raise RuntimeError(
            f"Insufficient filesystem capacity for {context}: "
            f"{free_gb:.2f} GiB free and {free_inodes:,} free inodes "
            f"at {path}. Required free disk: {minimum_gb:.2f} GiB. "
            "Remove stale *.pt.tmp files and unneeded completed-run "
            "last.pt checkpoints, then retry with --skip-existing."
        )


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def existing_run_request_is_compatible(
    config_path: Path,
    evaluation_path: Path,
    requested_config: dict[str, Any],
    requested_evaluation: dict[str, Any],
) -> bool:
    if not config_path.is_file() or not evaluation_path.is_file():
        return False
    existing_config = comparable_model_config(load_yaml(config_path))
    requested = comparable_model_config(requested_config)
    return (
        object_sha256(existing_config) == object_sha256(requested)
        and object_sha256(load_yaml(evaluation_path))
        == object_sha256(requested_evaluation)
    )


def comparable_model_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Remove only training-fitted metadata before request comparison."""

    comparable = copy.deepcopy(raw)
    comparable.pop("_numerical_head_metadata", None)
    comparable.pop("_categorical_head_metadata", None)
    return comparable


def object_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_yaml(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_scalar) + "\n",
        encoding="utf-8",
    )


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def python() -> str:
    return sys.executable


def format_value(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "NA"


if __name__ == "__main__":
    main()
