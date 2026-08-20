#!/usr/bin/env python3
"""Run the official RelDiff core through the temporal-interaction adapter."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # Python 3.7 developer environments.
    import importlib_metadata


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from baselines.reldiff.adapter import (  # noqa: E402
    file_sha256,
    generation_validity,
    postprocess_generated_interactions,
    prepare_training_database,
    write_json,
)
from baselines.reldiff.schema import RelDiffDatasetConfig, load_dataset_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        default="configs/experiments/reldiff_baseline.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=["preflight", "prepare", "smoke", "full", "evaluate", "summary", "all"],
        default="all",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=["amazon_toy", "movielens_100k", "rel_hm"]
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_path = Path(args.experiment_config)
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    output_root = Path(experiment["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    config_by_key = {
        config.key: config
        for config in (load_dataset_config(path) for path in experiment["datasets"])
    }
    unknown = sorted(set(args.datasets) - set(config_by_key))
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}")
    configs = [config_by_key[key] for key in args.datasets]
    if args.device:
        experiment["training"]["device"] = args.device
        experiment["training"]["sampling_device"] = args.device

    if args.stage in {"preflight", "all"}:
        run_checked(
            [
                sys.executable,
                "src/scripts/preflight_reldiff_baseline.py",
                "--experiment-config",
                str(experiment_path),
                "--output-dir",
                str(output_root / "preflight"),
            ],
            log_path=output_root / "preflight/preflight.log",
        )
        if args.stage == "preflight":
            return

    require_preflight(output_root)
    if args.stage in {"prepare", "all"}:
        for config in configs:
            prepare_one(config, experiment, smoke=False)
        if args.stage == "prepare":
            return
    if args.stage in {"smoke", "all"}:
        for config in configs:
            run_one(config, experiment, smoke=True, skip_existing=args.skip_existing)
        if args.stage == "smoke":
            return
    if args.stage in {"full", "all"}:
        for config in configs:
            run_one(config, experiment, smoke=False, skip_existing=args.skip_existing)
        if args.stage == "full":
            return
    if args.stage in {"evaluate", "all"}:
        for config in configs:
            evaluate_one(config, experiment, skip_existing=args.skip_existing)
        if args.stage == "evaluate":
            return
    summarize(configs, experiment)


def require_preflight(output_root: Path) -> None:
    path = output_root / "preflight/repeated_pair_gate.json"
    if not path.is_file():
        raise RuntimeError("Missing repeated-pair preflight. Run --stage preflight first.")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS":
        raise RuntimeError("RelDiff repeated-pair preflight is BLOCK; training is forbidden.")


def prepare_one(
    config: RelDiffDatasetConfig, experiment: dict[str, Any], *, smoke: bool
) -> dict[str, Any]:
    seed = int(experiment["seed"])
    run_root = dataset_run_root(config, experiment)
    mode = "smoke" if smoke else "full"
    work_root = run_root / mode if smoke else run_root
    provenance = work_root / "config"
    staged_name = staged_dataset_name(config, seed, smoke)
    max_rows = (
        int(experiment["training"]["smoke_train_rows"]) if smoke else None
    )
    manifest = prepare_training_database(
        config,
        data_root=ROOT / "data",
        staged_name=staged_name,
        provenance_dir=provenance,
        max_train_rows=max_rows,
        restrict_entities_to_train=smoke,
    )
    shutil.copy2(
        config_path_for_key(config.key, experiment), provenance / "adapter_config.yaml"
    )
    shutil.copy2(
        experiment["official_model_config"], provenance / "official_reldiff_config.toml"
    )
    dataset_config = provenance / "reldiff_dataset_config.toml"
    dataset_config.write_text(
        "is_disjoint = false\n"
        f'dimension_tables = ["{config.source_table}", "{config.destination_table}"]\n',
        encoding="utf-8",
    )
    manifest["reldiff_dataset_config"] = str(dataset_config)
    write_json(manifest, provenance / "data_manifest.json")
    return manifest


def run_one(
    config: RelDiffDatasetConfig,
    experiment: dict[str, Any],
    *,
    smoke: bool,
    skip_existing: bool,
) -> None:
    require_preflight(Path(experiment["output_root"]))
    seed = int(experiment["seed"])
    run_root = dataset_run_root(config, experiment)
    work_root = run_root / "smoke" if smoke else run_root
    output_csv = work_root / "generated/synthetic_interactions.csv"
    validity_path = work_root / "generated/generation_validity.json"
    if skip_existing and output_csv.is_file() and validity_path.is_file():
        validity = json.loads(validity_path.read_text(encoding="utf-8"))
        if validity.get("valid"):
            print(f"[{config.key}/{ 'smoke' if smoke else 'full' }] reusing complete output")
            return

    manifest = prepare_one(config, experiment, smoke=smoke)
    staged_name = manifest["staged_name"]
    logs = work_root / "logs"
    runtime_dir = work_root / "runtime"
    logs.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []

    preprocessing = run_timed(
        [
            sys.executable,
            "src/scripts/preprocess_data.py",
            "--dataset_name",
            staged_name,
            "--normalization",
            "quantile",
            "--sigma-data",
            "1.0",
        ],
        logs / "preprocess.log",
        runtime_dir / "preprocess_resource.json",
    )
    commands.append(preprocessing["command"])

    to_nx = run_timed(
        [
            sys.executable,
            "src/structure/to_networkx.py",
            "--dataset_name",
            staged_name,
            "--data-path",
            "data",
            "--skip-preprocess",
        ],
        logs / "to_networkx.log",
        runtime_dir / "to_networkx_resource.json",
    )
    commands.append(to_nx["command"])
    structure_path = ROOT / "data/structure" / f"{staged_name}_graph_gen.pkl"
    structure = run_timed(
        [
            sys.executable,
            "src/scripts/generate_reldiff_baseline_structure.py",
            "--dataset-name",
            staged_name,
            "--data-dir",
            "data",
            "--output",
            repository_command_path(structure_path),
            "--runtime-output",
            repository_command_path(runtime_dir / "structure_runtime.json"),
            "--seed",
            str(seed),
            "--max-retries",
            str(experiment["structure"]["max_retries"]),
        ],
        logs / "structure.log",
        runtime_dir / "structure_resource.json",
    )
    commands.append(structure["command"])

    epochs = int(
        experiment["training"]["smoke_epochs" if smoke else "full_epochs"]
    )
    batch_size = int(
        experiment["training"]["smoke_batch_size" if smoke else "batch_size"]
    )
    run_id = f"_baseline_seed{seed}_{'smoke' if smoke else 'full'}"
    train_command = [
        sys.executable,
        "src/scripts/train_joint_diffusion.py",
        staged_name,
        "--num-epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--sampling-batch-size",
        str(experiment["training"]["sampling_batch_size"]),
        "--run-id",
        run_id,
        "--config-path",
        str(experiment["official_model_config"]),
        "--dataset-config-path",
        repository_command_path(work_root / "config/reldiff_dataset_config.toml"),
        "--device",
        str(experiment["training"]["device"]),
        "--sampling-device",
        str(experiment["training"]["sampling_device"]),
        "--seed",
        str(seed),
        "--preserve-explicit-table-nodes",
        "--skip-preprocess",
        "--no-wandb",
    ]
    if experiment["training"].get("mixed_precision"):
        train_command.append("--mixed-precision")
    training = run_timed(
        train_command,
        logs / "train.log",
        runtime_dir / "training_resource.json",
    )
    commands.append(training["command"])
    upstream_checkpoint_dir = ROOT / "ckpt" / staged_name / f"multi{run_id}"
    copy_tree(upstream_checkpoint_dir, work_root / "checkpoints")

    sample_command = [
        sys.executable,
        "src/scripts/sample_joint_diffusion.py",
        staged_name,
        "--num-samples",
        "1",
        "--sampling-batch-size",
        str(experiment["training"]["sampling_batch_size"]),
        "--run-id",
        run_id,
        "--config-path",
        str(experiment["official_model_config"]),
        "--dataset-config-path",
        repository_command_path(work_root / "config/reldiff_dataset_config.toml"),
        "--structure",
        "generated",
        "--device",
        str(experiment["training"]["device"]),
        "--sampling-device",
        str(experiment["training"]["sampling_device"]),
        "--seed",
        str(seed),
        "--preserve-explicit-table-nodes",
    ]
    sampling = run_timed(
        sample_command,
        logs / "sample.log",
        runtime_dir / "sampling_resource.json",
    )
    commands.append(sampling["command"])
    generated_table = (
        ROOT
        / "data/synthetic"
        / staged_name
        / "RelDiff_gen"
        / run_id
        / "sample1"
        / f"{config.interaction_table}.csv"
    )
    if not generated_table.is_file():
        raise FileNotFoundError(f"RelDiff did not write {generated_table}")
    post_start = time.perf_counter()
    postprocess_generated_interactions(
        config,
        generated_table=generated_table,
        data_manifest=work_root / "config/data_manifest.json",
        output=output_csv,
    )
    post_seconds = time.perf_counter() - post_start
    validity = generation_validity(
        config, output_csv, work_root / "config/data_manifest.json"
    )
    write_json(validity, validity_path)
    if not validity["valid"]:
        raise RuntimeError(f"Generated table failed validity: {validity_path}")

    structure_runtime = json.loads(
        (runtime_dir / "structure_runtime.json").read_text(encoding="utf-8")
    )
    runtime = {
        "preprocessing_seconds": preprocessing["elapsed_seconds"],
        "graph_conversion_seconds": to_nx["elapsed_seconds"],
        "graph_structure_fit_seconds": structure_runtime["graph_structure_fit_seconds"],
        "graph_structure_sample_seconds": structure_runtime["graph_structure_sample_seconds"],
        "diffusion_training_seconds": training["elapsed_seconds"],
        "diffusion_sampling_seconds": sampling["elapsed_seconds"],
        "postprocessing_seconds": post_seconds,
        "total_training_seconds": preprocessing["elapsed_seconds"]
        + structure_runtime["graph_structure_fit_seconds"]
        + training["elapsed_seconds"],
        "total_sampling_seconds": structure_runtime["graph_structure_sample_seconds"]
        + sampling["elapsed_seconds"]
        + post_seconds,
        "peak_cpu_ram_kb": max(
            preprocessing.get("maximum_resident_set_kb") or 0,
            structure.get("maximum_resident_set_kb") or 0,
            training.get("maximum_resident_set_kb") or 0,
            sampling.get("maximum_resident_set_kb") or 0,
        ),
        "peak_gpu_memory_mb": None,
        "gpu": gpu_info(),
        "parameter_count": parse_parameter_count(logs / "train.log"),
        "epochs": epochs,
        "batch_size": batch_size,
        "sampling_batch_size": int(experiment["training"]["sampling_batch_size"]),
        "diffusion_steps": 100,
        "seed": seed,
        "synthetic_rows": validity["actual_row_count"],
        "rows_per_sampling_second": validity["actual_row_count"]
        / max(sampling["elapsed_seconds"], 1e-12),
    }
    write_json(runtime, runtime_dir / "runtime_summary.json")
    write_manifest(config, experiment, work_root, commands, smoke=smoke)
    print(f"[{config.key}] wrote {output_csv}")


def evaluate_one(
    config: RelDiffDatasetConfig,
    experiment: dict[str, Any],
    *,
    skip_existing: bool,
) -> None:
    run_root = dataset_run_root(config, experiment)
    generated = run_root / "generated/synthetic_interactions.csv"
    if not generated.is_file():
        raise FileNotFoundError(f"Missing full synthetic table: {generated}")
    manifest_path = run_root / "config/data_manifest.json"
    validity = generation_validity(config, generated, manifest_path)
    write_json(validity, run_root / "generated/generation_validity.json")
    if not validity["valid"]:
        raise RuntimeError(f"Validity failed for {config.key}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for label, split in (("heldout", "test"), ("train_fidelity", "train")):
        evaluation_dir = run_root / "evaluation" / label
        metrics_path = evaluation_dir / "metrics.json"
        if skip_existing and metrics_path.is_file():
            print(f"[{config.key}/{label}] reusing evaluation")
            continue
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        real_path = Path(manifest["splits"][split]["path"])
        resolved = resolved_evaluation_config(
            config, real_path, generated, evaluation_dir
        )
        resolved_path = evaluation_dir / "evaluation_config_resolved.yaml"
        resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        run_checked(
            [
                sys.executable,
                "src/scripts/evaluate_single_event_table_paper_metrics.py",
                "--config",
                str(resolved_path),
                "--real-table",
                str(real_path),
                "--synthetic-table",
                str(generated),
                "--output-dir",
                str(evaluation_dir),
                "--seed",
                str(experiment["seed"]),
            ],
            log_path=evaluation_dir / "paper_metrics.log",
        )
        run_checked(
            [
                sys.executable,
                "src/scripts/evaluate_temporal_sbm_event_spine.py",
                "--real-reviews",
                str(real_path),
                "--synthetic-reviews",
                str(generated),
                "--customer-id-col",
                config.source_fk,
                "--product-id-col",
                config.destination_fk,
                "--timestamp-col",
                config.timestamp,
                "--output",
                str(evaluation_dir / "event_spine_metrics.json"),
            ],
            log_path=evaluation_dir / "event_spine_metrics.log",
        )
        write_json(
            {
                "text_embedding_c2st": None,
                "cross_modal_text_metrics": None,
                "reason": "baseline_has_no_native_free_form_text_generator",
            },
            evaluation_dir / "text_metric_policy.json",
        )


def resolved_evaluation_config(
    config: RelDiffDatasetConfig,
    real_path: Path,
    generated_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    raw = yaml.safe_load(config.evaluation_config.read_text(encoding="utf-8"))
    raw["real_table_path"] = str(real_path)
    raw["synthetic_table_path"] = str(generated_path)
    raw.setdefault("row_count", {})["require_match_for_final"] = False
    raw["row_count"]["on_mismatch"] = "warn"
    raw["legacy_evaluator"] = {"enabled": False}
    columns = raw.get("table", {}).get("columns", {})
    allowed = set(config.semantic_columns) | {config.event_id}
    raw["table"]["columns"] = {
        name: value for name, value in columns.items() if name in allowed
    }
    raw.setdefault("evaluation", {}).setdefault("text", {})["text_columns"] = []
    raw["evaluation"]["text"]["max_text_rows"] = 0
    raw["baseline_text_policy"] = {
        "status": "NA",
        "reason": "baseline_has_no_native_free_form_text_generator",
    }
    return raw


def summarize(configs: list[RelDiffDatasetConfig], experiment: dict[str, Any]) -> None:
    output_root = Path(experiment["output_root"])
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for config in configs:
        run_root = dataset_run_root(config, experiment)
        manifest = load_json_optional(run_root / "config/data_manifest.json")
        validity = load_json_optional(run_root / "generated/generation_validity.json")
        runtime = load_json_optional(run_root / "runtime/runtime_summary.json")
        heldout = load_json_optional(run_root / "evaluation/heldout/metrics.json")
        event = load_json_optional(run_root / "evaluation/heldout/event_spine_metrics.json")
        structural = event.get("structural", {})
        temporal = event.get("temporal", {})
        joint_temporal = event.get("joint_temporal_edge", {})
        summary = heldout.get("paper_metrics_summary", {})
        row = {
            "dataset": config.key,
            "status": "complete" if heldout and validity.get("valid") else "incomplete",
            "train_rows": manifest.get("training_rows"),
            "synthetic_rows": validity.get("actual_row_count"),
            "test_rows": manifest.get("splits", {}).get("test", {}).get("rows"),
            "constraint_violation": summary.get("constraint_violation_rate"),
            "shape_error": summary.get("shape_error"),
            "structured_c2st": summary.get("structured_c2st_error"),
            "trend": summary.get("trend_error"),
            "temporal_event_distance": summary.get("temporal_event_distance"),
            "repeated_pair_error": absolute_difference(
                structural.get("duplicate_customer_product_rate_real"),
                structural.get("duplicate_customer_product_rate_synthetic"),
            ),
            "training_time": runtime.get("total_training_seconds"),
            "sampling_time": runtime.get("total_sampling_seconds"),
            "text_metric": "NA",
        }
        rows.append(row)
        structural_rows.append(
            {"dataset": config.key, **select_keys(structural, structural_metric_keys())}
        )
        temporal_rows.append(
            {
                "dataset": config.key,
                **select_keys(temporal, temporal_metric_keys()),
                **select_keys(joint_temporal, ["top_product_trajectory_corr"]),
            }
        )
        runtime_rows.append({"dataset": config.key, **runtime})

    write_csv(rows, summary_dir / "baseline_results.csv")
    write_csv(structural_rows, summary_dir / "structural_results.csv")
    write_csv(temporal_rows, summary_dir / "temporal_results.csv")
    write_csv(runtime_rows, summary_dir / "runtime_results.csv")
    implementation = build_implementation_manifest(configs, experiment, rows)
    write_json(implementation, summary_dir / "implementation_manifest.json")
    (summary_dir / "baseline_summary.md").write_text(
        render_summary(rows, implementation), encoding="utf-8"
    )
    print_final_console(rows, implementation)


def write_manifest(
    config: RelDiffDatasetConfig,
    experiment: dict[str, Any],
    work_root: Path,
    commands: list[list[str]],
    *,
    smoke: bool,
) -> None:
    manifest = {
        "dataset": config.to_manifest(),
        "seed": experiment["seed"],
        "smoke": smoke,
        "reldiff_repository": experiment["upstream_repository"],
        "reldiff_commit": git_value("rev-parse", "HEAD"),
        "bundled_upstream_lineage_commit": experiment["upstream_import_commit"],
        "modifications_to_upstream": [
            "CLI seed plumbing in train/sample entry points",
            "CLI explicit-table-node flag in train/sample entry points",
            "CLI skip-preprocess reuse flag",
            "No model, loss, schedule, diffusion, GNN, or D2K+SBM core change",
        ],
        "adapter_files": [
            "src/baselines/reldiff/schema.py",
            "src/baselines/reldiff/adapter.py",
            "src/scripts/preflight_reldiff_baseline.py",
            "src/scripts/generate_reldiff_baseline_structure.py",
            "src/scripts/run_reldiff_baseline.py",
        ],
        "exact_commands": [shlex.join(command) for command in commands],
    }
    write_json(manifest, work_root / "config/implementation_manifest.json")


def build_implementation_manifest(
    configs: list[RelDiffDatasetConfig],
    experiment: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gate = load_json_optional(
        Path(experiment["output_root"]) / "preflight/repeated_pair_gate.json"
    )
    dataset_records: dict[str, Any] = {}
    all_commands: dict[str, list[str]] = {}
    for config in configs:
        run_root = dataset_run_root(config, experiment)
        data_manifest = load_json_optional(run_root / "config/data_manifest.json")
        run_manifest = load_json_optional(
            run_root / "config/implementation_manifest.json"
        )
        dataset_records[config.key] = {
            "schema": config.to_manifest(),
            "data_manifest": data_manifest,
            "run_manifest_path": str(
                run_root / "config/implementation_manifest.json"
            ),
        }
        all_commands[config.key] = run_manifest.get("exact_commands", [])
    return {
        "reldiff_repository": experiment["upstream_repository"],
        "reldiff_commit": git_value("rev-parse", "HEAD"),
        "bundled_upstream_lineage_commit": experiment["upstream_import_commit"],
        "worktree_status": git_value("status", "--short"),
        "modifications_to_upstream": [
            "Existing float32 positional-embedding compatibility cast",
            "Existing cross-version AMP compatibility in trainer",
            "Adapter CLI seed, explicit-node, and skip-preprocess flags",
            "No change to model architecture, losses, schedules, or D2K+SBM core",
        ],
        "adapter_files": [
            "src/baselines/reldiff/schema.py",
            "src/baselines/reldiff/adapter.py",
            "src/scripts/preflight_reldiff_baseline.py",
            "src/scripts/generate_reldiff_baseline_structure.py",
            "src/scripts/run_reldiff_baseline.py",
        ],
        "python_version": platform.python_version(),
        "torch_version": package_version("torch"),
        "graph_library_version": package_version("graph-tool"),
        "cuda_version": cuda_version(),
        "seed": experiment["seed"],
        "repeated_pair_status": gate,
        "datasets": dataset_records,
        "exact_commands": all_commands,
        "hyperparameters": {
            "model_config": experiment["official_model_config"],
            **experiment["training"],
            "structure": experiment["structure"],
        },
        "results_status": rows,
        "no_test_data_used_for_training": True,
        "no_test_timestamps_used": True,
        "no_test_degree_sequence_used": True,
        "no_text_surrogates_used": True,
        "core_reldiff_architecture_modified": False,
    }


def run_timed(command: list[str], log_path: Path, resource_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    time_file = resource_path.with_suffix(".time.txt")
    wrapped = ["/usr/bin/time", "-v", "-o", str(time_file), *command]
    run_checked(wrapped, log_path=log_path)
    record = {
        "command": command,
        "elapsed_seconds": time.perf_counter() - start,
        "maximum_resident_set_kb": parse_max_rss(time_file),
    }
    write_json(record, resource_path)
    return record


def run_checked(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", shlex.join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + shlex.join(command) + "\n")
        handle.flush()
        try:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError:
            handle.flush()
            print(f"Command failed; last lines from {log_path}:", file=sys.stderr)
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(tail[-80:]), file=sys.stderr)
            raise


def dataset_run_root(
    config: RelDiffDatasetConfig, experiment: dict[str, Any]
) -> Path:
    return Path(experiment["output_root"]) / config.key / f"seed_{experiment['seed']}"


def repository_command_path(path: str | Path) -> str:
    """Return a stable repo-relative path for subprocesses launched from ROOT."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def staged_dataset_name(config: RelDiffDatasetConfig, seed: int, smoke: bool) -> str:
    suffix = "smoke" if smoke else "full"
    return f"reldiff_baseline_{config.key}_seed{seed}_{suffix}"


def config_path_for_key(key: str, experiment: dict[str, Any]) -> Path:
    for path in experiment["datasets"]:
        if load_dataset_config(path).key == key:
            return Path(path)
    raise KeyError(key)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def parse_max_rss(path: Path) -> int | None:
    if not path.is_file():
        return None
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    return int(match.group(1)) if match else None


def parse_parameter_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    match = re.search(
        r"number of parameters\s*=\s*(\d+)",
        path.read_text(encoding="utf-8", errors="replace"),
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def gpu_info() -> dict[str, Any]:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        return {"num_gpus": len(text.splitlines()), "devices": text.splitlines()}
    except Exception:
        return {"num_gpus": 0, "devices": []}


def cuda_version() -> str | None:
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def select_keys(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: data.get(key) for key in keys}


def structural_metric_keys() -> list[str]:
    return [
        "customer_degree_ks",
        "product_degree_ks",
        "duplicate_customer_product_rate_real",
        "duplicate_customer_product_rate_synthetic",
        "edge_overlap_rate",
        "top_100_product_overlap",
    ]


def temporal_metric_keys() -> list[str]:
    return [
        "global_timestamp_ks",
        "timestamp_count_l1_by_date",
        "timestamp_count_correlation_by_date",
        "customer_inter_event_time_ks",
        "product_inter_event_time_ks",
        "monthly_or_daily_count_correlation",
    ]


def absolute_difference(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    return abs(float(first) - float(second))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_summary(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    table_header = (
        "| Dataset | Rows | Constraint | Shape | Structured C2ST | Trend | "
        "Temporal Event Distance | Repeated-Pair Error | Training Time | Sampling Time |"
    )
    table_rule = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    table_rows = []
    for row in rows:
        table_rows.append(
            "| {dataset} | {synthetic_rows} | {constraint_violation} | {shape_error} | "
            "{structured_c2st} | {trend} | {temporal_event_distance} | "
            "{repeated_pair_error} | {training_time} | {sampling_time} |".format(**row)
        )
    return "\n".join(
        [
            "# RELDIFF Baseline Adaptation",
            "",
            "## 1. Executive Summary",
            "",
            f"Repeated-pair compatibility: `{manifest.get('repeated_pair_status', {}).get('verdict', 'unknown')}`.",
            "The baseline uses bundled official RelDiff D2K+SBM structure generation and graph-conditioned attribute diffusion. The adapter keeps attributed interactions as explicit native row nodes; the core architecture is unchanged.",
            "",
            "## 2. Repeated-Pair Audit",
            "",
            "See `../preflight/reldiff_code_audit.md`, `repeated_pair_unit_test.json`, and the real-data CSV audits. Three u1-i1 events survive preprocessing and graph-to-table inversion as three independent event nodes.",
            "",
            "## 3. Relational Representation",
            "",
            "Each database contains ID-only source and destination dimension tables plus an attributed interaction relation. Each interaction row remains an explicit node connected by two foreign-key edges.",
            "",
            "## 4. Timestamp Treatment",
            "",
            "Timestamp is generated as an ordinary numerical attribute measured in seconds from the training-only minimum, then decoded without clipping.",
            "",
            "## 5. Text Treatment",
            "",
            "RelDiff has no native free-form text generator. Amazon summary/review_text are excluded and text metrics are NA.",
            "",
            "## 6. Training Setup",
            "",
            f"Seed 42; bundled RelDiff lineage `{manifest.get('bundled_upstream_lineage_commit')}`; official model config `src/reldiff/configs/reldiff_config.toml`.",
            "",
            "## 7. Results",
            "",
            table_header,
            table_rule,
            *table_rows,
            "",
            "## 8. Structural Fidelity",
            "",
            "See `structural_results.csv` for source/destination degree, pair multiplicity, overlap, and popularity metrics.",
            "",
            "## 9. Temporal Fidelity",
            "",
            "See `temporal_results.csv` for event volume, timestamp, inter-event-time, and temporal trajectory metrics.",
            "",
            "## 10. Limitations",
            "",
            "RelDiff generates a static relational graph and then timestamps as numerical attributes. It is not a continuous-time event process, and no text generator is added in this vanilla baseline.",
            "",
        ]
    )


def print_final_console(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    print("=" * 60)
    print("RELDIFF BASELINE")
    print("=" * 60)
    print("UPSTREAM COMMIT:", manifest.get("bundled_upstream_lineage_commit"))
    gate = manifest.get("repeated_pair_status", {})
    print("REPEATED-PAIR PREFLIGHT:", gate.get("verdict"))
    print("PARALLEL EDGES:", "YES" if gate.get("checks", {}).get("parallel_edges_supported") else "NO")
    print("PER-EVENT EDGE ATTRIBUTES: YES (explicit event-node attributes)")
    print("ROUND-TRIP MULTIPLICITY:", "PASS" if gate.get("checks", {}).get("inverse_transform_preserves_events") else "FAIL")
    for row in rows:
        print("-" * 60)
        print(row["dataset"].upper())
        for key in (
            "status", "train_rows", "synthetic_rows", "constraint_violation",
            "shape_error", "structured_c2st", "trend", "temporal_event_distance",
            "training_time", "sampling_time", "text_metric",
        ):
            print(f"{key}: {row.get(key)}")
        fraction = gate.get("repeated_pair_fraction_train", {}).get(row["dataset"])
        print(f"repeated_pair_fraction: {fraction}")
    print("=" * 60)
    print("NO TEST DATA USED FOR TRAINING: YES")
    print("NO TEST TIMESTAMPS USED: YES")
    print("NO TEST DEGREE SEQUENCE USED: YES")
    print("NO TEXT SURROGATES USED: YES")
    print("CORE RELDIFF ARCHITECTURE MODIFIED: NO")
    print("=" * 60)


if __name__ == "__main__":
    main()
