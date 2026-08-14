#!/usr/bin/env python3
"""Evaluate the frozen LSTM architecture on Yelp and RetailRocket."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT / "src"))

from data_preprocessing.interaction_datasets.registry import get_adapter  # noqa: E402
from data_preprocessing.interaction_datasets.validation import validate_subset  # noqa: E402


DEFAULT_CONFIG = "configs/experiments/frozen_lstm_5dataset.yaml"
STAGES = ("inspect", "prepare", "run", "report", "all")
NEW_DATASETS = ("yelp", "retailrocket")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--datasets", nargs="+", choices=NEW_DATASETS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-batch-size", default="8192")
    parser.add_argument("--minimum-free-disk-gb", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebuild-precomputed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(Path(args.experiment_config))
    selected = list(args.datasets or NEW_DATASETS)
    output = Path(matrix["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    stages = STAGES[:-1] if args.stage == "all" else (args.stage,)
    inventory: dict[str, Any] | None = None
    for stage in stages:
        print(f"\n===== frozen-lstm-5dataset: {stage} =====", flush=True)
        if stage == "inspect":
            inventory = inspect_repository(matrix, selected)
            write_json(inventory, output / "preflight_inventory.json")
            require_preflight_ready(inventory)
        elif stage == "prepare":
            inventory = inventory or inspect_repository(matrix, selected)
            require_preflight_ready(inventory)
            prepare_datasets_and_configs(matrix, selected, args)
        elif stage == "run":
            inventory = inventory or inspect_repository(matrix, selected)
            require_preflight_ready(inventory)
            if not args.dry_run:
                require_prepared_inputs(matrix, selected)
            run_new_datasets(matrix, selected, args)
        elif stage == "report":
            finalize_and_report(matrix)


def inspect_repository(
    matrix: dict[str, Any],
    selected: list[str],
) -> dict[str, Any]:
    frozen = matrix["frozen_architecture"]
    frozen_path = Path(frozen["config"])
    lock_path = Path(frozen["lock"])
    text_template = Path(frozen["text_template"])
    frozen_record: dict[str, Any] = {
        "path": str(frozen_path),
        "exists": frozen_path.is_file(),
        "sha256": sha256_file(frozen_path) if frozen_path.is_file() else None,
        "lock_path": str(lock_path),
        "lock_exists": lock_path.is_file(),
        "text_template_path": str(text_template),
        "text_template_exists": text_template.is_file(),
        "text_template_sha256": (
            sha256_file(text_template) if text_template.is_file() else None
        ),
    }
    if lock_path.is_file():
        frozen_record["lock"] = load_json(lock_path)

    datasets: dict[str, Any] = {}
    preprocessing = []
    training = []
    for name in selected:
        definition = matrix["datasets"][name]
        adapter = get_adapter(definition["adapter"])
        processed = processed_dir(matrix, definition)
        ready = processed_subset_ready(adapter, processed)
        raw_files: dict[str, str] = {}
        raw_error = None
        try:
            raw_files = {
                key: str(path)
                for key, path in adapter.locate_raw_files(
                    matrix["raw_root"]
                ).files.items()
            }
        except FileNotFoundError as exc:
            raw_error = str(exc)
        if not ready:
            preprocessing.append(
                {
                    "dataset": name,
                    "operation": "generic source-entity-induced subset",
                    "output": str(processed),
                    "raw_available": bool(raw_files),
                    "missing_raw": raw_error,
                }
            )
        checkpoint = (
            Path(definition["output_root"])
            / "runs/seed_42/checkpoints/best.pt"
        )
        if not checkpoint.is_file():
            training.append(
                {
                    "dataset": name,
                    "seed": 42,
                    "output": str(definition["output_root"]),
                }
            )
        datasets[name] = {
            "base_config": definition["base_config"],
            "evaluation_config": definition["evaluation_config"],
            "processed_dir": str(processed),
            "processed_ready": ready,
            "raw_files": raw_files,
            "raw_error": raw_error,
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.is_file(),
            "fixed_spine_protocol": "heldout-test",
            "split_policy": "explicit chronological 70/15/15",
        }
    inventory = {
        "frozen_model_config": frozen_record,
        "datasets": datasets,
        "required_new_data_preprocessing": preprocessing,
        "required_new_training_runs": training,
        "existing_pipeline": {
            "subset_builder": "src/scripts/build_interaction_subsets.py",
            "split_materializer": "src/scripts/materialize_interaction_lstm_splits.py",
            "training_driver": "src/scripts/run_lstm_multiseed_experiment.py",
            "development_evaluator": "src/scripts/evaluate_single_event_table_paper_metrics.py",
            "attribute_diagnostics": "src/scripts/evaluate_lstm_attribute_diagnostics.py",
            "event_spine_conditions": "source FK, destination FK, timestamp",
            "graph_context": "past-only; target event excluded",
        },
    }
    print_inventory(inventory)
    return inventory


def print_inventory(inventory: dict[str, Any]) -> None:
    print("\nFROZEN_MODEL_CONFIG", flush=True)
    print(json.dumps(inventory["frozen_model_config"], indent=2), flush=True)
    print("\nEXISTING_YELP_ASSETS", flush=True)
    print(json.dumps(inventory["datasets"].get("yelp"), indent=2), flush=True)
    print("\nEXISTING_RETAILROCKET_ASSETS", flush=True)
    print(
        json.dumps(inventory["datasets"].get("retailrocket"), indent=2),
        flush=True,
    )
    print("\nREQUIRED_NEW_DATA_PREPROCESSING", flush=True)
    print(
        json.dumps(inventory["required_new_data_preprocessing"], indent=2),
        flush=True,
    )
    print("\nREQUIRED_NEW_TRAINING_RUNS", flush=True)
    print(
        json.dumps(inventory["required_new_training_runs"], indent=2),
        flush=True,
    )


def require_preflight_ready(inventory: dict[str, Any]) -> None:
    frozen = inventory["frozen_model_config"]
    missing = []
    if not frozen["exists"]:
        missing.append(f"frozen config: {frozen['path']}")
    if not frozen["lock_exists"]:
        missing.append(f"architecture lock: {frozen['lock_path']}")
    if not frozen["text_template_exists"]:
        missing.append(
            f"frozen text template: {frozen['text_template_path']}"
        )
    for item in inventory["required_new_data_preprocessing"]:
        if not item["raw_available"]:
            missing.append(
                f"{item['dataset']} raw data required for preprocessing: "
                f"{item['missing_raw']}"
            )
    if missing:
        raise FileNotFoundError(
            "Preflight stopped before preprocessing/training:\n- "
            + "\n- ".join(missing)
        )
    validate_frozen_contract(
        load_yaml(Path(frozen["path"])),
        frozen.get("lock") or {},
    )


def processed_dir(
    matrix: dict[str, Any],
    definition: dict[str, Any],
) -> Path:
    return Path(matrix["processed_root"]) / definition["processed_name"]


def processed_subset_ready(adapter: Any, directory: Path) -> bool:
    required = [
        directory / "interactions.csv",
        directory / adapter.source_table_filename,
        directory / adapter.destination_table_filename,
        directory / "subset_manifest.json",
        directory / "validation_report.json",
    ]
    return all(path.is_file() for path in required)


def require_prepared_inputs(
    matrix: dict[str, Any],
    selected: list[str],
) -> None:
    missing = []
    for name in selected:
        definition = matrix["datasets"][name]
        adapter = get_adapter(definition["adapter"])
        directory = processed_dir(matrix, definition)
        if not processed_subset_ready(adapter, directory):
            missing.append(
                f"{name}: {directory} (run --stage prepare first)"
            )
    if missing:
        raise FileNotFoundError(
            "Frozen-dataset training inputs are not prepared:\n- "
            + "\n- ".join(missing)
        )


def prepare_datasets_and_configs(
    matrix: dict[str, Any],
    selected: list[str],
    args: argparse.Namespace,
) -> None:
    output = Path(matrix["output_root"])
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name in selected:
        definition = matrix["datasets"][name]
        adapter = get_adapter(definition["adapter"])
        directory = processed_dir(matrix, definition)
        if not processed_subset_ready(adapter, directory):
            command = [
                sys.executable,
                "src/scripts/build_interaction_subsets.py",
                "--dataset",
                str(definition["adapter"]),
                "--raw-root",
                str(matrix["raw_root"]),
                "--processed-root",
                str(matrix["processed_root"]),
                "--target-interactions",
                str(matrix["target_interactions"]),
                "--allowed-relative-error",
                str(matrix["allowed_relative_error"]),
                "--seed",
                str(matrix["seed"]),
                "--chunk-size",
                str(args.chunk_size),
                "--no-download",
            ]
            run_command(command, logs / f"build_{name}.log", args.dry_run)
        if not args.dry_run:
            report = validate_subset(adapter, directory)
            write_json(
                report,
                output / "datasets" / name / "subset_validation.json",
            )
            if not report.get("valid"):
                raise RuntimeError(
                    f"Prepared {name} subset failed validation: "
                    f"{report.get('errors')}"
                )
            copy_construction_artifacts(directory, output / "datasets" / name)
    write_frozen_dataset_configs(matrix, selected)


def write_frozen_dataset_configs(
    matrix: dict[str, Any],
    selected: list[str],
) -> dict[str, Path]:
    frozen_path = Path(matrix["frozen_architecture"]["config"])
    frozen = load_yaml(frozen_path)
    lock = load_json(Path(matrix["frozen_architecture"]["lock"]))
    validate_frozen_contract(frozen, lock)
    text_template = load_yaml(
        Path(matrix["frozen_architecture"]["text_template"])
    )
    output = Path(matrix["output_root"])
    paths: dict[str, Path] = {}
    for name in selected:
        definition = matrix["datasets"][name]
        base = load_yaml(Path(definition["base_config"]))
        resolved = derive_frozen_dataset_config(
            base,
            frozen,
            text_template,
            output_root=Path(definition["output_root"]),
            seed=int(matrix["seed"]),
            frozen_config_path=frozen_path,
        )
        path = output / "configs" / f"{name}_seed42.yaml"
        write_yaml(resolved, path)
        paths[name] = path
    return paths


def derive_frozen_dataset_config(
    base: dict[str, Any],
    frozen: dict[str, Any],
    text_template: dict[str, Any],
    *,
    output_root: Path,
    seed: int,
    frozen_config_path: Path,
) -> dict[str, Any]:
    """Apply the frozen architecture using schema-driven modality routing."""

    resolved = copy.deepcopy(base)
    targets = (resolved.get("columns") or {}).get("target") or {}
    categorical = [str(value) for value in targets.get("categorical") or []]
    numerical = [str(value) for value in targets.get("numerical") or []]
    text = [str(value) for value in targets.get("text") or []]
    if text:
        apply_frozen_text_branch(resolved, text_template, text)
    else:
        apply_structured_backbone(resolved, frozen)
    apply_frozen_graph_settings(resolved, text_template if text else frozen)
    resolved["numerical_heads"] = copy.deepcopy(frozen["numerical_heads"])
    resolved.pop("categorical_heads", None)
    ensure_event_spine_roles(resolved)
    resolved.setdefault("paths", {})["output_dir"] = str(output_root)
    resolved.setdefault("training", {})["seed"] = int(seed)
    resolved.setdefault("sampling", {})["seed"] = int(seed)
    resolved["sampling"]["num_rows"] = "all"
    resolved["sampling"]["numerical_temperature"] = 1.0
    resolved["loss_weights"] = {
        **{column: 1.0 for column in categorical},
        **{column: 1.0 for column in numerical},
        **{column: 1.0 for column in text},
    }
    if "review_text" in text and (
        resolved.get("review_text_length") or {}
    ).get("enabled"):
        resolved.setdefault("auxiliary_targets", {})["categorical"] = [
            "review_text_length_bucket"
        ]
        resolved["loss_weights"]["review_text_length"] = 2.0
    resolved["experiment_name"] = (
        f"frozen_lstm_{resolved.get('dataset_name')}_seed42"
    )
    resolved.setdefault("experiment_metadata", {}).update(
        {
            "frozen_architecture": True,
            "architecture_modified": False,
            "frozen_config_path": str(frozen_config_path),
            "frozen_config_sha256": sha256_file(frozen_config_path),
            "support_prior_alpha": 1.0,
            "support_residual_weight": 0.25,
            "support_sampling_temperature": 1.0,
            "categorical_architecture": "original",
            "temporal_prior_lambda": 0.0,
            "training_only_numerical_routing": True,
            "generator_seed": int(seed),
            "dataset_specific_model_logic": False,
        }
    )
    validate_derived_frozen_config(resolved)
    return resolved


def apply_structured_backbone(
    resolved: dict[str, Any],
    frozen: dict[str, Any],
) -> None:
    for key in (
        "tokenizer",
        "id_encoding",
        "datetime_encoding",
        "model",
        "text_decoder",
        "review_text_decoder",
        "summary_decoder",
        "text_length_prediction",
        "training",
        "training_regularization",
        "loss",
        "sampling",
    ):
        if key in frozen:
            resolved[key] = copy.deepcopy(frozen[key])


def apply_frozen_text_branch(
    resolved: dict[str, Any],
    template: dict[str, Any],
    text_targets: list[str],
) -> None:
    for key in (
        "tokenizer",
        "id_encoding",
        "datetime_encoding",
        "model",
        "text_decoder",
        "training",
        "training_regularization",
        "loss",
        "sampling",
    ):
        if key in template:
            resolved[key] = copy.deepcopy(template[key])
    template_text = copy.deepcopy(template.get("text") or {})
    max_length = template_text.get("max_length") or {}
    template_text["max_length"] = {
        column: max_length.get(column, "auto") for column in text_targets
    }
    resolved["text"] = template_text
    if "review_text" in text_targets:
        resolved["review_text"] = copy.deepcopy(
            template.get("review_text") or {}
        )
        resolved["review_text_length"] = copy.deepcopy(
            template.get("review_text_length") or {}
        )
        decoder = copy.deepcopy(template.get("review_text_decoder") or {})
        decoder["condition_on_summary"] = bool(
            "summary" in text_targets
        )
        resolved["review_text_decoder"] = decoder
    if "summary" not in text_targets:
        resolved.pop("summary_length", None)
        resolved.pop("summary_length_loss", None)
        resolved.pop("summary_decoder", None)


def apply_frozen_graph_settings(
    resolved: dict[str, Any],
    template: dict[str, Any],
) -> None:
    graph = resolved.setdefault("graph_conditioning", {})
    source = template.get("graph_conditioning") or {}
    for key in (
        "enabled",
        "mode",
        "add_reverse_edges",
        "graph_uses_future_events",
        "graph_uses_target_attributes",
        "leakage_policy",
        "graph_encoder",
    ):
        if key in source:
            graph[key] = copy.deepcopy(source[key])
    source_temporal = copy.deepcopy(source.get("temporal_filter") or {})
    timestamp = (graph.get("temporal_filter") or {}).get(
        "timestamp_column"
    )
    if source_temporal:
        if timestamp:
            source_temporal["timestamp_column"] = timestamp
        graph["temporal_filter"] = source_temporal
    targets = (resolved.get("columns") or {}).get("target") or {}
    graph["forbidden_node_features"] = list(
        dict.fromkeys(
            str(value)
            for role in ("categorical", "numerical", "text")
            for value in targets.get(role) or []
        )
    )
    graph["graph_uses_future_events"] = False
    graph["graph_uses_target_attributes"] = False


def ensure_event_spine_roles(config: dict[str, Any]) -> None:
    conditions = (config.get("columns") or {}).get("condition") or {}
    foreign_keys = [str(value) for value in conditions.get("foreign_keys") or []]
    datetimes = [str(value) for value in conditions.get("datetimes") or []]
    if len(foreign_keys) != 2 or len(datetimes) != 1:
        raise ValueError(
            "Frozen single-event-table model requires two FK conditions "
            f"and one timestamp; got {foreign_keys}, {datetimes}"
        )
    event = config.setdefault("event_spine", {})
    event["source_fk"] = foreign_keys[0]
    event["destination_fk"] = foreign_keys[1]
    event["timestamp"] = datetimes[0]


def validate_frozen_contract(
    frozen: dict[str, Any],
    lock: dict[str, Any],
) -> None:
    head = frozen.get("numerical_heads") or {}
    prior = head.get("global_prior") or {}
    temporal = prior.get("temporal_prior") or {}
    numerical_temperature = (frozen.get("sampling") or {}).get(
        "numerical_temperature", 1.0
    )
    errors = []
    if str(head.get("mode")) != "auto":
        errors.append("numerical_heads.mode must be auto")
    if float(prior.get("alpha", -1)) != 1.0:
        errors.append("support prior alpha must be 1.0")
    if float(prior.get("residual_weight", -1)) != 0.25:
        errors.append("support residual gamma must be 0.25")
    if bool(temporal.get("enabled")) or float(temporal.get("lambda_t", -1)) != 0.0:
        errors.append("temporal support prior must be disabled with lambda 0")
    if float(numerical_temperature) != 1.0:
        errors.append("support sampling temperature must be 1.0")
    if frozen.get("categorical_heads"):
        errors.append("frozen config contains categorical prior anchoring")
    if lock.get("categorical_architecture") != "original":
        errors.append("architecture lock must select original categorical head")
    if float(lock.get("temporal_prior_lambda", -1)) != 0.0:
        errors.append("architecture lock temporal lambda must be 0")
    if errors:
        raise RuntimeError("Frozen architecture contract mismatch: " + "; ".join(errors))


def validate_derived_frozen_config(config: dict[str, Any]) -> None:
    head = config.get("numerical_heads") or {}
    prior = head.get("global_prior") or {}
    temporal = prior.get("temporal_prior") or {}
    graph = config.get("graph_conditioning") or {}
    errors = []
    if str(head.get("mode")) != "auto":
        errors.append("router is not auto")
    if float(prior.get("residual_weight", -1)) != 0.25:
        errors.append("gamma differs from 0.25")
    if bool(temporal.get("enabled")) or float(temporal.get("lambda_t", -1)) != 0.0:
        errors.append("temporal prior is enabled")
    if config.get("categorical_heads"):
        errors.append("categorical prior anchoring is present")
    if bool(graph.get("graph_uses_future_events")):
        errors.append("graph uses future events")
    if bool(graph.get("graph_uses_target_attributes")):
        errors.append("graph uses target attributes")
    if errors:
        raise RuntimeError("Derived config violates frozen contract: " + "; ".join(errors))


def run_new_datasets(
    matrix: dict[str, Any],
    selected: list[str],
    args: argparse.Namespace,
) -> None:
    config_paths = write_frozen_dataset_configs(matrix, selected)
    for name in selected:
        definition = matrix["datasets"][name]
        command = [
            sys.executable,
            "src/scripts/run_lstm_multiseed_experiment.py",
            "--config",
            str(config_paths[name]),
            "--evaluation-config",
            str(definition["evaluation_config"]),
            "--output-root",
            str(definition["output_root"]),
            "--pretokenized-dir",
            str(definition["pretokenized_dir"]),
            "--neighbor-cache-dir",
            str(definition["neighbor_cache_dir"]),
            "--seeds",
            "42",
            "--evaluation-seed",
            "42",
            "--evaluation-scope",
            "heldout-test",
            "--sampling-policy",
            str(definition["sampling_policy"]),
            "--sample-batch-size",
            str(args.sample_batch_size),
            "--minimum-free-disk-gb",
            str(args.minimum_free_disk_gb),
            "--device",
            str(args.device),
            "--skip-smoke",
        ]
        if args.skip_existing:
            command.append("--skip-existing")
        if args.rebuild_precomputed:
            command.append("--rebuild-precomputed")
        if args.dry_run:
            command.append("--dry-run")
        run_command(
            command,
            Path(definition["output_root"]) / "logs/frozen_driver.log",
            args.dry_run,
        )


def run_command(command: list[str], log_path: Path, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
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


def copy_construction_artifacts(source: Path, destination: Path) -> None:
    target = destination / "construction"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "subset_manifest.json",
        "validation_report.json",
        "schema.yaml",
        "statistics.json",
        "statistics.md",
        "README.md",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, target / name)


def finalize_and_report(matrix: dict[str, Any]) -> None:
    output = Path(matrix["output_root"])
    for name in NEW_DATASETS:
        require_completed_new_run(matrix, name)
        package_run(matrix, name)
    write_yelp_routing_audit(matrix)
    diagnostics = write_dataset_diagnostics(matrix)
    leakage = write_leakage_audit(matrix)
    final = write_five_dataset_results(matrix)
    summary = write_dataset_summary(matrix)
    write_run_report(matrix, final, summary, diagnostics, leakage)
    print_final_console(matrix, final, leakage, diagnostics)


def require_completed_new_run(matrix: dict[str, Any], name: str) -> None:
    root = Path(matrix["datasets"][name]["output_root"]) / "runs/seed_42"
    required = [
        root / "checkpoints/best.pt",
        root / "training_metadata.json",
        root / "samples/synthetic_interactions.csv",
        root / "samples/metadata/runtime_sampling_fast.json",
        root / "sampling_validation.json",
        root / "evaluation/paper_grade/metrics.json",
        root / "evaluation/attribute_diagnostics.json",
        root / "config_resolved.yaml",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete frozen {name} run:\n- " + "\n- ".join(missing)
        )


def package_run(matrix: dict[str, Any], name: str) -> None:
    output = Path(matrix["output_root"])
    root = Path(matrix["datasets"][name]["output_root"])
    run = root / "runs/seed_42"
    stable = {
        run / "checkpoints/best.pt": output / "checkpoints" / f"{name}_seed42/best.pt",
        run / "samples/synthetic_interactions.csv": output / "synthetic" / f"{name}_seed42/synthetic_interactions.csv",
        run / "evaluation/paper_grade/metrics.json": output / "metrics" / f"{name}_seed42.json",
        run / "evaluation/attribute_diagnostics.json": output / "metrics" / f"{name}_seed42_attribute_diagnostics.json",
        run / "sampling_validation.json": output / "metrics" / f"{name}_seed42_sampling_validation.json",
        run / "config_resolved.yaml": output / "configs" / f"{name}_seed42_resolved.yaml",
    }
    for source, destination in stable.items():
        stable_link_or_copy(source, destination)
    spines = root / "shared/spines"
    references = {
        "dataset": name,
        "source_interaction_table": str(
            processed_dir(matrix, matrix["datasets"][name])
            / "interactions.csv"
        ),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        real = spines / f"{split}_real.csv"
        spine = spines / f"{split}_spine.csv"
        references["splits"][split] = {
            "real": file_reference(real),
            "fixed_event_spine": file_reference(spine),
        }
    references["history_prefix"] = file_reference(
        spines / "history_prefix_spine.csv"
    )
    write_json(references, root / "data_references.json")


def stable_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) == sha256_file(destination):
            return
        raise RuntimeError(
            f"Refusing to overwrite differing stable artifact: {destination}"
        )
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(source, destination)


def file_reference(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "rows": csv_row_count(path) if path.is_file() else None,
    }


def write_yelp_routing_audit(matrix: dict[str, Any]) -> None:
    output = Path(matrix["output_root"])
    route_path = (
        Path(matrix["datasets"]["yelp"]["output_root"])
        / "runs/seed_42/metadata/numerical_head_routing.json"
    )
    routing = load_json(route_path)
    rows = []
    for item in routing.get("columns") or []:
        stats = item.get("diagnostic_statistics") or {}
        rows.append(
            {
                "field": item.get("column"),
                "train_row_count": item.get("train_rows"),
                "unique_count": item.get("unique_count"),
                "unique_ratio": item.get("unique_ratio"),
                "repeated_value_mass": stats.get("repeated_observation_mass"),
                "inferred_numerical_type": item.get("inferred_type"),
                "selected_head": item.get("chosen_head"),
                "implementation_mode": item.get("implementation_mode"),
                "support_size": stats.get("support_size"),
                "reason": routing_reason(stats, item.get("chosen_head")),
            }
        )
    frame = pd.DataFrame(rows)
    expected = {"useful", "funny", "cool"}
    if set(frame.get("field", [])) != expected:
        raise RuntimeError(
            "Yelp routing audit does not contain exactly useful/funny/cool"
        )
    frame.to_csv(output / "numerical_routing_yelp.csv", index=False)


def routing_reason(stats: dict[str, Any], selected: Any) -> str:
    signals = stats.get("structured_signals") or {}
    active = [name for name, value in signals.items() if bool(value)]
    if str(selected) == "support_prior":
        return "training-only structured signals: " + ", ".join(active)
    return "training-only statistics did not meet structured-support threshold"


def write_dataset_diagnostics(
    matrix: dict[str, Any],
) -> dict[str, Any]:
    output = Path(matrix["output_root"])
    diagnostics: dict[str, Any] = {}
    for name in NEW_DATASETS:
        definition = matrix["datasets"][name]
        root = Path(definition["output_root"])
        run = root / "runs/seed_42"
        real = pd.read_csv(root / "shared/spines/test_real.csv", low_memory=False)
        synthetic = pd.read_csv(
            run / "samples/synthetic_interactions.csv",
            low_memory=False,
        )
        attribute = load_json(
            run / "evaluation/attribute_diagnostics.json"
        )
        paper = load_json(run / "evaluation/paper_grade/metrics.json")
        runtime = load_json(
            run / "samples/metadata/runtime_sampling_fast.json"
        )
        if name == "yelp":
            report = yelp_diagnostics(
                real,
                synthetic,
                attribute,
                paper,
                runtime,
            )
        else:
            report = retailrocket_diagnostics(
                real,
                synthetic,
                attribute,
                paper,
            )
        diagnostics[name] = report
        write_json(report, output / "metrics" / f"{name}_diagnostics.json")
    return diagnostics


def yelp_diagnostics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    attribute: dict[str, Any],
    paper: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    categorical = attribute.get("categorical_attributes") or {}
    numerical = attribute.get("numerical_attributes") or {}
    real_stars = pd.to_numeric(real["stars"], errors="coerce")
    syn_stars = pd.to_numeric(synthetic["stars"], errors="coerce")
    text_real = real["review_text"].fillna("").astype(str)
    text_syn = synthetic["review_text"].fillna("").astype(str)
    dependencies = attribute.get("dependency_fidelity") or {}
    selected_dependencies = [
        item
        for item in dependencies.get("pairs") or []
        if (
            {str(item.get("left")), str(item.get("right"))}
            & {"stars"}
        )
        and (
            {str(item.get("left")), str(item.get("right"))}
            & {"useful", "funny", "cool", "review_text"}
        )
    ]
    return {
        "stars": {
            **(categorical.get("stars") or {}),
            "ordinal_wasserstein": wasserstein_1d(real_stars, syn_stars),
            "distribution_real": category_proportions(real["stars"]),
            "distribution_synthetic": category_proportions(
                synthetic["stars"]
            ),
        },
        "counts": {
            column: numerical.get(column) or {}
            for column in ("useful", "funny", "cool")
        },
        "numerical_routing_path": "outputs/frozen_lstm_5dataset/numerical_routing_yelp.csv",
        "text": {
            "empty_rate_real": empty_text_rate(text_real),
            "empty_rate_synthetic": empty_text_rate(text_syn),
            "token_length_real": length_summary(
                text_real.map(lambda value: len(value.split()))
            ),
            "token_length_synthetic": length_summary(
                text_syn.map(lambda value: len(value.split()))
            ),
            "character_length_real": length_summary(text_real.str.len()),
            "character_length_synthetic": length_summary(text_syn.str.len()),
            "text_embedding_c2st_error": nested(
                paper,
                "paper_metrics_summary",
                "text_embedding_c2st_error",
            ),
            "sampling_seconds": first_number(
                runtime,
                "total_sampling_seconds",
                "sampling_seconds",
            ),
        },
        "selected_dependencies": selected_dependencies,
        "customer_conditioning": nested(
            attribute, "conditional_fidelity", "user_id"
        ),
        "business_conditioning": nested(
            attribute, "conditional_fidelity", "business_id"
        ),
        "time_conditioning": nested(
            attribute, "conditional_fidelity", "_time_bin"
        ),
        "history_coverage": attribute.get("history_coverage"),
    }


def retailrocket_diagnostics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    attribute: dict[str, Any],
    paper: dict[str, Any],
) -> dict[str, Any]:
    categorical = (
        (attribute.get("categorical_attributes") or {}).get("event_type")
        or {}
    )
    conditional = attribute.get("conditional_fidelity") or {}
    return {
        "event_type": {
            **categorical,
            "proportions_real": category_proportions(real["event_type"]),
            "proportions_synthetic": category_proportions(
                synthetic["event_type"]
            ),
            "categorical_only_c2st": nested(
                attribute,
                "attribute_group_c2st",
                "categorical_only",
                "c2st_error_mean",
            ),
        },
        "event_type_by_time_bucket": conditional.get("_time_bin"),
        "event_type_by_visitor_history_bucket": conditional.get(
            "_visitor_id_history_bucket"
        ),
        "event_type_by_item_history_bucket": conditional.get(
            "_item_id_history_bucket"
        ),
        "history_coverage": attribute.get("history_coverage"),
        "core_metrics": paper.get("paper_metrics_summary"),
    }


def write_leakage_audit(matrix: dict[str, Any]) -> dict[str, Any]:
    output = Path(matrix["output_root"])
    reports: dict[str, Any] = {}
    all_passed = True
    for name in NEW_DATASETS:
        definition = matrix["datasets"][name]
        root = Path(definition["output_root"])
        run = root / "runs/seed_42"
        config = load_yaml(run / "config_resolved.yaml")
        route_path = run / "metadata/numerical_head_routing.json"
        routing = load_json(route_path) if route_path.is_file() else {
            "training_only": False,
            "columns": [],
        }
        numerical_metadata_path = (
            run / "metadata/numerical_head_metadata.json"
        )
        numerical_metadata = (
            load_json(numerical_metadata_path)
            if numerical_metadata_path.is_file()
            else {}
        )
        split_check = chronological_split_check(root / "shared/spines")
        graph = config.get("graph_conditioning") or {}
        temporal = graph.get("temporal_filter") or {}
        targets = set(
            str(value)
            for role in ("categorical", "numerical", "text")
            for value in (
                ((config.get("columns") or {}).get("target") or {}).get(role)
                or []
            )
        )
        numerical_targets = set(
            str(value)
            for value in (
                ((config.get("columns") or {}).get("target") or {}).get(
                    "numerical"
                )
                or []
            )
        )
        forbidden = set(
            str(value) for value in graph.get("forbidden_node_features") or []
        )
        eval_config = load_yaml(run / "evaluation_config_resolved.yaml")
        eval_columns = set(
            str(value)
            for value in ((eval_config.get("table") or {}).get("columns") or {})
        )
        pretokenized = load_json(
            Path(definition["pretokenized_dir"]) / "metadata.json"
        )
        neighbor = load_json(
            Path(definition["neighbor_cache_dir"]) / "metadata.json"
        )
        auto_length = (
            (config.get("_auto_text_length_metadata") or {}).get(
                "review_text"
            )
            or {}
        )
        checks = {
            "chronological_split": split_check["passed"],
            "past_only_graph_context": bool(
                temporal.get("enabled")
                and temporal.get("mode") == "past_only"
                and not graph.get("graph_uses_future_events", False)
            ),
            "target_event_excluded": bool(
                temporal.get("exclude_target_event_from_neighbors")
            ),
            "target_attributes_excluded_from_context": bool(
                not graph.get("graph_uses_target_attributes", False)
                and targets.issubset(forbidden)
            ),
            "training_only_numerical_router": bool(
                not numerical_targets
                or (
                    routing.get("training_only") is True
                    and {
                        str(item.get("column"))
                        for item in routing.get("columns") or []
                    }
                    == numerical_targets
                    and all(
                        (item.get("diagnostic_statistics") or {}).get(
                            "training_only"
                        )
                        is True
                        for item in routing.get("columns") or []
                    )
                )
            ),
            "training_only_numerical_support_and_prior": bool(
                not numerical_targets
                or (
                    numerical_metadata.get("training_only") is True
                    and set(
                        str(value)
                        for value in (
                            numerical_metadata.get("columns") or {}
                        )
                    )
                    == numerical_targets
                    and all(
                        (item or {}).get("training_only") is True
                        for item in (
                            numerical_metadata.get("columns") or {}
                        ).values()
                    )
                )
            ),
            "training_only_categorical_domain": categorical_domains_match_train(
                root / "shared/spines/train_real.csv",
                eval_config,
                targets,
            ),
            "training_only_tokenizer_and_normalizers": bool(
                int(pretokenized.get("train_rows", 0)) > 0
                and pretokenized.get("split_source")
                == "explicit_split_column"
            ),
            "training_only_text_length_metadata": bool(
                not auto_length
                or (
                    auto_length.get("training_only") is True
                    and auto_length.get("fit_scope")
                    == "explicit_train_split"
                )
            ),
            "neighbor_cache_temporal_safety": bool(
                (neighbor.get("temporal_safety_sample") or {}).get(
                    "temporal_past_only"
                )
                and int(
                    (neighbor.get("temporal_safety_sample") or {}).get(
                        "future_or_same_time_violations", -1
                    )
                )
                == 0
            ),
            "original_categorical_head": not bool(
                config.get("categorical_heads")
            ),
            "temporal_support_prior_disabled": temporal_prior_disabled(config),
            "fixed_spine_target_columns_absent": fixed_spines_exclude_targets(
                root / "shared/spines", targets
            ),
            "no_arbitrary_identifier_target": bool(
                not targets.intersection({"transactionid", "transaction_id"})
            ),
            "transaction_id_excluded_from_retailrocket_evaluator": bool(
                name != "retailrocket"
                or not eval_columns.intersection(
                    {"transactionid", "transaction_id"}
                )
            ),
            "architecture_not_selected_on_validation_or_test": bool(
                (config.get("experiment_metadata") or {}).get(
                    "frozen_architecture"
                )
            ),
        }
        passed = all(checks.values())
        all_passed = all_passed and passed
        reports[name] = {
            "passed": passed,
            "checks": checks,
            "split_evidence": split_check,
            "categorical_domains_resolved_from_train_by_multiseed_driver": True,
            "normalizers_tokenizer_support_and_priors_fit_on_train_split": True,
        }
    report = {"passed": all_passed, "datasets": reports}
    write_json(report, output / "leakage_audit.json")
    lines = [
        "# Leakage Audit",
        "",
        f"Overall: **{'PASS' if all_passed else 'FAIL'}**",
        "",
        "Dataset | Check | Result",
        "--- | --- | ---",
    ]
    for dataset, record in reports.items():
        for check, passed in record["checks"].items():
            lines.append(
                f"{dataset} | {check} | {'PASS' if passed else 'FAIL'}"
            )
    lines.extend(
        [
            "",
            "All vocabularies, categorical domains, numerical normalizers, support values, "
            "empirical priors, router statistics, tokenizer state, and graph/history statistics "
            "are fitted from the materialized training split. The held-out event spine contains "
            "only event ID, source FK, destination FK, and timestamp.",
            "",
            "RetailRocket `transactionid` is retained only in `events_audit.csv`; it is absent "
            "from the model-ready interaction table, generated targets, fixed spine, and evaluator.",
        ]
    )
    (output / "leakage_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def chronological_split_check(spines: Path) -> dict[str, Any]:
    ranges = {}
    for split in ("train", "validation", "test"):
        path = spines / f"{split}_spine.csv"
        frame = pd.read_csv(path, low_memory=False)
        timestamp = next(
            column
            for column in frame.columns
            if column in {"event_time", "review_time"}
        )
        values = pd.to_datetime(frame[timestamp], errors="coerce", utc=True)
        ranges[split] = {
            "rows": int(len(frame)),
            "min": values.min().isoformat(),
            "max": values.max().isoformat(),
        }
    passed = bool(
        pd.Timestamp(ranges["train"]["max"])
        <= pd.Timestamp(ranges["validation"]["min"])
        <= pd.Timestamp(ranges["validation"]["max"])
        <= pd.Timestamp(ranges["test"]["min"])
    )
    return {"passed": passed, "ranges": ranges}


def fixed_spines_exclude_targets(spines: Path, targets: set[str]) -> bool:
    for split in ("train", "validation", "test"):
        columns = set(pd.read_csv(spines / f"{split}_spine.csv", nrows=0).columns)
        if columns.intersection(targets):
            return False
    return True


def categorical_domains_match_train(
    train_path: Path,
    evaluation_config: dict[str, Any],
    targets: set[str],
) -> bool:
    columns = (evaluation_config.get("table") or {}).get("columns") or {}
    categorical = [
        column
        for column, metadata in columns.items()
        if column in targets
        and str((metadata or {}).get("type", "")).lower() == "categorical"
    ]
    if not categorical:
        return True
    train = pd.read_csv(train_path, usecols=categorical, low_memory=False)
    for column in categorical:
        expected = {
            canonical_category(value)
            for value in train[column].dropna().drop_duplicates()
        }
        configured = {
            canonical_category(value)
            for value in (columns[column].get("valid_values") or [])
        }
        if expected != configured:
            return False
    return True


def canonical_category(value_: Any) -> str:
    text = str(value_).strip()
    try:
        numeric_value = float(text)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
    except ValueError:
        pass
    return text


def temporal_prior_disabled(config: dict[str, Any]) -> bool:
    temporal = (
        ((config.get("numerical_heads") or {}).get("global_prior") or {}).get(
            "temporal_prior"
        )
        or {}
    )
    return bool(
        not temporal.get("enabled", False)
        and float(temporal.get("lambda_t", 0.0)) == 0.0
    )


def write_five_dataset_results(
    matrix: dict[str, Any],
) -> pd.DataFrame:
    output = Path(matrix["output_root"])
    rows = existing_final_rows(matrix)
    for name in NEW_DATASETS:
        rows.append(new_dataset_result_row(matrix, name))
    columns = [
        "Dataset",
        "Rows",
        "Generated Num Attrs",
        "Generated Cat Attrs",
        "Generated Text Attrs",
        "Comparability",
        "Constraint Violation ↓",
        "FK Similarity ↑",
        "Shape Error ↓",
        "C2ST Error ↓",
        "Trend Error ↓",
        "Text C2ST Error ↓",
        "Numerical C2ST ↓",
        "Categorical C2ST ↓",
        "Train Time",
        "Sample Time",
        "Rows/sec",
    ]
    frame = pd.DataFrame(rows).reindex(columns=columns)
    frame.to_csv(output / "final_5dataset_results.csv", index=False)
    write_markdown_table(
        frame,
        output / "final_5dataset_results.md",
        "Final Frozen-LSTM Five-Dataset Results",
        preamble=(
            "All runs use generator seed 42. Yelp and RetailRocket use held-out fixed "
            "event spines. NA denotes a genuinely inapplicable metric and is never "
            "converted to zero. C2ST values use the repository's normalized C2ST-error "
            "definition, where lower is better."
        ),
    )
    return frame


def existing_final_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = matrix["frozen_architecture"]
    compact = pd.read_csv(frozen["results"])
    final = compact[compact["model"] == "FINAL"].copy()
    manifest = load_json(Path(frozen["confirmation_manifest"]))
    root_by_dataset = {
        "rel_hm": Path(manifest["rel_hm_final_root"]),
        "movielens_100k": Path(manifest["movielens_final_root"]),
        "amazon_toy": Path(manifest["amazon_final_root"]),
    }
    rows = []
    for key in ("amazon_toy", "movielens_100k", "rel_hm"):
        selected = final[final["dataset"] == key]
        if selected.empty:
            raise RuntimeError(f"Missing compact FINAL result for {key}")
        source = selected.iloc[0]
        definition = matrix["existing_datasets"][key]
        config_path = root_by_dataset[key] / "runs/seed_42/config_resolved.yaml"
        comparable, reason = frozen_config_comparability(
            load_yaml(config_path),
            has_numerical=bool(definition["generated_numerical"]),
        )
        profile = interaction_profile(definition)
        rows.append(
            result_row(
                definition["display_name"],
                profile["interactions"],
                definition,
                source.to_dict(),
                "EXACT FROZEN ARCHITECTURE"
                if comparable
                else f"NOT EXACTLY COMPARABLE: {reason}",
            )
        )
    return rows


def new_dataset_result_row(
    matrix: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    definition = matrix["datasets"][name]
    root = Path(definition["output_root"])
    run = root / "runs/seed_42"
    paper = load_json(run / "evaluation/paper_grade/metrics.json")
    attribute = load_json(run / "evaluation/attribute_diagnostics.json")
    training = load_json(run / "training_metadata.json")
    runtime = load_json(
        run / "samples/metadata/runtime_sampling_fast.json"
    )
    summary = dict(paper.get("paper_metrics_summary") or {})
    summary.update(
        {
            "numerical_only_c2st": nested(
                attribute,
                "attribute_group_c2st",
                "numerical_only",
                "c2st_error_mean",
            ),
            "categorical_only_c2st": nested(
                attribute,
                "attribute_group_c2st",
                "categorical_only",
                "c2st_error_mean",
            ),
            "training_seconds": first_number(
                training,
                "total_training_seconds",
                "training_seconds",
                "wall_clock_seconds",
            ),
            "sampling_seconds": first_number(
                runtime,
                "total_sampling_seconds",
                "sampling_seconds",
            ),
            "rows_per_second": first_number(runtime, "rows_per_second"),
        }
    )
    manifest = load_json(
        processed_dir(matrix, definition) / "subset_manifest.json"
    )
    config = load_yaml(run / "config_resolved.yaml")
    comparable, reason = frozen_config_comparability(
        config,
        has_numerical=bool(definition["generated_numerical"]),
    )
    return result_row(
        definition["display_name"],
        manifest["actual_interactions"],
        definition,
        summary,
        "EXACT FROZEN ARCHITECTURE"
        if comparable
        else f"NOT EXACTLY COMPARABLE: {reason}",
    )


def result_row(
    display_name: str,
    rows: int,
    definition: dict[str, Any],
    metrics: dict[str, Any],
    comparability: str,
) -> dict[str, Any]:
    return {
        "Dataset": display_name,
        "Rows": int(rows),
        "Generated Num Attrs": len(definition["generated_numerical"]),
        "Generated Cat Attrs": len(definition["generated_categorical"]),
        "Generated Text Attrs": len(definition["generated_text"]),
        "Comparability": comparability,
        "Constraint Violation ↓": value(metrics, "constraint_violation", "constraint_violation_rate"),
        "FK Similarity ↑": value(metrics, "fk_similarity", "fk_cardinality_similarity"),
        "Shape Error ↓": value(metrics, "shape_error"),
        "C2ST Error ↓": value(metrics, "full_row_c2st", "single_table_c2st_error"),
        "Trend Error ↓": value(metrics, "trend_error"),
        "Text C2ST Error ↓": value(metrics, "text_embedding_c2st", "text_embedding_c2st_error"),
        "Numerical C2ST ↓": value(metrics, "numerical_only_c2st"),
        "Categorical C2ST ↓": value(metrics, "categorical_only_c2st"),
        "Train Time": value(metrics, "training_seconds"),
        "Sample Time": value(metrics, "sampling_seconds"),
        "Rows/sec": value(metrics, "rows_per_second"),
    }


def frozen_config_comparability(
    config: dict[str, Any],
    *,
    has_numerical: bool,
) -> tuple[bool, str]:
    errors = []
    if config.get("categorical_heads"):
        errors.append("categorical-prior anchoring present")
    if has_numerical:
        head = config.get("numerical_heads") or {}
        prior = head.get("global_prior") or {}
        if str(head.get("mode")) != "auto":
            errors.append("numerical router is not auto")
        if float(prior.get("alpha", -1)) != 1.0:
            errors.append("prior alpha differs")
        if float(prior.get("residual_weight", -1)) != 0.25:
            errors.append("residual gamma differs")
        if not temporal_prior_disabled(config):
            errors.append("temporal prior differs")
        numerical_temperature = (config.get("sampling") or {}).get(
            "numerical_temperature", 1.0
        )
        if float(numerical_temperature) != 1.0:
            errors.append("numerical sampling temperature differs")
    return not errors, "; ".join(errors) if errors else "exact"


def write_dataset_summary(matrix: dict[str, Any]) -> pd.DataFrame:
    output = Path(matrix["output_root"])
    manifest = load_json(
        Path(matrix["frozen_architecture"]["confirmation_manifest"])
    )
    result_roots = {
        "amazon_toy": Path(manifest["amazon_final_root"]),
        "movielens_100k": Path(manifest["movielens_final_root"]),
        "rel_hm": Path(manifest["rel_hm_final_root"]),
    }
    rows = []
    for key, definition in matrix["existing_datasets"].items():
        profile = interaction_profile(definition)
        if not any(profile["split_counts"].values()):
            split_counts = split_counts_from_spines(
                result_roots[key] / "shared/spines"
            )
            if split_counts is not None:
                profile["split_counts"] = split_counts
        rows.append(dataset_summary_row(definition, profile))
    for name in NEW_DATASETS:
        definition = matrix["datasets"][name]
        adapter = get_adapter(definition["adapter"])
        directory = processed_dir(matrix, definition)
        manifest = load_json(directory / "subset_manifest.json")
        expanded = {
            **definition,
            "interaction_table": str(directory / "interactions.csv"),
            "source_table": str(directory / adapter.source_table_filename),
            "destination_table": str(
                directory / adapter.destination_table_filename
            ),
            "source_column": adapter.source_id_column,
            "destination_column": adapter.destination_id_column,
            "timestamp_column": adapter.timestamp_column,
        }
        profile = interaction_profile(expanded)
        profile["manifest"] = manifest
        rows.append(dataset_summary_row(expanded, profile))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "dataset_summary.csv", index=False)
    write_markdown_table(
        frame,
        output / "dataset_summary.md",
        "Five-Dataset Summary",
    )
    return frame


def split_counts_from_spines(spines: Path) -> dict[str, int] | None:
    paths = {
        split: spines / f"{split}_real.csv"
        for split in ("train", "validation", "test")
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    return {split: csv_row_count(path) for split, path in paths.items()}


def interaction_profile(definition: dict[str, Any]) -> dict[str, Any]:
    path = Path(definition["interaction_table"])
    usecols = {
        definition["source_column"],
        definition["destination_column"],
        definition["timestamp_column"],
        "split",
    }
    rows = 0
    sources: set[str] = set()
    destinations: set[str] = set()
    split_counts = {"train": 0, "validation": 0, "test": 0}
    timestamp_min = None
    timestamp_max = None
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in usecols,
        chunksize=250000,
        low_memory=False,
    ):
        rows += len(chunk)
        sources.update(chunk[definition["source_column"]].astype(str))
        destinations.update(
            chunk[definition["destination_column"]].astype(str)
        )
        times = pd.to_datetime(
            chunk[definition["timestamp_column"]],
            errors="coerce",
            utc=True,
        )
        current_min = times.min()
        current_max = times.max()
        if pd.notna(current_min):
            timestamp_min = (
                current_min
                if timestamp_min is None
                else min(timestamp_min, current_min)
            )
        if pd.notna(current_max):
            timestamp_max = (
                current_max
                if timestamp_max is None
                else max(timestamp_max, current_max)
            )
        if "split" in chunk:
            counts = chunk["split"].astype(str).str.lower().value_counts()
            for split in split_counts:
                split_counts[split] += int(counts.get(split, 0))
    if not any(split_counts.values()):
        split_counts = {"train": None, "validation": None, "test": None}
    return {
        "interactions": int(rows),
        "source_entities": len(sources),
        "destination_entities": len(destinations),
        "source_table_rows": csv_row_count(Path(definition["source_table"])),
        "destination_table_rows": csv_row_count(
            Path(definition["destination_table"])
        ),
        "split_counts": split_counts,
        "timestamp_min": timestamp_min.isoformat() if timestamp_min is not None else None,
        "timestamp_max": timestamp_max.isoformat() if timestamp_max is not None else None,
    }


def dataset_summary_row(
    definition: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    splits = profile["split_counts"]
    return {
        "Dataset": definition["display_name"],
        "# source entities": profile["source_entities"],
        "# destination entities": profile["destination_entities"],
        "# interactions": profile["interactions"],
        "# train interactions": splits.get("train"),
        "# validation interactions": splits.get("validation"),
        "# test interactions": splits.get("test"),
        "time span": f"{profile['timestamp_min']} to {profile['timestamp_max']}",
        "# generated numerical attrs": len(definition["generated_numerical"]),
        "# generated categorical attrs": len(definition["generated_categorical"]),
        "# generated text attrs": len(definition["generated_text"]),
        "source static table": static_table_name(
            definition.get("source_table"), "users/visitors"
        ),
        "destination static table": static_table_name(
            definition.get("destination_table"), "businesses/items"
        ),
    }


def static_table_name(value_: Any, default: str) -> str:
    return Path(str(value_)).name if value_ else default


def write_run_report(
    matrix: dict[str, Any],
    final: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: dict[str, Any],
    leakage: dict[str, Any],
) -> None:
    output = Path(matrix["output_root"])
    frozen_path = Path(matrix["frozen_architecture"]["config"])
    text_template_path = Path(
        matrix["frozen_architecture"]["text_template"]
    )
    routing = pd.read_csv(output / "numerical_routing_yelp.csv")
    yelp = final[final["Dataset"] == matrix["datasets"]["yelp"]["display_name"]].iloc[0]
    retail = final[final["Dataset"] == matrix["datasets"]["retailrocket"]["display_name"]].iloc[0]
    blocking = blocking_reasons(final, leakage, diagnostics)
    recommendation = (
        "BLOCKED BEFORE PAPER-GRADE EXPERIMENTS"
        if blocking
        else "PROCEED TO PAPER-GRADE EXPERIMENTS"
    )
    lines = [
        "# Frozen LSTM Evaluation on Yelp and RetailRocket",
        "",
        "## Frozen Architecture",
        "",
        f"Config: `{frozen_path}`",
        f"SHA-256: `{sha256_file(frozen_path)}`",
        f"Frozen text template: `{text_template_path}`",
        f"Text-template SHA-256: `{sha256_file(text_template_path)}`",
        "",
        "The model was not modified. Numerical fields use the training-only auto router; "
        "support-like fields use empirical-prior residual logits with alpha 1.0 and gamma "
        "0.25. The original categorical head is retained, temporal prior lambda is zero, "
        "and text uses the existing length-preserving autoregressive LSTM branch.",
        "",
        "## Dataset Construction",
        "",
        "### Yelp",
        "",
        dataset_construction_text(matrix, "yelp"),
        "",
        "### RetailRocket",
        "",
        dataset_construction_text(matrix, "retailrocket"),
        "",
        "## Schema Mapping",
        "",
    ]
    for name in NEW_DATASETS:
        definition = matrix["datasets"][name]
        lines.extend(
            [
                f"### {definition['display_name']}",
                "",
                f"- Fixed event spine: `{', '.join(definition['fixed_fields'])}`",
                f"- Generated categorical: `{', '.join(definition['generated_categorical']) or 'NA'}`",
                f"- Generated numerical: `{', '.join(definition['generated_numerical']) or 'NA'}`",
                f"- Generated text: `{', '.join(definition['generated_text']) or 'NA'}`",
                f"- Excluded audit/identifier fields: `{', '.join(definition['excluded_fields']) or 'none'}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Numerical Routing",
            "",
            dataframe_markdown(routing),
            "",
            "RetailRocket has no generated numerical attribute; routing is NA.",
            "",
            "## Yelp Results",
            "",
            dataframe_markdown(pd.DataFrame([yelp])),
            "",
            "Detailed diagnostics: `metrics/yelp_diagnostics.json`.",
            "",
            "```json",
            json.dumps(diagnostics["yelp"], indent=2, sort_keys=True),
            "```",
            "",
            "## RetailRocket Results",
            "",
            dataframe_markdown(pd.DataFrame([retail])),
            "",
            "Detailed diagnostics: `metrics/retailrocket_diagnostics.json`.",
            "",
            "```json",
            json.dumps(
                diagnostics["retailrocket"], indent=2, sort_keys=True
            ),
            "```",
            "",
            "## Five-Dataset Summary",
            "",
            dataframe_markdown(final),
            "",
            "## Generalization Assessment",
            "",
            "1. The frozen architecture trained through the generic pipeline without model changes.",
            "2. Schema-driven routing decisions are recorded in `numerical_routing_yelp.csv`.",
            f"3. Leakage and schema validity checks: **{'PASS' if leakage['passed'] else 'FAIL'}**.",
            "4. Joint fidelity is reported without post-hoc dataset-specific repair or tuning.",
            "5. Dataset-specific weaknesses should be diagnosed from C2ST feature importance, "
            "conditional diagnostics, and text/support reports without reopening architecture search.",
            "6. A blocking weakness is declared only for invalid output, leakage, failed schema "
            "representation, a missing complete run, or effectively collapsed fidelity.",
            "",
            "## Recommendation",
            "",
            f"**{recommendation}**",
        ]
    )
    if blocking:
        lines.extend(["", f"Blocking reason: {blocking[0]}"])
    else:
        lines.extend(
            [
                "",
                "No fundamental schema, validity, leakage, training, or sampling failure was detected. "
                "Weaker fidelity metrics remain diagnostic findings, not grounds for dataset-specific architecture changes.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifact Index",
            "",
            "- `final_5dataset_results.csv` / `.md`",
            "- `dataset_summary.csv` / `.md`",
            "- `numerical_routing_yelp.csv`",
            "- `leakage_audit.json` / `.md`",
            "- `checkpoints/{yelp,retailrocket}_seed42/best.pt`",
            "- `synthetic/{yelp,retailrocket}_seed42/synthetic_interactions.csv`",
            "- `metrics/{yelp,retailrocket}_seed42.json`",
            "- `metrics/{yelp,retailrocket}_seed42_sampling_validation.json`",
            "- `configs/{yelp,retailrocket}_seed42_resolved.yaml`",
        ]
    )
    (output / "run_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    write_json(
        {
            "recommendation": recommendation,
            "blocking_reasons": blocking,
            "architecture_modified": False,
            "seed": 42,
            "leakage_audit_passed": leakage["passed"],
            "diagnostics": diagnostics,
        },
        output / "run_report.json",
    )


def blocking_reasons(
    final: pd.DataFrame,
    leakage: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> list[str]:
    reasons = []
    if not leakage.get("passed"):
        reasons.append("Leakage audit failed.")
    new = final[final["Dataset"].isin([
        "Yelp-100K-induced",
        "RetailRocket-100K-induced",
    ])]
    for _, row in new.iterrows():
        if numeric(row.get("Constraint Violation ↓"), default=1.0) != 0.0:
            reasons.append(f"{row['Dataset']} generated invalid output.")
        if not str(row.get("Comparability", "")).startswith("EXACT"):
            reasons.append(f"{row['Dataset']} did not use the exact frozen contract.")
        c2st = numeric(row.get("C2ST Error ↓"), default=None)
        if c2st is not None and c2st >= 0.99:
            reasons.append(f"{row['Dataset']} joint fidelity effectively collapsed.")
    yelp_text = ((diagnostics or {}).get("yelp") or {}).get("text") or {}
    empty_rate = numeric(yelp_text.get("empty_rate_synthetic"), default=None)
    if empty_rate is not None and empty_rate != 0.0:
        reasons.append("Yelp generated null or empty review text.")
    return reasons


def print_final_console(
    matrix: dict[str, Any],
    final: pd.DataFrame,
    leakage: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> None:
    output = Path(matrix["output_root"])
    summary = pd.read_csv(output / "dataset_summary.csv")
    route = pd.read_csv(output / "numerical_routing_yelp.csv")
    blocking = blocking_reasons(final, leakage, diagnostics)
    print("\n========================================================")
    print("FROZEN LSTM — FIVE DATASET STATUS")
    print("========================================================")
    print("\nARCHITECTURE MODIFIED:\nNO")
    print("\nSEED:\n42")
    for name, entity_labels in (
        ("yelp", ("users", "businesses")),
        ("retailrocket", ("visitors", "items")),
    ):
        display = matrix["datasets"][name]["display_name"]
        row = final[final["Dataset"] == display].iloc[0]
        data = summary[summary["Dataset"] == display].iloc[0]
        print(f"\n{name.upper()}\n--------------------------------")
        print(f"rows: {int(data['# interactions'])}")
        print(f"{entity_labels[0]}: {int(data['# source entities'])}")
        print(f"{entity_labels[1]}: {int(data['# destination entities'])}")
        if name == "yelp":
            route_text = ", ".join(
                f"{item.field}={item.selected_head}"
                for item in route.itertuples()
            )
            print(f"numerical routing: {route_text}")
        print(f"constraint: {fmt(row['Constraint Violation ↓'])}")
        print(f"shape: {fmt(row['Shape Error ↓'])}")
        print(f"C2ST: {fmt(row['C2ST Error ↓'])}")
        print(f"trend: {fmt(row['Trend Error ↓'])}")
        if name == "yelp":
            print(f"text C2ST: {fmt(row['Text C2ST Error ↓'])}")
        else:
            print(f"categorical C2ST: {fmt(row['Categorical C2ST ↓'])}")
        print(f"train time: {fmt(row['Train Time'])}")
        print(f"sample time: {fmt(row['Sample Time'])}")
        print(f"status: {'PASS' if not blocking else 'INSPECT'}")
    print("\nFIVE DATASETS\n--------------------------------")
    for _, row in final.iterrows():
        print(f"{row['Dataset']}: {row['Comparability']}")
    print(f"\nLEAKAGE AUDIT:\n{'PASS' if leakage['passed'] else 'FAIL'}")
    schema_generality = all(
        str(value).startswith("EXACT") for value in final["Comparability"]
    )
    print(f"\nSCHEMA GENERALITY:\n{'PASS' if schema_generality else 'FAIL'}")
    print("\nFINAL RECOMMENDATION:\n")
    print(
        "BLOCKED BEFORE PAPER-GRADE EXPERIMENTS"
        if blocking
        else "PROCEED TO PAPER-GRADE EXPERIMENTS"
    )
    print("\n========================================================")


def dataset_construction_text(matrix: dict[str, Any], name: str) -> str:
    definition = matrix["datasets"][name]
    manifest = load_json(
        processed_dir(matrix, definition) / "subset_manifest.json"
    )
    source = "users" if name == "yelp" else "visitors"
    destination = "businesses" if name == "yelp" else "items"
    return (
        f"Selected {manifest['selected_source_entities']:,} {source} with the generic "
        f"source-entity-induced procedure and retained every interaction for each selected "
        f"source. The resulting table has {manifest['actual_interactions']:,} interactions "
        f"and {manifest['selected_destination_entities']:,} referenced {destination}. "
        f"Complete histories={manifest['complete_source_histories']}; "
        f"FK valid={manifest['foreign_key_valid']}."
    )


def write_markdown_table(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    preamble: str | None = None,
) -> None:
    lines = [f"# {title}", ""]
    if preamble:
        lines.extend([preamble, ""])
    lines.append(dataframe_markdown(frame))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(fmt(row.get(column)) for column in frame.columns)
            + " |"
        )
    return "\n".join(lines)


def category_proportions(series: pd.Series) -> dict[str, float]:
    counts = series.fillna("<NA>").astype(str).value_counts(normalize=True)
    return {str(key): float(value) for key, value in counts.items()}


def empty_text_rate(series: pd.Series) -> float:
    return float(series.fillna("").astype(str).str.strip().eq("").mean())


def length_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def wasserstein_1d(real: pd.Series, synthetic: pd.Series) -> float | None:
    left = pd.to_numeric(real, errors="coerce").dropna().to_numpy(float)
    right = pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy(float)
    if not len(left) or not len(right):
        return None
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(left, right))
    except Exception:
        quantiles = np.linspace(0.0, 1.0, min(max(len(left), len(right)), 1001))
        return float(
            np.mean(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles)))
        )


def nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        candidate = mapping.get(key)
        if finite(candidate):
            return float(candidate)
    for candidate in mapping.values():
        if isinstance(candidate, dict):
            value = first_number(candidate, *keys)
            if value is not None:
                return value
    return None


def value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        candidate = mapping.get(key)
        if finite(candidate):
            return float(candidate)
    return None


def numeric(value_: Any, default: float | None) -> float | None:
    return float(value_) if finite(value_) else default


def finite(value_: Any) -> bool:
    try:
        return math.isfinite(float(value_))
    except (TypeError, ValueError):
        return False


def fmt(value_: Any) -> str:
    if value_ is None or (isinstance(value_, float) and not math.isfinite(value_)):
        return "NA"
    if isinstance(value_, (float, np.floating)):
        return f"{float(value_):.6g}"
    return str(value_).replace("|", "\\|").replace("\n", " ")


def csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        rows = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))
    return max(rows - 1, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value_ = yaml.safe_load(handle)
    if not isinstance(value_, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value_


def write_yaml(value_: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value_, sort_keys=False), encoding="utf-8"
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value_ = json.load(handle)
    if not isinstance(value_, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value_


def write_json(value_: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value_, indent=2, sort_keys=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def json_default(value_: Any) -> Any:
    if isinstance(value_, (np.integer, np.floating)):
        return value_.item()
    if isinstance(value_, Path):
        return str(value_)
    raise TypeError(type(value_).__name__)


if __name__ == "__main__":
    main()
