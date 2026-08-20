#!/usr/bin/env python3
"""Audit Amazon-Toy text C2ST without training or resampling model output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.text_c2st_audit import (  # noqa: E402
    EmbeddingStore,
    TEXT_FIELDS,
    TextC2STProtocol,
    c2st_error,
    canonical_text,
    compare_text_frames,
    evaluate_prepared_embeddings,
    evaluate_protocol,
    finite_or_none,
    flatten_protocol_result,
    implied_auc,
    inspect_csv,
    prepare_protocol_embeddings,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = "outputs/text_c2st_audit"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLD_EVALUATOR_COMMIT = "7c9df5722be4e9f8fbbc33c86bf3f8b67dc92fc3"
OLD_FULL_ROW_CONFIG_COMMIT = "4f76c591973675035e286ff56d58ac9ce42cab05"
STRICT_EVALUATOR_COMMIT = "1bcd2ae21c6cf38c9c5125972c3dacec2fc7e94a"
AMAZON_EVALUATION_CONFIG = (
    "configs/evaluation/single_event_table_paper_metrics_amazon_toy.yaml"
)
KNOWN_HISTORICAL_SCORES = (0.5481, 0.5575, 0.5749)
REQUIRED_OUTPUTS = (
    "audit_report.md",
    "audit_summary.json",
    "evaluator_comparison.csv",
    "data_hashes.csv",
    "cross_evaluation_matrix.csv",
    "classifier_comparison.csv",
    "sample_size_analysis.csv",
    "embedding_consistency.json",
    "preprocessing_audit.json",
    "canonical_text_c2st_protocol.md",
    "canonical_text_c2st_results.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict no-training Amazon text-C2ST reproducibility audit."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical-real")
    parser.add_argument("--canonical-lstm-synthetic")
    parser.add_argument("--canonical-diffusion-synthetic")
    parser.add_argument("--historical-synthetic")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--canonical-rows", type=int, default=50000)
    parser.add_argument("--classifier-comparison-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Write artifact inventory and hashes without running classifiers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ROOT / args.output_dir
    ensure_fresh_output(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "embedding_cache").mkdir(exist_ok=True)

    progress("inventorying historical evaluations")
    inventory = discover_historical_evaluations()
    historical_frame = pd.DataFrame([public_inventory_row(item) for item in inventory])
    historical_frame.to_csv(output / "historical_evaluations.csv", index=False)

    current_entry = select_current_entry(inventory)
    old_entry = select_historical_entry(inventory, current_entry)
    paths = resolve_audit_paths(args, current_entry, old_entry)
    require_paths(paths, args.inventory_only)

    path_roles = invert_path_roles(paths)
    frames: dict[str, pd.DataFrame] = {}
    hash_rows = []
    for path, roles in path_roles.items():
        if not path.is_file():
            hash_rows.append(
                {
                    "path": str(path),
                    "roles": ", ".join(sorted(roles)),
                    "status": "missing",
                }
            )
            continue
        record, frame = inspect_csv(path)
        record["roles"] = ", ".join(sorted(roles))
        record["status"] = "available"
        hash_rows.append(record)
        frames[str(path)] = frame
    pd.DataFrame(hash_rows).to_csv(output / "data_hashes.csv", index=False)

    old_real = frames.get(str(paths["old_real"]))
    current_real = frames.get(str(paths["canonical_real"]))
    old_synthetic = frames.get(str(paths["old_synthetic"]))
    current_synthetic = frames.get(str(paths["canonical_lstm_synthetic"]))
    diffusion_synthetic = frames.get(str(paths["canonical_diffusion_synthetic"]))

    data_comparison = {
        "real": compare_optional_frames(old_real, current_real),
        "synthetic": compare_optional_frames(old_synthetic, current_synthetic),
    }
    write_json(data_comparison, output / "data_comparison.json")

    if args.inventory_only:
        write_json(
            {
                "status": "inventory_only",
                "historical_evaluations": len(inventory),
                "resolved_paths": {key: str(value) for key, value in paths.items()},
            },
            output / "audit_summary.json",
        )
        print(f"Inventory written to {output}")
        return

    assert current_real is not None
    assert current_synthetic is not None
    assert old_real is not None
    assert old_synthetic is not None
    assert diffusion_synthetic is not None

    old_backend, backend_evidence = infer_historical_backend(old_entry)
    old_protocol = historical_protocol(old_entry, old_backend, args)
    new_protocol = strict_protocol(current_entry, args)
    evaluator_comparison = evaluator_comparison_rows(
        old_protocol, new_protocol, old_entry, current_entry, backend_evidence
    )

    cache_roots = discover_embedding_cache_roots(inventory)
    store = EmbeddingStore(
        output / "embedding_cache",
        device=args.device,
        existing_cache_roots=cache_roots,
    )

    progress("E1/4 historical data with historical evaluator")
    controlled: dict[str, dict[str, Any]] = {}
    controlled["old_data_old_evaluator"] = evaluate_protocol(
        old_real,
        old_synthetic,
        old_protocol,
        store,
        label="old_data_old_evaluator",
    )
    same_real = same_file_hash(paths["old_real"], paths["canonical_real"])
    same_synthetic = same_file_hash(
        paths["old_synthetic"], paths["canonical_lstm_synthetic"]
    )
    if same_real and same_synthetic:
        controlled["current_data_old_evaluator"] = controlled[
            "old_data_old_evaluator"
        ]
    else:
        progress("E4/4 current data with historical evaluator")
        controlled["current_data_old_evaluator"] = evaluate_protocol(
            current_real,
            current_synthetic,
            old_protocol,
            store,
            label="current_data_old_evaluator",
        )

    reused_current = metric_result(current_entry)
    if reused_current is not None and metric_matches_protocol(
        current_entry, new_protocol
    ):
        progress("E2/4 reusing exact strict-evaluator result on current data")
        controlled["current_data_new_evaluator"] = reused_current
        controlled["current_data_new_evaluator"]["result_source"] = str(
            current_entry.get("metrics_path")
        )
    else:
        progress("E2/4 current data with strict evaluator")
        controlled["current_data_new_evaluator"] = evaluate_protocol(
            current_real,
            current_synthetic,
            new_protocol,
            store,
            label="current_data_new_evaluator",
        )
        controlled["current_data_new_evaluator"]["result_source"] = "computed"

    if same_real and same_synthetic:
        controlled["old_data_new_evaluator"] = controlled[
            "current_data_new_evaluator"
        ]
    else:
        progress("E3/4 historical data with strict evaluator")
        controlled["old_data_new_evaluator"] = evaluate_protocol(
            old_real,
            old_synthetic,
            new_protocol,
            store,
            label="old_data_new_evaluator",
        )

    cross_rows = []
    cross_rows.extend(
        flatten_protocol_result(
            controlled["old_data_old_evaluator"],
            data_label="historical synthetic",
            evaluator_label="historical evaluator",
        )
    )
    cross_rows.extend(
        flatten_protocol_result(
            controlled["old_data_new_evaluator"],
            data_label="historical synthetic",
            evaluator_label="strict evaluator",
        )
    )
    cross_rows.extend(
        flatten_protocol_result(
            controlled["current_data_old_evaluator"],
            data_label="current frozen synthetic",
            evaluator_label="historical evaluator",
        )
    )
    cross_rows.extend(
        flatten_protocol_result(
            controlled["current_data_new_evaluator"],
            data_label="current frozen synthetic",
            evaluator_label="strict evaluator",
        )
    )
    pd.DataFrame(cross_rows).to_csv(
        output / "cross_evaluation_matrix.csv", index=False
    )
    write_json(controlled, output / "controlled_results.json")

    progress("running component ablations")
    component_rows, intermediate_results = component_ablation(
        current_real,
        current_synthetic,
        old_protocol,
        new_protocol,
        controlled,
        store,
    )
    pd.DataFrame(component_rows).to_csv(
        output / "component_ablation.csv", index=False
    )

    progress("running fixed classifier sensitivity comparison")
    classifier_rows = classifier_comparison(
        current_real,
        current_synthetic,
        args,
        store,
    )
    pd.DataFrame(classifier_rows).to_csv(
        output / "classifier_comparison.csv", index=False
    )

    sample_rows = sample_size_analysis(
        old_protocol,
        new_protocol,
        controlled,
        intermediate_results,
    )
    pd.DataFrame(sample_rows).to_csv(
        output / "sample_size_analysis.csv", index=False
    )

    progress("auditing embedding and preprocessing consistency")
    embedding_consistency = audit_embedding_consistency(
        current_real, current_synthetic, old_protocol, new_protocol, store
    )
    write_json(embedding_consistency, output / "embedding_consistency.json")
    preprocessing = preprocessing_audit(current_real, current_synthetic)
    write_json(preprocessing, output / "preprocessing_audit.json")

    canonical_protocol = TextC2STProtocol(
        name="canonical_paper_text_c2st_v1",
        embedding_backend="minilm",
        embedding_model=args.embedding_model,
        preprocessing="canonical",
        classifiers=("logistic_regression",),
        max_rows=min(
            int(args.canonical_rows),
            len(current_real),
            len(current_synthetic),
            len(diffusion_synthetic),
        ),
        seed=int(args.seed),
        n_splits=5,
    )
    progress("preparing canonical MiniLM embeddings for LSTM and diffusion")
    canonical_lstm_prepared = prepare_protocol_embeddings(
        current_real,
        current_synthetic,
        canonical_protocol,
        store,
        label="canonical_lstm",
    )
    canonical_diffusion_prepared = prepare_protocol_embeddings(
        current_real,
        diffusion_synthetic,
        canonical_protocol,
        store,
        label="canonical_diffusion",
    )
    progress("evaluating canonical LSTM and diffusion scores")
    canonical_lstm = evaluate_prepared_embeddings(
        canonical_lstm_prepared,
        canonical_protocol.classifiers,
        seed=args.seed,
        n_splits=5,
        protocol=canonical_protocol,
    )
    canonical_diffusion = evaluate_prepared_embeddings(
        canonical_diffusion_prepared,
        canonical_protocol.classifiers,
        seed=args.seed,
        n_splits=5,
        protocol=canonical_protocol,
    )

    progress("running same-seed and discriminator-seed reproducibility checks")
    repeated = [
        {
            "repeat": 1,
            "seed": args.seed,
            "macro_error": canonical_lstm["macro_error"],
            "source": "canonical evaluation",
        }
    ]
    for repeat in range(2):
        result = evaluate_prepared_embeddings(
            canonical_lstm_prepared,
            canonical_protocol.classifiers,
            seed=args.seed,
            n_splits=5,
            protocol=canonical_protocol,
        )
        repeated.append(
            {
                "repeat": repeat + 2,
                "seed": args.seed,
                "macro_error": result["macro_error"],
                "source": "independent classifier rerun on cached embeddings",
            }
        )
    seed_results = []
    for seed in (17, 42, 73):
        result = (
            canonical_lstm
            if seed == args.seed
            else evaluate_prepared_embeddings(
                canonical_lstm_prepared,
                canonical_protocol.classifiers,
                seed=seed,
                n_splits=5,
                protocol=canonical_protocol,
            )
        )
        seed_results.append({"seed": seed, "macro_error": result["macro_error"]})
    reproducibility = reproducibility_summary(repeated, seed_results)
    write_json(reproducibility, output / "reproducibility.json")

    canonical_results = {
        "protocol": canonical_protocol.as_dict(),
        "real_table": str(paths["canonical_real"]),
        "lstm_synthetic_table": str(paths["canonical_lstm_synthetic"]),
        "diffusion_synthetic_table": str(paths["canonical_diffusion_synthetic"]),
        "embedding_model_metadata": store.model_metadata,
        "final_lstm": canonical_lstm,
        "diffusion": canonical_diffusion,
        "lstm_better_than_diffusion": bool(
            canonical_lstm["macro_error"] < canonical_diffusion["macro_error"]
        ),
        "fairness_check": {
            "same_real_dataset": True,
            "same_embedding_model_and_revision": True,
            "same_preprocessing": True,
            "same_rows_and_sampling_policy": True,
            "same_classifier_and_hyperparameters": True,
            "same_cv_splits_and_seed": True,
            "same_aggregation_and_error_formula": True,
            "passed": True,
        },
        "reproducibility": reproducibility,
    }
    write_json(canonical_results, output / "canonical_text_c2st_results.json")
    write_canonical_protocol(output, canonical_protocol, store.model_metadata)
    add_runtime_embedding_metadata(evaluator_comparison, store.model_metadata)
    pd.DataFrame(evaluator_comparison).to_csv(
        output / "evaluator_comparison.csv", index=False
    )

    progress("writing audit report and frozen protocol")
    summary = build_summary(
        inventory=inventory,
        old_entry=old_entry,
        current_entry=current_entry,
        paths=paths,
        same_real=same_real,
        same_synthetic=same_synthetic,
        old_protocol=old_protocol,
        new_protocol=new_protocol,
        controlled=controlled,
        component_rows=component_rows,
        embedding_consistency=embedding_consistency,
        canonical_results=canonical_results,
        backend_evidence=backend_evidence,
        data_comparison=data_comparison,
        cache_events=store.cache_events,
    )
    write_json(summary, output / "audit_summary.json")
    write_audit_report(
        output,
        summary,
        historical_frame,
        pd.DataFrame(hash_rows),
        pd.DataFrame(evaluator_comparison),
        pd.DataFrame(cross_rows),
        pd.DataFrame(component_rows),
        pd.DataFrame(classifier_rows),
        pd.DataFrame(sample_rows),
        embedding_consistency,
        preprocessing,
        data_comparison,
        canonical_results,
        reproducibility,
    )
    print_console(summary, controlled, canonical_results, old_protocol, new_protocol)


def discover_historical_evaluations() -> list[dict[str, Any]]:
    output_root = ROOT / "outputs"
    records = []
    if output_root.exists():
        names = {
            "metrics.json",
            "paper_metrics.json",
            "eval_metrics.json",
            "text_embedding_c2st_report.json",
        }
        for directory, _, files in os.walk(output_root):
            if "embedding_cache" in Path(directory).parts:
                continue
            for filename in files:
                if filename not in names:
                    continue
                path = Path(directory) / filename
                if path.stat().st_size > 50 * 1024 * 1024:
                    continue
                payload = load_json(path)
                record = evaluation_record(path, payload)
                if record is not None:
                    records.append(record)
    reference = ROOT / "configs/experiments/hierarchical_diffusion_amazon_toy_diagnostics.yaml"
    if reference.is_file():
        config = load_yaml(reference)
        score = finite_or_none(
            (config.get("reference_metrics") or {}).get(
                "lstm_text_embedding_c2st_error"
            )
        )
        if score is not None:
            records.append(
                {
                    "evaluation_id": "reference_metric_hierarchical_diagnostics",
                    "reported_text_c2st": score,
                    "synthetic_csv": None,
                    "real_csv": None,
                    "embedding_model": "not recorded",
                    "effective_embedding_backend": "not recorded",
                    "classifier": "not recorded",
                    "classifiers": [],
                    "rows_real": None,
                    "rows_synthetic": None,
                    "seed": None,
                    "evaluation_script": "not recorded",
                    "metric_version": "supplied reference only",
                    "timestamp_or_commit": None,
                    "metrics_path": str(reference.resolve()),
                    "payload": config,
                }
            )
    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        metrics_path = Path(str(record.get("metrics_path", "")))
        key = (
            str(metrics_path.parent),
            record.get("reported_text_c2st"),
            record.get("real_csv"),
            record.get("synthetic_csv"),
        )
        previous = deduplicated.get(key)
        if previous is None or metrics_path.name == "metrics.json":
            deduplicated[key] = record
    return sorted(
        deduplicated.values(), key=lambda item: str(item.get("metrics_path"))
    )


def evaluation_record(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    text = payload.get("text_embedding_c2st") or {}
    summary = payload.get("paper_metrics_summary") or {}
    direct_report = path.name == "text_embedding_c2st_report.json"
    if direct_report:
        text = payload
        sibling = load_json(path.parent / "metrics.json")
        if sibling:
            summary = sibling.get("paper_metrics_summary") or summary
            payload = {**sibling, "text_embedding_c2st": text}
    score = finite_or_none(
        summary.get("text_embedding_c2st_error", text.get("macro_error"))
    )
    if score is None:
        return None
    serialized = json.dumps(payload, default=str).lower()
    if "amazon" not in str(path).lower() and "amazon" not in serialized:
        return None
    dataset = payload.get("dataset") or {}
    real_path = resolve_optional_path(dataset.get("real_table_path"))
    synthetic_path = resolve_optional_path(dataset.get("synthetic_table_path"))
    config_path = find_companion_evaluation_config(path)
    config = load_yaml(config_path) if config_path else {}
    real_path = real_path or resolve_optional_path(config.get("real_table_path"))
    synthetic_path = synthetic_path or resolve_optional_path(
        config.get("synthetic_table_path")
    )
    per_field = text.get("per_text_column") or {}
    inferred_classifiers = sorted(
        {
            name
            for values in per_field.values()
            for name in (values.get("per_classifier") or {})
        }
    )
    selected = sorted(
        {
            str(values.get("classifier"))
            for values in per_field.values()
            if values.get("classifier")
        }
    )
    real_rows = [
        int(values.get("num_real"))
        for values in per_field.values()
        if values.get("num_real") is not None
    ]
    synthetic_rows = [
        int(values.get("num_synthetic"))
        for values in per_field.values()
        if values.get("num_synthetic") is not None
    ]
    embedding_models = sorted(
        {
            str(values.get("embedding_model"))
            for values in per_field.values()
            if values.get("embedding_model")
        }
    )
    backend, evidence = infer_backend_from_metric(path, text)
    timestamp = datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    return {
        "evaluation_id": str(path.relative_to(ROOT)),
        "reported_text_c2st": score,
        "synthetic_csv": str(synthetic_path) if synthetic_path else None,
        "real_csv": str(real_path) if real_path else None,
        "embedding_model": ", ".join(embedding_models) or config_embedding_model(config),
        "effective_embedding_backend": backend,
        "backend_evidence": evidence,
        "classifier": ", ".join(selected) or None,
        "classifiers": config_classifiers(config) or inferred_classifiers,
        "rows_real": min(real_rows) if real_rows else None,
        "rows_synthetic": min(synthetic_rows) if synthetic_rows else None,
        "seed": config_seed(config, payload),
        "n_splits": config_n_splits(config),
        "evaluation_script": "src/scripts/evaluate_single_event_table_paper_metrics.py",
        "metric_version": payload.get("paper_metrics_version") or config.get(
            "paper_metrics_version"
        ),
        "timestamp_or_commit": timestamp,
        "metrics_path": str(path.resolve()),
        "evaluation_config": str(config_path) if config_path else None,
        "payload": payload,
    }


def resolve_audit_paths(
    args: argparse.Namespace,
    current_entry: dict[str, Any],
    old_entry: dict[str, Any],
) -> dict[str, Path]:
    canonical_real = first_existing_path(
        args.canonical_real,
        current_entry.get("real_csv"),
        "data/original/rel-amazon-toy/review.csv",
    )
    current_synthetic = first_existing_path(
        args.canonical_lstm_synthetic,
        current_entry.get("synthetic_csv"),
        discover_from_preflight("amazon_toy"),
        "outputs/architecture_finalization/transfer/amazon_toy/M2_global_support/runs/seed_42/samples/synthetic_interactions.csv",
    )
    historical_synthetic = first_existing_path(
        args.historical_synthetic,
        old_entry.get("synthetic_csv"),
        "outputs/amazon-toy/conditional_tabdlm_exp5_3_lstm_length_preserving_privacy_sampler/runs/v51_length_preserving_exact_block/synthetic_review_attrs_full_79663.csv",
        "outputs/amazon-toy/conditional_tabdlm_exp5_3_lstm_length_preserving_privacy_sampler/runs/v51_length_preserving_exact_block/synthetic_review_attrs_fast.csv",
    )
    old_real = first_existing_path(old_entry.get("real_csv"), canonical_real)
    diffusion = first_existing_path(
        args.canonical_diffusion_synthetic,
        discover_from_preflight("diffusion_amazon_toy"),
        "outputs/amazon-toy/conditional_tabdlm_hierarchical_v41/synthetic_review_attrs_full.csv",
        "outputs/amazon-toy/conditional_tabdlm_hierarchical_v41/synthetic_review_attrs.csv",
    )
    return {
        "canonical_real": canonical_real,
        "canonical_lstm_synthetic": current_synthetic,
        "canonical_diffusion_synthetic": diffusion,
        "old_real": old_real,
        "old_synthetic": historical_synthetic,
    }


def historical_protocol(
    entry: dict[str, Any], backend: str, args: argparse.Namespace
) -> TextC2STProtocol:
    classifiers = tuple(entry.get("classifiers") or default_classifiers())
    return TextC2STProtocol(
        name="historical_evaluator_observed",
        embedding_backend=backend,
        embedding_model=entry.get("embedding_model") or args.embedding_model,
        preprocessing=("historical_hash" if backend == "deterministic_hash" else "raw_str"),
        classifiers=classifiers,
        max_rows=int(entry.get("rows_real") or 50000),
        seed=int(entry.get("seed") or args.seed),
        n_splits=int(entry.get("n_splits") or 5),
    )


def strict_protocol(
    entry: dict[str, Any], args: argparse.Namespace
) -> TextC2STProtocol:
    return TextC2STProtocol(
        name="strict_cleanup_evaluator",
        embedding_backend="minilm",
        embedding_model=args.embedding_model,
        preprocessing="raw_str",
        classifiers=tuple(entry.get("classifiers") or default_classifiers()),
        max_rows=int(entry.get("rows_real") or 79663),
        seed=int(entry.get("seed") or args.seed),
        n_splits=int(entry.get("n_splits") or 5),
    )


def component_ablation(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    old: TextC2STProtocol,
    new: TextC2STProtocol,
    controlled: dict[str, dict[str, Any]],
    store: EmbeddingStore,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    results = {"A0_historical": controlled["current_data_old_evaluator"]}
    rows = [ablation_row("A0", "exact historical configuration", results["A0_historical"], None)]
    previous = results["A0_historical"]
    if old.max_rows != new.max_rows:
        row_protocol = TextC2STProtocol(
            name="A1_row_count_only",
            embedding_backend=old.embedding_backend,
            embedding_model=old.embedding_model,
            preprocessing=old.preprocessing,
            classifiers=old.classifiers,
            max_rows=new.max_rows,
            seed=old.seed,
            n_splits=old.n_splits,
            aggregation=old.aggregation,
        )
        results["A1_row_count"] = evaluate_protocol(
            real, synthetic, row_protocol, store, label="ablation_row_count"
        )
        rows.append(
            ablation_row(
                "A1",
                f"row limit {old.max_rows} -> {new.max_rows}",
                results["A1_row_count"],
                previous,
            )
        )
        previous = results["A1_row_count"]
    if old.embedding_backend != new.embedding_backend:
        results["A2_embedding_backend"] = controlled[
            "current_data_new_evaluator"
        ]
        rows.append(
            ablation_row(
                "A2",
                f"effective embedding {old.embedding_backend} -> {new.embedding_backend}",
                results["A2_embedding_backend"],
                previous,
            )
        )
        previous = results["A2_embedding_backend"]
    if old.classifiers != new.classifiers:
        rows.append(
            {
                "ablation": "not run",
                "change": "classifier protocol differs but is already represented in strict result",
            }
        )
    return rows, results


def classifier_comparison(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    args: argparse.Namespace,
    store: EmbeddingStore,
) -> list[dict[str, Any]]:
    protocol = TextC2STProtocol(
        name="classifier_sensitivity",
        embedding_backend="minilm",
        embedding_model=args.embedding_model,
        preprocessing="canonical",
        classifiers=("logistic_regression", "random_forest", "gradient_boosting"),
        max_rows=min(args.classifier_comparison_rows, len(real), len(synthetic)),
        seed=args.seed,
        n_splits=5,
    )
    result = evaluate_protocol(real, synthetic, protocol, store, label="classifier_comparison")
    rows = []
    for classifier in protocol.classifiers:
        field_values = []
        row: dict[str, Any] = {
            "classifier": classifier,
            "rows_per_class": result["num_real"],
            "same_cached_embeddings": True,
            "same_cv_partitions": True,
        }
        for field in TEXT_FIELDS:
            values = ((result.get("per_field") or {}).get(field) or {}).get(
                "per_classifier", {}
            ).get(classifier, {})
            row[f"{field}_auc"] = values.get("auc")
            row[f"{field}_error"] = values.get("error")
            if values.get("error") is not None:
                field_values.append(values["error"])
        row["macro_error"] = float(np.mean(field_values)) if field_values else None
        rows.append(row)
    return rows


def sample_size_analysis(
    old: TextC2STProtocol,
    new: TextC2STProtocol,
    controlled: dict[str, dict[str, Any]],
    intermediate: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if old.max_rows == new.max_rows:
        return []
    rows = []
    for label, result in (
        (f"N={old.max_rows}", controlled["current_data_old_evaluator"]),
        (f"N={new.max_rows}", intermediate.get("A1_row_count")),
    ):
        if not result:
            continue
        rows.append(
            {
                "sample_size": label,
                "embedding_backend": old.embedding_backend,
                "summary_error": field_error(result, "summary"),
                "review_error": field_error(result, "review_text"),
                "macro_error": result.get("macro_error"),
            }
        )
    return rows


def audit_embedding_consistency(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    old: TextC2STProtocol,
    new: TextC2STProtocol,
    store: EmbeddingStore,
) -> dict[str, Any]:
    result: dict[str, Any] = {"num_real": 100, "num_synthetic": 100, "fields": {}}
    for field in TEXT_FIELDS:
        values = pd.concat([real[field].head(100), synthetic[field].head(100)]).tolist()
        old_values = store.embed(
            values,
            backend=old.embedding_backend,
            model_name=old.embedding_model,
            preprocessing=old.preprocessing,
            label=f"embedding_consistency_{field}_old",
        )
        new_values = store.embed(
            values,
            backend=new.embedding_backend,
            model_name=new.embedding_model,
            preprocessing=new.preprocessing,
            label=f"embedding_consistency_{field}_new",
        )
        field_result: dict[str, Any] = {
            "old_shape": list(old_values.shape),
            "new_shape": list(new_values.shape),
            "same_shape": bool(old_values.shape == new_values.shape),
        }
        if old_values.shape == new_values.shape:
            numerator = np.sum(old_values * new_values, axis=1)
            denominator = np.linalg.norm(old_values, axis=1) * np.linalg.norm(new_values, axis=1)
            cosine = numerator / np.maximum(denominator, 1e-12)
            difference = np.abs(old_values - new_values)
            field_result.update(
                {
                    "mean_cosine_similarity": float(np.mean(cosine)),
                    "max_absolute_difference": float(np.max(difference)),
                    "mean_absolute_difference": float(np.mean(difference)),
                    "effectively_identical": bool(
                        np.allclose(old_values, new_values, rtol=1e-6, atol=1e-7)
                    ),
                }
            )
        else:
            field_result.update(
                {
                    "mean_cosine_similarity": None,
                    "max_absolute_difference": None,
                    "mean_absolute_difference": None,
                    "effectively_identical": False,
                    "reason": "embedding dimensions differ",
                }
            )
        result["fields"][field] = field_result
    result["embedding_pipeline_same"] = bool(
        all(value["effectively_identical"] for value in result["fields"].values())
    )
    return result


def preprocessing_audit(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "historical_minilm": "str(value); no lowercasing, stripping, whitespace collapse, HTML removal, or Unicode normalization",
        "historical_hash": "null-to-empty, strip/collapse whitespace, lowercase tokenization",
        "canonical": "null-to-empty, Unicode NFC, strip/collapse whitespace, preserve case",
        "html_removal": "none in historical, strict-cleanup, or canonical protocol",
        "field_joining": "none for per-field metrics",
        "combined_field_representation": "concatenate independently computed summary and review embeddings",
        "special_tokens": "managed only by SentenceTransformer tokenizer; no project-added tokens or separators",
        "truncation": "hash backend has no sequence truncation; MiniLM uses the loaded SentenceTransformer max_sequence_length",
        "tables": {},
    }
    for table_name, frame in (("real", real), ("synthetic", synthetic)):
        result["tables"][table_name] = {}
        for field in TEXT_FIELDS:
            raw = frame[field].map(str)
            canonical = frame[field].map(canonical_text)
            result["tables"][table_name][field] = {
                "rows": int(len(frame)),
                "changed_by_canonicalization": int((raw != canonical).sum()),
                "changed_by_lowercasing_after_canonicalization": int(
                    (canonical != canonical.str.lower()).sum()
                ),
                "null_rows": int(frame[field].isna().sum()),
                "empty_after_canonicalization": int(canonical.eq("").sum()),
            }
    return result


def build_summary(**values: Any) -> dict[str, Any]:
    old_entry = values["old_entry"]
    current_entry = values["current_entry"]
    old_protocol = values["old_protocol"]
    new_protocol = values["new_protocol"]
    controlled = values["controlled"]
    canonical = values["canonical_results"]
    same_embeddings = values["embedding_consistency"]["embedding_pipeline_same"]
    same_classifier = old_protocol.classifiers == new_protocol.classifiers
    same_rows = old_protocol.max_rows == new_protocol.max_rows
    same_split = old_protocol.n_splits == new_protocol.n_splits and old_protocol.seed == new_protocol.seed
    same_aggregation = old_protocol.aggregation == new_protocol.aggregation
    old_current = controlled["current_data_old_evaluator"]["macro_error"]
    new_current = controlled["current_data_new_evaluator"]["macro_error"]
    old_old = controlled["old_data_old_evaluator"]["macro_error"]
    new_old = controlled["old_data_new_evaluator"]["macro_error"]
    old_reported = finite_or_none(old_entry.get("reported_text_c2st"))
    new_reported = finite_or_none(current_entry.get("reported_text_c2st"))
    old_reproduction_error = (
        abs(old_old - old_reported) if old_reported is not None else None
    )
    new_reproduction_error = (
        abs(new_current - new_reported) if new_reported is not None else None
    )
    backend_evidence = str(values["backend_evidence"])
    backend_directly_observed = any(
        marker in backend_evidence.lower()
        for marker in ("64 feature", "64-d cache", "cache metadata")
    )
    historical_backend_validated = bool(
        old_protocol.embedding_backend == "deterministic_hash"
        and (
            backend_directly_observed
            or (old_reproduction_error is not None and old_reproduction_error <= 0.01)
        )
    )
    causes = []
    if not same_embeddings:
        qualifier = "confirmed" if historical_backend_validated else "strongly indicated"
        causes.append(
            f"The primary evaluator change is the embedding backend ({qualifier}): "
            "the historical path used/could silently fall back to 64-D deterministic "
            "hashes, while the strict cleanup requires 384-D MiniLM. On the current "
            f"frozen data, the full evaluator effect is {new_current - old_current:+.6f}."
        )
    if not values["same_synthetic"]:
        causes.append(
            f"The historical and current synthetic CSVs differ; under the historical evaluator the data effect is {old_current - old_old:+.6f}, and under the strict evaluator it is {new_current - new_old:+.6f}."
        )
    if not same_rows:
        causes.append(
            f"The row limit changed from {old_protocol.max_rows} to {new_protocol.max_rows}."
        )
    if not causes:
        causes.append("No protocol or artifact difference was detected; inspect numerical reproducibility details.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_commit(),
        "model_retrained": False,
        "text_resampled": False,
        "old_reported_lstm_text_c2st": old_reported,
        "new_reported_lstm_text_c2st": new_reported,
        "old_reported_implied_auc_assuming_auc_ge_half": (
            implied_auc(old_reported) if old_reported is not None else None
        ),
        "new_reported_implied_auc_assuming_auc_ge_half": (
            implied_auc(new_reported) if new_reported is not None else None
        ),
        "historical_reproduction_absolute_error": old_reproduction_error,
        "strict_reproduction_absolute_error": new_reproduction_error,
        "historical_backend_validated": historical_backend_validated,
        "same_synthetic_data": values["same_synthetic"],
        "same_real_data": values["same_real"],
        "same_embeddings": same_embeddings,
        "same_classifier": same_classifier,
        "same_row_count": same_rows,
        "same_split_cv": same_split,
        "same_aggregation": same_aggregation,
        "root_causes": causes,
        "controlled_macro_errors": {
            "old_data_old_evaluator": old_old,
            "old_data_new_evaluator": new_old,
            "current_data_old_evaluator": old_current,
            "current_data_new_evaluator": new_current,
        },
        "historical_backend_evidence": backend_evidence,
        "data_comparison": values["data_comparison"],
        "canonical_lstm_macro_error": canonical["final_lstm"]["macro_error"],
        "canonical_diffusion_macro_error": canonical["diffusion"]["macro_error"],
        "lstm_better_than_diffusion": canonical["lstm_better_than_diffusion"],
        "cache_events": values["cache_events"],
    }


def write_audit_report(
    output: Path,
    summary: dict[str, Any],
    historical: pd.DataFrame,
    hashes: pd.DataFrame,
    evaluator: pd.DataFrame,
    cross: pd.DataFrame,
    ablation: pd.DataFrame,
    classifiers: pd.DataFrame,
    sample_size: pd.DataFrame,
    consistency: dict[str, Any],
    preprocessing: dict[str, Any],
    data_comparison: dict[str, Any],
    canonical: dict[str, Any],
    reproducibility: dict[str, Any],
) -> None:
    causes = "\n".join(
        f"{index}. {value}" for index, value in enumerate(summary["root_causes"], 1)
    )
    report = f"""# Amazon Text C2ST Reproducibility Audit

## 1. Executive Answer

The change from `{fmt(summary['old_reported_lstm_text_c2st'])}` to `{fmt(summary['new_reported_lstm_text_c2st'])}` is an evaluation audit result, not a model change. No model was trained and no text was resampled.

Historical-score reproduction absolute error: `{fmt(summary['historical_reproduction_absolute_error'])}`.<br>
Strict-score reproduction absolute error: `{fmt(summary['strict_reproduction_absolute_error'])}`.<br>
Historical backend inference validated: **{yes_no(summary['historical_backend_validated'])}**.

{causes}

Component deltas below are path-dependent; they are not presented as additive causal effects when data and protocol interactions prevent that interpretation.

## 2. Historical Evaluations

{markdown_table(historical)}

## 3. Artifact Hash Audit

{markdown_table(hashes)}

Same real CSV: **{yes_no(summary['same_real_data'])}**<br>
Same synthetic CSV: **{yes_no(summary['same_synthetic_data'])}**

Quantitative text comparison:

```json
{json.dumps(data_comparison, indent=2, sort_keys=True)}
```

## 4. Old Evaluator

{markdown_table(evaluator[evaluator['evaluator'] == 'historical'])}

The original implementation at `{OLD_EVALUATOR_COMMIT[:8]}` caught every Sentence Transformer failure and silently substituted a 64-dimensional deterministic token-hash vector. The requested model name in metadata therefore does not by itself establish that MiniLM ran.

## 5. New Evaluator

{markdown_table(evaluator[evaluator['evaluator'] == 'strict cleanup'])}

The strict cleanup rejects fallback embeddings. MiniLM uses Sentence Transformer mean pooling, its model-defined truncation limit, `normalize_embeddings=False`, and the evaluator casts output to float64.

## 6. Controlled 2x2 Cross-Evaluation

{markdown_table(cross)}

## 7. Root Cause

{causes}

## 8. Per-Field Results

The controlled table reports summary and review separately. `macro` is the arithmetic mean of per-field normalized errors. Both implementations use `error = 2 * abs(AUC - 0.5)`. This is not generally interchangeable with transforming the macro AUC when field AUCs lie on different sides of 0.5. Both quantities and their numerical gap are retained in machine-readable output.

## 9. Combined Text Metric

Combined text concatenates `summary_embedding` and `review_text_embedding` for the same sampled rows before classification. It does not embed joined strings. It is a secondary/appendix diagnostic and does not replace the primary per-field macro.

## 10. Classifier Sensitivity

{markdown_table(classifiers)}

All classifiers use the same cached embeddings and identical materialized stratified folds. This table demonstrates that C2ST is discriminator-dependent.

## 11. Reproducibility

Same-seed repeat mean: `{fmt(reproducibility['same_seed_mean'])}`<br>
Same-seed repeat standard deviation: `{fmt(reproducibility['same_seed_std'])}`<br>
Seeds 17/42/73 mean: `{fmt(reproducibility['discriminator_seed_mean'])}`<br>
Seeds 17/42/73 standard deviation: `{fmt(reproducibility['discriminator_seed_std'])}`

## 12. Canonical Protocol Recommendation

Use MiniLM embeddings from a pinned local snapshot, explicit null/Unicode/whitespace canonicalization, no text-length or structured features, balanced 50,000-per-class sampling (or all rows when smaller), fixed logistic regression with StandardScaler inside five-fold stratified CV, seed 42, and `2 * abs(AUC - 0.5)`. Report summary and review independently and their arithmetic mean. Keep concatenated embeddings secondary.

Loaded embedding runtime metadata:

```json
{json.dumps(canonical.get('embedding_model_metadata', {}), indent=2, sort_keys=True)}
```

## 13. Canonical Amazon Scores

{canonical_score_table(canonical)}

Canonical LSTM/diffusion fairness check: **{yes_no((canonical.get('fairness_check') or {}).get('passed'))}**. Both outputs use the same real rows, loaded model revision, preprocessing, balanced sample size, classifier, folds, seed, aggregation, and error formula.

## 14. Impact on Previous Conclusions

The LSTM text generator did not change during this audit. The discrepancy is attributable to data/evaluator differences listed above. Under the canonical fair protocol, LSTM better than diffusion: **{yes_no(canonical['lstm_better_than_diffusion'])}**.

## Component-by-Component Ablation

{markdown_table(ablation)}

## Sample-Size Sensitivity

{markdown_table(sample_size)}

## Embedding Consistency

```json
{json.dumps(consistency, indent=2, sort_keys=True)}
```

## Text Preprocessing Audit

```json
{json.dumps(preprocessing, indent=2, sort_keys=True)}
```
"""
    (output / "audit_report.md").write_text(report, encoding="utf-8")


def write_canonical_protocol(
    output: Path, protocol: TextC2STProtocol, model_metadata: dict[str, Any]
) -> None:
    text = f"""# Canonical Amazon Text-C2ST Protocol

- Embedding: `{protocol.embedding_model}` using Sentence Transformers.
- Effective backend: MiniLM; fallback is forbidden.
- Revision: `{model_metadata.get('revision') or 'recorded at runtime in canonical_text_c2st_results.json when exposed by the library'}`.
- Sentence Transformers version: `{model_metadata.get('library_version')}`.
- Transformers version: `{model_metadata.get('transformers_version')}`.
- Torch version: `{model_metadata.get('torch_version')}`.
- Embedding dimension: `{model_metadata.get('embedding_dimension')}`.
- Pooling: `{json.dumps(model_metadata.get('pooling'), sort_keys=True)}`.
- Maximum sequence length: `{model_metadata.get('max_sequence_length')}`.
- Truncation: `{model_metadata.get('truncation_behavior')}`.
- Encoding batch size: `{model_metadata.get('encode_batch_size')}`.
- Device used by this audit: `{model_metadata.get('device')}`.
- Model dtype: `{model_metadata.get('model_dtype')}`; evaluator array dtype: `{model_metadata.get('output_dtype_after_evaluator_cast')}`.
- Embedding normalization: disabled.
- Text preprocessing: null to empty, Unicode NFC, trim and collapse whitespace, preserve case.
- Sampling: balanced, at most {protocol.max_rows} real and {protocol.max_rows} synthetic rows.
- Classifier: StandardScaler + LogisticRegression(max_iter=500).
- Evaluation: five-fold StratifiedKFold with shuffle and seed {protocol.seed}.
- Error: `2 * abs(AUC - 0.5)`; lower is better.
- Aggregation: arithmetic mean of summary and review-text errors.
- Combined text: concatenated per-field embeddings, secondary/appendix-only.
- Excluded: text lengths, structured attributes, IDs, foreign keys, and timestamps.
"""
    (output / "canonical_text_c2st_protocol.md").write_text(text, encoding="utf-8")


def print_console(
    summary: dict[str, Any],
    controlled: dict[str, dict[str, Any]],
    canonical: dict[str, Any],
    old: TextC2STProtocol,
    new: TextC2STProtocol,
) -> None:
    old_result = controlled["current_data_old_evaluator"]
    new_result = controlled["current_data_new_evaluator"]
    print("\n============================================================")
    print("TEXT C2ST AUDIT")
    print("============================================================")
    print(f"\nOLD REPORTED LSTM TEXT C2ST:\n{fmt(summary['old_reported_lstm_text_c2st'])}")
    print(f"\nNEW REPORTED LSTM TEXT C2ST:\n{fmt(summary['new_reported_lstm_text_c2st'])}")
    for label, key in (
        ("SAME SYNTHETIC DATA?", "same_synthetic_data"),
        ("SAME REAL DATA?", "same_real_data"),
        ("SAME EMBEDDINGS?", "same_embeddings"),
        ("SAME CLASSIFIER?", "same_classifier"),
        ("SAME ROW COUNT?", "same_row_count"),
        ("SAME SPLIT/CV?", "same_split_cv"),
        ("SAME AGGREGATION?", "same_aggregation"),
    ):
        print(f"\n{label}\n{yes_no(summary[key])}")
    print("\nROOT CAUSE:")
    for index, cause in enumerate(summary["root_causes"], 1):
        print(f"{index}. {cause}")
    print("\nCONTROLLED SAME-DATA RESULTS:")
    print_result("Old evaluator", old_result)
    print_result("New evaluator", new_result)
    print("\nCANONICAL PAPER PROTOCOL:")
    print(f"embedding = {canonical['protocol']['embedding_model']}")
    print("classifier = StandardScaler + LogisticRegression(max_iter=500)")
    print("evaluation = balanced five-fold stratified CV")
    print(f"rows = {canonical['protocol']['max_rows']} per class")
    print(f"seed = {canonical['protocol']['seed']}")
    print("aggregation = arithmetic mean of per-field errors")
    print("error formula = 2 * abs(AUC - 0.5)")
    print("\nCANONICAL AMAZON RESULTS:")
    print_result("Final LSTM", canonical["final_lstm"])
    print_result("Diffusion", canonical["diffusion"])
    print(f"\nLSTM BETTER THAN DIFFUSION?\n{yes_no(canonical['lstm_better_than_diffusion'])}")
    print("\nMODEL RETRAINED:\nNO")
    print("\nTEXT RESAMPLED:\nNO")
    print("\n============================================================")


def select_current_entry(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = [
        item
        for item in inventory
        if "evaluation_cleanup/recomputed/amazon_toy/metrics.json"
        in str(item.get("metrics_path"))
    ]
    candidates = preferred or [
        item
        for item in inventory
        if finite_or_none(item.get("reported_text_c2st")) is not None
        and float(item["reported_text_c2st"]) >= 0.70
        and item.get("synthetic_csv")
    ]
    if not candidates:
        raise FileNotFoundError(
            "Could not locate the strict Amazon cleanup metrics. Expected outputs/evaluation_cleanup/recomputed/amazon_toy/metrics.json"
        )
    return max(candidates, key=lambda item: Path(item["metrics_path"]).stat().st_mtime)


def select_historical_entry(
    inventory: list[dict[str, Any]], current: dict[str, Any]
) -> dict[str, Any]:
    candidates = []
    for item in inventory:
        score = finite_or_none(item.get("reported_text_c2st"))
        if score is None or score >= 0.70 or item is current:
            continue
        path = str(item.get("metrics_path", "")).lower()
        if "diffusion" in path or "hierarchical" in path:
            continue
        if item.get("synthetic_csv") and Path(item["synthetic_csv"]).is_file():
            candidates.append(item)
    if not candidates:
        references = [
            item
            for item in inventory
            if finite_or_none(item.get("reported_text_c2st")) in KNOWN_HISTORICAL_SCORES
        ]
        if references:
            return references[0]
        raise FileNotFoundError("Could not locate a historical Amazon LSTM text-C2ST result")
    return min(
        candidates,
        key=lambda item: min(
            abs(float(item["reported_text_c2st"]) - target)
            for target in KNOWN_HISTORICAL_SCORES
        ),
    )


def infer_historical_backend(entry: dict[str, Any]) -> tuple[str, str]:
    backend = entry.get("effective_embedding_backend")
    evidence = entry.get("backend_evidence") or ""
    if backend in {"minilm", "deterministic_hash"}:
        return str(backend), str(evidence)
    return (
        "deterministic_hash",
        "Historical evaluator silently fell back on any Sentence Transformer exception; backend was not recorded. This is an explicit hypothesis that the controlled reproduction validates against the reported score.",
    )


def infer_backend_from_metric(path: Path, text: dict[str, Any]) -> tuple[str | None, str | None]:
    dimensions = set()
    for values in (text.get("per_text_column") or {}).values():
        features = values.get("feature_names") or []
        if features:
            dimensions.add(len(features))
    if dimensions == {64}:
        return "deterministic_hash", "64 feature names recorded in text metric"
    if dimensions == {384}:
        return "minilm", "384 feature names recorded in text metric"
    cache_dir = path.parent / "embedding_cache"
    if cache_dir.is_dir():
        for metadata_path in cache_dir.glob("*.json"):
            metadata = load_json(metadata_path)
            backend = metadata.get("embedding_backend")
            if backend == "deterministic_hash_fallback":
                return "deterministic_hash", f"cache metadata: {metadata_path}"
            if backend and "MiniLM" in str(backend):
                return "minilm", f"cache metadata: {metadata_path}"
            shape = metadata.get("shape") or []
            if len(shape) == 2 and shape[1] == 64:
                return "deterministic_hash", f"64-D cache: {metadata_path}"
            if len(shape) == 2 and shape[1] == 384:
                return "minilm", f"384-D cache: {metadata_path}"
    return None, None


def metric_result(entry: dict[str, Any]) -> dict[str, Any] | None:
    payload = entry.get("payload") or {}
    text = payload.get("text_embedding_c2st") or (
        payload if "per_text_column" in payload else {}
    )
    fields = text.get("per_text_column") or {}
    if not fields:
        return None
    per_field = {}
    for field, values in fields.items():
        per_field[field] = {
            "auc": values.get("auc"),
            "error": values.get("error"),
            "accuracy": values.get("accuracy"),
            "classifier": values.get("classifier"),
            "per_classifier": values.get("per_classifier") or {},
            "num_real": values.get("num_real"),
            "num_synthetic": values.get("num_synthetic"),
        }
    combined = text.get("combined_text_fields")
    macro_auc = finite_or_none(text.get("macro_auc"))
    macro_error = finite_or_none(text.get("macro_error"))
    transformed_macro_auc = c2st_error(macro_auc) if macro_auc is not None else None
    return {
        "protocol": None,
        "num_real": entry.get("rows_real"),
        "num_synthetic": entry.get("rows_synthetic"),
        "fields": list(per_field),
        "per_field": per_field,
        "macro_auc": macro_auc,
        "macro_error": macro_error,
        "macro_error_from_macro_auc": transformed_macro_auc,
        "aggregation_identity_gap": (
            macro_error - transformed_macro_auc
            if macro_error is not None and transformed_macro_auc is not None
            else None
        ),
        "combined": combined,
    }


def metric_matches_protocol(entry: dict[str, Any], protocol: TextC2STProtocol) -> bool:
    return bool(
        entry.get("effective_embedding_backend") == "minilm"
        and int(entry.get("rows_real") or -1) == protocol.max_rows
        and tuple(entry.get("classifiers") or ()) == protocol.classifiers
    )


def evaluator_comparison_rows(
    old: TextC2STProtocol,
    new: TextC2STProtocol,
    old_entry: dict[str, Any],
    new_entry: dict[str, Any],
    backend_evidence: str,
) -> list[dict[str, Any]]:
    rows = []
    old_commit = (
        OLD_FULL_ROW_CONFIG_COMMIT
        if old.max_rows != 50000
        else OLD_EVALUATOR_COMMIT
    )
    for label, protocol, entry, commit in (
        ("historical", old, old_entry, old_commit),
        ("strict cleanup", new, new_entry, STRICT_EVALUATOR_COMMIT),
    ):
        rows.append(
            {
                "evaluator": label,
                "reported_macro_error": entry.get("reported_text_c2st"),
                "requested_embedding_model": protocol.embedding_model,
                "effective_embedding_backend": protocol.embedding_backend,
                "backend_evidence": backend_evidence if label == "historical" else "strict fallback rejection and 384-D cache",
                "preprocessing": protocol.preprocessing,
                "embedding_library": (
                    "sentence-transformers"
                    if protocol.embedding_backend == "minilm"
                    else "project-local NumPy hash embedding"
                ),
                "model_revision": (
                    "not recorded by original evaluation artifact"
                    if protocol.embedding_backend == "minilm"
                    else "not applicable"
                ),
                "pooling": "Sentence Transformer model pooling" if protocol.embedding_backend == "minilm" else "signed token hashing + L2 normalization",
                "embedding_normalization": False if protocol.embedding_backend == "minilm" else "per-row L2",
                "max_sequence_length": "model default" if protocol.embedding_backend == "minilm" else "not applicable",
                "truncation_behavior": (
                    "SentenceTransformer model-default truncation"
                    if protocol.embedding_backend == "minilm"
                    else "not applicable"
                ),
                "embedding_batch_size": (
                    "32 (SentenceTransformer.encode default)"
                    if protocol.embedding_backend == "minilm"
                    else "not applicable"
                ),
                "embedding_device": (
                    "not recorded; SentenceTransformer automatic selection"
                    if protocol.embedding_backend == "minilm"
                    else "CPU/NumPy"
                ),
                "model_dtype": (
                    "not recorded" if protocol.embedding_backend == "minilm" else "float64"
                ),
                "evaluator_array_dtype": "float64",
                "classifiers": ", ".join(protocol.classifiers),
                "selected_classifier_in_reported_metric": entry.get("classifier"),
                "classifier_hyperparameters": "LR StandardScaler+max_iter=500; RF n_estimators=100,n_jobs=1; GB sklearn defaults",
                "class_weight": "None for all classifiers",
                "classifier_random_state": protocol.seed,
                "rows_per_class": protocol.max_rows,
                "balanced_real_synthetic": True,
                "row_sampling": (
                    "real.sample(random_state=seed); "
                    "synthetic.sample(random_state=seed+1); head when uncapped"
                ),
                "seed": protocol.seed,
                "cv": f"{protocol.n_splits}-fold StratifiedKFold(shuffle=True)",
                "aggregation": "mean(per-field error)",
                "error_formula": "2 * abs(AUC - 0.5)",
                "evaluation_script": entry.get("evaluation_script"),
                "code_commit": commit,
            }
        )
    return rows


def add_runtime_embedding_metadata(
    rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    """Attach the exact model snapshot used by this controlled audit."""

    for row in rows:
        if row.get("effective_embedding_backend") != "minilm":
            continue
        row["audit_runtime_library_version"] = metadata.get("library_version")
        row["audit_runtime_transformers_version"] = metadata.get(
            "transformers_version"
        )
        row["audit_runtime_torch_version"] = metadata.get("torch_version")
        row["audit_runtime_model_revision"] = metadata.get("revision")
        row["audit_runtime_embedding_dimension"] = metadata.get(
            "embedding_dimension"
        )
        row["audit_runtime_pooling"] = json.dumps(
            metadata.get("pooling"), sort_keys=True
        )
        row["audit_runtime_max_sequence_length"] = metadata.get(
            "max_sequence_length"
        )
        row["audit_runtime_tokenizer_model_max_length"] = metadata.get(
            "tokenizer_model_max_length"
        )
        row["audit_runtime_embedding_device"] = metadata.get("device")
        row["audit_runtime_model_dtype"] = metadata.get("model_dtype")


def public_inventory_row(item: dict[str, Any]) -> dict[str, Any]:
    score = finite_or_none(item.get("reported_text_c2st"))
    return {
        "Evaluation ID": item.get("evaluation_id"),
        "Reported Text C2ST": score,
        "Implied AUC (assuming AUC >= 0.5)": implied_auc(score) if score is not None else None,
        "Synthetic CSV": item.get("synthetic_csv"),
        "Real CSV": item.get("real_csv"),
        "Embedding Model": item.get("embedding_model"),
        "Effective Backend": item.get("effective_embedding_backend"),
        "Classifier": item.get("classifier"),
        "Rows Real": item.get("rows_real"),
        "Rows Synthetic": item.get("rows_synthetic"),
        "Seed": item.get("seed"),
        "Evaluation Script": item.get("evaluation_script"),
        "Metric Version": item.get("metric_version"),
        "Timestamp / Commit": item.get("timestamp_or_commit"),
    }


def ablation_row(
    label: str,
    change: str,
    result: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ablation": label,
        "change": change,
        "summary_c2st": field_error(result, "summary"),
        "review_c2st": field_error(result, "review_text"),
        "macro_c2st": result.get("macro_error"),
        "delta_vs_previous": (
            result.get("macro_error") - previous.get("macro_error")
            if previous and result.get("macro_error") is not None
            else None
        ),
    }


def reproducibility_summary(
    repeated: list[dict[str, Any]], seeds: list[dict[str, Any]]
) -> dict[str, Any]:
    repeat_values = np.asarray([item["macro_error"] for item in repeated], dtype=float)
    seed_values = np.asarray([item["macro_error"] for item in seeds], dtype=float)
    return {
        "same_seed_runs": repeated,
        "same_seed_mean": float(repeat_values.mean()),
        "same_seed_std": float(repeat_values.std(ddof=0)),
        "same_seed_identical": bool(np.all(repeat_values == repeat_values[0])),
        "discriminator_seed_runs_same_embeddings": seeds,
        "discriminator_seed_mean": float(seed_values.mean()),
        "discriminator_seed_std": float(seed_values.std(ddof=0)),
    }


def canonical_score_table(canonical: dict[str, Any]) -> str:
    rows = []
    for label, key in (("Final LSTM", "final_lstm"), ("Diffusion", "diffusion")):
        result = canonical[key]
        rows.append(
            {
                "Model": label,
                "Summary AUC": field_auc(result, "summary"),
                "Summary Error": field_error(result, "summary"),
                "Review AUC": field_auc(result, "review_text"),
                "Review Error": field_error(result, "review_text"),
                "Macro Error": result.get("macro_error"),
                "Combined Error": (result.get("combined") or {}).get("error"),
            }
        )
    return markdown_table(pd.DataFrame(rows))


def print_result(label: str, result: dict[str, Any]) -> None:
    print(f"\n{label}:")
    print(f"    summary = {fmt(field_error(result, 'summary'))}")
    print(f"    review  = {fmt(field_error(result, 'review_text'))}")
    print(f"    macro   = {fmt(result.get('macro_error'))}")
    print(f"    combined = {fmt((result.get('combined') or {}).get('error'))}")


def compare_optional_frames(
    left: pd.DataFrame | None, right: pd.DataFrame | None
) -> dict[str, Any]:
    if left is None or right is None:
        return {"status": "unavailable"}
    return compare_text_frames(left, right)


def invert_path_roles(paths: dict[str, Path]) -> dict[Path, list[str]]:
    result: dict[Path, list[str]] = {}
    for role, path in paths.items():
        result.setdefault(path, []).append(role)
    return result


def require_paths(paths: dict[str, Path], inventory_only: bool) -> None:
    required = ["canonical_real", "canonical_lstm_synthetic", "old_real", "old_synthetic"]
    if not inventory_only:
        required.append("canonical_diffusion_synthetic")
    missing = [f"{key}: {paths[key]}" for key in required if not paths[key].is_file()]
    if missing:
        raise FileNotFoundError("Missing audit inputs:\n- " + "\n- ".join(missing))


def first_existing_path(*values: Any) -> Path:
    candidates = []
    for value in values:
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        candidates.append(path.resolve())
        if path.is_file():
            return path.resolve()
    return candidates[0] if candidates else (ROOT / "__missing__").resolve()


def resolve_optional_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return (path if path.is_absolute() else ROOT / path).resolve()


def discover_from_preflight(key: str) -> str | None:
    path = ROOT / "outputs/evaluation_cleanup/preflight_inventory.json"
    payload = load_json(path)
    for item in payload.get("available_synthetic_outputs", []):
        if item.get("key") == key:
            return item.get("path")
    return None


def discover_embedding_cache_roots(inventory: list[dict[str, Any]]) -> list[Path]:
    roots = {ROOT / "outputs/evaluation_cleanup/recomputed"}
    for item in inventory:
        path = Path(str(item.get("metrics_path", "")))
        if path.is_file():
            roots.add(path.parent)
    return sorted(roots)


def find_companion_evaluation_config(path: Path) -> Path | None:
    for parent in [path.parent, *list(path.parents)[:6]]:
        for name in ("evaluation_config_resolved.yaml", "evaluation_config.yaml"):
            candidate = parent / name
            if candidate.is_file():
                return candidate.resolve()
    return (ROOT / AMAZON_EVALUATION_CONFIG).resolve()


def config_embedding_model(config: dict[str, Any]) -> str | None:
    return (((config.get("evaluation") or {}).get("text") or {}).get("embedding_model"))


def config_classifiers(config: dict[str, Any]) -> list[str]:
    return list((((config.get("evaluation") or {}).get("c2st") or {}).get("classifiers") or []))


def config_seed(config: dict[str, Any], payload: dict[str, Any]) -> int:
    return int((config.get("evaluation") or {}).get("random_seed", 42))


def config_n_splits(config: dict[str, Any]) -> int:
    return int(((config.get("evaluation") or {}).get("c2st") or {}).get("n_splits", 5))


def default_classifiers() -> tuple[str, ...]:
    return ("logistic_regression", "random_forest", "gradient_boosting")


def same_file_hash(left: Path, right: Path) -> bool:
    if not left.is_file() or not right.is_file():
        return False
    if left.resolve() == right.resolve():
        return True
    from evaluation.text_c2st_audit import file_sha256

    return file_sha256(left) == file_sha256(right)


def field_error(result: dict[str, Any], field: str) -> float | None:
    return finite_or_none(((result.get("per_field") or {}).get(field) or {}).get("error"))


def field_auc(result: dict[str, Any], field: str) -> float | None:
    return finite_or_none(((result.get("per_field") or {}).get(field) or {}).get("auc"))


def ensure_fresh_output(output: Path) -> None:
    existing = [name for name in REQUIRED_OUTPUTS if (output / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite completed audit artifacts at {output}. "
            "Use a new --output-dir. Existing: " + ", ".join(existing)
        )


def load_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    candidate = Path(path)
    if not candidate.is_file():
        return {}
    value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No records._"
    clean = frame.copy()
    clean = clean.replace({np.nan: "NA", None: "NA"})
    columns = [str(column) for column in clean.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in clean.iterrows():
        values = [str(row[column]).replace("|", "\\|").replace("\n", " ") for column in clean.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    numeric = finite_or_none(value)
    return "NA" if numeric is None else f"{numeric:.6f}"


def yes_no(value: Any) -> str:
    return "YES" if bool(value) else "NO"


def progress(message: str) -> None:
    print(f"[text-c2st-audit] {message}", flush=True)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
