#!/usr/bin/env python3
"""Run reproducible hierarchical-diffusion diagnostic matrices."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.dataset import load_text_tokenizer  # noqa: E402
from attribute_generation.conditional_tabdlm.diffusion_diagnostics import (  # noqa: E402
    current_git_commit,
    dataframe_fingerprint,
    file_sha256,
    safe_name,
    text_generation_diagnostics,
    unique_run_root,
    write_json,
)
from attribute_generation.conditional_tabdlm.hierarchical_sample import (  # noqa: E402
    hierarchical_sample_from_config,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from evaluation.paper_metrics.reporting import write_markdown_report  # noqa: E402
from evaluation.paper_metrics.c2st import single_table_c2st_metrics  # noqa: E402
from evaluation.paper_metrics.text_embedding import (  # noqa: E402
    text_embedding_c2st_metrics,
)
from evaluation.paper_metrics.utils import write_json as write_paper_json  # noqa: E402
from scripts.evaluate_single_event_table_paper_metrics import (  # noqa: E402
    evaluate_paper_metrics,
    write_legacy_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--matrices", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.experiment_config).open(encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    run_diagnostic_experiment(
        experiment,
        experiment_config_path=Path(args.experiment_config),
        device_override=args.device,
        seeds_override=args.seeds,
        matrices_override=args.matrices,
        dry_run=bool(args.dry_run),
        continue_on_error=bool(args.continue_on_error),
        max_runs=args.max_runs,
    )


def run_diagnostic_experiment(
    experiment: dict[str, Any],
    *,
    experiment_config_path: Path,
    device_override: str | None = None,
    seeds_override: list[int] | None = None,
    matrices_override: list[str] | None = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
    max_runs: int | None = None,
) -> Path:
    benchmark = load_and_verify_benchmark(
        Path(experiment["benchmark"]["manifest"])
    )
    model_config_path = Path(experiment["model"]["config"])
    checkpoint_path = Path(experiment["model"]["checkpoint"])
    evaluation_config_path = Path(experiment["evaluation"]["config"])
    require_file(model_config_path, "model config")
    require_file(checkpoint_path, "checkpoint")
    require_file(evaluation_config_path, "evaluation config")
    config = load_config(model_config_path)
    training_metadata_path = Path(
        experiment.get("model", {}).get(
            "training_metadata",
            checkpoint_path.parent.parent
            / "metadata"
            / "training_runtime.json",
        )
    )
    training_runtime = read_json_optional(training_metadata_path)
    if "minimum_text_content_tokens" in (experiment.get("sampling") or {}):
        config.raw.setdefault("sampling", {})[
            "minimum_text_content_tokens"
        ] = experiment["sampling"]["minimum_text_content_tokens"]
    tokenizer = load_text_tokenizer(config)
    seeds = [
        int(value)
        for value in (
            seeds_override
            if seeds_override
            else experiment.get("seeds", [42])
        )
    ]
    matrices = set(
        matrices_override
        if matrices_override
        else experiment.get("enabled_matrices", ["progressive_conditioning"])
    )
    specifications = build_run_specifications(experiment, matrices)
    planned = [
        dict(specification, seed=seed)
        for seed in seeds
        for specification in specifications
    ]
    if max_runs is not None:
        planned = planned[: int(max_runs)]
    output_root = unique_run_root(
        experiment["output_root"],
        experiment.get("experiment_name", "hierarchical_diffusion_diagnostics"),
    )
    resolved = copy.deepcopy(experiment)
    resolved["resolved"] = {
        "git_commit": current_git_commit(),
        "experiment_config_path": str(experiment_config_path),
        "experiment_config_sha256": file_sha256(experiment_config_path),
        "benchmark_manifest_sha256": file_sha256(
            experiment["benchmark"]["manifest"]
        ),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "model_parameter_count": checkpoint_parameter_count(checkpoint_path),
        "training_metadata_path": (
            str(training_metadata_path)
            if training_metadata_path.exists()
            else None
        ),
        "training_runtime": training_runtime,
        "planned_runs": planned,
        "dry_run": bool(dry_run),
    }
    write_json(output_root / "resolved_experiment.json", resolved)
    if dry_run:
        print(json.dumps({"output_root": str(output_root), "runs": planned}, indent=2))
        return output_root

    evaluation_real_path = benchmark_file(benchmark, "evaluation_real")
    evaluation_spine_path = benchmark_file(benchmark, "evaluation_spine")
    history_prefix_path = benchmark_file(benchmark, "graph_history_prefix")
    real = pd.read_csv(evaluation_real_path)
    evaluation_config = load_yaml(evaluation_config_path)
    coverage_path = optional_benchmark_file(
        benchmark, "evaluation_history_coverage"
    )
    history_coverage = (
        pd.read_csv(coverage_path) if coverage_path is not None else None
    )
    results: list[dict[str, Any]] = []
    generated_by_key: dict[tuple[int, str], pd.DataFrame] = {}
    for run_index, run_spec in enumerate(planned, start=1):
        seed = int(run_spec["seed"])
        label = safe_name(run_spec["label"])
        run_dir = output_root / f"{run_index:03d}_{label}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        print(
            f"[{run_index}/{len(planned)}] {run_spec['matrix']} "
            f"{run_spec['label']} seed={seed}"
        )
        started = time.perf_counter()
        try:
            output_path = run_one_sample(
                config=config,
                checkpoint_path=checkpoint_path,
                evaluation_spine_path=evaluation_spine_path,
                evaluation_real_path=evaluation_real_path,
                history_prefix_path=history_prefix_path,
                output_dir=run_dir,
                run_spec=run_spec,
                experiment=experiment,
                device_override=device_override,
                seed=seed,
            )
            synthetic = pd.read_csv(output_path)
            paper_metrics = evaluate_one(
                evaluation_config_path=evaluation_config_path,
                model_config_path=model_config_path,
                real_path=evaluation_real_path,
                synthetic_path=output_path,
                output_dir=run_dir / "evaluation",
                seed=seed,
            )
            text_metrics = text_generation_diagnostics(
                real,
                synthetic,
                schema=config.schema,
                tokenizer=tokenizer,
            )
            write_json(run_dir / "evaluation" / "text_diagnostics.json", text_metrics)
            subgroup_metrics = history_subgroup_metrics(
                real,
                synthetic,
                history_coverage,
                evaluation_config,
                output_dir=output_root / "shared_subgroup_embeddings",
                seed=seed,
                settings=experiment.get("graph_subgroup_evaluation") or {},
            )
            write_json(
                run_dir / "evaluation" / "history_subgroup_metrics.json",
                subgroup_metrics,
            )
            runtime = read_json_optional(
                run_dir
                / "metadata"
                / "runtime_hierarchical_sampling.json"
            )
            sample_metadata = read_json_optional(
                run_dir
                / "metadata"
                / "hierarchical_sample_metadata.json"
            )
            elapsed = float(time.perf_counter() - started)
            manifest = {
                "status": "completed",
                "matrix": run_spec["matrix"],
                "label": run_spec["label"],
                "seed": seed,
                "run_specification": run_spec,
                "git_commit": current_git_commit(),
                "benchmark_fingerprint": benchmark.get(
                    "dataframe_fingerprints", {}
                ).get("evaluation_real"),
                "evaluation_real_fingerprint": dataframe_fingerprint(real),
                "synthetic_fingerprint": dataframe_fingerprint(synthetic),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "model_parameter_count": resolved["resolved"][
                    "model_parameter_count"
                ],
                "wall_clock_seconds": elapsed,
                "runtime": runtime,
                "training_runtime": training_runtime,
                "sample_metadata": sample_metadata,
            }
            write_json(run_dir / "run_manifest.json", manifest)
            result = result_row(
                run_spec,
                seed,
                paper_metrics,
                text_metrics,
                subgroup_metrics,
                runtime,
                manifest,
            )
            results.append(result)
            generated_by_key[(seed, run_spec["label"])] = synthetic
        except Exception as exc:
            write_json(
                run_dir / "failure.json",
                {
                    "status": "failed",
                    "run_specification": run_spec,
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if not continue_on_error:
                raise
            results.append(
                {
                    "matrix": run_spec["matrix"],
                    "label": run_spec["label"],
                    "seed": seed,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    add_context_sensitivity(results, generated_by_key)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_root / "consolidated_results.csv", index=False)
    write_json(
        output_root / "consolidated_results.json",
        result_frame.where(pd.notna(result_frame), None).to_dict(
            orient="records"
        ),
    )
    aggregate = aggregate_results(result_frame)
    aggregate.to_csv(output_root / "aggregate_mean_std.csv", index=False)
    diagnosis = diagnose_results(aggregate, experiment)
    write_json(output_root / "diagnosis_and_recommendation.json", diagnosis)
    write_results_report(
        output_root / "results_report.md",
        result_frame,
        aggregate,
        diagnosis,
    )
    print(f"Wrote consolidated experiment to {output_root}")
    return output_root


def build_run_specifications(
    experiment: dict[str, Any], enabled_matrices: set[str]
) -> list[dict[str, Any]]:
    matrices = experiment.get("matrices") or {}
    specifications: list[dict[str, Any]] = []
    if "progressive_conditioning" in enabled_matrices:
        for mode in matrices.get(
            "progressive_conditioning", ["O1", "O2", "O3", "O4", "O5"]
        ):
            specifications.append(
                {
                    "matrix": "progressive_conditioning",
                    "label": str(mode).upper(),
                    "conditioning_mode": str(mode).upper(),
                    "graph_mode": None,
                    "decoding_policy": "current",
                }
            )
    if "graph_context" in enabled_matrices:
        for entry in matrices.get("graph_context", []):
            specifications.append(
                {
                    "matrix": "graph_context",
                    "label": str(entry["name"]),
                    "conditioning_mode": None,
                    "graph_mode": str(entry["mode"]),
                    "use_real_history_prefix": bool(
                        entry.get("use_real_history_prefix", True)
                    ),
                    "decoding_policy": "current",
                }
            )
    if "decoding_policy" in enabled_matrices:
        for entry in matrices.get("decoding_policy", []):
            specifications.append(
                {
                    "matrix": "decoding_policy",
                    "label": str(entry["name"]),
                    "conditioning_mode": "O4",
                    "graph_mode": None,
                    "decoding_policy": str(entry["policy"]),
                    "text_top_k": entry.get("top_k"),
                    "top_p": entry.get("top_p"),
                    "temperature": entry.get("temperature"),
                }
            )
    if not specifications:
        raise ValueError(
            f"No runs configured for enabled matrices {sorted(enabled_matrices)}"
        )
    return specifications


def run_one_sample(
    *,
    config: Any,
    checkpoint_path: Path,
    evaluation_spine_path: Path,
    evaluation_real_path: Path,
    history_prefix_path: Path,
    output_dir: Path,
    run_spec: dict[str, Any],
    experiment: dict[str, Any],
    device_override: str | None,
    seed: int,
) -> Path:
    sampling = experiment.get("sampling") or {}
    conditioning_mode = run_spec.get("conditioning_mode")
    oracle_path = (
        evaluation_real_path
        if conditioning_mode in {"O1", "O2", "O3"}
        else None
    )
    graph_prefix = None
    if conditioning_mode in {"O1", "O2", "O3"}:
        graph_prefix = history_prefix_path
    elif run_spec.get("use_real_history_prefix"):
        graph_prefix = history_prefix_path
    output_path = output_dir / "synthetic_table.csv"
    return hierarchical_sample_from_config(
        config,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        num_rows="all",
        sample_batch_size=int(sampling.get("batch_size", 128)),
        structured_steps=sampling.get("structured_steps", 25),
        text_steps=sampling.get("text_steps", 25),
        timestep_spacing=sampling.get("timestep_spacing", "uniform"),
        inference_dtype=sampling.get("inference_dtype", "bfloat16"),
        text_top_k=run_spec.get(
            "text_top_k", sampling.get("text_top_k", 512)
        ),
        temperature=run_spec.get(
            "temperature", sampling.get("temperature", 1.0)
        ),
        top_p=run_spec.get("top_p", sampling.get("top_p", 0.95)),
        graph_mode_override=run_spec.get("graph_mode") or "correct",
        device=device_override or sampling.get("device", "cuda"),
        seed=seed,
        synthetic_spine_path=evaluation_spine_path,
        profile=True,
        profile_output=output_dir
        / "metadata"
        / "runtime_hierarchical_sampling.json",
        debug_write_aux_targets=True,
        oracle_structured_table_path=oracle_path,
        conditioning_mode=conditioning_mode,
        graph_history_prefix_path=graph_prefix,
        decoding_policy=run_spec.get("decoding_policy", "current"),
    )


def evaluate_one(
    *,
    evaluation_config_path: Path,
    model_config_path: Path,
    real_path: Path,
    synthetic_path: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    with evaluation_config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["real_table_path"] = str(real_path)
    config["synthetic_table_path"] = str(synthetic_path)
    config.setdefault("evaluation", {})["random_seed"] = int(seed)
    config["evaluation"]["sample_size"] = None
    config.setdefault("legacy_evaluator", {})["enabled"] = True
    config["legacy_evaluator"]["config_path"] = str(model_config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = evaluate_paper_metrics(config, output_dir)
    write_paper_json(metrics, output_dir / "metrics.json")
    write_paper_json(metrics, output_dir / "paper_metrics.json")
    write_markdown_report(metrics, output_dir / "metrics.md")
    write_legacy_metrics(config, output_dir)
    return metrics


def result_row(
    run_spec: dict[str, Any],
    seed: int,
    paper: dict[str, Any],
    text: dict[str, Any],
    subgroup: dict[str, Any],
    runtime: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    summary = paper.get("paper_metrics_summary") or {}
    row: dict[str, Any] = {
        "matrix": run_spec["matrix"],
        "label": run_spec["label"],
        "seed": int(seed),
        "status": "completed",
        "constraint_violation_rate": summary.get("constraint_violation_rate"),
        "shape_error": summary.get("shape_error"),
        "trend_error": summary.get("trend_error"),
        "text_embedding_c2st_error": summary.get(
            "text_embedding_c2st_error"
        ),
        "single_table_c2st_error": summary.get(
            "single_table_c2st_error"
        ),
        "sampling_seconds": runtime.get("total_sampling_seconds"),
        "training_seconds": (
            manifest.get("training_runtime") or {}
        ).get("total_training_seconds"),
        "rows_per_second": runtime.get("rows_per_second"),
        "peak_gpu_memory_mb": runtime.get(
            "cuda_memory_peak_allocated_mb"
        ),
        "synthetic_fingerprint": manifest["synthetic_fingerprint"],
    }
    for column, values in (text.get("per_column") or {}).items():
        for metric in (
            "empty_rate",
            "special_token_leakage_rate",
            "padding_token_leakage_rate",
            "vocabulary_size",
            "distinct_1",
            "distinct_2",
            "repeated_ngram_rate",
            "token_count_ks",
            "character_count_ks",
            "text_length_ks",
            "exact_or_valid_length_satisfaction_rate",
            "exact_training_row_duplication_rate",
            "invalid_utf8_rate",
        ):
            row[f"{column}_{metric}"] = values.get(metric)
    cross_field = (
        text.get("cross_field", {}).get(
            "first_second_text_hash_cosine", {}
        )
    )
    if cross_field:
        real_cosine = cross_field.get("real_mean")
        synthetic_cosine = cross_field.get("synthetic_mean")
        row["summary_review_hash_cosine_real"] = real_cosine
        row["summary_review_hash_cosine_synthetic"] = synthetic_cosine
        row["summary_review_hash_cosine_absolute_error"] = (
            abs(float(real_cosine) - float(synthetic_cosine))
            if real_cosine is not None and synthetic_cosine is not None
            else None
        )
    coverage = subgroup.get("coverage") or {}
    for metric, metric_value in coverage.items():
        row[f"graph_{metric}"] = metric_value
    for group, values in (subgroup.get("groups") or {}).items():
        row[f"{group}_num_rows"] = values.get("num_rows")
        row[f"{group}_single_table_c2st_error"] = values.get(
            "single_table_c2st_error"
        )
        row[f"{group}_text_embedding_c2st_error"] = values.get(
            "text_embedding_c2st_error"
        )
    return row


def add_context_sensitivity(
    rows: list[dict[str, Any]],
    generated: dict[tuple[int, str], pd.DataFrame],
) -> None:
    for row in rows:
        seed = int(row.get("seed", -1))
        label = str(row.get("label"))
        reference_label = "graph_both"
        if label == "graph_shuffled" and (seed, reference_label) in generated:
            row["output_change_rate_vs_graph_both"] = output_change_rate(
                generated[(seed, reference_label)],
                generated[(seed, label)],
            )


def output_change_rate(first: pd.DataFrame, second: pd.DataFrame) -> float:
    common = [column for column in first.columns if column in second.columns]
    if not common or len(first) != len(second):
        return 1.0
    equal = (
        first[common].fillna("<missing>").astype(str).to_numpy()
        == second[common].fillna("<missing>").astype(str).to_numpy()
    )
    return float(1.0 - equal.all(axis=1).mean())


def history_subgroup_metrics(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    coverage: pd.DataFrame | None,
    evaluation_config: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if coverage is None:
        return {
            "status": "skipped",
            "reason": "benchmark_has_no_evaluation_history_coverage",
            "coverage": {},
            "groups": {},
        }
    if not bool(settings.get("enabled", True)):
        return {
            "status": "skipped",
            "reason": "disabled",
            "coverage": {},
            "groups": {},
        }
    if len(coverage) != len(real) or len(synthetic) != len(real):
        raise ValueError(
            "History subgroup evaluation requires row-aligned real, synthetic, "
            "and coverage tables"
        )
    required = {
        "customer_history_count",
        "product_history_count",
        "history_group",
    }
    missing = sorted(required.difference(coverage.columns))
    if missing:
        raise ValueError(
            f"Evaluation history coverage is missing columns: {missing}"
        )
    customer_counts = pd.to_numeric(
        coverage["customer_history_count"], errors="coerce"
    ).fillna(0.0)
    product_counts = pd.to_numeric(
        coverage["product_history_count"], errors="coerce"
    ).fillna(0.0)
    coverage_summary = {
        "num_rows": int(len(coverage)),
        "customer_history_rate": float((customer_counts > 0).mean()),
        "product_history_rate": float((product_counts > 0).mean()),
        "any_history_rate": float(
            ((customer_counts > 0) | (product_counts > 0)).mean()
        ),
        "mean_customer_history_count": float(customer_counts.mean()),
        "mean_product_history_count": float(product_counts.mean()),
        "p90_customer_history_count": float(
            customer_counts.quantile(0.9)
        ),
        "p90_product_history_count": float(product_counts.quantile(0.9)),
    }
    groups: dict[str, Any] = {}
    min_rows = int(settings.get("min_rows", 100))
    selected_groups = [
        str(item)
        for item in settings.get("groups", ["cold", "partial", "warm"])
    ]
    for group in selected_groups:
        positions = coverage.index[
            coverage["history_group"].astype(str) == group
        ].to_numpy(dtype=int)
        if len(positions) < min_rows:
            groups[group] = {
                "status": "skipped",
                "reason": f"fewer_than_{min_rows}_rows",
                "num_rows": int(len(positions)),
                "single_table_c2st_error": None,
                "text_embedding_c2st_error": None,
            }
            continue
        group_config = copy.deepcopy(evaluation_config)
        group_config.setdefault("evaluation", {})["random_seed"] = int(seed)
        max_rows = settings.get("max_rows")
        if max_rows is not None:
            group_config["evaluation"].setdefault("c2st", {})[
                "max_rows"
            ] = int(max_rows)
            group_config["evaluation"].setdefault("text", {})[
                "max_text_rows"
            ] = int(max_rows)
        group_real = real.iloc[positions].reset_index(drop=True)
        group_synthetic = synthetic.iloc[positions].reset_index(drop=True)
        table_c2st, _ = single_table_c2st_metrics(
            group_real, group_synthetic, group_config
        )
        text_c2st = text_embedding_c2st_metrics(
            group_real,
            group_synthetic,
            group_config,
            output_dir / group,
        )
        groups[group] = {
            "status": "completed",
            "num_rows": int(len(positions)),
            "single_table_c2st_error": table_c2st.get("error"),
            "text_embedding_c2st_error": text_c2st.get("macro_error"),
        }
    return {
        "status": "completed",
        "coverage": coverage_summary,
        "groups": groups,
    }


def aggregate_results(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame.loc[frame.get("status", "") == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()
    numeric = [
        column
        for column in completed.select_dtypes(include="number").columns
        if column != "seed"
    ]
    rows: list[dict[str, Any]] = []
    for (matrix, label), group in completed.groupby(["matrix", "label"]):
        row: dict[str, Any] = {
            "matrix": matrix,
            "label": label,
            "num_seeds": int(group["seed"].nunique()),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_mean"] = (
                float(values.mean()) if len(values) else None
            )
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose_results(
    aggregate: pd.DataFrame, experiment: dict[str, Any]
) -> dict[str, Any]:
    if aggregate.empty:
        return {
            "status": "not_yet_determined",
            "hypotheses": {},
            "recommendation": {
                "status": "not_yet_determined",
                "reason": "No completed ablation results are available.",
            },
        }
    by_label = {
        str(row["label"]): row
        for _, row in aggregate.iterrows()
    }
    metric = "text_embedding_c2st_error_mean"
    o1 = value(by_label.get("O1"), metric)
    o2 = value(by_label.get("O2"), metric)
    o3 = value(by_label.get("O3"), metric)
    o4 = value(by_label.get("O4"), metric)
    o5 = value(by_label.get("O5"), metric)
    shuffled_change = value(
        by_label.get("graph_shuffled"),
        "output_change_rate_vs_graph_both_mean",
    )
    hypotheses = {
        "text_diffusion_formulation": evidence_from_level(o1),
        "structured_error_propagation": evidence_from_gap(o1, o2),
        "length_error_propagation": evidence_from_gap(o2, o3),
        "graph_context_construction": evidence_from_gap(o3, o4),
        "absence_of_graph_context": evidence_from_gap(o4, o5),
        "ignored_graph_conditioning": evidence_from_context_sensitivity(
            shuffled_change
        ),
        "implementation_errors": {
            "status": "strongly_supported",
            "evidence": (
                "The audit found unconstrained special-token sampling and "
                "incorrect aggregate loss reporting; both are now fixed."
            ),
        },
        "train_inference_mismatch": {
            "status": "not_yet_determined",
            "evidence": "Requires clean/corrupted/mixed training comparison.",
        },
        "loss_imbalance": {
            "status": "not_yet_determined",
            "evidence": "Requires logged modality losses and gradient audit runs.",
        },
        "decoding_restriction": {
            "status": (
                "not_yet_determined"
                if not any(
                    row.get("matrix") == "decoding_policy"
                    for row in by_label.values()
                )
                else "see_decoding_ablation"
            ),
            "evidence": "Compare the decoding-policy matrix in the aggregate table.",
        },
    }
    reference = experiment.get("reference_metrics") or {}
    lstm_text = reference.get("lstm_text_embedding_c2st_error")
    recommendation: dict[str, Any]
    if o1 is None:
        recommendation = {
            "status": "not_yet_determined",
            "reason": "O1 oracle-conditioned results are required.",
        }
    elif o1 >= 0.40 and lstm_text is not None and float(lstm_text) < o1:
        recommendation = {
            "option": 4,
            "name": "Use the LSTM as the primary attribute generator",
            "reason": (
                "Text remains highly distinguishable even under O1 oracle "
                "conditioning, while the fixed LSTM reference is better."
            ),
        }
    elif o1 < 0.20 and o4 is not None and o4 - o1 >= 0.20:
        recommendation = {
            "option": 5,
            "name": "Use a hybrid generator",
            "reason": (
                "Oracle text is viable, but generated conditions introduce "
                "substantial degradation."
            ),
        }
    elif o1 < 0.20:
        recommendation = {
            "option": 1,
            "name": "Continue tuning the existing text diffusion model",
            "reason": "O1 indicates that the current text denoiser can model the target under correct conditions.",
        }
    else:
        recommendation = {
            "status": "not_yet_determined",
            "reason": (
                "The O1 result is inconclusive; run corruption, loss, and "
                "decoding matrices before replacing the text branch."
            ),
        }
    return {
        "status": "results_available",
        "metric_used_for_progressive_diagnosis": metric,
        "hypotheses": hypotheses,
        "recommendation": recommendation,
    }


def evidence_from_level(level: float | None) -> dict[str, Any]:
    if level is None:
        return {"status": "not_yet_determined", "evidence": "O1 is missing."}
    if level >= 0.50:
        status = "strongly_supported"
    elif level >= 0.30:
        status = "moderately_supported"
    elif level >= 0.15:
        status = "weakly_supported"
    else:
        status = "rejected"
    return {
        "status": status,
        "evidence": f"O1 text-embedding C2ST error={level:.4f}.",
    }


def evidence_from_gap(
    before: float | None, after: float | None
) -> dict[str, Any]:
    if before is None or after is None:
        return {
            "status": "not_yet_determined",
            "evidence": "Required adjacent progressive modes are missing.",
        }
    gap = after - before
    if gap >= 0.20:
        status = "strongly_supported"
    elif gap >= 0.10:
        status = "moderately_supported"
    elif gap >= 0.03:
        status = "weakly_supported"
    else:
        status = "rejected"
    return {"status": status, "evidence": f"C2ST degradation={gap:.4f}."}


def evidence_from_context_sensitivity(
    output_change_rate: float | None,
) -> dict[str, Any]:
    if output_change_rate is None:
        return {
            "status": "not_yet_determined",
            "evidence": (
                "Run graph_both and graph_shuffled with matching seeds to test "
                "whether graph context changes generated rows."
            ),
        }
    if output_change_rate <= 0.01:
        status = "strongly_supported"
    elif output_change_rate <= 0.05:
        status = "moderately_supported"
    elif output_change_rate <= 0.15:
        status = "weakly_supported"
    else:
        status = "rejected"
    return {
        "status": status,
        "evidence": (
            "Shuffling graph histories changed "
            f"{output_change_rate:.2%} of generated rows relative to graph_both."
        ),
    }


def write_results_report(
    path: Path,
    results: pd.DataFrame,
    aggregate: pd.DataFrame,
    diagnosis: dict[str, Any],
) -> None:
    lines = [
        "# Hierarchical Diffusion Diagnostic Results",
        "",
        "Lower C2ST error is better; zero corresponds to chance-level discrimination.",
        "",
        "## Completed Runs",
        "",
        dataframe_markdown(results),
        "",
        "## Mean and Standard Deviation",
        "",
        dataframe_markdown(aggregate),
        "",
        "## Bottleneck Diagnosis",
        "",
    ]
    for name, record in (diagnosis.get("hypotheses") or {}).items():
        lines.append(
            f"- **{name}**: {record.get('status')}. {record.get('evidence', '')}"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            json.dumps(
                diagnosis.get("recommendation"), indent=2, sort_keys=True
            ),
            "",
            "Oracle modes O1-O3 are diagnostic upper bounds and are not valid generative baselines.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No completed results._"
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(
            lambda item: "" if pd.isna(item) else f"{float(item):.5g}"
        )
    try:
        return display.to_markdown(index=False)
    except ImportError:
        return "```\n" + display.to_string(index=False) + "\n```"


def load_and_verify_benchmark(path: Path) -> dict[str, Any]:
    require_file(path, "benchmark manifest")
    manifest = read_json(path)
    for name, record in (manifest.get("files") or {}).items():
        file_path = Path(record["path"])
        require_file(file_path, f"benchmark file {name}")
        actual = file_sha256(file_path)
        if actual != record["sha256"]:
            raise ValueError(
                f"Benchmark file changed after materialization: {file_path}"
            )
    return manifest


def benchmark_file(manifest: dict[str, Any], name: str) -> Path:
    try:
        path = Path(manifest["files"][name]["path"])
    except KeyError:
        raise KeyError(f"Benchmark manifest has no {name!r} file")
    require_file(path, f"benchmark {name}")
    return path


def optional_benchmark_file(
    manifest: dict[str, Any], name: str
) -> Path | None:
    record = (manifest.get("files") or {}).get(name)
    if not record:
        return None
    path = Path(record["path"])
    require_file(path, f"benchmark {name}")
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def checkpoint_parameter_count(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    total = 0
    for key in ("model_state_dict", "graph_encoder_state_dict"):
        state = checkpoint.get(key) or {}
        total += sum(
            int(value.numel())
            for value in state.values()
            if torch.is_tensor(value)
        )
    return int(total)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_json_optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def value(record: Any, key: str) -> float | None:
    if record is None:
        return None
    raw = record.get(key)
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    return float(raw)


if __name__ == "__main__":
    main()
