#!/usr/bin/env python3
"""Run Phases 1-8 post-hoc diagnostics for a multi-seed LSTM experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.posthoc_diagnostics import (  # noqa: E402
    assert_aligned_spine,
    c2st_feature_ablation_suite,
    c2st_sanity_suite,
    classifier_importance_analysis,
    conditional_support_suite,
    dataframe_fingerprint,
    oracle_ablation_suite,
    projection_ablation_suite,
    write_json,
)
from attribute_generation.conditional_tabdlm.numerical_support import (  # noqa: E402
    numerical_support_profile,
)
from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMConfig,
    ConditionalTABDLMSchema,
)


PHASES = (
    "audit",
    "sanity",
    "feature_ablation",
    "support",
    "projection",
    "oracle",
    "importance",
    "conditional",
    "diagnosis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        required=True,
        help="Root produced by run_lstm_multiseed_experiment.py.",
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument(
        "--classifier-seeds",
        nargs="+",
        type=int,
        default=[11, 23, 37, 53, 71],
    )
    parser.add_argument(
        "--output-dir",
        help="Defaults to <experiment-root>/diagnostics/posthoc_v1.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=[*PHASES, "all"],
        default=["all"],
    )
    parser.add_argument(
        "--max-c2st-rows",
        type=int,
        default=10000,
        help="Rows per class for diagnostic C2ST runs; use 0 for all rows.",
    )
    parser.add_argument("--chance-tolerance", type=float, default=0.15)
    parser.add_argument("--projection-classifier-seed", type=int, default=42)
    parser.add_argument("--stochastic-neighbors", type=int, default=8)
    parser.add_argument("--stochastic-temperature", type=float, default=1.0)
    parser.add_argument("--min-entity-rows", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-failed-c2st-controls",
        action="store_true",
        help="Continue after failed S1/S2 controls. Results are marked untrusted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else experiment_root / "diagnostics" / "posthoc_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = resolve_requested_phases(args.phases)
    max_c2st_rows = (
        None if int(args.max_c2st_rows) <= 0 else int(args.max_c2st_rows)
    )

    model_raw = load_yaml(Path(args.model_config))
    schema = ConditionalTABDLMSchema.from_config_dict(model_raw)
    model_config = ConditionalTABDLMConfig(
        raw=model_raw,
        schema=schema,
        config_path=Path(args.model_config),
    )
    evaluation_config = load_yaml(Path(args.evaluation_config))
    paths = resolve_experiment_paths(
        experiment_root,
        [int(seed) for seed in args.seeds],
    )
    require_experiment_files(paths)
    frames = load_experiment_frames(paths)
    for seed, synthetic in frames["synthetic_by_seed"].items():
        assert_aligned_spine(frames["test"], synthetic, schema)

    manifest = {
        "experiment_root": str(experiment_root),
        "output_dir": str(output_dir),
        "model_config": str(args.model_config),
        "evaluation_config": str(args.evaluation_config),
        "generator_seeds": [int(seed) for seed in args.seeds],
        "classifier_seeds": [int(seed) for seed in args.classifier_seeds],
        "max_c2st_rows_per_class": max_c2st_rows,
        "phases_requested": sorted(selected),
        "retraining_performed": False,
    }
    write_json(manifest, output_dir / "run_manifest.json")

    audit_path = output_dir / "01_current_experiment_audit.json"
    if "audit" in selected and (args.force or not audit_path.exists()):
        print("[phase 1/8] auditing current experiment", flush=True)
        audit = current_experiment_audit(
            paths,
            frames,
            model_config,
            evaluation_config,
        )
        write_json(audit, audit_path)
        write_audit_markdown(audit, output_dir / "01_current_experiment_audit.md")
    elif audit_path.exists():
        audit = load_json(audit_path)
    else:
        audit = None

    sanity_path = output_dir / "02_c2st_sanity.json"
    if "sanity" in selected and (args.force or not sanity_path.exists()):
        print("[phase 2/8] running C2ST sanity controls", flush=True)
        sanity, sanity_details = c2st_sanity_suite(
            frames["train"],
            frames["test"],
            evaluation_config,
            schema,
            classifier_seeds=args.classifier_seeds,
            max_rows=max_c2st_rows,
            chance_tolerance=float(args.chance_tolerance),
            progress_dir=output_dir / "progress",
        )
        write_json(sanity, sanity_path)
        sanity_details.to_csv(
            output_dir / "02_c2st_sanity_classifier_runs.csv",
            index=False,
        )
    elif sanity_path.exists():
        sanity = load_json(sanity_path)
    else:
        sanity = None
    if (
        sanity is not None
        and not sanity.get("chance_controls_passed", False)
        and not args.allow_failed_c2st_controls
    ):
        failure = {
            "status": "stopped",
            "reason": "S1 or S2 C2ST chance control failed",
            "next_action": (
                "Inspect 02_c2st_sanity.json before trusting feature, "
                "projection, or oracle C2ST results."
            ),
        }
        write_json(failure, output_dir / "DIAGNOSTIC_STOP.json")
        raise RuntimeError(failure["reason"])

    feature_path = output_dir / "03_c2st_feature_ablation.csv"
    if "feature_ablation" in selected and (
        args.force or not feature_path.exists()
    ):
        print("[phase 3/8] running C2ST feature ablations", flush=True)
        feature_rows, feature_details = c2st_feature_ablation_suite(
            frames["train"],
            frames["test"],
            frames["synthetic_by_seed"],
            evaluation_config,
            schema,
            classifier_seeds=args.classifier_seeds,
            max_rows=max_c2st_rows,
            progress_dir=output_dir / "progress",
        )
        feature_rows.to_csv(feature_path, index=False)
        feature_details.to_csv(
            output_dir / "03_c2st_feature_ablation_classifier_runs.csv",
            index=False,
        )
        aggregate_numeric_table(
            feature_rows,
            "feature_set",
        ).to_csv(
            output_dir / "03_c2st_feature_ablation_aggregate.csv",
            index=False,
        )
    elif feature_path.exists():
        feature_rows = pd.read_csv(feature_path)
    else:
        feature_rows = None

    support_path = output_dir / "04_numerical_support.json"
    if "support" in selected and (args.force or not support_path.exists()):
        print("[phase 4/8] profiling numerical support", flush=True)
        support = numerical_support_suite(
            frames["train"],
            frames["test"],
            frames["synthetic_by_seed"],
            schema,
        )
        write_json(support, support_path)
        write_support_histograms(
            support,
            output_dir / "04_nearest_support_histograms.csv",
        )
    elif support_path.exists():
        support = load_json(support_path)
    else:
        support = None

    projection_path = output_dir / "05_price_projection_ablation.csv"
    projected_dir = output_dir / "projected_tables"
    if "projection" in selected and (
        args.force or not projection_path.exists()
    ):
        print("[phase 5/8] running numerical support projections", flush=True)
        projection_rows, projected_tables = projection_ablation_suite(
            frames["train"],
            frames["test"],
            frames["synthetic_by_seed"],
            frames["history_prefix"],
            evaluation_config,
            model_config,
            output_dir=projected_dir,
            c2st_seed=int(args.projection_classifier_seed),
            max_c2st_rows=max_c2st_rows,
            stochastic_neighbors=int(args.stochastic_neighbors),
            stochastic_temperature=float(args.stochastic_temperature),
            min_entity_rows=int(args.min_entity_rows),
            progress_path=(
                output_dir / "progress" / "projection_ablation_progress.csv"
            ),
        )
        projection_rows.to_csv(projection_path, index=False)
        aggregate_numeric_table(
            projection_rows,
            "projection",
        ).to_csv(
            output_dir / "05_price_projection_ablation_aggregate.csv",
            index=False,
        )
    elif projection_path.exists():
        projection_rows = pd.read_csv(projection_path)
        projected_tables = load_projected_tables(
            projected_dir,
            args.seeds,
        )
    else:
        projection_rows = None
        projected_tables = None

    oracle_path = output_dir / "06_oracle_attribute_ablation.csv"
    if "oracle" in selected and (args.force or not oracle_path.exists()):
        if projection_rows is None or projected_tables is None:
            raise RuntimeError(
                "Oracle phase requires completed projection outputs."
            )
        print("[phase 6/8] running oracle attribute ablations", flush=True)
        oracle_rows, best_projection = oracle_ablation_suite(
            frames["test"],
            frames["synthetic_by_seed"],
            projected_tables,
            projection_rows,
            evaluation_config,
            schema,
            c2st_seed=int(args.projection_classifier_seed),
            max_c2st_rows=max_c2st_rows,
            progress_path=(
                output_dir / "progress" / "oracle_ablation_progress.csv"
            ),
        )
        oracle_rows.to_csv(oracle_path, index=False)
        aggregate_numeric_table(
            oracle_rows,
            "oracle_variant",
        ).to_csv(
            output_dir / "06_oracle_attribute_ablation_aggregate.csv",
            index=False,
        )
        write_json(
            {"best_support_projection": best_projection},
            output_dir / "06_oracle_selection.json",
        )
    elif oracle_path.exists():
        oracle_rows = pd.read_csv(oracle_path)
        best_projection = load_json(
            output_dir / "06_oracle_selection.json"
        )["best_support_projection"]
    else:
        oracle_rows = None
        best_projection = None

    importance_path = output_dir / "07_c2st_feature_importance.json"
    if "importance" in selected and (
        args.force or not importance_path.exists()
    ):
        print("[phase 7/8] computing classifier importance", flush=True)
        importance = {}
        for seed, synthetic in frames["synthetic_by_seed"].items():
            print(f"[importance] generator seed={seed}", flush=True)
            importance[f"seed_{seed}"] = classifier_importance_analysis(
                frames["test"],
                synthetic,
                evaluation_config,
                schema,
                seed=int(seed),
                max_rows=max_c2st_rows,
                permutation_repeats=int(args.permutation_repeats),
            )
            write_json(importance, importance_path)
    elif importance_path.exists():
        importance = load_json(importance_path)
    else:
        importance = None

    conditional_path = output_dir / "08_conditional_support.json"
    if "conditional" in selected and (
        args.force or not conditional_path.exists()
    ):
        print("[phase 8/8] computing conditional support diagnostics", flush=True)
        conditional = conditional_support_suite(
            frames["train"],
            frames["test"],
            frames["synthetic_by_seed"],
            frames["history_prefix"],
            model_config,
        )
        write_json(conditional, conditional_path)
    elif conditional_path.exists():
        conditional = load_json(conditional_path)
    else:
        conditional = None

    diagnosis_path = output_dir / "diagnosis.json"
    if "diagnosis" in selected and (args.force or not diagnosis_path.exists()):
        required_values = {
            "sanity": sanity,
            "feature_ablation": feature_rows,
            "support": support,
            "projection": projection_rows,
            "oracle": oracle_rows,
            "importance": importance,
            "conditional": conditional,
            "best_projection": best_projection,
        }
        missing = [
            key for key, value in required_values.items() if value is None
        ]
        if missing:
            raise RuntimeError(
                f"Diagnosis phase is missing prerequisite outputs: {missing}"
            )
        diagnosis = build_diagnosis(
            sanity=sanity,
            feature_rows=feature_rows,
            support=support,
            projection_rows=projection_rows,
            oracle_rows=oracle_rows,
            importance=importance,
            conditional=conditional,
            schema=schema,
            best_projection=best_projection,
        )
        write_json(diagnosis, output_dir / "diagnosis.json")
        write_diagnosis_markdown(
            diagnosis,
            output_dir / "diagnosis_report.md",
        )
    elif diagnosis_path.exists():
        diagnosis = load_json(diagnosis_path)
    else:
        diagnosis = None

    if diagnosis is not None:
        print("\nPOST-HOC DIAGNOSTIC DECISION", flush=True)
        print(diagnosis["decision_statement"], flush=True)
        print(f"Report: {output_dir / 'diagnosis_report.md'}", flush=True)
    print(f"Results: {output_dir}", flush=True)


def resolve_experiment_paths(
    root: Path,
    seeds: list[int],
) -> dict[str, Any]:
    shared = root / "shared" / "spines"
    return {
        "root": root,
        "train": shared / "train_real.csv",
        "validation": shared / "validation_real.csv",
        "test": shared / "test_real.csv",
        "test_spine": shared / "test_spine.csv",
        "history_prefix": shared / "history_prefix_spine.csv",
        "seeds": {
            seed: {
                "root": root / "runs" / f"seed_{seed}",
                "config": root
                / "runs"
                / f"seed_{seed}"
                / "config_resolved.yaml",
                "evaluation_config": root
                / "runs"
                / f"seed_{seed}"
                / "evaluation_config_resolved.yaml",
                "checkpoint": root
                / "runs"
                / f"seed_{seed}"
                / "checkpoints"
                / "best.pt",
                "synthetic": root
                / "runs"
                / f"seed_{seed}"
                / "samples"
                / "synthetic_interactions.csv",
                "metrics": root
                / "runs"
                / f"seed_{seed}"
                / "evaluation"
                / "paper_grade"
                / "metrics.json",
                "attribute_diagnostics": root
                / "runs"
                / f"seed_{seed}"
                / "evaluation"
                / "attribute_diagnostics.json",
            }
            for seed in seeds
        },
    }


def resolve_requested_phases(requested: list[str]) -> set[str]:
    if "all" in requested:
        return set(PHASES)
    dependencies = {
        "audit": set(),
        "sanity": {"audit"},
        "feature_ablation": {"audit", "sanity"},
        "support": {"audit"},
        "projection": {"audit", "sanity", "support"},
        "oracle": {"audit", "sanity", "support", "projection"},
        "importance": {"audit", "sanity"},
        "conditional": {"audit"},
        "diagnosis": set(PHASES) - {"diagnosis"},
    }
    selected = set(requested)
    changed = True
    while changed:
        changed = False
        for phase in list(selected):
            before = len(selected)
            selected.update(dependencies[phase])
            changed = changed or len(selected) != before
    return selected


def require_experiment_files(paths: dict[str, Any]) -> None:
    required = [
        paths["train"],
        paths["validation"],
        paths["test"],
        paths["test_spine"],
        paths["history_prefix"],
    ]
    for seed_paths in paths["seeds"].values():
        required.extend(
            [
                seed_paths["config"],
                seed_paths["evaluation_config"],
                seed_paths["checkpoint"],
                seed_paths["synthetic"],
                seed_paths["metrics"],
                seed_paths["attribute_diagnostics"],
            ]
        )
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "The completed multi-seed experiment is incomplete:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def load_experiment_frames(paths: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": pd.read_csv(paths["train"], low_memory=False),
        "validation": pd.read_csv(paths["validation"], low_memory=False),
        "test": pd.read_csv(paths["test"], low_memory=False),
        "test_spine": pd.read_csv(paths["test_spine"], low_memory=False),
        "history_prefix": pd.read_csv(
            paths["history_prefix"],
            low_memory=False,
        ),
        "synthetic_by_seed": {
            int(seed): pd.read_csv(seed_paths["synthetic"], low_memory=False)
            for seed, seed_paths in paths["seeds"].items()
        },
    }


def current_experiment_audit(
    paths: dict[str, Any],
    frames: dict[str, Any],
    config: ConditionalTABDLMConfig,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    schema = config.schema
    test = frames["test"]
    primary_key = evaluation_config.get("table", {}).get("primary_key")
    primary_keys = (
        [primary_key]
        if isinstance(primary_key, str)
        else list(primary_key or [])
    )
    event_columns = [
        column
        for column in [*primary_keys, *schema.condition_columns]
        if column in test
    ]
    seed_reports = {}
    spine_fingerprints = {}
    checkpoint_reports = {}
    real_spine_fingerprint = dataframe_fingerprint(test[event_columns])
    for seed, synthetic in frames["synthetic_by_seed"].items():
        seed_paths = paths["seeds"][seed]
        assert_aligned_spine(test, synthetic, schema)
        spine_fingerprint = dataframe_fingerprint(synthetic[event_columns])
        spine_fingerprints[str(seed)] = spine_fingerprint
        seed_reports[str(seed)] = {
            "evaluation_rows_real": int(len(test)),
            "evaluation_rows_synthetic": int(len(synthetic)),
            "same_row_count": bool(len(test) == len(synthetic)),
            "event_spine_aligned": True,
            "event_spine_fingerprint": spine_fingerprint,
            "synthetic_sha256": file_sha256(seed_paths["synthetic"]),
            "resolved_config_sha256": file_sha256(seed_paths["config"]),
            "evaluation_config_sha256": file_sha256(
                seed_paths["evaluation_config"]
            ),
            "checkpoint_sha256": file_sha256(seed_paths["checkpoint"]),
            "missingness": {
                column: {
                    "real_rate": float(test[column].isna().mean()),
                    "synthetic_rate": float(synthetic[column].isna().mean()),
                }
                for column in schema.target_columns
            },
        }
        checkpoint_reports[str(seed)] = inspect_checkpoint(
            seed_paths["checkpoint"],
            frames["train"],
            synthetic,
            schema,
        )
    all_spines_same = len(set(spine_fingerprints.values())) == 1
    all_spines_match_real = all(
        fingerprint == real_spine_fingerprint
        for fingerprint in spine_fingerprints.values()
    )
    evaluation_columns = (
        evaluation_config.get("table", {}).get("columns", {}) or {}
    )
    suspicious = [
        column
        for column in evaluation_columns
        if str(column).lower().startswith("unnamed:")
        or str(column).lower() in {"index", "row_index", "file_order"}
        or str(column).startswith("__")
    ]
    return {
        "status": (
            "passed"
            if all_spines_same and all_spines_match_real and not suspicious
            else "failed"
        ),
        "row_counts": {
            "train": int(len(frames["train"])),
            "validation": int(len(frames["validation"])),
            "test": int(len(test)),
        },
        "schema": {
            "event_spine_columns": list(schema.condition_columns),
            "generated_attribute_columns": list(schema.target_columns),
            "numerical_targets": list(schema.numerical_targets),
            "categorical_targets": list(schema.categorical_targets),
            "evaluation_columns": list(evaluation_columns),
            "primary_key": primary_key,
        },
        "per_seed": seed_reports,
        "checkpoints": checkpoint_reports,
        "all_seeds_use_same_heldout_event_spine": all_spines_same,
        "all_synthetic_spines_match_real_test_rows": all_spines_match_real,
        "real_test_event_spine_fingerprint": real_spine_fingerprint,
        "temporary_or_order_columns_entering_c2st": suspicious,
        "c2st": {
            "same_rows_per_class": True,
            "balanced_sample_size_definition": (
                "min(real rows, synthetic rows, configured max rows)"
            ),
            "numerical_preprocessing": (
                "NaN -> 0 plus missingness indicator; StandardScaler is fit "
                "inside each classifier CV training fold."
            ),
            "categorical_preprocessing": (
                "Shared canonicalization and stateless stable hash buckets "
                "are applied after concatenating real and synthetic rows."
            ),
            "row_order_feature_included": False,
            "primary_key_excluded": bool(primary_key),
            "direction": (
                "AUC 0.5 and C2ST error 0 are chance; "
                "C2ST error = 2 * abs(AUC - 0.5), lower is better."
            ),
            "classifiers": (
                evaluation_config.get("evaluation", {})
                .get("c2st", {})
                .get("classifiers", ["logistic_regression"])
            ),
        },
        "decoding": {
            "numerical": (
                "Checkpoint train-only mean/std inverse transform, optional "
                "log1p inverse, then configured train-range clipping."
            ),
            "categorical": (
                "Checkpoint train-derived categorical vocabulary decoded by "
                "token ID; invalid generated values are audited below."
            ),
        },
    }


def inspect_checkpoint(
    path: Path,
    train: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
) -> dict[str, Any]:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        return {"status": "failed_to_load", "reason": str(exc)}
    numerical = {}
    stored_numerical = checkpoint.get("numerical_metadata") or {}
    for column in schema.numerical_targets:
        metadata = dict(stored_numerical.get(column) or {})
        train_values = pd.to_numeric(train[column], errors="coerce").dropna()
        transformed = (
            np.log1p(np.clip(train_values.to_numpy(float), 0.0, None))
            if str(metadata.get("preprocessing", "")).startswith("log1p")
            else train_values.to_numpy(float)
        )
        generated = pd.to_numeric(synthetic[column], errors="coerce")
        expected_mean = float(np.mean(transformed))
        expected_std = float(np.std(transformed))
        numerical[column] = {
            "metadata": metadata,
            "expected_train_mean": expected_mean,
            "expected_train_std": expected_std,
            "stored_mean_matches_train": bool(
                np.isclose(
                    float(metadata.get("mean", np.nan)),
                    expected_mean,
                    rtol=1e-7,
                    atol=1e-10,
                )
            ),
            "stored_std_matches_train": bool(
                np.isclose(
                    float(metadata.get("std", np.nan)),
                    expected_std,
                    rtol=1e-7,
                    atol=1e-10,
                )
            ),
            "synthetic_nonfinite_count": int(
                (~np.isfinite(generated.to_numpy(float))).sum()
            ),
            "synthetic_outside_stored_train_range_count": int(
                (
                    (generated < float(metadata.get("min_train", -np.inf)))
                    | (
                        generated
                        > float(metadata.get("max_train", np.inf))
                    )
                ).sum()
            ),
        }
    categorical = {}
    stored_vocabs = checkpoint.get("categorical_vocabs") or {}
    for column in schema.categorical_targets:
        vocab = dict(stored_vocabs.get(column) or {})
        tokens = set((vocab.get("token_to_id") or {}).keys())
        generated = set(synthetic[column].dropna().astype(str))
        categorical[column] = {
            "vocabulary_size": int(len(tokens)),
            "invalid_generated_values": sorted(generated - tokens),
            "invalid_generated_value_count": int(len(generated - tokens)),
            "train_values_missing_from_vocabulary": sorted(
                set(train[column].dropna().astype(str)) - tokens
            ),
        }
    model_config = checkpoint.get("model_config") or {}
    report = {
        "status": "loaded",
        "checkpoint_epoch_or_step": checkpoint.get("epoch"),
        "validation_metrics": checkpoint.get("valid_metrics"),
        "numerical": numerical,
        "categorical": categorical,
        "numerical_head": {
            "distribution": "heteroscedastic Gaussian",
            "parameters_per_row": ["mean", "log_standard_deviation"],
            "loss": "Gaussian negative log likelihood",
            "sampling": (
                "mean + Normal(0,1) * predicted_std * numerical_temperature"
            ),
            "model_config": model_config,
        },
    }
    del checkpoint
    return report


def numerical_support_suite(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synthetic_by_seed: dict[int, pd.DataFrame],
    schema: ConditionalTABDLMSchema,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in schema.numerical_targets:
        output[column] = {}
        for seed, synthetic in synthetic_by_seed.items():
            output[column][f"seed_{seed}"] = numerical_support_profile(
                train[column],
                test[column],
                synthetic[column],
            )
    return output


def write_support_histograms(
    support: dict[str, Any],
    path: Path,
) -> None:
    rows = []
    for column, by_seed in support.items():
        for seed, report in by_seed.items():
            histogram = report["nearest_distance_histogram"]
            edges = histogram["edges"]
            for index in range(len(edges) - 1):
                rows.append(
                    {
                        "column": column,
                        "seed": seed,
                        "left_edge": edges[index],
                        "right_edge": edges[index + 1],
                        "real_test_count": histogram["test_counts"][index],
                        "synthetic_count": histogram["synthetic_counts"][
                            index
                        ],
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def load_projected_tables(
    root: Path,
    seeds: list[int],
) -> dict[tuple[int, str], pd.DataFrame]:
    mode_names = {
        "P0": "none",
        "P1": "global_nearest",
        "P2": "global_stochastic",
        "P3": "entity_nearest",
        "P4": "learned_bins",
    }
    output = {}
    for seed in seeds:
        for label, mode in mode_names.items():
            path = root / f"seed_{seed}" / f"{label}_{mode}.csv"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing projected table required by oracle phase: {path}"
                )
            output[(int(seed), label)] = pd.read_csv(
                path,
                low_memory=False,
            )
    return output


def aggregate_numeric_table(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"generator_seed", "classifier_seed"}
    ]
    rows = []
    for group, values in frame.groupby(group_column, sort=False):
        row: dict[str, Any] = {
            group_column: group,
            "num_generator_seeds": int(
                values["generator_seed"].nunique()
            )
            if "generator_seed" in values
            else None,
        }
        for column in numeric:
            valid = pd.to_numeric(values[column], errors="coerce").dropna()
            row[f"{column}_mean"] = (
                float(valid.mean()) if len(valid) else None
            )
            row[f"{column}_std"] = (
                float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_diagnosis(
    *,
    sanity: dict[str, Any],
    feature_rows: pd.DataFrame,
    support: dict[str, Any],
    projection_rows: pd.DataFrame,
    oracle_rows: pd.DataFrame,
    importance: dict[str, Any],
    conditional: dict[str, Any],
    schema: ConditionalTABDLMSchema,
    best_projection: str,
) -> dict[str, Any]:
    feature = grouped_metric(
        feature_rows,
        "feature_set",
        "c2st_error_mean",
    )
    projection = grouped_metric(
        projection_rows,
        "projection",
        "single_table_c2st_error",
    )
    oracle = grouped_metric(
        oracle_rows,
        "oracle_variant",
        "c2st_error",
    )
    baseline = oracle.get("O1_both_generated")
    real_price = oracle.get(
        "O2_real_numerical_generated_categorical"
    )
    real_category = oracle.get(
        "O3_generated_numerical_real_categorical"
    )
    both_real = oracle.get("O4_both_real")
    projected = oracle.get(
        "O5_projected_numerical_generated_categorical"
    )
    price_gain = improvement(baseline, real_price)
    category_gain = improvement(baseline, real_category)
    support_gain = improvement(baseline, projected)
    support_share = (
        float(support_gain / price_gain)
        if price_gain is not None
        and price_gain > 1e-12
        and support_gain is not None
        else None
    )
    identity_price_signal = increase(
        feature.get("F1_numerical_only"),
        feature.get("F4_numerical_plus_destination"),
    )
    identity_category_signal = increase(
        feature.get("F2_categorical_only"),
        feature.get("F5_categorical_plus_source"),
    )
    temporal_signal = increase(
        feature.get("F3_generated_attributes"),
        feature.get("F6_generated_attributes_plus_time"),
    )
    raw_identity_signal = increase(
        feature.get("F8_full_without_entity_ids"),
        feature.get("F7_full_transaction_row"),
    )
    support_profiles = []
    for column in schema.numerical_targets:
        for seed_report in support.get(column, {}).values():
            support_profiles.append(seed_report)
    synthetic_support_overlap = mean_nested(
        support_profiles,
        ("synthetic", "exact_training_support_overlap_rate"),
    )
    real_support_overlap = mean_nested(
        support_profiles,
        ("test", "exact_training_support_overlap_rate"),
    )
    quantized = any(
        bool(
            report.get("training_support", {})
            .get("inferred_support_kind", {})
            .get("quantized")
        )
        for report in support_profiles
    )
    source = schema.foreign_key_columns[0]
    destination = (
        schema.foreign_key_columns[1]
        if len(schema.foreign_key_columns) > 1
        else source
    )
    destination_numerical_error = conditional_metric_mean(
        conditional,
        report_family="numerical_by_entity",
        report_key=(
            f"{destination}__{schema.numerical_targets[0]}"
            if schema.numerical_targets
            else None
        ),
        value_path=(
            "entity_conditioned_errors",
            "weighted_group_mean_standardized_mae",
        ),
    )
    source_categorical_tv = conditional_metric_mean(
        conditional,
        report_family="categorical_by_entity",
        report_key=(
            f"{source}__{schema.categorical_targets[0]}"
            if schema.categorical_targets
            else None
        ),
        value_path=("weighted_entity_total_variation",),
    )
    importance_group_auc_drop = importance_group_summary(importance)
    c2st_trustworthy = bool(
        sanity.get("chance_controls_passed")
        and (both_real is None or both_real <= 0.30)
    )

    hypotheses = {
        "price_support_mismatch": classify_strength(
            support_share,
            strong=0.60,
            moderate=0.25,
            supported_when_high=True,
            fallback=(
                "moderately supported"
                if quantized
                and (synthetic_support_overlap or 0.0)
                < (real_support_overlap or 0.0)
                else "weakly supported"
            ),
        ),
        "inverse_transformation_or_precision_issue": (
            "moderately supported"
            if quantized and (synthetic_support_overlap or 0.0) < 0.50
            else "weakly supported"
        ),
        "weak_article_or_destination_conditioning": classify_strength(
            max_optional(
                identity_price_signal,
                (
                    destination_numerical_error / 2.5
                    if destination_numerical_error is not None
                    else None
                ),
            ),
            strong=0.20,
            moderate=0.08,
            supported_when_high=True,
        ),
        "weak_customer_or_source_conditioning": classify_strength(
            max_optional(
                identity_category_signal,
                source_categorical_tv,
            ),
            strong=0.20,
            moderate=0.08,
            supported_when_high=True,
        ),
        "weak_temporal_conditioning": classify_strength(
            temporal_signal,
            strong=0.15,
            moderate=0.05,
            supported_when_high=True,
        ),
        "categorical_head_underfitting": classify_strength(
            category_gain,
            strong=0.20,
            moderate=0.08,
            supported_when_high=True,
        ),
        "numerical_head_underfitting": classify_strength(
            price_gain,
            strong=0.20,
            moderate=0.08,
            supported_when_high=True,
        ),
        "c2st_leakage_or_artifact": (
            "rejected" if c2st_trustworthy else "strongly supported"
        ),
        "raw_identity_hash_signal": classify_strength(
            raw_identity_signal,
            strong=0.20,
            moderate=0.08,
            supported_when_high=True,
        ),
        "joint_dependency_failure": joint_dependency_status(
            baseline,
            real_price,
            real_category,
            both_real,
        ),
        "model_capacity_problem": "not determined",
    }
    if not c2st_trustworthy:
        recommendation = (
            "No model modification. Repair and revalidate C2ST first."
        )
    elif (
        price_gain is not None
        and price_gain >= 0.08
        and support_share is not None
        and support_share >= 0.60
    ):
        recommendation = (
            "Option A: schema-derived discrete-support numerical head."
        )
    elif (
        price_gain is not None
        and price_gain >= max(category_gain or 0.0, 0.08)
        and (support_share is None or support_share < 0.60)
    ):
        recommendation = (
            "Option C: strengthen the hierarchical destination-conditioned "
            "numerical head."
        )
    elif category_gain is not None and category_gain >= 0.08:
        recommendation = (
            "Option D: strengthen the hierarchical source/destination/time "
            "categorical head."
        )
    elif hypotheses["joint_dependency_failure"] in {
        "strongly supported",
        "moderately supported",
    }:
        recommendation = (
            "Option E: add a data-derived autoregressive dependency between "
            "generated attributes."
        )
    else:
        recommendation = (
            "No architecture change yet; the measured post-hoc effects do not "
            "justify one of Options A-E."
        )
    decisions = {
        "price_support_explains_most_of_c2st": bool(
            support_share is not None and support_share >= 0.60
        ),
        "entity_conditioning_explains_most_of_c2st": bool(
            max(
                identity_price_signal or 0.0,
                identity_category_signal or 0.0,
                (
                    destination_numerical_error / 2.5
                    if destination_numerical_error is not None
                    else 0.0
                ),
                source_categorical_tv or 0.0,
            )
            >= 0.20
        ),
        "sales_channel_is_major_bottleneck": bool(
            (category_gain is not None and category_gain >= 0.08)
            or (
                source_categorical_tv is not None
                and source_categorical_tv >= 0.20
            )
        ),
        "c2st_implementation_is_trustworthy": c2st_trustworthy,
        "single_model_modification_to_attempt_next": recommendation,
    }
    statement = (
        f"Price support explains most of C2ST: "
        f"{yes_no(decisions['price_support_explains_most_of_c2st'])}. "
        f"Entity conditioning explains most: "
        f"{yes_no(decisions['entity_conditioning_explains_most_of_c2st'])}. "
        f"Sales channel is a major bottleneck: "
        f"{yes_no(decisions['sales_channel_is_major_bottleneck'])}. "
        f"C2ST is trustworthy: "
        f"{yes_no(decisions['c2st_implementation_is_trustworthy'])}. "
        f"Next modification: {recommendation}"
    )
    return {
        "status": "completed",
        "retraining_performed": False,
        "best_support_projection": best_projection,
        "evidence": {
            "feature_ablation_mean_c2st_error": feature,
            "projection_mean_c2st_error": projection,
            "oracle_mean_c2st_error": oracle,
            "real_price_oracle_improvement": price_gain,
            "real_categorical_oracle_improvement": category_gain,
            "support_projection_improvement": support_gain,
            "support_share_of_price_oracle_gain": support_share,
            "destination_identity_increment_over_numerical": (
                identity_price_signal
            ),
            "source_identity_increment_over_categorical": (
                identity_category_signal
            ),
            "time_increment_over_generated_attributes": temporal_signal,
            "raw_identity_increment_over_full_without_ids": (
                raw_identity_signal
            ),
            "synthetic_exact_training_support_overlap": (
                synthetic_support_overlap
            ),
            "real_test_exact_training_support_overlap": (
                real_support_overlap
            ),
            "training_support_inferred_quantized": quantized,
            "destination_conditioned_numerical_standardized_mae": (
                destination_numerical_error
            ),
            "source_conditioned_categorical_total_variation": (
                source_categorical_tv
            ),
            "c2st_importance_group_auc_drop": importance_group_auc_drop,
        },
        "ranked_hypotheses": [
            {"hypothesis": key, "classification": value}
            for key, value in sorted(
                hypotheses.items(),
                key=lambda item: hypothesis_rank(item[1]),
            )
        ],
        "decisions": decisions,
        "decision_statement": statement,
        "recommendation_is_provisional": True,
        "importance_report_available": bool(importance),
        "conditional_report_available": bool(conditional),
    }


def grouped_metric(
    frame: pd.DataFrame,
    group_column: str,
    metric: str,
) -> dict[str, float | None]:
    return {
        str(group): (
            float(pd.to_numeric(values[metric], errors="coerce").mean())
            if pd.to_numeric(values[metric], errors="coerce").notna().any()
            else None
        )
        for group, values in frame.groupby(group_column, sort=False)
    }


def improvement(
    baseline: float | None,
    candidate: float | None,
) -> float | None:
    if baseline is None or candidate is None:
        return None
    return float(baseline - candidate)


def increase(
    baseline: float | None,
    augmented: float | None,
) -> float | None:
    if baseline is None or augmented is None:
        return None
    return float(augmented - baseline)


def mean_nested(
    records: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> float | None:
    values = []
    for record in records:
        value: Any = record
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def conditional_metric_mean(
    conditional: dict[str, Any],
    *,
    report_family: str,
    report_key: str | None,
    value_path: tuple[str, ...],
) -> float | None:
    if report_key is None:
        return None
    values = []
    by_seed = conditional.get(report_family, {}) or {}
    for seed_report in by_seed.values():
        value: Any = seed_report.get(report_key)
        for key in value_path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def importance_group_summary(
    importance: dict[str, Any],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for report in importance.values():
        for row in report.get("feature_group_ablation", []) or []:
            grouped.setdefault(str(row["group"]), []).append(
                float(row["auc_drop"])
            )
    return {
        group: float(np.mean(values))
        for group, values in grouped.items()
        if values
    }


def max_optional(*values: float | None) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return max(valid) if valid else None


def classify_strength(
    value: float | None,
    *,
    strong: float,
    moderate: float,
    supported_when_high: bool,
    fallback: str = "weakly supported",
) -> str:
    if value is None:
        return "not determined"
    score = float(value) if supported_when_high else -float(value)
    if score >= strong:
        return "strongly supported"
    if score >= moderate:
        return "moderately supported"
    return fallback


def joint_dependency_status(
    baseline: float | None,
    real_numerical: float | None,
    real_categorical: float | None,
    both_real: float | None,
) -> str:
    if None in {baseline, real_numerical, real_categorical, both_real}:
        return "not determined"
    total = float(baseline - both_real)
    best_single = max(
        float(baseline - real_numerical),
        float(baseline - real_categorical),
    )
    residual = total - best_single
    if residual >= 0.20:
        return "strongly supported"
    if residual >= 0.08:
        return "moderately supported"
    return "weakly supported"


def hypothesis_rank(status: str) -> int:
    return {
        "strongly supported": 0,
        "moderately supported": 1,
        "weakly supported": 2,
        "not determined": 3,
        "rejected": 4,
    }.get(status, 5)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def write_audit_markdown(report: dict[str, Any], path: Path) -> None:
    counts = report["row_counts"]
    lines = [
        "# Current experiment audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Train rows: {counts['train']:,}",
        f"- Validation rows: {counts['validation']:,}",
        f"- Test rows: {counts['test']:,}",
        (
            "- Same held-out event spine across seeds: "
            f"{report['all_seeds_use_same_heldout_event_spine']}"
        ),
        (
            "- Temporary/order columns entering C2ST: "
            f"{report['temporary_or_order_columns_entering_c2st']}"
        ),
        "",
        "Numerical values are decoded with checkpoint train-only transform "
        "metadata and clipped according to configuration. Categorical values "
        "are decoded through the checkpoint vocabulary.",
        "",
        "C2ST uses balanced classes. Its StandardScaler is fitted inside each "
        "classifier CV fold; categorical hashing is a shared stateless transform. "
        "Chance is AUC 0.5 and error 0.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnosis_markdown(
    diagnosis: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Relational attribute post-hoc diagnosis",
        "",
        "## Decision",
        "",
        diagnosis["decision_statement"],
        "",
        "## Ranked hypotheses",
        "",
        "| Hypothesis | Classification |",
        "|---|---|",
    ]
    for row in diagnosis["ranked_hypotheses"]:
        lines.append(
            f"| {row['hypothesis'].replace('_', ' ')} | "
            f"{row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "```json",
            json.dumps(diagnosis["evidence"], indent=2, sort_keys=True),
            "```",
            "",
            "No retraining or architecture modification was performed in this "
            "diagnostic pass.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        chunk = handle.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(1024 * 1024)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
