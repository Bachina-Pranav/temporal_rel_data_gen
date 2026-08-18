#!/usr/bin/env python3
"""Recompute structured C2ST and analyze text-length signal without training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.length_head_analysis import (  # noqa: E402
    TextDatasetSpec,
    analyze_text_dataset,
)
from evaluation.paper_metrics.c2st import (  # noqa: E402
    structured_c2st_feature_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/experiments/final_evaluation_cleanup.yaml"
COMPACT_DATASETS = {
    "amazon_toy": "amazon_final_root",
    "movielens_100k": "movielens_final_root",
    "rel_hm": "rel_hm_final_root",
}
DISPLAY_NAMES = {
    "amazon_toy": "Amazon-Toy",
    "movielens_100k": "MovieLens-100K",
    "rel_hm": "Rel-HM",
    "yelp": "Yelp",
    "retailrocket": "RetailRocket",
    "diffusion_amazon_toy": "Amazon-Toy Diffusion",
}


@dataclass
class EvaluationTarget:
    key: str
    display_name: str
    model_kind: str
    evaluation_config: str
    model_config: str | None
    real_table: str
    synthetic_table: str
    run_root: str | None
    checkpoint: str | None
    old_metrics_candidates: list[str]
    old_full_row_c2st: float | None = None

    @property
    def available(self) -> bool:
        required = [
            self.evaluation_config,
            self.real_table,
            self.synthetic_table,
        ]
        return all(Path(value).is_file() for value in required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final no-retraining C2ST cleanup and multi-dataset text-length "
            "analysis."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["inspect", "c2st", "length", "report", "all"],
        default="all",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--max-length-analysis-rows",
        type=int,
        default=None,
        help="Smoke-test cap only; omit for final analysis.",
    )
    parser.add_argument(
        "--evaluation-sample-size",
        type=int,
        default=None,
        help="Smoke-test cap only; omit for final paper-grade evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(ROOT / args.config)
    output = ROOT / matrix["output_root"]
    length_output = ROOT / matrix["length_output_root"]
    output.mkdir(parents=True, exist_ok=True)
    targets, inventory = inspect_repository(matrix)
    write_json(inventory, output / "preflight_inventory.json")
    print_preflight(inventory)
    if args.stage == "inspect":
        return
    if args.stage in {"c2st", "all"}:
        text_embedding_model = str(
            matrix.get(
                "text_embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        require_text_embedding_dependency(targets, text_embedding_model)
        run_c2st_evaluations(
            targets,
            output,
            skip_existing=args.skip_existing,
            sample_size=args.evaluation_sample_size,
            text_embedding_model=text_embedding_model,
        )
    if args.stage in {"length", "all"}:
        run_length_analysis(
            targets,
            length_output,
            matrix,
            max_rows=args.max_length_analysis_rows,
        )
    if args.stage in {"report", "all"}:
        write_c2st_reports(targets, output, matrix)
        if length_output.exists():
            write_length_reports(length_output)
    print_final_console(targets, output, length_output)


def require_text_embedding_dependency(
    targets: list[EvaluationTarget],
    model_name: str,
) -> None:
    """Fail before evaluation when a required sentence encoder is unavailable."""
    if model_name in {"dummy", "deterministic_hash", "hash"}:
        return
    if not any(target.available and target_has_text(target) for target in targets):
        return
    if importlib.util.find_spec("sentence_transformers") is None:
        raise RuntimeError(
            "Text embedding C2ST requires the 'sentence-transformers' package. "
            "Install it in the active environment with: "
            "python -m pip install sentence-transformers"
        )


def target_has_text(target: EvaluationTarget) -> bool:
    config = load_yaml(Path(target.evaluation_config))
    columns = ((config.get("table") or {}).get("columns") or {})
    return any(
        str((definition or {}).get("type")) == "text"
        for definition in columns.values()
    )


def inspect_repository(
    matrix: dict[str, Any],
) -> tuple[list[EvaluationTarget], dict[str, Any]]:
    frozen_matrix = load_yaml(ROOT / matrix["frozen_lstm"]["matrix"])
    targets: list[EvaluationTarget] = []
    missing: list[dict[str, Any]] = []
    compact_manifest_path = ROOT / matrix["frozen_lstm"]["compact_manifest"]
    compact_manifest = load_json_optional(compact_manifest_path)
    for dataset, manifest_key in COMPACT_DATASETS.items():
        definition = frozen_matrix["existing_datasets"][dataset]
        root_value = compact_manifest.get(manifest_key)
        if root_value:
            run_root = ROOT / root_value / "runs/seed_42"
        else:
            run_root = ROOT / "__missing_compact_confirmation_manifest__" / dataset
        target = target_from_run(
            dataset,
            DISPLAY_NAMES[dataset],
            run_root,
            ROOT / definition.get(
                "evaluation_config",
                default_evaluation_config(dataset),
            ),
        )
        targets.append(target)
    for dataset, definition in (frozen_matrix.get("datasets") or {}).items():
        run_root = ROOT / definition["output_root"] / "runs/seed_42"
        targets.append(
            target_from_run(
                dataset,
                str(
                    definition.get("display_name")
                    or DISPLAY_NAMES.get(dataset, dataset)
                ),
                run_root,
                ROOT / definition["evaluation_config"],
            )
        )
    targets.append(discover_diffusion_target(matrix))
    apply_historical_c2st(targets, matrix)
    for target in targets:
        if not target.available:
            absent = [
                name
                for name, value in {
                    "evaluation_config": target.evaluation_config,
                    "real_table": target.real_table,
                    "synthetic_table": target.synthetic_table,
                }.items()
                if not Path(value).is_file()
            ]
            missing.append(
                {
                    "key": target.key,
                    "display_name": target.display_name,
                    "missing": absent,
                    "synthetic_table": target.synthetic_table,
                    "checkpoint_present": bool(
                        target.checkpoint and Path(target.checkpoint).is_file()
                    ),
                }
            )
    text_bearing = []
    manifests: dict[str, Any] = {}
    for target in targets:
        if not Path(target.evaluation_config).is_file():
            continue
        evaluation = load_yaml(Path(target.evaluation_config))
        manifest = structured_c2st_feature_manifest(evaluation.get("table") or {})
        manifests[target.key] = manifest
        text_columns = schema_text_columns(evaluation)
        if target.model_kind == "lstm" and text_columns:
            text_bearing.append(
                {
                    "key": target.key,
                    "dataset": target.display_name,
                    "text_columns": text_columns,
                }
            )
    policy = matrix["feature_policy"]
    return targets, {
        "available_frozen_lstm_datasets": [
            target.display_name
            for target in targets
            if target.model_kind == "lstm" and target.available
        ],
        "available_synthetic_outputs": [
            {
                "key": target.key,
                "model_kind": target.model_kind,
                "path": target.synthetic_table,
            }
            for target in targets
            if Path(target.synthetic_table).is_file()
        ],
        "text_bearing_datasets": text_bearing,
        "c2st_feature_policy_before": policy["before"],
        "c2st_feature_policy_after": policy["after"],
        "c2st_feature_manifests": manifests,
        "missing_exact_outputs": missing,
        "required_retraining_runs": [],
        "required_resampling_runs": [
            item["key"]
            for item in missing
            if item["checkpoint_present"]
        ],
        "note": (
            "No trainer is called by this evaluation task. A missing exact "
            "synthetic table is not replaced with an older M2 output."
        ),
    }


def target_from_run(
    key: str,
    display_name: str,
    run_root: Path,
    fallback_evaluation_config: Path,
) -> EvaluationTarget:
    resolved_evaluation = run_root / "evaluation_config_resolved.yaml"
    has_resolved_evaluation = resolved_evaluation.is_file()
    evaluation_config = prefer_existing(
        resolved_evaluation,
        fallback_evaluation_config,
    )
    model_config = prefer_existing_optional(run_root / "config_resolved.yaml")
    evaluation = (
        load_yaml(evaluation_config)
        if evaluation_config.is_file()
        else {}
    )
    exact_real = run_root.parent.parent / "shared/spines/test_real.csv"
    exact_synthetic = run_root / "samples/synthetic_interactions.csv"
    if has_resolved_evaluation:
        real_table = str(
            path_from_config(evaluation.get("real_table_path"), exact_real)
        )
        synthetic_table = str(
            path_from_config(
                evaluation.get("synthetic_table_path"), exact_synthetic
            )
        )
    else:
        real_table = str(
            exact_real
            if exact_real.is_file()
            else path_from_config(evaluation.get("real_table_path"), exact_real)
        )
        synthetic_table = str(exact_synthetic)
    return EvaluationTarget(
        key=key,
        display_name=display_name,
        model_kind="lstm",
        evaluation_config=str(evaluation_config),
        model_config=str(model_config) if model_config else None,
        real_table=real_table,
        synthetic_table=synthetic_table,
        run_root=str(run_root),
        checkpoint=str(run_root / "checkpoints/best.pt"),
        old_metrics_candidates=[
            str(run_root / "evaluation/paper_grade/metrics.json"),
            str(run_root / "evaluation/paper_grade/paper_metrics.json"),
        ],
    )


def discover_diffusion_target(matrix: dict[str, Any]) -> EvaluationTarget:
    definition = matrix["diffusion_amazon_toy"]
    evaluation_path = ROOT / definition["evaluation_config"]
    evaluation = load_yaml(evaluation_path)
    candidates = [ROOT / path for path in definition["synthetic_candidates"]]
    synthetic = next((path for path in candidates if path.is_file()), candidates[0])
    checkpoint = (
        ROOT
        / "outputs/amazon-toy/conditional_tabdlm_hierarchical_v41/checkpoints/best.pt"
    )
    return EvaluationTarget(
        key="diffusion_amazon_toy",
        display_name=DISPLAY_NAMES["diffusion_amazon_toy"],
        model_kind="diffusion",
        evaluation_config=str(evaluation_path),
        model_config=str(
            ROOT
            / "configs/attribute_generation/conditional_tabdlm_amazon_toy_hierarchical_v41.yaml"
        ),
        real_table=str(path_from_config(evaluation.get("real_table_path"), Path(""))),
        synthetic_table=str(synthetic),
        run_root=str(synthetic.parent),
        checkpoint=str(checkpoint),
        old_metrics_candidates=[
            str(ROOT / path)
            for path in definition["old_metrics_candidates"]
        ],
    )


def apply_historical_c2st(
    targets: list[EvaluationTarget], matrix: dict[str, Any]
) -> None:
    compact_path = ROOT / matrix["frozen_lstm"]["compact_results"]
    compact = pd.read_csv(compact_path) if compact_path.is_file() else pd.DataFrame()
    for target in targets:
        value = None
        if target.key in COMPACT_DATASETS and not compact.empty:
            selected = compact[
                (compact.get("dataset") == target.key)
                & (compact.get("model") == "FINAL")
            ]
            if not selected.empty:
                value = finite_or_none(selected.iloc[0].get("full_row_c2st"))
        if value is None:
            for candidate in target.old_metrics_candidates:
                metrics = load_json_optional(Path(candidate))
                summary = metrics.get("paper_metrics_summary") or {}
                value = finite_or_none(summary.get("single_table_c2st_error"))
                if value is not None:
                    break
        target.old_full_row_c2st = value


def print_preflight(inventory: dict[str, Any]) -> None:
    sections = [
        (
            "AVAILABLE_FROZEN_LSTM_DATASETS",
            inventory["available_frozen_lstm_datasets"],
        ),
        (
            "AVAILABLE_SYNTHETIC_OUTPUTS",
            inventory["available_synthetic_outputs"],
        ),
        ("TEXT_BEARING_DATASETS", inventory["text_bearing_datasets"]),
        (
            "C2ST_FEATURE_POLICY_BEFORE",
            inventory["c2st_feature_policy_before"],
        ),
        (
            "C2ST_FEATURE_POLICY_AFTER",
            inventory["c2st_feature_policy_after"],
        ),
        ("REQUIRED_RETRAINING_RUNS", inventory["required_retraining_runs"]),
    ]
    for heading, value in sections:
        print(f"\n{heading}")
        if heading == "REQUIRED_RETRAINING_RUNS" and not value:
            print("ZERO")
        else:
            print(json.dumps(value, indent=2, sort_keys=True))
    if inventory["missing_exact_outputs"]:
        print("\nMISSING_EXACT_SYNTHETIC_OUTPUTS")
        print(json.dumps(inventory["missing_exact_outputs"], indent=2))


def run_c2st_evaluations(
    targets: list[EvaluationTarget],
    output: Path,
    *,
    skip_existing: bool,
    sample_size: int | None,
    text_embedding_model: str,
) -> None:
    resolved = output / "resolved_configs"
    resolved.mkdir(parents=True, exist_ok=True)
    for target in targets:
        if not target.available:
            print(f"[skip] {target.display_name}: exact artifacts unavailable")
            continue
        destination = output / "recomputed" / target.key
        metrics = destination / "metrics.json"
        if skip_existing and metrics.is_file():
            print(f"[reuse] {target.display_name}: {metrics}")
            continue
        config = load_yaml(Path(target.evaluation_config))
        config["real_table_path"] = target.real_table
        config["synthetic_table_path"] = target.synthetic_table
        config.setdefault("evaluation", {})["random_seed"] = 42
        if schema_text_columns(config):
            text_config = config["evaluation"].setdefault("text", {})
            text_config["embedding_model"] = text_embedding_model
            text_config["require_embedding_model"] = True
        if sample_size is not None:
            config["evaluation"]["sample_size"] = int(sample_size)
        resolved_path = resolved / f"{target.key}.yaml"
        write_yaml(config, resolved_path)
        command = [
            sys.executable,
            "src/scripts/evaluate_single_event_table_paper_metrics.py",
            "--config",
            str(resolved_path),
            "--real-table",
            target.real_table,
            "--synthetic-table",
            target.synthetic_table,
            "--output-dir",
            str(destination),
            "--seed",
            "42",
        ]
        print("$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def write_c2st_reports(
    targets: list[EvaluationTarget],
    output: Path,
    matrix: dict[str, Any],
) -> None:
    rows = []
    manifests: dict[str, Any] = {}
    manifest_dir = output / "c2st_feature_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for target in targets:
        metrics_path = output / "recomputed" / target.key / "metrics.json"
        metrics = load_json_optional(metrics_path)
        summary = metrics.get("paper_metrics_summary") or {}
        text = metrics.get("text_embedding_c2st") or {}
        manifest = (
            metrics.get("structured_c2st") or {}
        ).get("feature_manifest") or manifest_from_target(target)
        manifests[target.key] = manifest
        write_json(manifest, manifest_dir / f"{target.key}.json")
        row = {
            "Dataset": target.display_name,
            "Model": "Final LSTM" if target.model_kind == "lstm" else "Diffusion",
            "Artifact Status": "evaluated" if metrics else "exact output unavailable",
            "Constraint ↓": summary.get("constraint_violation_rate"),
            "FK Similarity ↑": summary.get("fk_cardinality_similarity"),
            "Shape ↓": summary.get("shape_error"),
            "Structured C2ST ↓": summary.get("structured_c2st_error"),
            "Trend ↓": summary.get("trend_error"),
            "Text C2ST ↓": summary.get("text_embedding_c2st_error"),
            "Summary Text C2ST ↓": nested_text_error(text, "summary"),
            "Review Text C2ST ↓": nested_text_error(text, "review_text"),
            "Combined Text C2ST ↓": finite_or_none(
                (text.get("combined_text_fields") or {}).get("error")
            ),
        }
        rows.append(row)
        audit_rows.append(
            {
                "Dataset": target.display_name,
                "Model": row["Model"],
                "Old full-row C2ST": target.old_full_row_c2st,
                "New structured C2ST": row["Structured C2ST ↓"],
                "Included fields": ", ".join(manifest.get("included_columns") or []),
                "Excluded fields": "; ".join(
                    f"{item['column']}: {item['reason']}"
                    for item in manifest.get("excluded_columns") or []
                ),
            }
        )
    write_json(manifests, output / "structured_c2st_feature_manifest.json")
    frame = pd.DataFrame(rows)
    lstm = frame[frame["Model"] == "Final LSTM"].copy() if not frame.empty else frame
    main_columns = [
        "Dataset",
        "Artifact Status",
        "Constraint ↓",
        "FK Similarity ↑",
        "Shape ↓",
        "Structured C2ST ↓",
        "Trend ↓",
        "Text C2ST ↓",
    ]
    write_csv_markdown(
        lstm.reindex(columns=main_columns),
        output / "final_lstm_structured_c2st_results.csv",
        output / "final_lstm_structured_c2st_results.md",
        "Final LSTM Structured-C2ST Results",
        "Structured C2ST uses generated numerical and categorical fields only. NA means not applicable or the exact frozen output was unavailable.",
    )
    text_columns = [
        "Dataset",
        "Model",
        "Artifact Status",
        "Summary Text C2ST ↓",
        "Review Text C2ST ↓",
        "Combined Text C2ST ↓",
        "Text C2ST ↓",
    ]
    text_details = frame.reindex(columns=text_columns)
    write_csv_markdown(
        text_details,
        output / "text_embedding_c2st_details.csv",
        output / "text_embedding_c2st_details.md",
        "Separate Text-Embedding C2ST Results",
        "Per-field and fused embedding discriminators are separate from Structured C2ST. Combined text is applicable only when a dataset has multiple generated text fields.",
    )
    comparison = amazon_comparison(frame)
    write_csv_markdown(
        comparison,
        output / "amazon_diffusion_vs_lstm_updated.csv",
        output / "amazon_diffusion_vs_lstm_updated.md",
        "Amazon-Toy: Final Diffusion vs Final LSTM",
        "All error metrics are lower-is-better. Structured C2ST excludes text, text length, IDs, foreign keys, and the fixed timestamp.",
    )
    audit = pd.DataFrame(audit_rows)
    policy = matrix["feature_policy"]
    body = [
        "# C2ST Feature-Policy Audit",
        "",
        "## Old policy",
        "",
        str(policy["before"]),
        "",
        "The historical featurizer admitted datetime, foreign-key, text hash, and text-length features when those columns were present in the schema.",
        "",
        "## Corrected primary policy",
        "",
        str(policy["after"]),
        "",
        "The primary metric is `structured_c2st_error`. It excludes raw text, embeddings, all text-derived length signals, IDs, foreign keys, and fixed event-spine timestamps. Missingness remains represented by the structured featurizer. The evaluator seed is fixed at 42, balanced sampling is retained, and scaling for logistic regression remains inside each CV fold.",
        "",
        "Text is evaluated separately as `text_embedding_c2st_error`; Amazon additionally reports per-field and combined embedding C2ST.",
        "",
        "## Resolved fields and old/new scores",
        "",
        markdown_table(audit),
        "",
        "Individual machine-readable manifests are in `c2st_feature_manifests/`.",
        "Missing exact artifacts are recorded in `preflight_inventory.json`; no older M2 result is substituted.",
    ]
    (output / "c2st_policy_audit.md").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )


def run_length_analysis(
    targets: list[EvaluationTarget],
    output: Path,
    matrix: dict[str, Any],
    *,
    max_rows: int | None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    accumulated = {
        "statistics": [],
        "predictive": [],
        "associations": [],
        "entity_effects": [],
        "conditional_distributions": [],
        "summary": [],
        "leakage": [],
    }
    unavailable = []
    for target in targets:
        if target.model_kind != "lstm" or not target.model_config:
            continue
        spec, reason = text_dataset_spec(target)
        if spec is None:
            if reason != "no generated free-form text":
                unavailable.append({"dataset": target.display_name, "reason": reason})
            continue
        print(
            f"[length] {spec.name}: {', '.join(spec.text_columns)}",
            flush=True,
        )
        result = analyze_text_dataset(
            spec,
            output / "datasets" / target.key,
            seed=int(matrix.get("seed", 42)),
            minimum_entity_interactions=int(
                matrix.get("minimum_entity_interactions", 5)
            ),
            max_rows=max_rows,
        )
        for key in accumulated:
            accumulated[key].extend(result[key])
    file_map = {
        "summary": "length_head_summary.csv",
        "predictive": "predictive_results.csv",
        "associations": "association_results.csv",
        "entity_effects": "entity_effects.csv",
        "conditional_distributions": "conditional_distribution_results.csv",
        "statistics": "dataset_field_statistics.csv",
    }
    for key, filename in file_map.items():
        frame = pd.DataFrame(accumulated[key])
        if frame.empty:
            frame = pd.DataFrame(columns=empty_length_columns(key))
        frame.to_csv(output / filename, index=False)
    write_json(
        {"records": accumulated["leakage"], "unavailable": unavailable},
        output / "leakage_audit.json",
    )
    write_feature_definitions(output / "feature_definitions.md")
    write_leakage_report(
        accumulated["leakage"], unavailable, output / "leakage_audit.md"
    )
    collect_figures(output)


def text_dataset_spec(
    target: EvaluationTarget,
) -> tuple[TextDatasetSpec | None, str]:
    evaluation = load_yaml(Path(target.evaluation_config))
    text_columns = schema_text_columns(evaluation)
    if not text_columns:
        return None, "no generated free-form text"
    config_path = Path(target.model_config or "")
    if not config_path.is_file():
        return None, f"missing resolved model config: {config_path}"
    model = load_yaml(config_path)
    condition = (model.get("columns") or {}).get("condition") or {}
    foreign_keys = list(condition.get("foreign_keys") or [])
    datetimes = list(condition.get("datetimes") or [])
    if len(foreign_keys) < 2 or not datetimes:
        return None, "model config does not identify two relational FKs and time"
    run_root = Path(target.run_root or "")
    shared = run_root.parent.parent / "shared/spines"
    split_paths = {
        split: shared / f"{split}_real.csv"
        for split in ("train", "validation", "test")
    }
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        return None, "missing chronological materialized splits: " + ", ".join(missing)
    manifest = structured_c2st_feature_manifest(evaluation.get("table") or {})
    return TextDatasetSpec(
        name=target.display_name,
        config_path=config_path,
        train_path=split_paths["train"],
        validation_path=split_paths["validation"],
        test_path=split_paths["test"],
        source_column=str(foreign_keys[0]),
        destination_column=str(foreign_keys[1]),
        timestamp_column=str(datetimes[0]),
        text_columns=tuple(text_columns),
        structured_columns=tuple(manifest["included_columns"]),
        table_columns=dict((evaluation.get("table") or {}).get("columns") or {}),
    ), "ok"


def write_length_reports(output: Path) -> None:
    summary_path = output / "length_head_summary.csv"
    if not summary_path.is_file():
        return
    summary = pd.read_csv(summary_path)
    if summary.empty:
        conclusion = "No complete text-bearing frozen datasets were available."
        lstm = "No recommendation can be made from missing data."
        diffusion = "No recommendation can be made from missing data."
    else:
        strong = summary[summary["Evidence_for_Conditional_Length_Head"] == "STRONG"]
        replicated_datasets = strong["dataset"].nunique()
        if replicated_datasets >= 2:
            conclusion = (
                "Conditional length signal is substantial and replicated across "
                "multiple text datasets."
            )
            lstm = (
                "Retaining an explicit conditional length head is defensible, "
                "although natural EOS decoding should remain an implementation "
                "ablation rather than an assumed improvement."
            )
            diffusion = (
                "A conditional length/mask predictor is statistically defensible "
                "in addition to being operationally useful for diffusion masking."
            )
        elif (summary["Evidence_for_Conditional_Length_Head"] == "WEAK").mean() >= 0.5:
            conclusion = (
                "Held-out context gains are mostly weak and do not replicate as a "
                "strong relational signal across datasets."
            )
            lstm = (
                "Prefer natural EOS decoding with a maximum-length cap; the data do "
                "not currently justify claiming a learned relational length head."
            )
            diffusion = (
                "Diffusion still needs a mask/length mechanism operationally, but "
                "an unconditional empirical length distribution is better supported "
                "than a claimed relational conditional head."
            )
        else:
            conclusion = (
                "Evidence is mixed or moderate: structured/time context helps in "
                "some fields, but strong relational-history signal is not replicated."
            )
            lstm = (
                "Treat the length head as optional and compare it against natural EOS "
                "decoding before changing the frozen architecture."
            )
            diffusion = (
                "Retain a length mechanism for masking, but describe conditional "
                "length modeling as tentative unless stronger replicated evidence "
                "appears."
            )
    columns = [
        "dataset",
        "text_field",
        "L0_MAE",
        "Structured_MAE",
        "Source_History_MAE",
        "Destination_History_MAE",
        "Full_MAE",
        "Full_R2",
        "Full_Spearman",
        "Source_Incremental_Signal",
        "Destination_Incremental_Signal",
        "Evidence_for_Conditional_Length_Head",
    ]
    report = [
        "# Text-Length Head Decision",
        "",
        "This is a held-out chronological data analysis; no generator was retrained and no architecture was modified.",
        "",
        "## Evidence summary",
        "",
        markdown_table(summary.reindex(columns=columns)),
        "",
        "## Cross-dataset conclusion",
        "",
        conclusion,
        "",
        "## LSTM recommendation",
        "",
        lstm,
        "",
        "## Diffusion recommendation",
        "",
        diffusion,
        "",
        "The diffusion recommendation distinguishes statistical evidence for conditional length from the separate implementation need for a finite mask or sequence extent.",
    ]
    (output / "length_head_decision.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    write_json(
        {
            "cross_dataset_conclusion": conclusion,
            "lstm_recommendation": lstm,
            "diffusion_recommendation": diffusion,
            "no_model_architecture_modified": True,
        },
        output / "length_head_decision.json",
    )


def empty_length_columns(kind: str) -> list[str]:
    common = ["dataset", "text_field"]
    columns = {
        "summary": common
        + [
            "L0_MAE",
            "Structured_MAE",
            "Source_History_MAE",
            "Destination_History_MAE",
            "Full_MAE",
            "Full_R2",
            "Full_Spearman",
            "Source_Incremental_Signal",
            "Destination_Incremental_Signal",
            "Evidence_for_Conditional_Length_Head",
        ],
        "predictive": common
        + [
            "model",
            "mae_token_length",
            "mae_log1p_token_length",
            "r2_log1p_token_length",
            "spearman",
        ],
        "associations": common + ["context", "context_type", "effect_size"],
        "entity_effects": common
        + ["entity_role", "between_entity_variance", "within_entity_variance", "icc_like"],
        "conditional_distributions": common
        + ["context", "max_pairwise_ks", "max_pairwise_wasserstein"],
        "statistics": common
        + ["length_definition", "n", "mean", "std", "median", "p95"],
    }
    return columns[kind]


def write_feature_definitions(path: Path) -> None:
    path.write_text(
        """# Text-Length Analysis Feature Definitions

## Target

`log1p(token_length)`, where `token_length` is computed by the generator's `SimpleTextTokenizer`. Word and character counts are descriptive secondary definitions.

## Predictors

- **L0 unconditional:** training-set median only.
- **L1 structured:** generated numerical/categorical attributes; no IDs or text-derived features.
- **L2 time:** normalized chronological time plus month and weekday cyclic features.
- **L3 source history:** strictly earlier source-event count, past mean length, recency, and activity rate.
- **L4 destination history:** the corresponding strictly earlier destination features.
- **L5 full:** L1 + L2 + L3 + L4.
- **History ablations:** L5 without source history and L5 without destination history.

The fixed generic predictor is XGBoost with 200 trees, depth 6, learning rate 0.05, histogram tree construction, and seed 42. If XGBoost is unavailable, the documented fallback is scikit-learn histogram gradient boosting. Predictors are fit on the chronological training split only and measured on test; validation remains untouched because no hyperparameter selection is performed.
""",
        encoding="utf-8",
    )


def write_leakage_report(
    records: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    path: Path,
) -> None:
    rows = [
        {
            "Dataset": record["dataset"],
            "Text field": record["text_field"],
            "Passed": record["passed"],
            "Checks": json.dumps(record["checks"], sort_keys=True),
        }
        for record in records
    ]
    content = [
        "# Length-Analysis Leakage Audit",
        "",
        "Histories are grouped by entity and timestamp; every event at the same timestamp receives only aggregates from strictly earlier timestamps. Raw entity IDs and current text are excluded from primary predictors.",
        "",
        markdown_table(pd.DataFrame(rows)),
    ]
    if unavailable:
        content.extend(
            ["", "## Unavailable analyses", "", "```json", json.dumps(unavailable, indent=2), "```"]
        )
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def collect_figures(output: Path) -> None:
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (output / "datasets").glob("*/figures/*.png"):
        shutil.copy2(source, destination / source.name)


def print_final_console(
    targets: list[EvaluationTarget], output: Path, length_output: Path
) -> None:
    print("\n=============================================================")
    print("C2ST EVALUATION CLEANUP")
    print("=============================================================")
    print("\nSTRUCTURED C2ST POLICY:")
    print("numerical + categorical generated attributes only")
    print("\nEXCLUDED:")
    print("text, embeddings, text lengths, IDs, FKs, fixed timestamps")
    frame_path = output / "final_lstm_structured_c2st_results.csv"
    if frame_path.is_file():
        frame = pd.read_csv(frame_path)
        print("\nLSTM RESULTS:")
        for _, row in frame.iterrows():
            print(f"{row['Dataset']}:")
            print(f"  structured C2ST = {format_value(row.get('Structured C2ST ↓'))}")
            if pd.notna(row.get("Text C2ST ↓")):
                print(f"  text C2ST = {format_value(row.get('Text C2ST ↓'))}")
    diffusion_path = output / "recomputed/diffusion_amazon_toy/metrics.json"
    diffusion = load_json_optional(diffusion_path)
    diffusion_target = next(
        (target for target in targets if target.key == "diffusion_amazon_toy"),
        None,
    )
    if diffusion:
        summary = diffusion.get("paper_metrics_summary") or {}
        print("\nDIFFUSION AMAZON:")
        print(
            "old full-row C2ST = "
            + format_value(
                diffusion_target.old_full_row_c2st if diffusion_target else None
            )
        )
        print(
            "new structured C2ST = "
            + format_value(summary.get("structured_c2st_error"))
        )
        print(
            "text C2ST = "
            + format_value(summary.get("text_embedding_c2st_error"))
        )
    print("\n=============================================================")
    print("LENGTH-HEAD DATA ANALYSIS")
    print("=============================================================")
    summary_path = length_output / "length_head_summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        for _, row in summary.iterrows():
            print(f"\n{row['dataset']} {row['text_field']}:")
            print(f"  baseline MAE = {format_value(row.get('L0_MAE'))}")
            print(f"  full-context MAE = {format_value(row.get('Full_MAE'))}")
            print(f"  R2 = {format_value(row.get('Full_R2'))}")
            print(
                "  source-history gain = "
                + format_value(row.get("Source_Incremental_Signal"))
            )
            print(
                "  destination-history gain = "
                + format_value(row.get("Destination_Incremental_Signal"))
            )
            print(
                "  evidence = "
                + str(row.get("Evidence_for_Conditional_Length_Head"))
            )
    decision = load_json_optional(length_output / "length_head_decision.json")
    if decision:
        print("\nCROSS-DATASET CONCLUSION:")
        print(decision["cross_dataset_conclusion"])
        print("\nLSTM LENGTH-HEAD RECOMMENDATION:")
        print(decision["lstm_recommendation"])
        print("\nDIFFUSION LENGTH-HEAD RECOMMENDATION:")
        print(decision["diffusion_recommendation"])
    print("\nNO MODEL ARCHITECTURE WAS MODIFIED.")
    print("=============================================================")


def amazon_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Metric", "Diffusion", "Final LSTM", "Better"])
    lstm = frame[
        (frame["Dataset"] == DISPLAY_NAMES["amazon_toy"])
        & (frame["Model"] == "Final LSTM")
    ]
    diffusion = frame[frame["Model"] == "Diffusion"]
    metrics = [
        ("Constraint Violation ↓", "Constraint ↓"),
        ("Shape Error ↓", "Shape ↓"),
        ("Structured C2ST ↓", "Structured C2ST ↓"),
        ("Trend Error ↓", "Trend ↓"),
        ("Text C2ST ↓", "Text C2ST ↓"),
    ]
    rows = []
    for label, column in metrics:
        left = finite_or_none(diffusion.iloc[0].get(column)) if not diffusion.empty else None
        right = finite_or_none(lstm.iloc[0].get(column)) if not lstm.empty else None
        if left is None or right is None:
            better = "NA"
        elif abs(left - right) <= 1e-12:
            better = "Tie"
        else:
            better = "Diffusion" if left < right else "Final LSTM"
        rows.append(
            {"Metric": label, "Diffusion": left, "Final LSTM": right, "Better": better}
        )
    return pd.DataFrame(rows)


def schema_text_columns(config: dict[str, Any]) -> list[str]:
    return [
        str(column)
        for column, field in (
            ((config.get("table") or {}).get("columns") or {}).items()
        )
        if str((field or {}).get("type", "")).lower() == "text"
        and (field or {}).get("generated", True) is not False
    ]


def nested_text_error(text: dict[str, Any], column: str) -> float | None:
    return finite_or_none(
        ((text.get("per_text_column") or {}).get(column) or {}).get("error")
    )


def manifest_from_target(target: EvaluationTarget) -> dict[str, Any]:
    path = Path(target.evaluation_config)
    if not path.is_file():
        return {}
    return structured_c2st_feature_manifest(
        (load_yaml(path).get("table") or {})
    )


def write_csv_markdown(
    frame: pd.DataFrame,
    csv_path: Path,
    markdown_path: Path,
    title: str,
    preamble: str,
) -> None:
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(
        f"# {title}\n\n{preamble}\n\n{markdown_table(frame)}\n",
        encoding="utf-8",
    )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No complete results available._"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(markdown_value(row.get(column)) for column in frame.columns)
            + " |"
        )
    return "\n".join(lines)


def markdown_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_value(value: Any) -> str:
    parsed = finite_or_none(value)
    return "NA" if parsed is None else f"{parsed:.6g}"


def finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def path_from_config(value: Any, fallback: Path) -> Path:
    if value:
        path = Path(str(value))
        return path if path.is_absolute() else ROOT / path
    return fallback


def prefer_existing(first: Path, fallback: Path) -> Path:
    return first if first.is_file() else fallback


def prefer_existing_optional(path: Path) -> Path | None:
    return path if path.is_file() else None


def default_evaluation_config(dataset: str) -> str:
    names = {
        "amazon_toy": "single_event_table_paper_metrics_amazon_toy.yaml",
        "movielens_100k": "single_event_table_paper_metrics_movielens_100k.yaml",
        "rel_hm": "single_event_table_paper_metrics_hm_10k_customers.yaml",
    }
    return "configs/evaluation/" + names[dataset]


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(value: dict[str, Any], path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def load_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
