#!/usr/bin/env python3
"""Select and evaluate a training-derived M2 support-logit correction."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.support_calibration import (  # noqa: E402
    corrected_logit_bias,
    support_calibration_metrics,
    support_probability_table,
)
from scripts.evaluate_lstm_attribute_diagnostics import numerical_metrics  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="8192")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = Path(args.experiment_root)
    run_root = experiment_root / "runs" / f"seed_{args.seed}"
    shared = experiment_root / "shared" / "spines"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_root / "checkpoints" / "best.pt"
    config = run_root / "config_resolved.yaml"
    evaluation_config = run_root / "evaluation_config_resolved.yaml"
    required = [
        checkpoint,
        config,
        evaluation_config,
        shared / "train_real.csv",
        shared / "validation_real.csv",
        shared / "validation_spine.csv",
        shared / "train_spine.csv",
        shared / "test_real.csv",
        shared / "test_spine.csv",
        shared / "history_prefix_spine.csv",
    ]
    require_files(required)
    payload = torch.load(checkpoint, map_location="cpu")
    columns = list(
        (
            payload["raw_config"]
            .get("_numerical_head_metadata", {})
            .get("columns", {})
        )
    )
    if not columns:
        raise ValueError("M2 checkpoint has no support numerical columns")
    train = pd.read_csv(shared / "train_real.csv", low_memory=False)
    validation = pd.read_csv(
        shared / "validation_real.csv",
        low_memory=False,
    )

    original_validation = (
        output_dir / "validation" / "lambda_0" / "synthetic.csv"
    )
    sample(
        config=config,
        checkpoint=checkpoint,
        spine=shared / "validation_spine.csv",
        history=shared / "train_spine.csv",
        output=original_validation,
        seed=int(args.seed),
        device=args.device,
        batch_size=str(args.batch_size),
    )
    generated_zero = pd.read_csv(
        original_validation,
        low_memory=False,
    )
    calibration_tables = {
        column: support_probability_table(
            train[column],
            validation[column],
            generated_zero[column],
        )
        for column in columns
    }
    for column, table in calibration_tables.items():
        table_path = (
            output_dir
            / "calibration"
            / f"{column}_support_probabilities.csv"
        )
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)

    rows = []
    scratch_checkpoint = (
        output_dir / "checkpoints" / "_validation_candidate.pt"
    )
    for strength in sorted(set(float(value) for value in args.lambdas)):
        label = lambda_label(strength)
        candidate_checkpoint = checkpoint
        if strength != 0.0:
            candidate_checkpoint = scratch_checkpoint
            write_biased_checkpoint(
                payload,
                calibration_tables,
                strength=strength,
                output=candidate_checkpoint,
            )
        synthetic_path = (
            output_dir / "validation" / label / "synthetic.csv"
        )
        if strength == 0.0:
            synthetic_path.parent.mkdir(parents=True, exist_ok=True)
            if synthetic_path != original_validation:
                synthetic_path.write_bytes(original_validation.read_bytes())
        else:
            sample(
                config=config,
                checkpoint=candidate_checkpoint,
                spine=shared / "validation_spine.csv",
                history=shared / "train_spine.csv",
                output=synthetic_path,
                seed=int(args.seed),
                device=args.device,
                batch_size=str(args.batch_size),
            )
        generated = pd.read_csv(synthetic_path, low_memory=False)
        row: dict[str, Any] = {
            "lambda": strength,
            "synthetic_validation": str(synthetic_path),
        }
        objective = 0.0
        for column in columns:
            numerical = numerical_metrics(
                train[column],
                validation[column],
                generated[column],
            )
            support = support_calibration_metrics(
                support_probability_table(
                    train[column],
                    validation[column],
                    generated[column],
                )
            )
            prefix = f"{column}."
            row.update(
                {
                    prefix + "ks": numerical["ks_distance"],
                    prefix
                    + "wasserstein": numerical["wasserstein_distance"],
                    prefix + "quantile_mae": numerical["quantile_mae"],
                    prefix
                    + "support_tv": support[
                        "total_variation_train_vs_generated"
                    ],
                    prefix
                    + "entropy": support["generated_entropy_nats"],
                }
            )
            scale = max(
                float(
                    pd.to_numeric(
                        validation[column],
                        errors="coerce",
                    ).std()
                ),
                1e-12,
            )
            objective += (
                float(numerical["ks_distance"])
                + float(numerical["wasserstein_distance"]) / scale
                + float(numerical["quantile_mae"]) / scale
                + float(
                    support["total_variation_train_vs_generated"]
                )
            )
        row["validation_objective"] = objective / max(len(columns), 1)
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            output_dir / "validation_grid_progress.csv",
            index=False,
        )

    grid = pd.DataFrame(rows).sort_values("lambda")
    grid.to_csv(output_dir / "validation_grid.csv", index=False)
    best_row = min(
        rows,
        key=lambda row: (
            float(row["validation_objective"]),
            float(row["lambda"]),
        ),
    )
    best_lambda = float(best_row["lambda"])
    selected_checkpoint = output_dir / "checkpoints" / "best.pt"
    write_biased_checkpoint(
        payload,
        calibration_tables,
        strength=best_lambda,
        output=selected_checkpoint,
    )
    scratch_checkpoint.unlink(missing_ok=True)
    test_output = output_dir / "test" / "synthetic_interactions.csv"
    sample(
        config=config,
        checkpoint=selected_checkpoint,
        spine=shared / "test_spine.csv",
        history=shared / "history_prefix_spine.csv",
        output=test_output,
        seed=int(args.seed),
        device=args.device,
        batch_size=str(args.batch_size),
    )
    evaluate_test(
        config=config,
        evaluation_config=evaluation_config,
        train=shared / "train_real.csv",
        real=shared / "test_real.csv",
        synthetic=test_output,
        history=shared / "history_prefix_spine.csv",
        output_dir=output_dir / "test" / "evaluation",
        seed=int(args.seed),
    )
    test_metrics = collect_test_metrics(
        output_dir / "test" / "evaluation",
        columns,
        train_path=shared / "train_real.csv",
        real_path=shared / "test_real.csv",
        synthetic_path=test_output,
    )
    selection = {
        "experiment": "M2C_posthoc_calibrated_support",
        "seed": int(args.seed),
        "lambda_grid": sorted(
            set(float(value) for value in args.lambdas)
        ),
        "best_lambda": best_lambda,
        "best_validation_objective": float(
            best_row["validation_objective"]
        ),
        "target_distribution_source": "training_split_only",
        "m2_distribution_estimation_source": (
            "generated_validation_spine; no validation target values "
            "enter delta_k"
        ),
        "lambda_selection_source": "validation_split_only",
        "test_evaluations_after_selection": 1,
        "test_labels_used_during_selection": False,
        "test_synthetic": str(test_output),
        "selected_checkpoint": str(selected_checkpoint),
        "test_metrics": test_metrics,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "m2c_posthoc_calibration_report.md").write_text(
        m2c_markdown(selection, grid),
        encoding="utf-8",
    )
    print(output_dir / "validation_grid.csv")
    print(output_dir / "selection.json")
    print(output_dir / "m2c_posthoc_calibration_report.md")
    print(output_dir / "test" / "evaluation")


def write_biased_checkpoint(
    original: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    *,
    strength: float,
    output: Path,
) -> None:
    payload = dict(original)
    payload["raw_config"] = copy.deepcopy(original["raw_config"])
    metadata = payload["raw_config"]["_numerical_head_metadata"]
    for column, table in tables.items():
        column_metadata = metadata["columns"][column]
        support = np.asarray(
            column_metadata["support_values_original"],
            dtype=float,
        )
        ordered = table.set_index("support_value").reindex(support)
        if ordered["delta_log_probability"].isna().any():
            raise ValueError(
                f"Calibration support does not match checkpoint for {column}"
            )
        global_prior = dict(
            column_metadata.get("global_prior") or {}
        )
        counts = np.asarray(
            column_metadata["support_counts"],
            dtype=float,
        )
        global_prior.update(
            {
                "enabled": bool(global_prior.get("enabled", False)),
                "probabilities": (
                    counts / counts.sum()
                ).tolist(),
                "runtime_logit_bias": corrected_logit_bias(
                    ordered.reset_index(),
                    strength=float(strength),
                ).tolist(),
                "posthoc_training_derived": True,
                "posthoc_strength": float(strength),
            }
        )
        column_metadata["global_prior"] = global_prior
    payload["raw_config"].setdefault(
        "experiment_metadata",
        {},
    )["posthoc_support_calibration"] = {
        "enabled": True,
        "strength": float(strength),
        "test_labels_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def sample(
    *,
    config: Path,
    checkpoint: Path,
    spine: Path,
    history: Path,
    output: Path,
    seed: int,
    device: str,
    batch_size: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "src/scripts/sample_lstm_joint_full_review_text_fast.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--synthetic-spine",
            str(spine),
            "--graph-history-prefix",
            str(history),
            "--output",
            str(output),
            "--num-rows",
            "all",
            "--batch-size",
            batch_size,
            "--device",
            device,
            "--seed",
            str(seed),
            "--mixed-precision",
            "--cache-graph-context",
            "--profile",
        ]
    )


def evaluate_test(
    *,
    config: Path,
    evaluation_config: Path,
    train: Path,
    real: Path,
    synthetic: Path,
    history: Path,
    output_dir: Path,
    seed: int,
) -> None:
    paper = output_dir / "paper_grade"
    run(
        [
            sys.executable,
            "src/scripts/evaluate_single_event_table_paper_metrics.py",
            "--config",
            str(evaluation_config),
            "--real-table",
            str(real),
            "--synthetic-table",
            str(synthetic),
            "--output-dir",
            str(paper),
            "--seed",
            str(seed),
        ]
    )
    run(
        [
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
            "--graph-history-prefix",
            str(history),
            "--evaluation-config",
            str(evaluation_config),
            "--output",
            str(output_dir / "attribute_diagnostics.json"),
            "--seed",
            str(seed),
        ]
    )


def collect_test_metrics(
    evaluation_dir: Path,
    numerical_columns: list[str],
    *,
    train_path: Path,
    real_path: Path,
    synthetic_path: Path,
) -> dict[str, Any]:
    paper = json.loads(
        (
            evaluation_dir / "paper_grade" / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    attribute = json.loads(
        (
            evaluation_dir / "attribute_diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    summary = paper.get("paper_metrics_summary") or {}
    output: dict[str, Any] = {
        "full_row_c2st": summary.get("single_table_c2st_error"),
        "shape_error": summary.get("shape_error"),
        "trend_error": summary.get("trend_error"),
        "temporal_event_distance": summary.get(
            "temporal_event_distance"
        ),
        "numerical_only_c2st": (
            (
                attribute.get("attribute_group_c2st") or {}
            ).get("numerical_only")
            or {}
        ).get("c2st_error_mean"),
        "numerical_attributes": {},
        "conditional_fidelity": attribute.get(
            "conditional_fidelity"
        ),
    }
    train = pd.read_csv(
        train_path,
        usecols=numerical_columns,
        low_memory=False,
    )
    real = pd.read_csv(
        real_path,
        usecols=numerical_columns,
        low_memory=False,
    )
    synthetic = pd.read_csv(
        synthetic_path,
        usecols=numerical_columns,
        low_memory=False,
    )
    for column in numerical_columns:
        metrics = (
            attribute.get("numerical_attributes") or {}
        ).get(column, {})
        support_metrics = support_calibration_metrics(
            support_probability_table(
                train[column],
                real[column],
                synthetic[column],
            )
        )
        output["numerical_attributes"][column] = {
            key: metrics.get(key)
            for key in (
                "ks_distance",
                "wasserstein_distance",
                "quantile_mae",
                "synthetic_training_support_overlap_rate",
                "support_entropy_synthetic",
            )
        }
        output["numerical_attributes"][column].update(
            {
                "support_tv_train_vs_generated": (
                    support_metrics[
                        "total_variation_train_vs_generated"
                    ]
                ),
                "support_entropy_generated": (
                    support_metrics["generated_entropy_nats"]
                ),
            }
        )
    return output


def m2c_markdown(
    selection: dict[str, Any],
    grid: pd.DataFrame,
) -> str:
    columns = [
        column
        for column in grid.columns
        if column != "synthetic_validation"
    ]
    lines = [
        "# M2C Post-hoc Support Calibration",
        "",
        "The correction target uses the training split only. Lambda is "
        "selected on validation and test is evaluated once.",
        "",
        f"- Selected lambda: {selection['best_lambda']}",
        f"- Validation objective: "
        f"{selection['best_validation_objective']:.6g}",
        f"- Test evaluations after selection: "
        f"{selection['test_evaluations_after_selection']}",
        "",
        "## Validation Grid",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in grid[columns].itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(format_markdown_value(value) for value in row)
            + " |"
        )
    lines.extend(
        [
            "",
            "## One-shot Test Metrics",
            "",
            "```json",
            json.dumps(
                selection["test_metrics"],
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required M2C files:\n- " + "\n- ".join(missing)
        )


def lambda_label(value: float) -> str:
    return "lambda_" + str(value).replace(".", "p")


if __name__ == "__main__":
    main()
