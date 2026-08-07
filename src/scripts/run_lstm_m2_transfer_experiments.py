#!/usr/bin/env python3
"""Run the controlled seed-42 M2 transfer on Amazon-toy and MovieLens-toy."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import (  # noqa: E402
    ConditionalTABDLMSchema,
    resolve_auto_review_text_config,
)


DEFAULT_EXPERIMENT = (
    "configs/experiments/lstm_m2_transfer_amazon_movielens.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--stage",
        choices=["inventory", "run", "summarize", "all"],
        default="all",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Defaults to every dataset in the experiment config.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-batch-size", default="8192")
    parser.add_argument("--minimum-free-disk-gb", type=float, default=5.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebuild-precomputed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_yaml(args.experiment_config)
    selected = selected_datasets(matrix, args.datasets)
    resolved_configs = write_m2_configs(matrix, selected)

    if args.stage in {"inventory", "all"}:
        inventory = build_inventory(matrix, selected, resolved_configs)
        output = Path(matrix["comparison_output_dir"])
        write_json(inventory, output / "dataset_inventory.json")
        print(json.dumps(inventory, indent=2, sort_keys=True), flush=True)
        if args.stage == "inventory":
            return

    if args.stage in {"run", "all"}:
        for name in selected:
            run_dataset(
                matrix,
                name,
                resolved_configs[name],
                args,
            )

    if args.stage in {"summarize", "all"} and not args.dry_run:
        summarize(matrix, selected)


def selected_datasets(
    matrix: dict[str, Any],
    requested: list[str] | None,
) -> list[str]:
    available = list((matrix.get("datasets") or {}).keys())
    selected = list(requested or available)
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(
            f"Unknown datasets {unknown}; available datasets are {available}"
        )
    return selected


def write_m2_configs(
    matrix: dict[str, Any],
    selected: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in selected:
        definition = matrix["datasets"][name]
        base = load_yaml(definition["base_config"])
        derived = derive_m2_config(
            base,
            numerical_attributes=definition["numerical_attributes"],
            numerical_head=matrix["numerical_head"],
            output_root=definition["output_root"],
            seed=int(matrix.get("seed", 42)),
            numerical_sampling_temperature=float(
                matrix.get("numerical_sampling_temperature", 1.0)
            ),
            variant_name=f"{name}_M2_global_support",
        )
        output = Path(definition["output_root"]) / "shared"
        path = output / "config_m2_global_support.yaml"
        write_yaml(derived, path)
        paths[name] = path
    return paths


def derive_m2_config(
    base: dict[str, Any],
    *,
    numerical_attributes: list[str],
    numerical_head: dict[str, Any],
    output_root: str | Path,
    seed: int,
    numerical_sampling_temperature: float,
    variant_name: str,
) -> dict[str, Any]:
    """Move only selected numeric-valued targets to the M2 support head."""

    derived = copy.deepcopy(base)
    targets = derived.setdefault("columns", {}).setdefault("target", {})
    categorical = [str(value) for value in targets.get("categorical", [])]
    numerical = [str(value) for value in targets.get("numerical", [])]
    text = [str(value) for value in targets.get("text", [])]
    selected = [str(value) for value in numerical_attributes]
    missing = sorted(
        set(selected).difference([*categorical, *numerical])
    )
    if missing:
        raise ValueError(
            f"M2 numerical attributes are not generated targets: {missing}"
        )
    targets["categorical"] = [
        value for value in categorical if value not in selected
    ]
    targets["numerical"] = list(
        dict.fromkeys([*numerical, *selected])
    )
    targets["text"] = text
    derived["numerical_heads"] = copy.deepcopy(numerical_head)
    derived["experiment_name"] = variant_name
    derived["base_experiment"] = base.get("experiment_name")
    derived.setdefault("paths", {})["output_dir"] = str(output_root)
    derived.setdefault("training", {})["seed"] = int(seed)
    derived.setdefault("sampling", {})["seed"] = int(seed)
    derived["sampling"]["numerical_temperature"] = float(
        numerical_sampling_temperature
    )
    derived.setdefault("experiment_metadata", {}).update(
        {
            "numerical_head_variant": "M2_global_support",
            "derived_from_config": base.get("experiment_name"),
            "only_numerical_head_changed": True,
            "training_support_only": True,
            "destination_conditioned_numerical_head": False,
            "global_prior_enabled": False,
            "seed": int(seed),
        }
    )
    validate_target_partition(derived)
    return derived


def validate_target_partition(config: dict[str, Any]) -> None:
    targets = config["columns"]["target"]
    values = [
        str(value)
        for role in ("categorical", "numerical", "text")
        for value in targets.get(role, [])
    ]
    duplicates = sorted(
        {value for value in values if values.count(value) > 1}
    )
    if duplicates:
        raise ValueError(
            f"Generated targets occur in multiple roles: {duplicates}"
        )


def build_inventory(
    matrix: dict[str, Any],
    selected: list[str],
    configs: dict[str, Path],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "experiment_name": matrix["experiment_name"],
        "seed": int(matrix.get("seed", 42)),
        "datasets": {},
    }
    for name in selected:
        definition = matrix["datasets"][name]
        config = resolve_auto_review_text_config(load_yaml(configs[name]))
        schema = ConditionalTABDLMSchema.from_config_dict(config)
        table = Path(config["paths"]["train_data_path"])
        spine = Path(config["paths"]["synthetic_spine_path"])
        require_file(table, f"prepared {name} interaction table")
        require_file(spine, f"fixed {name} evaluation spine")
        frame = pd.read_csv(
            table,
            usecols=lambda column: column in {
                "split",
                *schema.datetime_columns,
            },
            low_memory=False,
        )
        split_counts, split_source = inventory_split_counts(frame, schema)
        output_root = Path(definition["output_root"])
        previous_metrics = first_existing(
            definition.get("previous_metrics") or []
        )
        output["datasets"][name] = {
            "display_name": definition["display_name"],
            "dataset_path": str(table),
            "fixed_evaluation_spine": str(spine),
            "dataset_rows": int(len(frame)),
            "evaluation_spine_rows": csv_row_count(spine),
            "split_source": split_source,
            "split_row_counts": split_counts,
            "train_path": str(output_root / "shared/spines/train_real.csv"),
            "validation_path": str(
                output_root / "shared/spines/validation_real.csv"
            ),
            "test_path": str(output_root / "shared/spines/test_real.csv"),
            "generated_attributes": list(schema.target_columns),
            "numerical_attributes": list(schema.numerical_targets),
            "categorical_attributes": list(schema.categorical_targets),
            "text_attributes": list(schema.text_targets),
            "m2_config": str(configs[name]),
            "previous_lstm_metrics": previous_metrics,
            "new_output_root": str(output_root),
        }
    return output


def inventory_split_counts(
    frame: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
) -> tuple[dict[str, int], str]:
    if "split" in frame:
        aliases = {
            "train": "train",
            "training": "train",
            "valid": "validation",
            "val": "validation",
            "validation": "validation",
            "test": "test",
        }
        labels = frame["split"].astype(str).str.lower().map(aliases)
        unknown = sorted(set(frame.loc[labels.isna(), "split"].astype(str)))
        if unknown:
            raise ValueError(f"Unknown split labels: {unknown}")
        return (
            {
                label: int((labels == label).sum())
                for label in ("train", "validation", "test")
            },
            "explicit_split_column",
        )
    n = len(frame)
    train_end = int(n * 0.90)
    validation_end = int(n * 0.95)
    return (
        {
            "train": train_end,
            "validation": validation_end - train_end,
            "test": n - validation_end,
        },
        "legacy_time_aware_90_5_5",
    )


def run_dataset(
    matrix: dict[str, Any],
    name: str,
    config_path: Path,
    args: argparse.Namespace,
) -> None:
    definition = matrix["datasets"][name]
    command = [
        sys.executable,
        "src/scripts/run_lstm_multiseed_experiment.py",
        "--config",
        str(config_path),
        "--evaluation-config",
        str(definition["evaluation_config"]),
        "--output-root",
        str(definition["output_root"]),
        "--pretokenized-dir",
        str(definition["pretokenized_dir"]),
        "--neighbor-cache-dir",
        str(definition["neighbor_cache_dir"]),
        "--seeds",
        str(int(matrix.get("seed", 42))),
        "--device",
        args.device,
        "--sample-batch-size",
        str(args.sample_batch_size),
        "--minimum-free-disk-gb",
        str(args.minimum_free_disk_gb),
        "--evaluation-scope",
        "configured-spine",
        "--sampling-policy",
        str(definition.get("sampling_policy", "fast")),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    if args.rebuild_precomputed:
        command.append("--rebuild-precomputed")
    if args.dry_run:
        command.append("--dry-run")
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize(matrix: dict[str, Any], selected: list[str]) -> None:
    output_dir = Path(matrix["comparison_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    core_rows: list[dict[str, Any]] = []
    numerical_rows: list[dict[str, Any]] = []
    dataset_specific_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    seed = int(matrix.get("seed", 42))
    for name in selected:
        definition = matrix["datasets"][name]
        run_root = (
            Path(definition["output_root"])
            / "runs"
            / f"seed_{seed}"
        )
        previous_paper_path = first_existing(
            definition.get("previous_metrics") or []
        )
        previous_attribute_path = first_existing(
            definition.get("previous_attribute_diagnostics") or []
        )
        previous_legacy_path = first_existing(
            definition.get("previous_legacy_metrics") or []
        )
        previous_sampling_path = first_existing(
            definition.get("previous_sampling_metadata") or []
        )
        new_paper_path = run_root / "evaluation/paper_grade/metrics.json"
        new_attribute_path = run_root / "evaluation/attribute_diagnostics.json"
        new_legacy_path = (
            run_root
            / "evaluation/paper_grade/legacy_diagnostic_metrics.json"
        )
        new_sampling_path = (
            run_root
            / "samples/metadata/runtime_sampling_fast.json"
        )
        previous = collect_metrics(
            definition["display_name"],
            "LSTM v5.3",
            previous_paper_path,
            previous_attribute_path,
            previous_legacy_path,
            previous_sampling_path,
        )
        current = collect_metrics(
            definition["display_name"],
            "M2 global support",
            str(new_paper_path),
            str(new_attribute_path),
            str(new_legacy_path),
            str(new_sampling_path),
        )
        core_rows.extend([previous["core"], current["core"]])
        numerical_rows.extend(previous["numerical"])
        numerical_rows.extend(current["numerical"])
        dataset_specific_rows.extend(previous["dataset_specific"])
        dataset_specific_rows.extend(current["dataset_specific"])
        details[name] = comparison_decision(previous, current)

    core = pd.DataFrame(core_rows)
    numerical = pd.DataFrame(numerical_rows)
    dataset_specific = pd.DataFrame(dataset_specific_rows)
    core.to_csv(output_dir / "model_comparison.csv", index=False)
    numerical.to_csv(
        output_dir / "numerical_attribute_comparison.csv",
        index=False,
    )
    dataset_specific.to_csv(
        output_dir / "dataset_specific_metric_comparison.csv",
        index=False,
    )
    payload = {
        "experiment_name": matrix["experiment_name"],
        "seed": seed,
        "dataset_decisions": details,
        "conclusion": transfer_conclusion(details),
        "core_comparison_csv": str(output_dir / "model_comparison.csv"),
        "numerical_comparison_csv": str(
            output_dir / "numerical_attribute_comparison.csv"
        ),
        "dataset_specific_comparison_csv": str(
            output_dir / "dataset_specific_metric_comparison.csv"
        ),
    }
    write_json(payload, output_dir / "comparison.json")
    (output_dir / "report.md").write_text(
        markdown_report(core, numerical, dataset_specific, payload),
        encoding="utf-8",
    )
    for path in (
        output_dir / "model_comparison.csv",
        output_dir / "numerical_attribute_comparison.csv",
        output_dir / "dataset_specific_metric_comparison.csv",
        output_dir / "comparison.json",
        output_dir / "report.md",
    ):
        print(path)


def collect_metrics(
    dataset: str,
    model: str,
    paper_path: str | None,
    attribute_path: str | None,
    legacy_path: str | None,
    sampling_path: str | None,
) -> dict[str, Any]:
    resolved_paper_path = (
        str(paper_path)
        if paper_path and Path(paper_path).is_file()
        else None
    )
    resolved_attribute_path = (
        str(attribute_path)
        if attribute_path and Path(attribute_path).is_file()
        else None
    )
    resolved_legacy_path = (
        str(legacy_path)
        if legacy_path and Path(legacy_path).is_file()
        else None
    )
    resolved_sampling_path = (
        str(sampling_path)
        if sampling_path and Path(sampling_path).is_file()
        else None
    )
    paper = load_json(resolved_paper_path) if resolved_paper_path else {}
    attribute = (
        load_json(resolved_attribute_path)
        if resolved_attribute_path
        else {}
    )
    legacy = load_json(resolved_legacy_path) if resolved_legacy_path else {}
    sampling = (
        load_json(resolved_sampling_path)
        if resolved_sampling_path
        else {}
    )
    summary = paper.get("paper_metrics_summary") or {}
    per_column_shape = (paper.get("shape") or {}).get("per_column") or {}
    core = {
        "Dataset": dataset,
        "Model": model,
        "Shape": summary.get("shape_error"),
        "Full C2ST": summary.get("single_table_c2st_error"),
        "Text C2ST": summary.get("text_embedding_c2st_error"),
        "Trend": summary.get("trend_error"),
        "Constraint": summary.get("constraint_violation_rate"),
        "FK Similarity": summary.get("fk_cardinality_similarity"),
        "Temporal Event Distance": summary.get("temporal_event_distance"),
        "Sampling Seconds": sampling.get("total_sampling_seconds"),
        "Rows Per Second": sampling.get("rows_per_second"),
        "metrics_path": resolved_paper_path,
    }
    numerical = []
    for column, metrics in (
        attribute.get("numerical_attributes") or {}
    ).items():
        numerical.append(
            {
                "Dataset": dataset,
                "Model": model,
                "Attribute": column,
                "KS": metrics.get("ks_distance"),
                "TV": metrics.get("support_total_variation"),
                "Wasserstein": metrics.get("wasserstein_distance"),
                "Mean Error": metrics.get("mean_absolute_error"),
                "Quantile Error": metrics.get("quantile_mae"),
                "Invalid Rate": metrics.get("invalid_rate"),
                "Support Overlap": metrics.get(
                    "synthetic_training_support_overlap_rate"
                ),
                "Unique Generated": metrics.get(
                    "num_unique_synthetic"
                ),
                "Numerical C2ST": (
                    (
                        attribute.get("attribute_group_c2st") or {}
                    ).get("numerical_only")
                    or {}
                ).get("c2st_error_mean"),
                "diagnostics_path": resolved_attribute_path,
            }
        )
    if not numerical:
        rating_shape = (
            (paper.get("shape") or {}).get("per_column") or {}
        ).get("rating") or {}
        categorical_rating = (
            attribute.get("categorical_attributes") or {}
        ).get("rating") or {}
        if rating_shape or categorical_rating:
            numerical.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Attribute": "rating",
                    "KS": None,
                    "TV": categorical_rating.get(
                        "total_variation_distance",
                        rating_shape.get("shape_error"),
                    ),
                    "Wasserstein": (
                        rating_shape.get("secondary_statistics") or {}
                    ).get("ordinal_wasserstein_distance"),
                    "Mean Error": None,
                    "Quantile Error": None,
                    "Invalid Rate": categorical_rating.get(
                        "invalid_category_rate"
                    ),
                    "Support Overlap": None,
                    "Unique Generated": None,
                    "Numerical C2ST": None,
                    "diagnostics_path": resolved_attribute_path,
                }
            )
    dataset_specific = [
        {
            "Dataset": dataset,
            "Model": model,
            "Metric": metric,
            "Value": value,
            "metrics_path": resolved_legacy_path,
        }
        for metric, value in flatten_numeric_scalars(legacy).items()
    ]
    return {
        "core": core,
        "numerical": numerical,
        "dataset_specific": dataset_specific,
        "per_column_shape": per_column_shape,
    }


def comparison_decision(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    old = previous["core"]
    new = current["core"]
    metric_status = {
        metric: lower_is_better_status(old.get(metric), new.get(metric))
        for metric in ("Shape", "Full C2ST", "Text C2ST", "Trend", "Constraint")
    }
    available = [value for value in metric_status.values() if value != "not_available"]
    if not available:
        overall = "not_available"
    elif available.count("degraded") > available.count("improved"):
        overall = "degraded"
    elif available.count("improved") > available.count("degraded"):
        overall = "improved"
    else:
        overall = "matched"
    old_num = first_attribute(previous["numerical"])
    new_num = first_attribute(current["numerical"])
    numerical_status = {
        metric: lower_is_better_status(
            old_num.get(metric) if old_num else None,
            new_num.get(metric) if new_num else None,
        )
        for metric in ("KS", "TV", "Wasserstein", "Mean Error", "Quantile Error", "Invalid Rate")
    }
    numerical_attributes = {
        row.get("Attribute") for row in current["numerical"]
    }
    shared_columns = set(previous["per_column_shape"]).intersection(
        current["per_column_shape"]
    )
    text_shape_status = []
    categorical_shape_status = []
    for column in shared_columns:
        old_shape = previous["per_column_shape"][column]
        new_shape = current["per_column_shape"][column]
        status = lower_is_better_status(
            old_shape.get("shape_error"),
            new_shape.get("shape_error"),
        )
        if old_shape.get("type") == "text":
            text_shape_status.append(status)
        elif (
            old_shape.get("type") == "categorical"
            and column not in numerical_attributes
        ):
            categorical_shape_status.append(status)
    text_status = aggregate_status(
        [metric_status["Text C2ST"], *text_shape_status]
    )
    categorical_status = aggregate_status(categorical_shape_status)
    numerical_overall = aggregate_status(list(numerical_status.values()))
    return {
        "overall": overall,
        "core_metric_status": metric_status,
        "numerical_metric_status": numerical_status,
        "numerical_overall": numerical_overall,
        "text_status": text_status,
        "categorical_status": categorical_status,
        "previous_metrics_available": bool(old.get("metrics_path")),
        "new_metrics_available": bool(new.get("metrics_path")),
    }


def lower_is_better_status(old: Any, new: Any) -> str:
    if old is None or new is None:
        return "not_available"
    old_value = float(old)
    new_value = float(new)
    tolerance = max(1e-6, 0.01 * max(abs(old_value), 1e-6))
    if new_value < old_value - tolerance:
        return "improved"
    if new_value > old_value + tolerance:
        return "degraded"
    return "matched"


def aggregate_status(statuses: list[str]) -> str:
    available = [status for status in statuses if status != "not_available"]
    if not available:
        return "not_available"
    if "degraded" in available:
        return "degraded"
    if "improved" in available:
        return "improved"
    return "matched"


def transfer_conclusion(details: dict[str, Any]) -> dict[str, Any]:
    completed = [
        value for value in details.values()
        if value.get("overall") != "not_available"
    ]
    regressions = [
        name for name, value in details.items()
        if (
            value.get("overall") == "degraded"
            or value.get("numerical_overall") == "degraded"
            or value.get("text_status") == "degraded"
            or value.get("categorical_status") == "degraded"
        )
    ]
    amazon = details.get("amazon_toy") or {}
    movielens = details.get("movielens_toy") or {}
    numerical_improved = [
        name for name, value in details.items()
        if value.get("numerical_overall") == "improved"
    ]
    text_or_categorical_regressions = [
        name for name, value in details.items()
        if "degraded" in {
            value.get("text_status"),
            value.get("categorical_status"),
        }
    ]
    return {
        "both_datasets_comparable": len(completed) == len(details),
        "obvious_regressions": regressions,
        "amazon_transfer_successful": (
            amazon.get("overall") in {"improved", "matched"}
            and "amazon_toy" not in regressions
        ),
        "movielens_transfer_successful": (
            movielens.get("overall") in {"improved", "matched"}
            and "movielens_toy" not in regressions
        ),
        "numerical_generation_improved_for": numerical_improved,
        "amazon_text_status": amazon.get("text_status", "not_available"),
        "amazon_categorical_status": amazon.get(
            "categorical_status", "not_available"
        ),
        "text_or_categorical_regressions": (
            text_or_categorical_regressions
        ),
        "m2_appears_dataset_agnostic": (
            len(completed) == len(details) and not regressions
        ),
        "architecture_decision": (
            "continue_with_M2"
            if len(completed) == len(details) and not regressions
            else "inspect_comparison_before_freezing"
        ),
    }


def markdown_report(
    core: pd.DataFrame,
    numerical: pd.DataFrame,
    dataset_specific: pd.DataFrame,
    payload: dict[str, Any],
) -> str:
    lines = [
        "# M2 Global-Support Transfer Experiment",
        "",
        "Seed: 42",
        "",
        "## Core comparison",
        "",
        markdown_table(core),
        "",
        "## Numerical attribute comparison",
        "",
        markdown_table(numerical),
        "",
        "## Existing dataset-specific metrics",
        "",
        markdown_table(dataset_specific),
        "",
        "## Dataset decisions",
        "",
    ]
    for dataset, decision in payload["dataset_decisions"].items():
        lines.append(f"- {dataset}: **{decision['overall']}**")
    conclusion = payload["conclusion"]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Obvious regressions: {conclusion['obvious_regressions'] or 'none detected'}",
            f"- Dataset-agnostic transfer supported: {conclusion['m2_appears_dataset_agnostic']}",
            f"- Amazon transfer successful: {conclusion['amazon_transfer_successful']}",
            f"- MovieLens transfer successful: {conclusion['movielens_transfer_successful']}",
            f"- Numerical generation improved for: {conclusion['numerical_generation_improved_for'] or 'none'}",
            f"- Amazon text status: {conclusion['amazon_text_status']}",
            f"- Amazon categorical status: {conclusion['amazon_categorical_status']}",
            f"- Text/categorical regressions: {conclusion['text_or_categorical_regressions'] or 'none detected'}",
            f"- Decision: `{conclusion['architecture_decision']}`",
            "",
            "The conclusion is automated and provisional; inspect the per-column and text reports before freezing the head.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No completed metrics were found."
    columns = [
        column for column in frame.columns
        if not column.endswith("_path")
    ]
    display = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in display.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("N/A")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def first_attribute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0] if rows else {}


def flatten_numeric_scalars(
    payload: dict[str, Any],
    prefix: str = "",
) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_numeric_scalars(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[name] = value
    return flattened


def first_existing(paths: list[str]) -> str | None:
    for raw in paths:
        if Path(raw).is_file():
            return str(raw)
    return None


def csv_row_count(path: Path) -> int:
    return sum(
        len(chunk)
        for chunk in pd.read_csv(path, usecols=[0], chunksize=500_000)
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_yaml(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


if __name__ == "__main__":
    main()
