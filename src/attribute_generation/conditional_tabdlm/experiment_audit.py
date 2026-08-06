"""Strict, schema-driven audit for temporal interaction attribute experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import ConditionalTABDLMConfig


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "validation",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def audit_interaction_experiment(
    config: ConditionalTABDLMConfig,
    evaluation_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit one prepared event table before training or evaluation."""

    table_path = config.train_data_path
    frame = pd.read_csv(table_path, low_memory=False)
    table_cfg = dict(evaluation_config.get("table") or {})
    columns_cfg = dict(table_cfg.get("columns") or {})
    primary_key = table_cfg.get("primary_key")
    primary_keys = [primary_key] if isinstance(primary_key, str) else list(primary_key or [])
    timestamp = config.schema.datetime_columns[0]
    targets = list(config.schema.target_columns)
    conditions = list(config.schema.condition_columns)
    errors: list[str] = []
    warnings: list[str] = []

    required = list(dict.fromkeys([*primary_keys, *conditions, *targets, "split"]))
    missing_required = [column for column in required if column not in frame.columns]
    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")

    role_rows = schema_role_rows(frame, config, table_cfg)
    accidental_index_columns = [
        column
        for column in frame.columns
        if str(column).strip().lower() in {"index", "level_0"}
        or str(column).strip().lower().startswith("unnamed:")
    ]
    if accidental_index_columns:
        errors.append(f"Accidental index columns found: {accidental_index_columns}")

    column_profiles = {
        column: column_profile(frame[column])
        for column in frame.columns
    }
    for column in targets:
        if column not in frame:
            continue
        profile = column_profiles[column]
        if profile["all_missing"]:
            errors.append(f"Generated attribute {column!r} has no valid observations")
        elif profile["constant"]:
            errors.append(f"Generated attribute {column!r} is constant and must be excluded")

    key_report = primary_key_report(frame, primary_keys)
    if not key_report["valid"]:
        errors.extend(key_report["errors"])
    fk_report = foreign_key_report(frame, columns_cfg)
    errors.extend(fk_report.pop("errors"))

    parsed_time = (
        pd.to_datetime(frame[timestamp], errors="coerce", utc=True)
        if timestamp in frame
        else pd.Series(pd.NaT, index=frame.index)
    )
    timestamp_parse_error_rate = float(parsed_time.isna().mean()) if len(frame) else 0.0
    if timestamp_parse_error_rate:
        errors.append(
            f"Timestamp {timestamp!r} has parse error rate {timestamp_parse_error_rate:.6g}"
        )
    split_report = split_integrity_report(
        frame,
        timestamp,
        parsed_time,
        primary_keys,
    )
    errors.extend(split_report.pop("errors"))
    warnings.extend(split_report.pop("warnings"))

    categorical = categorical_target_report(frame, config)
    numerical = numerical_target_report(frame, config)
    errors.extend(numerical.pop("errors"))
    history = history_coverage_report(frame, config)
    leakage = leakage_report(config)
    errors.extend(leakage.pop("errors"))

    duplicate_columns = [
        column
        for column in [*conditions, *targets]
        if column in frame.columns
    ]
    duplicate_count = (
        int(frame.duplicated(subset=duplicate_columns, keep=False).sum())
        if duplicate_columns
        else 0
    )
    previous_outputs = inspect_previous_outputs(config.output_dir)
    report = {
        "status": "passed" if not errors else "failed",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "dataset_name": config.raw.get("dataset_name"),
        "paths": {
            "interaction_table": str(table_path),
            "configured_output_dir": str(config.output_dir),
            "model_config": str(config.config_path) if config.config_path else None,
        },
        "fingerprints": {
            "interaction_table_sha256": file_sha256(table_path),
            "model_config_sha256": (
                file_sha256(config.config_path)
                if config.config_path and config.config_path.exists()
                else None
            ),
            "git_commit": git_revision(),
        },
        "scale": {
            "rows": int(len(frame)),
            "unique_entities": {
                column: int(frame[column].nunique(dropna=True))
                for column in config.schema.foreign_key_columns
                if column in frame
            },
            "timestamp_min": iso_or_none(parsed_time.min()),
            "timestamp_max": iso_or_none(parsed_time.max()),
        },
        "schema": {
            "primary_keys": primary_keys,
            "event_spine_columns": conditions,
            "generated_attributes": targets,
            "text_targets": list(config.schema.text_targets),
            "accidental_index_columns": accidental_index_columns,
        },
        "column_profiles": column_profiles,
        "primary_key_integrity": key_report,
        "foreign_key_integrity": fk_report,
        "timestamp_parse_error_rate": timestamp_parse_error_rate,
        "split_integrity": split_report,
        "categorical_targets": categorical,
        "numerical_targets": numerical,
        "history_coverage": history,
        "target_leakage": leakage,
        "duplicate_events": {
            "columns": duplicate_columns,
            "duplicate_row_count_including_all_copies": duplicate_count,
            "duplicate_rate": float(duplicate_count / len(frame)) if len(frame) else 0.0,
        },
        "previous_outputs": previous_outputs,
    }
    return report, pd.DataFrame(role_rows)


def schema_role_rows(
    frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    table_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = dict((config.raw.get("schema") or {}).get("fields") or {})
    primary_key = table_cfg.get("primary_key")
    primary_keys = {primary_key} if isinstance(primary_key, str) else set(primary_key or [])
    rows = []
    for column in frame.columns:
        field = dict(fields.get(column) or {})
        if column in primary_keys:
            role = "primary_key"
        elif column in config.schema.foreign_key_columns:
            role = field.get("role", "foreign_key")
        elif column in config.schema.datetime_columns:
            role = field.get("role", "timestamp")
        elif column in config.schema.target_columns:
            role = "generated_attribute"
        elif column == "split":
            role = "split"
        else:
            role = field.get("role", "unused")
        generated = column in config.schema.target_columns
        reason = (
            "schema-declared generated target"
            if generated
            else {
                "primary_key": "stable event identifier",
                "source_foreign_key": "event-spine foreign key",
                "destination_foreign_key": "event-spine foreign key",
                "foreign_key": "event-spine foreign key",
                "timestamp": "event-spine timestamp",
                "split": "administrative split label",
            }.get(str(role), "not selected by the model schema")
        )
        rows.append(
            {
                "column": str(column),
                "table": str(table_cfg.get("name", "interactions")),
                "data_type": str(frame[column].dtype),
                "role": str(role),
                "generated": bool(generated),
                "reason": reason,
            }
        )
    return rows


def column_profile(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    return {
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "missing_count": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()) if len(series) else 0.0,
        "num_unique_non_null": int(non_null.nunique(dropna=True)),
        "all_missing": bool(non_null.empty),
        "constant": bool(len(non_null) > 0 and non_null.nunique(dropna=True) <= 1),
    }


def primary_key_report(
    frame: pd.DataFrame,
    primary_keys: list[str],
) -> dict[str, Any]:
    errors = []
    if not primary_keys:
        return {
            "valid": False,
            "errors": ["No primary key configured for the interaction table"],
        }
    missing = [column for column in primary_keys if column not in frame]
    if missing:
        return {
            "valid": False,
            "errors": [f"Missing primary-key columns: {missing}"],
        }
    null_count = int(frame[primary_keys].isna().any(axis=1).sum())
    duplicate_count = int(frame.duplicated(subset=primary_keys, keep=False).sum())
    if null_count:
        errors.append(f"Primary key has {null_count} null rows")
    if duplicate_count:
        errors.append(f"Primary key has {duplicate_count} duplicate rows")
    return {
        "valid": not errors,
        "columns": primary_keys,
        "null_row_count": null_count,
        "duplicate_row_count": duplicate_count,
        "errors": errors,
    }


def foreign_key_report(
    frame: pd.DataFrame,
    columns_cfg: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    errors: list[str] = []
    for column, cfg in columns_cfg.items():
        if str((cfg or {}).get("type", "")).lower() != "foreign_key":
            continue
        parent_path = Path(str((cfg or {}).get("parent_table_path", "")))
        parent_column = str(((cfg or {}).get("references") or {}).get("column", column))
        if column not in frame:
            errors.append(f"Missing foreign-key column {column!r}")
            continue
        if not parent_path.exists():
            errors.append(f"Missing parent table for {column!r}: {parent_path}")
            continue
        parent = pd.read_csv(parent_path, usecols=[parent_column], low_memory=False)
        valid = set(parent[parent_column].dropna().astype(str))
        child = frame[column].astype(str)
        invalid = ~child.isin(valid)
        report[column] = {
            "parent_table": str(parent_path),
            "parent_column": parent_column,
            "parent_rows": int(len(parent)),
            "child_non_null_rows": int(frame[column].notna().sum()),
            "invalid_count": int(invalid.sum()),
            "valid_rate": float((~invalid).mean()) if len(frame) else 1.0,
        }
        if invalid.any():
            errors.append(
                f"Foreign key {column!r} has {int(invalid.sum())} values absent from its parent"
            )
    report["errors"] = errors
    return report


def split_integrity_report(
    frame: pd.DataFrame,
    timestamp_column: str,
    parsed_time: pd.Series,
    primary_keys: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if "split" not in frame:
        return {
            "valid": False,
            "errors": ["Missing explicit split column"],
            "warnings": [],
        }
    normalized = frame["split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
    unknown = sorted(set(frame.loc[normalized.isna(), "split"].astype(str)))
    if unknown:
        errors.append(f"Unknown split labels: {unknown}")
    split_frames = {
        name: frame.loc[normalized == name]
        for name in ("train", "validation", "test")
    }
    for name, split in split_frames.items():
        if split.empty:
            errors.append(f"Split {name!r} is empty")
    bounds: dict[str, Any] = {}
    for name, split in split_frames.items():
        times = parsed_time.loc[split.index]
        bounds[name] = {
            "rows": int(len(split)),
            "timestamp_min": iso_or_none(times.min()),
            "timestamp_max": iso_or_none(times.max()),
        }
    if all(not split.empty for split in split_frames.values()):
        train_max = parsed_time.loc[split_frames["train"].index].max()
        valid_min = parsed_time.loc[split_frames["validation"].index].min()
        valid_max = parsed_time.loc[split_frames["validation"].index].max()
        test_min = parsed_time.loc[split_frames["test"].index].min()
        if train_max > valid_min or valid_max > test_min:
            errors.append("Train/validation/test timestamps are not chronologically separated")
        if train_max == valid_min or valid_max == test_min:
            warnings.append(
                "Adjacent splits share a boundary timestamp; stable event ordering is required"
            )
    overlap: dict[str, int] = {}
    if primary_keys and all(column in frame for column in primary_keys):
        key_sets = {
            name: set(map(tuple, split.loc[:, primary_keys].astype(str).to_numpy()))
            for name, split in split_frames.items()
        }
        for left, right in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            count = len(key_sets[left].intersection(key_sets[right]))
            overlap[f"{left}_vs_{right}"] = int(count)
            if count:
                errors.append(f"Primary-key overlap between {left} and {right}: {count}")
    sorted_frame = frame.assign(_audit_time=parsed_time)
    sort_columns = ["_audit_time", *primary_keys]
    expected_order = sorted_frame.sort_values(sort_columns, kind="mergesort").index.to_numpy()
    chronological_ordered = bool(np.array_equal(expected_order, frame.index.to_numpy()))
    if not chronological_ordered:
        warnings.append(
            "Source rows are not globally sorted by timestamp and primary key; split membership remains authoritative"
        )
    return {
        "valid": not errors,
        "split_source": "explicit_split_column",
        "bounds": bounds,
        "primary_key_overlap": overlap,
        "globally_chronological_row_order": chronological_ordered,
        "timestamp_column": timestamp_column,
        "errors": errors,
        "warnings": warnings,
    }


def categorical_target_report(
    frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
) -> dict[str, Any]:
    labels = frame["split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
    out = {}
    for column in config.schema.categorical_targets:
        if column not in frame:
            continue
        domains = {
            split: sorted(frame.loc[labels == split, column].dropna().astype(str).unique())
            for split in ("train", "validation", "test")
        }
        train = set(domains["train"])
        out[column] = {
            "domains": domains,
            "validation_unseen_in_train": sorted(set(domains["validation"]) - train),
            "test_unseen_in_train": sorted(set(domains["test"]) - train),
        }
    return out


def numerical_target_report(
    frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    errors: list[str] = []
    labels = frame["split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
    for column in config.schema.numerical_targets:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values)
        train = values.loc[labels == "train"].dropna()
        if invalid.any():
            errors.append(f"Numerical target {column!r} has {int(invalid.sum())} invalid values")
        if train.empty:
            errors.append(f"Numerical target {column!r} has no valid training observations")
        out[column] = {
            "invalid_count": int(invalid.sum()),
            "min": float(values.min()) if values.notna().any() else None,
            "max": float(values.max()) if values.notna().any() else None,
            "mean": float(values.mean()) if values.notna().any() else None,
            "std": float(values.std()) if values.notna().any() else None,
            "train_min": float(train.min()) if len(train) else None,
            "train_max": float(train.max()) if len(train) else None,
        }
    out["errors"] = errors
    return out


def history_coverage_report(
    frame: pd.DataFrame,
    config: ConditionalTABDLMConfig,
) -> dict[str, Any]:
    timestamp = config.schema.datetime_columns[0]
    labels = frame["split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
    test_mask = labels == "test"
    if not test_mask.any():
        return {"status": "unavailable", "reason": "test split is empty"}
    times = pd.to_datetime(frame[timestamp], errors="coerce", utc=True)
    counts: dict[str, np.ndarray] = {}
    per_entity: dict[str, Any] = {}
    for column in config.schema.foreign_key_columns:
        prior = strict_prior_counts(frame[column], times)
        test_counts = prior[test_mask.to_numpy()]
        counts[column] = test_counts
        per_entity[column] = history_count_summary(test_counts)
    first = counts[config.schema.foreign_key_columns[0]]
    second = (
        counts[config.schema.foreign_key_columns[1]]
        if len(config.schema.foreign_key_columns) > 1
        else np.zeros_like(first)
    )
    cold = (first == 0) & (second == 0)
    partial = (first > 0) ^ (second > 0)
    warm = (first > 0) & (second > 0)
    return {
        "status": "completed",
        "evaluated_split": "test",
        "num_rows": int(test_mask.sum()),
        "per_entity_role": per_entity,
        "cold_events": int(cold.sum()),
        "partial_history_events": int(partial.sum()),
        "warm_events": int(warm.sum()),
        "any_history_coverage_rate": float((~cold).mean()),
        "warm_history_coverage_rate": float(warm.mean()),
        "definition": "strictly earlier timestamps only; same-timestamp events are excluded",
    }


def strict_prior_counts(entity: pd.Series, timestamps: pd.Series) -> np.ndarray:
    temp = pd.DataFrame(
        {
            "_entity": entity.astype(str).to_numpy(),
            "_time": timestamps.to_numpy(),
            "_row": np.arange(len(entity), dtype=np.int64),
        }
    )
    grouped = (
        temp.groupby(["_entity", "_time"], dropna=False)
        .size()
        .rename("_at_time")
        .reset_index()
        .sort_values(["_entity", "_time"], kind="mergesort")
    )
    grouped["_prior"] = (
        grouped.groupby("_entity")["_at_time"].cumsum() - grouped["_at_time"]
    )
    merged = temp.merge(
        grouped[["_entity", "_time", "_prior"]],
        on=["_entity", "_time"],
        how="left",
        sort=False,
    ).sort_values("_row")
    return merged["_prior"].fillna(0).to_numpy(dtype=np.int64)


def history_count_summary(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "coverage_rate": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
    return {
        "coverage_rate": float(np.mean(values > 0)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": int(np.max(values)),
    }


def leakage_report(config: ConditionalTABDLMConfig) -> dict[str, Any]:
    targets = set(config.schema.target_columns)
    conditions = set(config.schema.condition_columns)
    graph = dict(config.raw.get("graph_conditioning") or {})
    forbidden = set(graph.get("forbidden_node_features") or [])
    overlap = sorted(targets.intersection(conditions))
    missing_forbidden = sorted(targets - forbidden) if graph.get("enabled") else []
    future = bool(graph.get("graph_uses_future_events", False))
    target_features = bool(graph.get("graph_uses_target_attributes", False))
    errors = []
    if overlap:
        errors.append(f"Generated targets also appear as condition columns: {overlap}")
    if missing_forbidden:
        errors.append(
            f"Generated targets are not forbidden from graph features: {missing_forbidden}"
        )
    if future:
        errors.append("graph_uses_future_events must be false")
    if target_features:
        errors.append("graph_uses_target_attributes must be false")
    return {
        "valid": not errors,
        "target_condition_overlap": overlap,
        "targets_missing_from_forbidden_graph_features": missing_forbidden,
        "graph_uses_future_events": future,
        "graph_uses_target_attributes": target_features,
        "errors": errors,
    }


def inspect_previous_outputs(output_dir: Path) -> dict[str, Any]:
    if not output_dir.exists():
        return {
            "output_dir_exists": False,
            "checkpoints": [],
            "synthetic_tables": [],
            "logs": [],
        }
    return {
        "output_dir_exists": True,
        "checkpoints": sorted(str(path) for path in output_dir.rglob("*.pt")),
        "synthetic_tables": sorted(
            str(path)
            for path in output_dir.rglob("*.csv")
            if "synthetic" in path.name
        ),
        "logs": sorted(
            str(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix in {".log", ".jsonl"}
        ),
    }


def write_audit_outputs(
    report: dict[str, Any],
    roles: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    roles.to_csv(output_dir / "schema_roles.csv", index=False)
    (output_dir / "experiment_validation_report.md").write_text(
        audit_markdown(report, roles),
        encoding="utf-8",
    )


def audit_markdown(report: dict[str, Any], roles: pd.DataFrame) -> str:
    scale = report["scale"]
    split = report["split_integrity"].get("bounds", {})
    history = report["history_coverage"]
    lines = [
        f"# {report.get('dataset_name')} LSTM Experiment Audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Rows: {scale['rows']:,}",
        f"- Time range: {scale['timestamp_min']} to {scale['timestamp_max']}",
        f"- Generated attributes: {', '.join(report['schema']['generated_attributes'])}",
        "",
        "## Split",
        "",
    ]
    for name in ("train", "validation", "test"):
        item = split.get(name, {})
        lines.append(
            f"- {name}: {item.get('rows', 0):,} rows, "
            f"{item.get('timestamp_min')} to {item.get('timestamp_max')}"
        )
    lines.extend(["", "## Schema Roles", "", markdown_table(roles), ""])
    if history.get("status") == "completed":
        lines.extend(
            [
                "## Test History Coverage",
                "",
                f"- Cold events: {history['cold_events']:,}",
                f"- Partial-history events: {history['partial_history_events']:,}",
                f"- Warm events: {history['warm_events']:,}",
                f"- Any-history coverage: {history['any_history_coverage_rate']:.6f}",
                "",
            ]
        )
    if report["errors"]:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {message}" for message in report["errors"])
        lines.append("")
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {message}" for message in report["warnings"])
        lines.append("")
    return "\n".join(lines) + "\n"


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in frame.fillna("").itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def iso_or_none(value: Any) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")
