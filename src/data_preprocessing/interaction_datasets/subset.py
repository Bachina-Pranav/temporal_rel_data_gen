"""Generic source-entity-induced subset construction."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import InteractionDatasetAdapter, file_hashes, utc_now_iso, write_csv
from .schemas import write_schema_yaml
from .statistics import compute_statistics, write_statistics_markdown
from .validation import validate_subset


@dataclass(frozen=True)
class SelectionResult:
    selected_ids: tuple[str, ...]
    requested_count: int
    initial_count: int
    final_count: int
    absolute_deviation: int
    relative_deviation: float
    tolerance_achieved: bool
    refinement_operations: list[dict[str, Any]]


def count_source_histories(
    adapter: InteractionDatasetAdapter,
    raw_root: str | Path,
    *,
    chunk_size: int = 250_000,
) -> pd.Series:
    counts: dict[str, int] = {}
    rows_seen = 0
    for chunk_idx, source in enumerate(adapter.iter_source_id_chunks(raw_root, chunk_size=chunk_size), start=1):
        source = source.astype(str)
        rows_seen += int(len(source))
        for key, value in source.value_counts().items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(
                f"[count] {adapter.dataset_name}: chunks={chunk_idx:,} rows={rows_seen:,} source_entities={len(counts):,}",
                flush=True,
            )
    return pd.Series(counts, dtype="int64").sort_index()


def select_source_entities(
    counts: pd.Series | dict[str, int],
    *,
    target_interactions: int,
    allowed_relative_error: float = 0.01,
    seed: int = 42,
) -> SelectionResult:
    series = pd.Series(counts, dtype="int64")
    series.index = series.index.astype(str)
    if series.empty:
        raise ValueError("Cannot select source entities from empty counts")
    ordered = list(series.index)
    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    selected: list[str] = []
    total = 0
    for source_id in ordered:
        candidate = total + int(series[source_id])
        if abs(candidate - target_interactions) <= abs(total - target_interactions):
            selected.append(source_id)
            total = candidate
        else:
            break
        if total >= target_interactions:
            break
    initial_total = int(total)
    operations: list[dict[str, Any]] = []
    selected_set = set(selected)
    total = refine_selection(series, selected_set, target_interactions, operations)
    tolerance = abs(total - target_interactions) <= target_interactions * float(allowed_relative_error)
    return SelectionResult(
        selected_ids=tuple(sorted(selected_set)),
        requested_count=int(target_interactions),
        initial_count=int(initial_total),
        final_count=int(total),
        absolute_deviation=int(abs(total - target_interactions)),
        relative_deviation=float(abs(total - target_interactions) / max(int(target_interactions), 1)),
        tolerance_achieved=bool(tolerance),
        refinement_operations=operations,
    )


def refine_selection(series: pd.Series, selected: set[str], target: int, operations: list[dict[str, Any]]) -> int:
    total = int(series.loc[list(selected)].sum()) if selected else 0
    improved = True
    while improved:
        improved = False
        current_error = abs(total - target)
        if total < target:
            candidates = [sid for sid in series.index if sid not in selected]
            best = min(candidates, key=lambda sid: abs((total + int(series[sid])) - target), default=None)
            if best is not None:
                new_total = total + int(series[best])
                if abs(new_total - target) < current_error:
                    selected.add(str(best))
                    total = int(new_total)
                    operations.append({"op": "add", "source_id": str(best), "count": int(series[best]), "total": total})
                    improved = True
        else:
            best = min(selected, key=lambda sid: abs((total - int(series[sid])) - target), default=None)
            if best is not None:
                new_total = total - int(series[best])
                if abs(new_total - target) < current_error:
                    selected.remove(str(best))
                    total = int(new_total)
                    operations.append({"op": "remove", "source_id": str(best), "count": int(series[best]), "total": total})
                    improved = True
    return int(total)


def build_interaction_subset(
    adapter: InteractionDatasetAdapter,
    *,
    raw_root: str | Path,
    processed_root: str | Path,
    target_interactions: int = 100_000,
    allowed_relative_error: float = 0.01,
    seed: int = 42,
    chunk_size: int = 250_000,
    memory_budget_mb: int | None = None,
    temp_dir: str | Path | None = None,
) -> dict[str, Any]:
    del memory_budget_mb, temp_dir
    raw_root = Path(raw_root)
    output_dir = Path(processed_root) / adapter.benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = adapter.locate_raw_files(raw_root)
    counts = count_source_histories(adapter, raw_root, chunk_size=chunk_size)
    selection = select_source_entities(
        counts,
        target_interactions=target_interactions,
        allowed_relative_error=allowed_relative_error,
        seed=seed,
    )
    selected = set(selection.selected_ids)
    chunks: list[pd.DataFrame] = []
    rows_seen = 0
    rows_retained = 0
    for chunk_idx, chunk in enumerate(adapter.iter_interaction_chunks(raw_root, chunk_size=chunk_size), start=1):
        rows_seen += int(len(chunk))
        mask = chunk[adapter.source_id_column].astype(str).isin(selected)
        if bool(mask.any()):
            retained = chunk.loc[mask].copy()
            rows_retained += int(len(retained))
            chunks.append(retained)
        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(
                f"[materialize] {adapter.dataset_name}: chunks={chunk_idx:,} rows={rows_seen:,} retained={rows_retained:,}",
                flush=True,
            )
    if not chunks:
        raise ValueError(f"No interactions retained for {adapter.dataset_name}")
    interactions = pd.concat(chunks, ignore_index=True)
    audit_paths: dict[str, Path] = {}
    audit_columns = [column for column in interactions.columns if str(column).startswith("_audit_")]
    if audit_columns:
        audit = interactions[[adapter.event_id_column, *audit_columns]].copy()
        audit.columns = [str(column)[len("_audit_") :] if str(column).startswith("_audit_") else str(column) for column in audit.columns]
        audit_paths["events_audit"] = write_csv(audit, output_dir / "events_audit.csv")
        interactions = interactions.drop(columns=audit_columns)
    interactions[adapter.source_id_column] = interactions[adapter.source_id_column].astype(str)
    interactions[adapter.destination_id_column] = interactions[adapter.destination_id_column].astype(str)
    interactions[adapter.timestamp_column] = pd.to_datetime(interactions[adapter.timestamp_column], utc=True, errors="coerce")
    interactions = assign_chronological_splits(interactions, adapter.timestamp_column, adapter.event_id_column)
    destination_ids = set(interactions[adapter.destination_id_column].astype(str))
    source_frame = adapter.load_source_entities(raw_root, selected)
    destination_frame = adapter.load_destination_entities(raw_root, destination_ids)

    paths = {
        "interactions": write_csv(interactions, output_dir / "interactions.csv"),
        adapter.source_table_filename: write_csv(source_frame, output_dir / adapter.source_table_filename),
        adapter.destination_table_filename: write_csv(destination_frame, output_dir / adapter.destination_table_filename),
    }
    paths.update(audit_paths)
    paths.update(adapter.load_extra_support_tables(raw_root, destination_ids, output_dir))
    schema = adapter.schema_metadata()
    write_schema_yaml(schema, output_dir / "schema.yaml")
    validation = validate_subset(adapter, output_dir, raw_counts=counts.loc[list(selected)].to_dict())
    statistics = compute_statistics(adapter, interactions, full_counts=counts)
    manifest = build_manifest(adapter, raw_files.files, paths, interactions, selection, validation, seed=seed)
    write_json(manifest, output_dir / "subset_manifest.json")
    write_json(validation, output_dir / "validation_report.json")
    write_json(statistics, output_dir / "statistics.json")
    write_statistics_markdown(statistics, output_dir / "statistics.md")
    write_readme(adapter, manifest, output_dir / "README.md")
    return manifest


def assign_chronological_splits(frame: pd.DataFrame, timestamp_col: str, event_id_col: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[event_id_col] = frame[event_id_col].astype(str)
    frame = frame.sort_values([timestamp_col, event_id_col], kind="mergesort").reset_index(drop=True)
    n = len(frame)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    split = np.empty(n, dtype=object)
    split[:train_end] = "train"
    split[train_end:valid_end] = "validation"
    split[valid_end:] = "test"
    frame["split"] = split
    return frame


def build_manifest(
    adapter: InteractionDatasetAdapter,
    raw_files: dict[str, Path],
    processed_paths: dict[str, Path],
    interactions: pd.DataFrame,
    selection: SelectionResult,
    validation: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    timestamps = pd.to_datetime(interactions[adapter.timestamp_column], utc=True, errors="coerce")
    split_cutoffs = {
        split: timestamps[interactions["split"] == split].max().isoformat()
        for split in ["train", "validation", "test"]
        if bool((interactions["split"] == split).any())
    }
    return {
        "dataset_name": adapter.benchmark_name,
        "source_dataset": adapter.source_dataset,
        "source_version": None,
        "target_table": "interactions.csv",
        "source_entity_table": adapter.source_table_filename,
        "destination_entity_table": adapter.destination_table_filename,
        "source_id_column": adapter.source_id_column,
        "destination_id_column": adapter.destination_id_column,
        "timestamp_column": adapter.timestamp_column,
        "target_interactions": int(selection.requested_count),
        "actual_interactions": int(len(interactions)),
        "absolute_target_deviation": int(selection.absolute_deviation),
        "relative_target_deviation": float(selection.relative_deviation),
        "selected_source_entities": int(interactions[adapter.source_id_column].nunique()),
        "selected_destination_entities": int(interactions[adapter.destination_id_column].nunique()),
        "selection_seed": int(seed),
        "complete_source_histories": bool(validation.get("complete_source_histories", False)),
        "foreign_key_valid": bool(validation.get("foreign_key_valid", False)),
        "generated_attributes": list(adapter.generated_attributes),
        "attribute_types": dict(adapter.attribute_types),
        "support_only_columns": [],
        "excluded_columns": list(adapter.excluded_columns),
        "exclusion_reasons": dict(adapter.excluded_columns),
        "timestamp_min": timestamps.min().isoformat(),
        "timestamp_max": timestamps.max().isoformat(),
        "split_cutoffs": split_cutoffs,
        "raw_file_hashes": file_hashes(raw_files.values()),
        "processed_file_hashes": file_hashes(processed_paths.values()),
        "created_at": utc_now_iso(),
        "code_version": git_revision(),
        "selection": {
            "initial_count": int(selection.initial_count),
            "final_count": int(selection.final_count),
            "tolerance_achieved": bool(selection.tolerance_achieved),
            "refinement_operations": selection.refinement_operations,
        },
    }


def git_revision() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def write_readme(adapter: InteractionDatasetAdapter, manifest: dict[str, Any], path: str | Path) -> None:
    text = f"""# {adapter.benchmark_name}

This subset synthesizes one designated temporal interaction table while retaining source and destination entity tables as fixed support tables.

- Source entity: `{adapter.source_entity_name}`
- Interaction table: `interactions.csv`
- Destination entity: `{adapter.destination_entity_name}`
- Actual interactions: {manifest['actual_interactions']:,}
- Complete source histories: {manifest['complete_source_histories']}
- Foreign-key valid: {manifest['foreign_key_valid']}
"""
    Path(path).write_text(text, encoding="utf-8")
