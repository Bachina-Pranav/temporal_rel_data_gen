"""Dataset adaptation and postprocessing without changing RelDiff's model core."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import RelDiffDatasetConfig


SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "validation",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_interaction_splits(
    config: RelDiffDatasetConfig,
) -> tuple[dict[str, pd.DataFrame], str]:
    frame = pd.read_csv(config.interaction_path)
    required = set(config.semantic_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{config.interaction_path} is missing columns: {missing}")

    timestamps = pd.to_datetime(frame[config.timestamp], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(
            f"{config.timestamp} contains {int(timestamps.isna().sum())} invalid timestamps"
        )
    frame = frame.copy()
    frame[config.timestamp] = timestamps
    labels, split_source = resolve_split_labels(frame, config.timestamp)
    frame["__reldiff_split"] = labels
    frame["__reldiff_input_order"] = np.arange(len(frame), dtype=np.int64)

    result: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        selected = frame.loc[frame["__reldiff_split"] == split].copy()
        selected = selected.sort_values(
            [config.timestamp, "__reldiff_input_order"], kind="mergesort"
        ).reset_index(drop=True)
        result[split] = selected
    return result, split_source


def resolve_split_labels(
    frame: pd.DataFrame, timestamp_col: str
) -> tuple[pd.Series, str]:
    if "split" in frame.columns:
        labels = frame["split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES)
        unknown = sorted(set(frame.loc[labels.isna(), "split"].astype(str)))
        if unknown:
            raise ValueError(f"Unknown split labels: {unknown}")
        return labels, "explicit_split_column"

    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    order = timestamps.sort_values(kind="mergesort").index.to_numpy()
    train_end = int(len(frame) * 0.90)
    validation_end = int(len(frame) * 0.95)
    labels = pd.Series(index=frame.index, dtype="object")
    labels.loc[order[:train_end]] = "train"
    labels.loc[order[train_end:validation_end]] = "validation"
    labels.loc[order[validation_end:]] = "test"
    return labels, "legacy_time_aware_90_5_5"


def semantic_split_frame(
    frame: pd.DataFrame, config: RelDiffDatasetConfig
) -> pd.DataFrame:
    columns = [column for column in config.semantic_columns if column in frame]
    result = frame.loc[:, columns].copy()
    if config.event_id not in result:
        result.insert(0, config.event_id, np.arange(len(result), dtype=np.int64))
    return result


def encode_training_timestamp(
    frame: pd.DataFrame,
    config: RelDiffDatasetConfig,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    encoded = frame.copy()
    timestamps = pd.to_datetime(encoded[config.timestamp], errors="coerce", utc=True)
    encoded[config.timestamp] = (timestamps - origin).dt.total_seconds().astype(float)
    return encoded


def decode_generated_timestamp(
    values: pd.Series, origin: pd.Timestamp
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return origin + pd.to_timedelta(numeric, unit="s")


def load_entity_ids(path: Path, primary_key: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[primary_key])
    if frame[primary_key].isna().any():
        raise ValueError(f"Null primary keys in {path}:{primary_key}")
    if frame[primary_key].duplicated().any():
        raise ValueError(f"Duplicate primary keys in {path}:{primary_key}")
    return frame.reset_index(drop=True)


def prepare_training_database(
    config: RelDiffDatasetConfig,
    *,
    data_root: str | Path,
    staged_name: str,
    provenance_dir: str | Path,
    max_train_rows: int | None = None,
) -> dict[str, Any]:
    """Materialize a train-only three-table database for the upstream pipeline."""

    data_root = Path(data_root)
    provenance_dir = Path(provenance_dir)
    original_dir = data_root / "original" / staged_name
    if original_dir.exists():
        shutil.rmtree(original_dir)
    original_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    splits, split_source = read_interaction_splits(config)
    if max_train_rows is not None:
        splits["train"] = splits["train"].iloc[:max_train_rows].copy()
    train = semantic_split_frame(splits["train"], config)
    origin = pd.to_datetime(train[config.timestamp], utc=True).min()
    if pd.isna(origin):
        raise ValueError("Training split is empty or has no valid timestamp")

    source = load_entity_ids(config.source_entity_path, config.source_pk)
    destination = load_entity_ids(config.destination_entity_path, config.destination_pk)
    source_domain = set(source[config.source_pk])
    destination_domain = set(destination[config.destination_pk])
    if not train[config.source_fk].isin(source_domain).all():
        raise ValueError("Training source FKs are outside the supplied entity universe")
    if not train[config.destination_fk].isin(destination_domain).all():
        raise ValueError("Training destination FKs are outside the supplied entity universe")

    encoded_train = encode_training_timestamp(train, config, origin)
    encoded_train[config.event_id] = np.arange(len(encoded_train), dtype=np.int64)
    tables = {
        config.source_table: source,
        config.destination_table: destination,
        config.interaction_table: encoded_train,
    }
    metadata = create_syntherela_metadata(tables, config)
    from syntherela.data import save_tables

    save_tables(tables, str(original_dir), metadata=metadata, save_metadata=True)

    mappings_dir = provenance_dir / "id_mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    source_mapping = pd.DataFrame(
        {"internal_id": np.arange(len(source)), "original_id": source[config.source_pk]}
    )
    destination_mapping = pd.DataFrame(
        {
            "internal_id": np.arange(len(destination)),
            "original_id": destination[config.destination_pk],
        }
    )
    source_mapping.to_csv(mappings_dir / f"{config.source_table}.csv", index=False)
    destination_mapping.to_csv(
        mappings_dir / f"{config.destination_table}.csv", index=False
    )

    split_dir = provenance_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, Any] = {}
    for name, split_frame in splits.items():
        output = semantic_split_frame(split_frame, config)
        path = split_dir / f"{name}.csv"
        output.to_csv(path, index=False)
        split_manifest[name] = {
            "path": str(path),
            "rows": int(len(output)),
            "sha256": file_sha256(path),
            "timestamp_min": _timestamp_string(output[config.timestamp].min()),
            "timestamp_max": _timestamp_string(output[config.timestamp].max()),
        }

    manifest = {
        "dataset": config.to_manifest(),
        "staged_name": staged_name,
        "staged_database_dir": str(original_dir),
        "split_source": split_source,
        "splits": split_manifest,
        "entity_tables": {
            "source": _file_record(config.source_entity_path, len(source)),
            "destination": _file_record(config.destination_entity_path, len(destination)),
        },
        "training_time_origin": origin.isoformat(),
        "time_unit": "seconds",
        "clipped": False,
        "training_rows": int(len(train)),
        "source_entities": int(len(source)),
        "destination_entities": int(len(destination)),
        "text_or_surrogate_columns_included": [],
    }
    write_json(manifest, provenance_dir / "data_manifest.json")
    return manifest


def create_syntherela_metadata(
    tables: dict[str, pd.DataFrame], config: RelDiffDatasetConfig
):
    from syntherela.metadata import Metadata

    metadata = Metadata()
    metadata.detect_from_dataframes(tables)
    for relationship in metadata.relationships.copy():
        metadata.remove_relationship(
            parent_table_name=relationship["parent_table_name"],
            child_table_name=relationship["child_table_name"],
        )

    for table, key in (
        (config.source_table, config.source_pk),
        (config.destination_table, config.destination_pk),
        (config.interaction_table, config.event_id),
    ):
        metadata.update_column(table, key, sdtype="id")
        metadata.set_primary_key(table, key)

    metadata.update_column(config.interaction_table, config.source_fk, sdtype="id")
    metadata.update_column(config.interaction_table, config.destination_fk, sdtype="id")
    metadata.update_column(
        config.interaction_table,
        config.timestamp,
        sdtype="numerical",
        computer_representation="Float",
    )
    for column in config.numerical_attributes:
        metadata.update_column(
            config.interaction_table,
            column,
            sdtype="numerical",
            computer_representation="Float",
        )
    for column in config.categorical_attributes:
        metadata.update_column(config.interaction_table, column, sdtype="categorical")

    metadata.add_relationship(
        parent_table_name=config.source_table,
        child_table_name=config.interaction_table,
        parent_primary_key=config.source_pk,
        child_foreign_key=config.source_fk,
    )
    metadata.add_relationship(
        parent_table_name=config.destination_table,
        child_table_name=config.interaction_table,
        parent_primary_key=config.destination_pk,
        child_foreign_key=config.destination_fk,
    )
    metadata.validate()
    metadata.validate_data(tables)
    return metadata


def reconstruct_foreign_keys(
    dataset: Any,
    metadata: Any,
    table_name: str,
    row_count: int,
) -> pd.DataFrame:
    """Exercise the same child-sorted FK inversion used by MultiTableSampler."""

    result = pd.DataFrame(index=np.arange(row_count))
    for parent in metadata.get_parents(table_name):
        for foreign_key in metadata.get_foreign_keys(parent, table_name):
            edges = dataset.edge_index_dict[(table_name, foreign_key, parent)].cpu().numpy()
            child_ids = edges[0].tolist()
            if child_ids != sorted(child_ids):
                raise AssertionError(f"Unsorted child edge index for {foreign_key}")
            result[foreign_key] = pd.Series(dtype="Int64")
            result.loc[child_ids, foreign_key] = edges[1]
    return result


def postprocess_generated_interactions(
    config: RelDiffDatasetConfig,
    *,
    generated_table: str | Path,
    data_manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    generated = pd.read_csv(generated_table)
    manifest = json.loads(Path(data_manifest).read_text(encoding="utf-8"))
    provenance_dir = Path(data_manifest).parent
    source_map = pd.read_csv(
        provenance_dir / "id_mappings" / f"{config.source_table}.csv"
    ).set_index("internal_id")["original_id"]
    destination_map = pd.read_csv(
        provenance_dir / "id_mappings" / f"{config.destination_table}.csv"
    ).set_index("internal_id")["original_id"]

    generated[config.source_fk] = pd.to_numeric(
        generated[config.source_fk], errors="raise"
    ).astype(int).map(source_map)
    generated[config.destination_fk] = pd.to_numeric(
        generated[config.destination_fk], errors="raise"
    ).astype(int).map(destination_map)
    origin = pd.Timestamp(manifest["training_time_origin"])
    raw_generated_time = pd.to_numeric(generated[config.timestamp], errors="coerce")
    generated[config.timestamp] = decode_generated_timestamp(raw_generated_time, origin)
    generated[config.event_id] = np.arange(len(generated), dtype=np.int64)

    columns = [config.event_id, *config.semantic_columns]
    generated = generated.loc[:, columns]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output, index=False)

    train_min = 0.0
    train_max = (
        pd.Timestamp(manifest["splits"]["train"]["timestamp_max"]) - origin
    ).total_seconds()
    finite = raw_generated_time[np.isfinite(raw_generated_time)]
    out_of_range = (
        float(((finite < train_min) | (finite > train_max)).mean())
        if len(finite)
        else None
    )
    report = {
        "output": str(output),
        "rows": int(len(generated)),
        "training_time_origin": origin.isoformat(),
        "time_unit": "seconds",
        "generated_min_numeric": _finite_min(finite),
        "generated_max_numeric": _finite_max(finite),
        "train_min_numeric": train_min,
        "train_max_numeric": float(train_max),
        "out_of_train_range_fraction": out_of_range,
        "timestamp_clipped": False,
    }
    write_json(report, output.parent / "timestamp_postprocessing.json")
    return report


def repeated_pair_summary(
    frame: pd.DataFrame, source: str, destination: str
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    counts = frame.groupby([source, destination], dropna=False).size().rename("multiplicity")
    repeated = counts[counts > 1]
    repeated_rows = int(repeated.sum())
    p95 = float(np.quantile(counts.to_numpy(), 0.95)) if len(counts) else 0.0
    summary = {
        "num_rows": int(len(frame)),
        "num_unique_pairs": int(len(counts)),
        "num_repeated_pair_rows": repeated_rows,
        "fraction_rows_in_repeated_pairs": float(repeated_rows / len(frame)) if len(frame) else 0.0,
        "max_pair_multiplicity": int(counts.max()) if len(counts) else 0,
        "p95_pair_multiplicity": p95,
        "mean_multiplicity_given_repeated": float(repeated.mean()) if len(repeated) else 0.0,
    }
    top = counts.sort_values(ascending=False).head(20).reset_index()
    examples = (
        frame.merge(repeated.reset_index(), on=[source, destination], how="inner")
        .sort_values("multiplicity", ascending=False, kind="mergesort")
        .head(100)
    )
    return summary, top, examples


def generation_validity(
    config: RelDiffDatasetConfig,
    generated_path: str | Path,
    data_manifest_path: str | Path,
) -> dict[str, Any]:
    generated = pd.read_csv(generated_path)
    manifest = json.loads(Path(data_manifest_path).read_text(encoding="utf-8"))
    source_ids = set(pd.read_csv(config.source_entity_path, usecols=[config.source_pk])[config.source_pk])
    destination_ids = set(
        pd.read_csv(config.destination_entity_path, usecols=[config.destination_pk])[config.destination_pk]
    )
    timestamps = pd.to_datetime(generated[config.timestamp], errors="coerce", utc=True)
    categorical = {}
    train = pd.read_csv(manifest["splits"]["train"]["path"])
    for column in config.categorical_attributes:
        train_domain = set(train[column].dropna().astype(str))
        generated_domain = set(generated[column].dropna().astype(str))
        categorical[column] = {
            "valid": generated_domain <= train_domain,
            "unexpected_values": sorted(generated_domain - train_domain),
        }
    numerical = {}
    for column in config.numerical_attributes:
        values = pd.to_numeric(generated[column], errors="coerce")
        numerical[column] = {"finite_fraction": float(np.isfinite(values).mean())}
    text_present = sorted(set(config.ignored_attributes) & set(generated.columns))
    pair_sizes = generated.groupby(
        [config.source_fk, config.destination_fk], dropna=False
    )[config.source_fk].transform("size")
    report = {
        "source_fk_valid": bool(generated[config.source_fk].isin(source_ids).all()),
        "destination_fk_valid": bool(generated[config.destination_fk].isin(destination_ids).all()),
        "expected_row_count": int(manifest["training_rows"]),
        "actual_row_count": int(len(generated)),
        "row_count_valid": int(len(generated)) == int(manifest["training_rows"]),
        "lost_generated_edges": int(manifest["training_rows"]) - int(len(generated)),
        "repeated_pair_rows": int(pair_sizes.gt(1).sum()) if len(generated) else 0,
        "categorical": categorical,
        "numerical": numerical,
        "timestamp_parse_error_rate": float(timestamps.isna().mean()),
        "text_or_surrogate_columns_present": text_present,
    }
    report["valid"] = bool(
        report["source_fk_valid"]
        and report["destination_fk_valid"]
        and report["row_count_valid"]
        and report["timestamp_parse_error_rate"] == 0.0
        and not text_present
        and all(value["valid"] for value in categorical.values())
        and all(value["finite_fraction"] == 1.0 for value in numerical.values())
    )
    return report


def _file_record(path: Path, rows: int) -> dict[str, Any]:
    return {"path": str(path), "rows": int(rows), "sha256": file_sha256(path)}


def _timestamp_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _finite_min(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(min(values)) if values else None


def _finite_max(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(max(values)) if values else None
