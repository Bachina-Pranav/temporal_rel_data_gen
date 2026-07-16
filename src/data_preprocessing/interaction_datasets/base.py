"""Base classes and file utilities for interaction-table benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd


@dataclass(frozen=True)
class DownloadResult:
    dataset_name: str
    status: str
    raw_dir: Path
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawFileBundle:
    files: dict[str, Path]

    def require(self, *names: str) -> None:
        missing = [name for name in names if name not in self.files or not self.files[name].exists()]
        if missing:
            raise FileNotFoundError(f"Missing required raw files: {missing}")


class InteractionDatasetAdapter:
    """Dataset-specific adapter; generic code operates only through this API."""

    dataset_name: str
    benchmark_name: str
    source_dataset: str
    source_entity_name: str
    destination_entity_name: str
    source_entity_table: str = ""
    destination_entity_table: str = ""
    interaction_name: str
    source_id_column: str
    destination_id_column: str
    timestamp_column: str = "event_time"
    event_id_column: str = "event_id"
    generated_attributes: tuple[str, ...] = ()
    attribute_types: dict[str, str] = {}
    support_tables: tuple[str, ...] = ()
    excluded_columns: dict[str, str] = {}
    domain: str = ""

    def raw_dir(self, raw_root: str | Path) -> Path:
        return Path(raw_root) / self.dataset_name

    @property
    def source_table_filename(self) -> str:
        return self.source_entity_table or f"{self.source_entity_name}s.csv"

    @property
    def destination_table_filename(self) -> str:
        return self.destination_entity_table or f"{self.destination_entity_name}s.csv"

    def download(
        self,
        raw_root: str | Path,
        *,
        force: bool = False,
        verify_only: bool = False,
        archive: str | Path | None = None,
        **_: Any,
    ) -> DownloadResult:
        raise NotImplementedError

    def locate_raw_files(self, raw_root: str | Path) -> RawFileBundle:
        raise NotImplementedError

    def iter_interaction_chunks(
        self,
        raw_root: str | Path,
        *,
        chunk_size: int = 250_000,
    ) -> Iterator[pd.DataFrame]:
        raise NotImplementedError

    def load_source_entities(self, raw_root: str | Path, selected_ids: set[str]) -> pd.DataFrame:
        return pd.DataFrame({self.source_id_column: sorted(str(value) for value in selected_ids)})

    def load_destination_entities(self, raw_root: str | Path, selected_ids: set[str]) -> pd.DataFrame:
        return pd.DataFrame({self.destination_id_column: sorted(str(value) for value in selected_ids)})

    def load_extra_support_tables(self, raw_root: str | Path, destination_ids: set[str], output_dir: Path) -> dict[str, Path]:
        return {}

    def schema_metadata(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            self.source_id_column: {"role": "source_foreign_key", "semantic_type": "foreign_key", "nullable": False},
            self.destination_id_column: {"role": "destination_foreign_key", "semantic_type": "foreign_key", "nullable": False},
            self.timestamp_column: {"role": "timestamp", "semantic_type": "datetime", "nullable": False},
        }
        for column in self.generated_attributes:
            fields[column] = {
                "role": "generated_attribute",
                "semantic_type": self.attribute_types[column],
                "nullable": False,
                "valid_domain": None,
            }
        return {
            "dataset_name": self.benchmark_name,
            "method_scope": "single_designated_temporal_interaction_table",
            "source_dataset": self.source_dataset,
            "target_table": "interactions.csv",
            "source_entity_table": self.source_table_filename,
            "destination_entity_table": self.destination_table_filename,
            "source_id_column": self.source_id_column,
            "destination_id_column": self.destination_id_column,
            "timestamp_column": self.timestamp_column,
            "event_id_column": self.event_id_column,
            "generated_attributes": list(self.generated_attributes),
            "attribute_types": dict(self.attribute_types),
            "fields": fields,
            "support_tables": list(self.support_tables),
            "excluded_columns": dict(self.excluded_columns),
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if p.exists() and p.is_file():
            out[str(p)] = sha256_file(p)
    return out


def safe_extract_archive(archive: str | Path, output_dir: str | Path) -> list[Path]:
    archive = Path(archive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = safe_member_path(output_dir, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target)
        return extracted
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = safe_member_path(output_dir, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target)
        return extracted
    raise ValueError(f"Unsupported archive format: {archive}")


def safe_member_path(root: Path, member_name: str) -> Path:
    if os.path.isabs(member_name):
        raise ValueError(f"Unsafe absolute archive member path: {member_name}")
    target = (root / member_name).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError(f"Unsafe archive member path traversal: {member_name}")
    return target


def download_url(url: str, output_path: str | Path, *, force: bool = False) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - trusted adapter URL.
    tmp.replace(output_path)
    return output_path


def kaggle_download(dataset_slug: str, output_dir: str | Path, *, force: bool = False, competition: bool = False) -> tuple[bool, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if competition:
        cmd = ["kaggle", "competitions", "download", "-c", dataset_slug, "-p", str(output_dir)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(output_dir)]
    if force:
        cmd.append("--force")
    try:
        completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return False, "Kaggle CLI is not installed. Install/configure kaggle or provide a local archive."
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout).strip()
    return True, (completed.stdout or "downloaded").strip()


def write_download_metadata(result: DownloadResult, files: Iterable[Path], output_path: Path) -> None:
    payload = {
        "dataset_name": result.dataset_name,
        "status": result.status,
        "message": result.message,
        "raw_dir": str(result.raw_dir),
        "created_at": utc_now_iso(),
        "metadata": result.metadata,
        "file_hashes": file_hashes(files),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv_chunks_with_row_number(path: str | Path, *, chunk_size: int, **kwargs: Any) -> Iterator[pd.DataFrame]:
    offset = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size, **kwargs):
        chunk = chunk.copy()
        chunk["_raw_row_number"] = range(offset, offset + len(chunk))
        offset += len(chunk)
        yield chunk


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
