"""Yelp Open Dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import DownloadResult, InteractionDatasetAdapter, RawFileBundle, kaggle_download, safe_extract_archive, write_csv, write_download_metadata


class YelpAdapter(InteractionDatasetAdapter):
    dataset_name = "yelp"
    benchmark_name = "yelp_100k"
    source_dataset = "Yelp Open Dataset"
    source_entity_name = "user"
    destination_entity_name = "business"
    destination_entity_table = "businesses.csv"
    source_id_column = "user_id"
    destination_id_column = "business_id"
    generated_attributes = ("stars", "useful", "funny", "cool", "review_text")
    attribute_types = {
        "stars": "ordinal_categorical",
        "useful": "count_numerical",
        "funny": "count_numerical",
        "cool": "count_numerical",
        "review_text": "text",
    }
    support_tables = ("users.csv", "businesses.csv")
    excluded_columns = {
        "user.average_stars": "full-history aggregate; excluded from interaction attributes",
        "user.review_count": "full-history aggregate; excluded from interaction attributes",
        "business.stars": "full-history aggregate; support metadata only",
        "business.review_count": "full-history aggregate; support metadata only",
    }
    domain = "local business reviews"
    kaggle_slug = "yelp-dataset/yelp-dataset"

    def download(self, raw_root, *, force=False, verify_only=False, archive=None, **kwargs) -> DownloadResult:
        del kwargs
        raw_dir = self.raw_dir(raw_root)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if archive is not None:
            archive = Path(archive)
            if not archive.exists():
                return DownloadResult(
                    self.dataset_name,
                    "blocked_missing_archive",
                    raw_dir,
                    f"Yelp archive not found: {archive}. Replace the placeholder with the real yelp_dataset.tar path.",
                    {"archive": str(archive), "manual_acceptance_required": True},
                )
            if not verify_only:
                safe_extract_archive(archive, raw_dir)
            result = DownloadResult(self.dataset_name, "loaded_from_local_archive", raw_dir, f"Loaded {archive}", {"archive": str(archive), "manual_acceptance_required": True})
            write_download_metadata(result, list(raw_dir.rglob("*")), raw_dir / "download_metadata.json")
            return result
        if verify_only:
            self.locate_raw_files(raw_root).require("review", "user", "business")
            return DownloadResult(self.dataset_name, "verified_existing", raw_dir, "Yelp raw files already exist")
        ok, message = kaggle_download(self.kaggle_slug, raw_dir, force=force)
        if not ok:
            return DownloadResult(
                self.dataset_name,
                "blocked_missing_credentials_or_license_acceptance",
                raw_dir,
                "Yelp requires Kaggle credentials/license acceptance or --archive /path/to/yelp_dataset.tar. " + message,
                {"kaggle_slug": self.kaggle_slug, "manual_acceptance_required": True},
            )
        for archive_path in raw_dir.glob("*.zip"):
            safe_extract_archive(archive_path, raw_dir)
        result = DownloadResult(self.dataset_name, "downloaded_automatically", raw_dir, message, {"kaggle_slug": self.kaggle_slug, "manual_acceptance_required": True})
        write_download_metadata(result, list(raw_dir.rglob("*")), raw_dir / "download_metadata.json")
        return result

    def locate_raw_files(self, raw_root) -> RawFileBundle:
        raw_dir = self.raw_dir(raw_root)
        files = {
            "review": find_file(raw_dir, "yelp_academic_dataset_review.json"),
            "user": find_file(raw_dir, "yelp_academic_dataset_user.json"),
            "business": find_file(raw_dir, "yelp_academic_dataset_business.json"),
        }
        if not all(files.values()):
            raise FileNotFoundError(f"Missing Yelp JSON files under {raw_dir}")
        return RawFileBundle({key: path for key, path in files.items() if path is not None})

    def iter_interaction_chunks(self, raw_root, *, chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
        path = self.locate_raw_files(raw_root).files["review"]
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                rows.append(
                    {
                        "event_id": str(item["review_id"]),
                        "user_id": str(item["user_id"]),
                        "business_id": str(item["business_id"]),
                        "event_time": pd.to_datetime(item["date"], utc=True),
                        "stars": str(item["stars"]),
                        "useful": int(item.get("useful", 0)),
                        "funny": int(item.get("funny", 0)),
                        "cool": int(item.get("cool", 0)),
                        "review_text": str(item.get("text", "")),
                    }
                )
                if len(rows) >= chunk_size:
                    yield pd.DataFrame(rows)
                    rows = []
        if rows:
            yield pd.DataFrame(rows)

    def iter_source_id_chunks(self, raw_root, *, chunk_size: int = 100_000) -> Iterator[pd.Series]:
        path = self.locate_raw_files(raw_root).files["review"]
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                rows.append(str(json.loads(line)["user_id"]))
                if len(rows) >= chunk_size:
                    yield pd.Series(rows, dtype=str)
                    rows = []
        if rows:
            yield pd.Series(rows, dtype=str)

    def load_source_entities(self, raw_root, selected_ids: set[str]) -> pd.DataFrame:
        return filter_jsonl(self.locate_raw_files(raw_root).files["user"], "user_id", selected_ids)

    def load_destination_entities(self, raw_root, selected_ids: set[str]) -> pd.DataFrame:
        return filter_jsonl(self.locate_raw_files(raw_root).files["business"], "business_id", selected_ids)


def find_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def filter_jsonl(path: Path, id_column: str, ids: set[str]) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if str(item.get(id_column)) in ids:
                rows.append(item)
    if not rows:
        return pd.DataFrame({id_column: sorted(ids)})
    frame = pd.DataFrame(rows)
    frame[id_column] = frame[id_column].astype(str)
    return frame.reset_index(drop=True)
