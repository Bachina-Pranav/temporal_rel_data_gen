"""H&M transaction adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import DownloadResult, InteractionDatasetAdapter, RawFileBundle, kaggle_download, read_csv_chunks_with_row_number, safe_extract_archive, write_download_metadata


class HMAdapter(InteractionDatasetAdapter):
    dataset_name = "hm"
    benchmark_name = "hm_100k"
    source_dataset = "H&M Personalized Fashion Recommendations"
    source_entity_name = "customer"
    destination_entity_name = "article"
    source_id_column = "customer_id"
    destination_id_column = "article_id"
    generated_attributes = ("price", "sales_channel_id")
    attribute_types = {"price": "continuous_numerical", "sales_channel_id": "categorical"}
    support_tables = ("customers.csv", "articles.csv")
    domain = "seasonal retail purchases"
    kaggle_slug = "h-and-m-personalized-fashion-recommendations"

    def download(self, raw_root, *, force=False, verify_only=False, archive=None, **kwargs) -> DownloadResult:
        del kwargs
        raw_dir = self.raw_dir(raw_root)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if archive is not None:
            if not verify_only:
                safe_extract_archive(archive, raw_dir)
            result = DownloadResult(self.dataset_name, "loaded_from_local_archive", raw_dir, f"Loaded {archive}", {"archive": str(archive), "manual_acceptance_required": True})
            write_download_metadata(result, list(raw_dir.rglob("*")), raw_dir / "download_metadata.json")
            return result
        if verify_only:
            self.locate_raw_files(raw_root).require("transactions", "customers", "articles")
            return DownloadResult(self.dataset_name, "verified_existing", raw_dir, "H&M raw files already exist")
        ok, message = kaggle_download(self.kaggle_slug, raw_dir, force=force, competition=True)
        if not ok:
            return DownloadResult(self.dataset_name, "blocked_missing_credentials_or_license_acceptance", raw_dir, message, {"kaggle_slug": self.kaggle_slug, "manual_acceptance_required": True})
        for archive_path in raw_dir.glob("*.zip"):
            safe_extract_archive(archive_path, raw_dir)
        result = DownloadResult(self.dataset_name, "downloaded_automatically", raw_dir, message, {"kaggle_slug": self.kaggle_slug, "manual_acceptance_required": True})
        write_download_metadata(result, list(raw_dir.rglob("*")), raw_dir / "download_metadata.json")
        return result

    def locate_raw_files(self, raw_root) -> RawFileBundle:
        raw_dir = self.raw_dir(raw_root)
        files = {
            "transactions": find_file(raw_dir, "transactions_train.csv"),
            "customers": find_file(raw_dir, "customers.csv"),
            "articles": find_file(raw_dir, "articles.csv"),
        }
        if not all(files.values()):
            raise FileNotFoundError(f"Missing H&M CSV files under {raw_dir}")
        return RawFileBundle({key: path for key, path in files.items() if path is not None})

    def iter_interaction_chunks(self, raw_root, *, chunk_size: int = 250_000) -> Iterator[pd.DataFrame]:
        for chunk in read_csv_chunks_with_row_number(self.locate_raw_files(raw_root).files["transactions"], chunk_size=chunk_size):
            yield pd.DataFrame(
                {
                    "event_id": "hm-transaction-" + chunk["_raw_row_number"].astype(str),
                    "customer_id": chunk["customer_id"].astype(str),
                    "article_id": chunk["article_id"].astype(str),
                    "event_time": pd.to_datetime(chunk["t_dat"], utc=True),
                    "price": pd.to_numeric(chunk["price"], errors="coerce"),
                    "sales_channel_id": chunk["sales_channel_id"].astype(str),
                }
            )

    def iter_source_id_chunks(self, raw_root, *, chunk_size: int = 250_000) -> Iterator[pd.Series]:
        path = self.locate_raw_files(raw_root).files["transactions"]
        for chunk in pd.read_csv(path, usecols=["customer_id"], chunksize=chunk_size):
            yield chunk["customer_id"].astype(str)

    def load_source_entities(self, raw_root, selected_ids: set[str]) -> pd.DataFrame:
        return filter_csv_by_ids(self.locate_raw_files(raw_root).files["customers"], "customer_id", selected_ids)

    def load_destination_entities(self, raw_root, selected_ids: set[str]) -> pd.DataFrame:
        return filter_csv_by_ids(self.locate_raw_files(raw_root).files["articles"], "article_id", selected_ids)


def find_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def filter_csv_by_ids(path: Path, column: str, ids: set[str]) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        chunk[column] = chunk[column].astype(str)
        filtered = chunk[chunk[column].isin(ids)]
        if len(filtered):
            frames.append(filtered.copy())
    if not frames:
        return pd.DataFrame({column: sorted(ids)})
    return pd.concat(frames, ignore_index=True)
