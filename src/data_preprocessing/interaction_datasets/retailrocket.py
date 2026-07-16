"""RetailRocket e-commerce event adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import DownloadResult, InteractionDatasetAdapter, RawFileBundle, kaggle_download, read_csv_chunks_with_row_number, safe_extract_archive, write_csv, write_download_metadata


class RetailRocketAdapter(InteractionDatasetAdapter):
    dataset_name = "retailrocket"
    benchmark_name = "retailrocket_100k"
    source_dataset = "RetailRocket recommender-system dataset"
    source_entity_name = "visitor"
    destination_entity_name = "item"
    source_id_column = "visitor_id"
    destination_id_column = "item_id"
    generated_attributes = ("event_type",)
    attribute_types = {"event_type": "categorical"}
    support_tables = ("visitors.csv", "items.csv", "item_properties.csv", "category_tree.csv")
    excluded_columns = {"transactionid": "audit-only identifier; excluded from model-ready interaction table"}
    domain = "e-commerce events"
    kaggle_slug = "retailrocket/ecommerce-dataset"

    def download(self, raw_root, *, force=False, verify_only=False, archive=None, **kwargs) -> DownloadResult:
        del kwargs
        raw_dir = self.raw_dir(raw_root)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if archive is not None:
            if not verify_only:
                safe_extract_archive(archive, raw_dir)
            result = DownloadResult(self.dataset_name, "loaded_from_local_archive", raw_dir, f"Loaded {archive}", {"archive": str(archive)})
            write_download_metadata(result, list(raw_dir.rglob("*")), raw_dir / "download_metadata.json")
            return result
        if verify_only:
            self.locate_raw_files(raw_root).require("events")
            return DownloadResult(self.dataset_name, "verified_existing", raw_dir, "RetailRocket raw files already exist")
        ok, message = kaggle_download(self.kaggle_slug, raw_dir, force=force)
        if not ok:
            return DownloadResult(self.dataset_name, "blocked_missing_credentials_or_license_acceptance", raw_dir, message, {"kaggle_slug": self.kaggle_slug})
        for archive_path in raw_dir.glob("*.zip"):
            safe_extract_archive(archive_path, raw_dir)
        result = DownloadResult(self.dataset_name, "downloaded_automatically", raw_dir, message, {"kaggle_slug": self.kaggle_slug})
        write_download_metadata(result, list(raw_dir.rglob("*")), raw_dir / "download_metadata.json")
        return result

    def locate_raw_files(self, raw_root) -> RawFileBundle:
        raw_dir = self.raw_dir(raw_root)
        files = {
            "events": find_file(raw_dir, "events.csv"),
            "item_properties_part1": find_file(raw_dir, "item_properties_part1.csv"),
            "item_properties_part2": find_file(raw_dir, "item_properties_part2.csv"),
            "category_tree": find_file(raw_dir, "category_tree.csv"),
        }
        if files["events"] is None:
            raise FileNotFoundError(f"Missing RetailRocket events.csv under {raw_dir}")
        return RawFileBundle({key: path for key, path in files.items() if path is not None})

    def iter_interaction_chunks(self, raw_root, *, chunk_size: int = 250_000) -> Iterator[pd.DataFrame]:
        for chunk in read_csv_chunks_with_row_number(self.locate_raw_files(raw_root).files["events"], chunk_size=chunk_size):
            audit_transaction = chunk.get("transactionid")
            out = pd.DataFrame(
                {
                    "event_id": "rr-event-" + chunk["_raw_row_number"].astype(str),
                    "visitor_id": chunk["visitorid"].astype(str),
                    "item_id": chunk["itemid"].astype(str),
                    "event_time": pd.to_datetime(chunk["timestamp"], unit="ms", utc=True),
                    "event_type": chunk["event"].astype(str).map(normalize_event_type),
                }
            )
            if audit_transaction is not None:
                out["_audit_transactionid"] = audit_transaction
            yield out

    def load_destination_entities(self, raw_root, selected_ids: set[str]) -> pd.DataFrame:
        return pd.DataFrame({"item_id": sorted(str(value) for value in selected_ids)})

    def load_extra_support_tables(self, raw_root, destination_ids: set[str], output_dir: Path) -> dict[str, Path]:
        files = self.locate_raw_files(raw_root).files
        out: dict[str, Path] = {}
        parts = [files.get("item_properties_part1"), files.get("item_properties_part2")]
        frames = []
        for path in parts:
            if path is not None and path.exists():
                for frame in pd.read_csv(path, chunksize=250_000):
                    frame["itemid"] = frame["itemid"].astype(str)
                    filtered = frame[frame["itemid"].isin(destination_ids)]
                    if len(filtered):
                        frames.append(filtered.copy())
        if frames:
            out["item_properties"] = write_csv(pd.concat(frames, ignore_index=True), output_dir / "item_properties.csv")
        if files.get("category_tree") is not None:
            out["category_tree"] = write_csv(pd.read_csv(files["category_tree"]), output_dir / "category_tree.csv")
        return out


def normalize_event_type(value: object) -> str:
    text = str(value).strip().lower()
    mapping = {"view": "view", "addtocart": "addtocart", "transaction": "transaction"}
    if text not in mapping:
        raise ValueError(f"Unknown RetailRocket event type: {value!r}")
    return mapping[text]


def find_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None
