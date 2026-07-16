"""MovieLens-25M adapter for induced 100K interaction benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import (
    DownloadResult,
    InteractionDatasetAdapter,
    RawFileBundle,
    download_url,
    read_csv_chunks_with_row_number,
    safe_extract_archive,
    write_download_metadata,
)


class MovieLensAdapter(InteractionDatasetAdapter):
    dataset_name = "movielens"
    benchmark_name = "movielens_100k"
    source_dataset = "MovieLens 25M"
    source_entity_name = "user"
    destination_entity_name = "movie"
    source_id_column = "user_id"
    destination_id_column = "movie_id"
    generated_attributes = ("rating",)
    attribute_types = {"rating": "ordinal_categorical"}
    support_tables = ("users.csv", "movies.csv")
    domain = "movie ratings"
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

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
            self.locate_raw_files(raw_root).require("ratings", "movies")
            return DownloadResult(self.dataset_name, "verified_existing", raw_dir, "MovieLens raw files already exist")
        archive_path = download_url(self.url, raw_dir / "ml-25m.zip", force=force)
        safe_extract_archive(archive_path, raw_dir)
        result = DownloadResult(self.dataset_name, "downloaded_automatically", raw_dir, "Downloaded MovieLens 25M", {"url": self.url})
        write_download_metadata(result, [archive_path, *raw_dir.rglob("*.csv"), *raw_dir.rglob("README.txt")], raw_dir / "download_metadata.json")
        return result

    def locate_raw_files(self, raw_root) -> RawFileBundle:
        raw_dir = self.raw_dir(raw_root)
        candidates = [raw_dir / "ml-25m", raw_dir]
        for base in candidates:
            ratings = base / "ratings.csv"
            movies = base / "movies.csv"
            if ratings.exists() and movies.exists():
                return RawFileBundle({"ratings": ratings, "movies": movies})
        raise FileNotFoundError(f"Missing MovieLens 25M files under {raw_dir}; expected ml-25m/ratings.csv and movies.csv")

    def iter_interaction_chunks(self, raw_root, *, chunk_size: int = 250_000) -> Iterator[pd.DataFrame]:
        files = self.locate_raw_files(raw_root)
        for chunk in read_csv_chunks_with_row_number(files.files["ratings"], chunk_size=chunk_size):
            out = pd.DataFrame(
                {
                    "event_id": "ml-rating-" + chunk["_raw_row_number"].astype(str),
                    "user_id": chunk["userId"].astype(str),
                    "movie_id": chunk["movieId"].astype(str),
                    "event_time": pd.to_datetime(chunk["timestamp"], unit="s", utc=True),
                    "rating": chunk["rating"].astype(str),
                }
            )
            yield out

    def iter_source_id_chunks(self, raw_root, *, chunk_size: int = 250_000) -> Iterator[pd.Series]:
        path = self.locate_raw_files(raw_root).files["ratings"]
        for chunk in pd.read_csv(path, usecols=["userId"], chunksize=chunk_size):
            yield chunk["userId"].astype(str)

    def load_destination_entities(self, raw_root, selected_ids: set[str]) -> pd.DataFrame:
        movies = pd.read_csv(self.locate_raw_files(raw_root).files["movies"])
        movies = movies.rename(columns={"movieId": "movie_id"})
        movies["movie_id"] = movies["movie_id"].astype(str)
        return movies[movies["movie_id"].isin(selected_ids)].reset_index(drop=True)
