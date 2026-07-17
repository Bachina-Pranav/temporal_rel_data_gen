#!/usr/bin/env python3
"""Build a RelBench rel-hm induced subset from complete active-customer histories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preprocessing.interaction_datasets.base import (  # noqa: E402
    file_hashes,
    read_csv_chunks_with_row_number,
    utc_now_iso,
    write_csv,
)
from data_preprocessing.interaction_datasets.hm import HM10KCustomersAdapter, HMAdapter  # noqa: E402
from data_preprocessing.interaction_datasets.statistics import degree_summary, gini  # noqa: E402
from data_preprocessing.interaction_datasets.validation import validate_subset  # noqa: E402


REQUIRED_TRANSACTION_COLUMNS = ["customer_id", "article_id", "t_dat", "price", "sales_channel_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the RelBench rel-hm 10k-active-customer induced subset.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--relbench-root", default="data/original")
    parser.add_argument("--processed-root", default="data/processed/interaction_benchmarks")
    parser.add_argument("--output-name", default="hm_10k_customers")
    parser.add_argument("--num-customers", "--num-source-entities", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--archive", default=None, help="Optional legacy Kaggle H&M zip/archive to extract if RelBench CSVs are unavailable.")
    parser.add_argument("--force-download", action="store_true", help="Force legacy adapter download when applicable.")
    parser.add_argument("--no-download", action="store_true", help="Fail immediately if raw CSVs are missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_hm_induced_subset(
        raw_root=args.raw_root,
        relbench_root=args.relbench_root,
        processed_root=args.processed_root,
        output_name=args.output_name,
        num_customers=int(args.num_customers),
        seed=int(args.seed),
        chunk_size=int(args.chunk_size),
        archive=args.archive,
        force_download=bool(args.force_download),
        download_if_missing=not bool(args.no_download),
    )
    print(json.dumps(manifest, sort_keys=True, default=str))


def build_hm_induced_subset(
    *,
    raw_root: str | Path,
    processed_root: str | Path,
    relbench_root: str | Path = "data/original",
    output_name: str = "hm_10k_customers",
    num_customers: int = 10_000,
    seed: int = 42,
    chunk_size: int = 500_000,
    archive: str | Path | None = None,
    force_download: bool = False,
    download_if_missing: bool = True,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    relbench_root = Path(relbench_root)
    output_dir = Path(processed_root) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = locate_or_download_hm_raw(
        raw_root,
        relbench_root=relbench_root,
        archive=archive,
        force_download=force_download,
        download_if_missing=download_if_missing,
    )

    print("[hm] scanning raw tables", flush=True)
    raw_stats, raw_customer_counts = scan_raw_tables(raw_files, chunk_size=chunk_size)
    active_ids = np.asarray(sorted(raw_customer_counts), dtype=object)
    if len(active_ids) < int(num_customers):
        raise ValueError(f"Requested {num_customers:,} active customers but only found {len(active_ids):,}")
    rng = np.random.default_rng(int(seed))
    selected = sorted(str(value) for value in rng.choice(active_ids, size=int(num_customers), replace=False))
    selected_set = set(selected)
    (output_dir / "selected_customer_ids.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"[hm] materializing complete histories for {len(selected):,} customers", flush=True)
    interactions = materialize_transactions(raw_files["transactions"], selected_set, chunk_size=chunk_size)
    interactions = assign_chronological_splits(interactions)
    referenced_articles = set(interactions["article_id"].astype(str))

    customers = filter_entity_table(raw_files["customers"], "customer_id", selected_set, chunk_size=chunk_size)
    articles = filter_entity_table(raw_files["articles"], "article_id", referenced_articles, chunk_size=chunk_size)

    paths = {
        "interactions": write_csv(interactions, output_dir / "interactions.csv"),
        "customers.csv": write_csv(customers, output_dir / "customers.csv"),
        "articles.csv": write_csv(articles, output_dir / "articles.csv"),
    }
    write_schema_yaml(output_dir / "schema.yaml")
    validation = strict_validation_report(
        output_dir,
        interactions,
        customers,
        articles,
        raw_customer_counts,
        selected,
        raw_stats,
    )
    statistics = compute_hm_statistics(interactions, customers, articles, raw_stats)
    manifest = build_manifest(
        output_name,
        raw_files,
        paths,
        interactions,
        customers,
        articles,
        validation,
        statistics,
        selected,
        seed,
        raw_stats,
    )
    write_json(validation, output_dir / "validation_report.json")
    write_json(statistics, output_dir / "statistics.json")
    write_json(manifest, output_dir / "subset_manifest.json")
    write_statistics_markdown(statistics, output_dir / "statistics.md")
    write_readme(output_dir / "README.md", manifest)
    print(f"[hm] wrote {output_dir}", flush=True)
    return manifest


def locate_or_download_hm_raw(
    raw_root: Path,
    *,
    relbench_root: Path,
    archive: str | Path | None = None,
    force_download: bool = False,
    download_if_missing: bool = True,
) -> dict[str, Path]:
    first_error: Exception | None = None
    try:
        return locate_hm_source_files(raw_root=raw_root, relbench_root=relbench_root, allow_legacy=False)
    except FileNotFoundError as exc:
        first_error = exc
    if archive is not None:
        adapter = HMAdapter()
        print(f"[hm] source files missing; extracting local archive {archive}", flush=True)
        result = adapter.download(raw_root, force=force_download, archive=archive)
        print_download_result(result)
        return locate_hm_source_files(raw_root=raw_root, relbench_root=relbench_root, allow_legacy=True)
    if not download_if_missing:
        raise FileNotFoundError(str(first_error))
    target_dir = relbench_root / "rel-hm"
    print(f"[hm] RelBench rel-hm files missing; attempting RelBench download/cache into {target_dir}", flush=True)
    download_relbench_hm(target_dir)
    try:
        return locate_hm_source_files(raw_root=raw_root, relbench_root=relbench_root, allow_legacy=False)
    except FileNotFoundError as second_error:
        message = (
            f"{second_error}. Attempted RelBench get_dataset('rel-hm') after: {first_error}. "
            "Install relbench dependencies and ensure the dataset can be downloaded/cached, "
            "or pass --archive only if you intentionally want to use the legacy Kaggle H&M CSV layout."
        )
        raise FileNotFoundError(message) from second_error


def locate_hm_source_files(*, raw_root: Path, relbench_root: Path, allow_legacy: bool = False) -> dict[str, Path]:
    candidates = [
        {
            "layout": "relbench",
            "root": relbench_root / "rel-hm",
            "transactions": "transactions.csv",
            "customers": "customer.csv",
            "articles": "article.csv",
        },
        {
            "layout": "relbench",
            "root": raw_root / "rel-hm",
            "transactions": "transactions.csv",
            "customers": "customer.csv",
            "articles": "article.csv",
        },
    ]
    if allow_legacy:
        candidates.append(
            {
                "layout": "legacy_kaggle",
                "root": raw_root / "hm",
                "transactions": "transactions_train.csv",
                "customers": "customers.csv",
                "articles": "articles.csv",
            }
        )
    attempted = []
    for spec in candidates:
        root = Path(spec["root"])
        files = {
            "transactions": find_named_file(root, str(spec["transactions"])),
            "customers": find_named_file(root, str(spec["customers"])),
            "articles": find_named_file(root, str(spec["articles"])),
        }
        attempted.append({key: str(value) if value is not None else str(root / str(spec[key])) for key, value in files.items()})
        if all(files.values()):
            print(f"[hm] using {spec['layout']} source files under {root}", flush=True)
            return {key: path for key, path in files.items() if path is not None}
    raise FileNotFoundError(f"Missing RelBench rel-hm CSV files. Tried: {attempted}")


def find_named_file(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    direct = root / filename
    if direct.exists():
        return direct
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def download_relbench_hm(output_dir: Path) -> None:
    try:
        from relbench.datasets import get_dataset
    except ImportError as exc:
        raise RuntimeError("relbench is required to download rel-hm. Install project dependencies including relbench[full].") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset("rel-hm")
    db = dataset.get_db(upto_test_timestamp=True)
    summary = {}
    for table_name, table in db.table_dict.items():
        frame = table.df.copy()
        if table_name == "customer" and "postal_code" in frame.columns:
            frame = frame.drop(columns=["postal_code"])
        path = output_dir / f"{table_name}.csv"
        frame.to_csv(path, index=False)
        summary[str(table_name)] = {"rows": int(len(frame)), "columns": list(frame.columns), "path": str(path)}
        print(f"[hm] wrote RelBench table {path} rows={len(frame):,}", flush=True)
    write_json(
        {
            "dataset_name": "rel-hm",
            "source": "relbench.datasets.get_dataset('rel-hm')",
            "upto_test_timestamp": True,
            "tables": summary,
        },
        output_dir / "relbench_export_summary.json",
    )


def print_download_result(result) -> None:
    print(
        json.dumps(
            {
                "dataset_name": result.dataset_name,
                "download_status": result.status,
                "raw_dir": str(result.raw_dir),
                "message": result.message,
                "metadata": result.metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def scan_raw_tables(raw_files: dict[str, Path], *, chunk_size: int) -> tuple[dict[str, Any], dict[str, int]]:
    customer_schema, num_customers = csv_schema_and_count(raw_files["customers"], chunk_size=chunk_size)
    article_schema, num_articles = csv_schema_and_count(raw_files["articles"], chunk_size=chunk_size)
    counts: dict[str, int] = {}
    rows = 0
    missing = {column: 0 for column in REQUIRED_TRANSACTION_COLUMNS}
    price_parts: list[np.ndarray] = []
    channel_counts: dict[str, int] = {}
    date_min = None
    date_max = None
    chunk_duplicate_like_rows = 0
    schema: dict[str, str] | None = None
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            raw_files["transactions"],
            usecols=REQUIRED_TRANSACTION_COLUMNS,
            dtype={"customer_id": "string", "article_id": "string", "sales_channel_id": "string"},
            chunksize=chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        if schema is None:
            schema = {str(column): str(dtype) for column, dtype in chunk.dtypes.items()}
        rows += int(len(chunk))
        for column in REQUIRED_TRANSACTION_COLUMNS:
            missing[column] += int(chunk[column].isna().sum())
        customer_values = chunk["customer_id"].astype(str)
        for key, value in customer_values.value_counts().items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
        dates = pd.to_datetime(chunk["t_dat"], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates):
            cmin = valid_dates.min()
            cmax = valid_dates.max()
            date_min = cmin if date_min is None or cmin < date_min else date_min
            date_max = cmax if date_max is None or cmax > date_max else date_max
        prices = pd.to_numeric(chunk["price"], errors="coerce").to_numpy(dtype=float)
        price_parts.append(prices[np.isfinite(prices)])
        for key, value in chunk["sales_channel_id"].astype(str).value_counts(dropna=False).items():
            channel_counts[str(key)] = channel_counts.get(str(key), 0) + int(value)
        chunk_duplicate_like_rows += int(chunk.duplicated(subset=REQUIRED_TRANSACTION_COLUMNS, keep="first").sum())
        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(f"[hm] raw scan chunks={chunk_idx:,} transactions={rows:,} active_customers={len(counts):,}", flush=True)
    price_values = np.concatenate(price_parts) if price_parts else np.asarray([], dtype=float)
    stats = {
        "raw_paths": {key: str(path) for key, path in raw_files.items()},
        "customers": {"row_count": int(num_customers), "schema": customer_schema},
        "articles": {"row_count": int(num_articles), "schema": article_schema},
        "transactions": {
            "row_count": int(rows),
            "schema": schema or {},
            "active_customer_count": int(len(counts)),
            "timestamp_min": date_min.isoformat() if date_min is not None else None,
            "timestamp_max": date_max.isoformat() if date_max is not None else None,
            "missing_required_values": {key: int(value) for key, value in missing.items()},
            "duplicate_like_rows_within_chunks": int(chunk_duplicate_like_rows),
        },
        "price": price_summary(price_values),
        "sales_channel_id": {
            "domain": sorted(channel_counts),
            "counts": {str(key): int(value) for key, value in sorted(channel_counts.items())},
        },
    }
    return stats, counts


def csv_schema_and_count(path: Path, *, chunk_size: int) -> tuple[dict[str, str], int]:
    first = pd.read_csv(path, nrows=0, low_memory=False)
    schema = {str(column): str(dtype) for column, dtype in first.dtypes.items()}
    rows = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        rows += int(len(chunk))
    return schema, rows


def materialize_transactions(path: Path, selected: set[str], *, chunk_size: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    retained = 0
    rows = 0
    for chunk_idx, chunk in enumerate(
        read_csv_chunks_with_row_number(
            path,
            chunk_size=chunk_size,
            usecols=REQUIRED_TRANSACTION_COLUMNS,
            dtype={"customer_id": "string", "article_id": "string", "sales_channel_id": "string"},
            low_memory=False,
        ),
        start=1,
    ):
        rows += int(len(chunk))
        mask = chunk["customer_id"].astype(str).isin(selected)
        if bool(mask.any()):
            kept = chunk.loc[mask].copy()
            event_time = pd.to_datetime(kept["t_dat"], errors="coerce")
            canonical = pd.DataFrame(
                {
                    "event_id": "hm-transaction-" + kept["_raw_row_number"].astype(str),
                    "customer_id": kept["customer_id"].astype(str),
                    "article_id": kept["article_id"].astype(str),
                    "event_time": event_time.dt.strftime("%Y-%m-%d"),
                    "price": pd.to_numeric(kept["price"], errors="coerce"),
                    "sales_channel_id": kept["sales_channel_id"].astype(str),
                }
            )
            pieces.append(canonical)
            retained += int(len(canonical))
        if chunk_idx == 1 or chunk_idx % 10 == 0:
            print(f"[hm] materialize chunks={chunk_idx:,} rows={rows:,} retained={retained:,}", flush=True)
    if not pieces:
        raise ValueError("No Rel-H&M transactions retained")
    return pd.concat(pieces, ignore_index=True)


def filter_entity_table(path: Path, column: str, ids: set[str], *, chunk_size: int) -> pd.DataFrame:
    pieces = []
    for chunk in pd.read_csv(path, dtype={column: "string"}, chunksize=chunk_size, low_memory=False):
        chunk[column] = chunk[column].astype(str)
        filtered = chunk.loc[chunk[column].isin(ids)].copy()
        if len(filtered):
            pieces.append(filtered)
    if not pieces:
        return pd.DataFrame(columns=[column])
    return pd.concat(pieces, ignore_index=True)


def assign_chronological_splits(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_event_time_sort"] = pd.to_datetime(out["event_time"], errors="coerce")
    out = out.sort_values(["_event_time_sort", "event_id"], kind="mergesort").drop(columns=["_event_time_sort"]).reset_index(drop=True)
    n = len(out)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    split = np.empty(n, dtype=object)
    split[:train_end] = "train"
    split[train_end:valid_end] = "validation"
    split[valid_end:] = "test"
    out["split"] = split
    return out


def strict_validation_report(
    output_dir: Path,
    interactions: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    raw_customer_counts: dict[str, int],
    selected: list[str],
    raw_stats: dict[str, Any],
) -> dict[str, Any]:
    adapter = HM10KCustomersAdapter()
    raw_selected_counts = {customer_id: raw_customer_counts[customer_id] for customer_id in selected}
    generic = validate_subset(adapter, output_dir, raw_counts=raw_selected_counts)
    errors = list(generic.get("errors", []))
    warnings = list(generic.get("warnings", []))
    selected_set = set(selected)
    customer_ids = customers["customer_id"].astype(str) if "customer_id" in customers else pd.Series(dtype=str)
    article_ids = articles["article_id"].astype(str) if "article_id" in articles else pd.Series(dtype=str)
    transaction_customers = set(interactions["customer_id"].astype(str))
    transaction_articles = set(interactions["article_id"].astype(str))
    if len(customers) != len(selected):
        errors.append(f"Expected {len(selected):,} customer rows, got {len(customers):,}")
    if set(customer_ids) != selected_set:
        missing = sorted(selected_set.difference(set(customer_ids)))[:20]
        extra = sorted(set(customer_ids).difference(selected_set))[:20]
        errors.append(f"Customer table does not exactly match selected IDs; missing={missing}, extra={extra}")
    if transaction_customers != selected_set:
        missing = sorted(selected_set.difference(transaction_customers))[:20]
        extra = sorted(transaction_customers.difference(selected_set))[:20]
        errors.append(f"Transaction customers do not exactly match selected IDs; missing={missing}, extra={extra}")
    if not transaction_articles.issubset(set(article_ids)):
        errors.append("Not every referenced article is present in articles.csv")
    if customer_ids.duplicated().any():
        errors.append("customers.customer_id is not unique")
    if article_ids.duplicated().any():
        errors.append("articles.article_id is not unique")
    for column in ["event_id", "customer_id", "article_id", "event_time", "price", "sales_channel_id", "split"]:
        if column not in interactions or interactions[column].isna().any():
            errors.append(f"{column} contains nulls or is missing")
    timestamps = pd.to_datetime(interactions["event_time"], errors="coerce")
    if timestamps.isna().any():
        errors.append("event_time contains unparsable timestamps")
    price = pd.to_numeric(interactions["price"], errors="coerce")
    if price.isna().any() or not np.isfinite(price.to_numpy(dtype=float)).all() or (price < 0).any():
        errors.append("price values must be numeric, finite, and nonnegative")
    train = interactions.loc[interactions["split"] == "train"].copy()
    train_channels = set(train["sales_channel_id"].astype(str))
    val_channels = set(interactions.loc[interactions["split"] == "validation", "sales_channel_id"].astype(str))
    test_channels = set(interactions.loc[interactions["split"] == "test", "sales_channel_id"].astype(str))
    duplicate_like = int(interactions.duplicated(subset=["customer_id", "article_id", "event_time", "price", "sales_channel_id"], keep="first").sum())
    return {
        **generic,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
        "requested_customer_rows": int(len(selected)),
        "selected_customer_rows": int(len(customers)),
        "selected_unique_customers": int(len(set(customer_ids))),
        "selected_customers_active_in_raw": bool(all(raw_customer_counts.get(customer_id, 0) > 0 for customer_id in selected)),
        "complete_source_histories": not any(
            int(interactions["customer_id"].astype(str).value_counts().get(customer_id, 0)) != int(raw_customer_counts[customer_id])
            for customer_id in selected
        ),
        "event_id_unique": bool(not interactions["event_id"].astype(str).duplicated().any()),
        "customer_id_unique": bool(not customer_ids.duplicated().any()),
        "article_id_unique": bool(not article_ids.duplicated().any()),
        "required_fields_non_null": bool(not interactions[["event_id", "customer_id", "article_id", "event_time", "price", "sales_channel_id", "split"]].isna().any().any()),
        "timestamp_parse_error_rate": float(timestamps.isna().mean()) if len(timestamps) else 0.0,
        "price": {
            "split_ranges": price_ranges_by_split(interactions),
            "summary": price_summary(price.to_numpy(dtype=float)),
            "uses_training_split_for_model_transform": True,
        },
        "sales_channel_id": {
            "train_domain": sorted(train_channels),
            "validation_unseen_in_train": sorted(val_channels.difference(train_channels)),
            "test_unseen_in_train": sorted(test_channels.difference(train_channels)),
        },
        "duplicate_like_transactions_preserved": duplicate_like,
        "raw_duplicate_like_rows_within_chunks": int(raw_stats["transactions"]["duplicate_like_rows_within_chunks"]),
    }


def compute_hm_statistics(
    interactions: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    raw_stats: dict[str, Any],
) -> dict[str, Any]:
    timestamps = pd.to_datetime(interactions["event_time"], errors="coerce")
    customer_degree = interactions.groupby("customer_id").size()
    article_degree = interactions.groupby("article_id").size()
    unique_pairs = interactions[["customer_id", "article_id"]].drop_duplicates().shape[0]
    repeat_counts = interactions.groupby(["customer_id", "article_id"]).size()
    price = pd.to_numeric(interactions["price"], errors="coerce")
    daily = counts_by_period(timestamps, "D")
    weekly = counts_by_period(timestamps, "W")
    monthly = counts_by_period(timestamps, "M")
    top_article_counts = article_degree.sort_values(ascending=False)
    return {
        "dataset_name": "hm_10k_customers",
        "raw_dataset": raw_stats,
        "scale": {
            "customers": int(len(customers)),
            "articles": int(len(articles)),
            "transactions": int(len(interactions)),
            "timestamp_min": timestamps.min().isoformat() if len(timestamps) else None,
            "timestamp_max": timestamps.max().isoformat() if len(timestamps) else None,
            "interaction_sparsity": float(1.0 - unique_pairs / max(len(customers) * len(articles), 1)),
        },
        "degree_distributions": {
            "customer": degree_summary(customer_degree),
            "article": degree_summary(article_degree),
            "customer_gini": gini(customer_degree),
            "article_gini": gini(article_degree),
        },
        "temporal_behavior": {
            "transactions_per_day": daily,
            "transactions_per_week": weekly,
            "transactions_per_month": monthly,
            "customer_inter_event_days": inter_event_summary(interactions, "customer_id"),
            "article_inter_event_days": inter_event_summary(interactions, "article_id"),
            "customer_active_window_days": active_window_summary(interactions, "customer_id"),
            "article_active_window_days": active_window_summary(interactions, "article_id"),
        },
        "pair_behavior": {
            "unique_customer_article_pairs": int(unique_pairs),
            "repeated_pair_rate": float(1.0 - unique_pairs / max(len(interactions), 1)),
            "maximum_repeat_count": int(repeat_counts.max()) if len(repeat_counts) else 0,
            "top_article_share": float(top_article_counts.iloc[0] / max(len(interactions), 1)) if len(top_article_counts) else 0.0,
            "top_100_article_share": float(top_article_counts.head(100).sum() / max(len(interactions), 1)) if len(top_article_counts) else 0.0,
        },
        "price_behavior": {
            **price_summary(price.to_numpy(dtype=float)),
            "histogram": histogram(price),
            "log_price_histogram": histogram(np.log1p(price[price >= 0])),
            "by_sales_channel": numeric_by_group(interactions, "sales_channel_id", "price"),
            "by_article_popularity_bucket": numeric_by_bucket(interactions, "article_id", "price", article_degree),
            "by_customer_activity_bucket": numeric_by_bucket(interactions, "customer_id", "price", customer_degree),
            "monthly_mean": monthly_numeric_mean(interactions, "price"),
        },
        "channel_behavior": {
            "distribution": value_counts(interactions["sales_channel_id"]),
            "monthly_distribution": categorical_by_month(interactions, "sales_channel_id"),
            "by_customer_activity_bucket": categorical_by_bucket(interactions, "customer_id", "sales_channel_id", customer_degree),
            "by_article_popularity_bucket": categorical_by_bucket(interactions, "article_id", "sales_channel_id", article_degree),
        },
        "split_counts": value_counts(interactions["split"]),
        "duplicate_like_transactions": int(interactions.duplicated(subset=["customer_id", "article_id", "event_time", "price", "sales_channel_id"], keep="first").sum()),
    }


def price_summary(values: np.ndarray | pd.Series) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    return {
        "count": int(arr.size),
        "dtype": "float",
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "zero_rate": float(np.mean(arr == 0.0)),
        "quantiles": {str(q): float(np.quantile(arr, q)) for q in quantiles},
    }


def price_ranges_by_split(interactions: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for split, frame in interactions.groupby("split"):
        out[str(split)] = price_summary(pd.to_numeric(frame["price"], errors="coerce").to_numpy(dtype=float))
    return out


def counts_by_period(timestamps: pd.Series, freq: str) -> dict[str, int]:
    if timestamps.empty:
        return {}
    return {str(k): int(v) for k, v in timestamps.dt.to_period(freq).astype(str).value_counts().sort_index().items()}


def inter_event_summary(frame: pd.DataFrame, entity_col: str) -> dict[str, Any]:
    work = frame[[entity_col, "event_time"]].copy()
    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce")
    diffs = work.sort_values([entity_col, "event_time"]).groupby(entity_col)["event_time"].diff().dt.total_seconds() / 86400.0
    return numeric_summary(diffs.dropna())


def active_window_summary(frame: pd.DataFrame, entity_col: str) -> dict[str, Any]:
    work = frame[[entity_col, "event_time"]].copy()
    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce")
    grouped = work.groupby(entity_col)["event_time"]
    windows = (grouped.max() - grouped.min()).dt.total_seconds() / 86400.0
    return numeric_summary(windows.dropna())


def numeric_summary(values: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def histogram(values: pd.Series | np.ndarray, bins: int = 20) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"counts": [], "bin_edges": []}
    counts, edges = np.histogram(arr, bins=bins)
    return {"counts": [int(value) for value in counts], "bin_edges": [float(value) for value in edges]}


def numeric_by_group(frame: pd.DataFrame, group_col: str, value_col: str, limit: int = 50) -> dict[str, Any]:
    out = {}
    for key, group in frame.groupby(group_col):
        out[str(key)] = numeric_summary(pd.to_numeric(group[value_col], errors="coerce"))
        if len(out) >= limit:
            break
    return out


def bucket_labels(degrees: pd.Series) -> pd.Series:
    ranks = degrees.rank(method="first")
    try:
        return pd.qcut(ranks, q=min(5, len(degrees)), labels=False, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(["0"] * len(degrees), index=degrees.index)


def numeric_by_bucket(frame: pd.DataFrame, entity_col: str, value_col: str, degrees: pd.Series) -> dict[str, Any]:
    labels = bucket_labels(degrees)
    work = frame[[entity_col, value_col]].copy()
    work["_bucket"] = work[entity_col].map(labels).fillna("missing")
    return {str(key): numeric_summary(pd.to_numeric(group[value_col], errors="coerce")) for key, group in work.groupby("_bucket")}


def categorical_by_bucket(frame: pd.DataFrame, entity_col: str, value_col: str, degrees: pd.Series) -> dict[str, Any]:
    labels = bucket_labels(degrees)
    work = frame[[entity_col, value_col]].copy()
    work["_bucket"] = work[entity_col].map(labels).fillna("missing")
    return {str(key): value_counts(group[value_col]) for key, group in work.groupby("_bucket")}


def monthly_numeric_mean(frame: pd.DataFrame, value_col: str) -> dict[str, float]:
    work = frame[["event_time", value_col]].copy()
    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    return {str(k): float(v) for k, v in work.set_index("event_time")[value_col].resample("M").mean().dropna().items()}


def categorical_by_month(frame: pd.DataFrame, value_col: str) -> dict[str, dict[str, int]]:
    work = frame[["event_time", value_col]].copy()
    work["month"] = pd.to_datetime(work["event_time"], errors="coerce").dt.to_period("M").astype(str)
    return {str(month): value_counts(group[value_col]) for month, group in work.groupby("month")}


def value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.astype(str).value_counts(dropna=False).sort_index().items()}


def build_manifest(
    output_name: str,
    raw_files: dict[str, Path],
    processed_paths: dict[str, Path],
    interactions: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    validation: dict[str, Any],
    statistics: dict[str, Any],
    selected: list[str],
    seed: int,
    raw_stats: dict[str, Any],
) -> dict[str, Any]:
    timestamps = pd.to_datetime(interactions["event_time"], errors="coerce")
    split_cutoffs = {
        split: timestamps[interactions["split"] == split].max().isoformat()
        for split in ["train", "validation", "test"]
        if bool((interactions["split"] == split).any())
    }
    return {
        "dataset_name": output_name,
        "source_dataset": "rel-hm",
        "target_table": "transactions",
        "source_entity_table": "customers",
        "destination_entity_table": "articles",
        "source_id_column": "customer_id",
        "destination_id_column": "article_id",
        "timestamp_column": "event_time",
        "requested_source_entities": int(len(selected)),
        "selected_source_entities": int(customers["customer_id"].astype(str).nunique()) if "customer_id" in customers else 0,
        "selected_destination_entities": int(articles["article_id"].astype(str).nunique()) if "article_id" in articles else 0,
        "actual_interactions": int(len(interactions)),
        "selection_seed": int(seed),
        "active_customers_only": True,
        "complete_source_histories": bool(validation.get("complete_source_histories", False)),
        "foreign_key_valid": bool(validation.get("foreign_key_valid", False)),
        "generated_attributes": ["price", "sales_channel_id"],
        "attribute_types": {"price": "continuous_numerical", "sales_channel_id": "categorical"},
        "timestamp_min": timestamps.min().isoformat() if len(timestamps) else None,
        "timestamp_max": timestamps.max().isoformat() if len(timestamps) else None,
        "split_cutoffs": split_cutoffs,
        "split_counts": value_counts(interactions["split"]),
        "raw_file_hashes": file_hashes(raw_files.values()),
        "processed_file_hashes": file_hashes(processed_paths.values()),
        "selected_customer_ids_path": "selected_customer_ids.txt",
        "created_at": utc_now_iso(),
        "code_version": git_revision(),
        "subset": {
            "num_customers": int(len(selected)),
            "seed": int(seed),
            "require_active_customers": True,
            "preserve_complete_histories": True,
            "selection_rule": "uniform_sample_from_deterministically_sorted_active_customers",
        },
        "raw_summary": raw_stats,
        "statistics_summary": {
            "mean_transactions_per_customer": statistics["degree_distributions"]["customer"]["mean"],
            "median_transactions_per_customer": statistics["degree_distributions"]["customer"]["median"],
        },
    }


def write_schema_yaml(path: Path) -> None:
    schema = {
        "dataset_name": "hm_10k_customers",
        "method_scope": "single_designated_temporal_interaction_table",
        "source_dataset": "rel-hm",
        "target_table": "interactions.csv",
        "source_entity_table": "customers.csv",
        "destination_entity_table": "articles.csv",
        "source_id_column": "customer_id",
        "destination_id_column": "article_id",
        "timestamp_column": "event_time",
        "event_id_column": "event_id",
        "generated_attributes": ["price", "sales_channel_id"],
        "attribute_types": {"price": "continuous_numerical", "sales_channel_id": "categorical"},
        "fields": {
            "event_id": {"role": "primary_key", "semantic_type": "id", "nullable": False},
            "customer_id": {"role": "source_foreign_key", "semantic_type": "foreign_key", "nullable": False},
            "article_id": {"role": "destination_foreign_key", "semantic_type": "foreign_key", "nullable": False},
            "event_time": {"role": "timestamp", "semantic_type": "datetime", "nullable": False, "source_column": "t_dat"},
            "price": {"role": "generated_attribute", "semantic_type": "continuous_numerical", "nullable": False, "preprocessing": "standardize", "output_distribution": "gaussian"},
            "sales_channel_id": {"role": "generated_attribute", "semantic_type": "categorical", "nullable": False},
            "split": {"role": "split", "semantic_type": "categorical", "nullable": False},
        },
        "support_tables": ["customers.csv", "articles.csv"],
    }
    path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_statistics_markdown(stats: dict[str, Any], path: Path) -> None:
    lines = [
        "# Rel-H&M 10k-Customer LSTM Subset Statistics",
        "",
        f"- Customers: {stats['scale']['customers']:,}",
        f"- Articles: {stats['scale']['articles']:,}",
        f"- Transactions: {stats['scale']['transactions']:,}",
        f"- Time span: {stats['scale']['timestamp_min']} to {stats['scale']['timestamp_max']}",
        f"- Customer-degree mean/median/max: {stats['degree_distributions']['customer']['mean']:.3f} / {stats['degree_distributions']['customer']['median']:.3f} / {stats['degree_distributions']['customer']['max']:.0f}",
        f"- Article-degree mean/median/max: {stats['degree_distributions']['article']['mean']:.3f} / {stats['degree_distributions']['article']['median']:.3f} / {stats['degree_distributions']['article']['max']:.0f}",
        f"- Repeated-pair rate: {stats['pair_behavior']['repeated_pair_rate']:.6f}",
        f"- Top-100 article share: {stats['pair_behavior']['top_100_article_share']:.6f}",
        "",
        "## Price",
        "",
        f"- Min/max: {stats['price_behavior']['min']} / {stats['price_behavior']['max']}",
        f"- Mean/median/std: {stats['price_behavior']['mean']} / {stats['price_behavior']['median']} / {stats['price_behavior']['std']}",
        f"- Zero rate: {stats['price_behavior']['zero_rate']}",
        "",
        "## Sales Channel",
        "",
    ]
    for key, value in stats["channel_behavior"]["distribution"].items():
        lines.append(f"- {key}: {value:,}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    text = f"""# Rel-H&M 10k-Customer Induced Subdatabase

This directory contains a complete-history induced subset for RelBench `rel-hm`.

- Selected active customers: {manifest['selected_source_entities']:,}
- Referenced articles: {manifest['selected_destination_entities']:,}
- Retained transactions: {manifest['actual_interactions']:,}
- Event spine: `customer_id`, `article_id`, `event_time`
- Generated attributes: `price`, `sales_channel_id`
- Split rule: chronological 70/15/15 sorted by `event_time`, then `event_id`
- Customer selection: uniform sample from sorted active customer IDs with seed {manifest['selection_seed']}
- Duplicate-looking transaction rows are preserved.
"""
    path.write_text(text, encoding="utf-8")


def git_revision() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


if __name__ == "__main__":
    main()
