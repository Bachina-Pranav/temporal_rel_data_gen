#!/usr/bin/env python3
"""Create a smaller rel-amazon database from RelBench.

The script samples customers, keeps every review for those customers, then keeps
every product referenced by those reviews. It writes CSV files under
data/original/rel-amazon-toy by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CUSTOMER_ALIASES = ("customers", "customer")
PRODUCT_ALIASES = ("products", "product")
REVIEW_ALIASES = ("reviews", "review")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download rel-amazon if needed and create a customer-induced toy subset."
    )
    parser.add_argument("--dataset-name", default="rel-amazon")
    parser.add_argument("--num-customers", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="data/original/rel-amazon-toy",
        help="Directory where the toy CSV files and schema summary are written.",
    )
    parser.add_argument(
        "--sample-from",
        choices=["reviewed", "all"],
        default="reviewed",
        help=(
            "reviewed samples customers that have at least one review in the loaded "
            "database; all samples from the full customer table."
        ),
    )
    parser.add_argument(
        "--full-database",
        action="store_true",
        help="Include data after the RelBench test timestamp. By default only the train/validation-time database is loaded.",
    )
    parser.add_argument(
        "--no-syntherela-metadata",
        action="store_true",
        help="Skip best-effort Syntherela metadata.json creation.",
    )
    return parser.parse_args()


def load_relbench_db(dataset_name: str, upto_test_timestamp: bool):
    try:
        from relbench.datasets import get_dataset
    except ImportError as exc:
        raise SystemExit(
            "relbench is required. Install the project dependencies first, for example:\n"
            "    pip install -e .\n"
            "or:\n"
            "    pip install 'relbench[full]'"
        ) from exc

    print(f"Loading {dataset_name}; RelBench will download/cache it if needed.")
    try:
        dataset = get_dataset(dataset_name, download=True)
    except TypeError:
        dataset = get_dataset(dataset_name)

    try:
        return dataset.get_db(upto_test_timestamp=upto_test_timestamp)
    except TypeError:
        return dataset.get_db()


def resolve_table_name(table_dict: dict[str, Any], aliases: tuple[str, ...]) -> str:
    for name in aliases:
        if name in table_dict:
            return name
    available = ", ".join(sorted(table_dict))
    raise SystemExit(
        f"Could not find any of {aliases} in RelBench tables. Available tables: {available}"
    )


def table_df(table: Any) -> pd.DataFrame:
    return table.df.copy()


def find_foreign_key(
    db: Any,
    child_table_name: str,
    parent_table_name: str,
    fallback_columns: tuple[str, ...],
) -> str:
    child_table = db.table_dict[child_table_name]
    child_columns = set(child_table.df.columns)
    fkey_map = getattr(child_table, "fkey_col_to_pkey_table", {}) or {}
    for column, target_table in fkey_map.items():
        if target_table == parent_table_name and column in child_columns:
            return column

    parent_pkey = getattr(db.table_dict[parent_table_name], "pkey_col", None)
    for column in (parent_pkey, *fallback_columns):
        if column is not None and column in child_columns:
            return column

    raise SystemExit(
        f"Could not identify a foreign key from {child_table_name} to {parent_table_name}."
    )


def normalize_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), default=str)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return value


def normalize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if df[column].dtype != "object":
            continue
        sample = df[column].dropna().head(1000)
        if sample.map(lambda value: isinstance(value, (list, tuple, set, dict))).any():
            df[column] = df[column].map(normalize_cell)
    return df


def row_counts(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: len(table) for name, table in tables.items()}


def print_row_counts(title: str, counts: dict[str, int]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for table_name, count in counts.items():
        print(f"{table_name}: {count:,}")
    print(f"total: {sum(counts.values()):,}")


def build_relationships(db: Any, table_names: set[str]) -> list[dict[str, str]]:
    relationships = []
    for child_table_name in sorted(table_names):
        child_table = db.table_dict[child_table_name]
        fkey_map = getattr(child_table, "fkey_col_to_pkey_table", {}) or {}
        for child_foreign_key, parent_table_name in fkey_map.items():
            if parent_table_name not in table_names:
                continue
            parent_table = db.table_dict[parent_table_name]
            parent_primary_key = getattr(parent_table, "pkey_col", None)
            if parent_primary_key is None:
                continue
            relationships.append(
                {
                    "parent_table_name": parent_table_name,
                    "child_table_name": child_table_name,
                    "parent_primary_key": parent_primary_key,
                    "child_foreign_key": child_foreign_key,
                }
            )
    return relationships


def maybe_write_syntherela_metadata(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    db: Any,
    relationships: list[dict[str, str]],
) -> bool:
    try:
        from syntherela.data import save_tables
        from syntherela.metadata import Metadata
    except ImportError:
        print("Syntherela is not installed; wrote CSV files without metadata.json.")
        return False

    try:
        metadata = Metadata()
        metadata.detect_from_dataframes(tables)

        for relationship in metadata.relationships.copy():
            metadata.remove_relationship(
                parent_table_name=relationship["parent_table_name"],
                child_table_name=relationship["child_table_name"],
            )

        for table_name, df in tables.items():
            pkey_col = getattr(db.table_dict[table_name], "pkey_col", None)
            if pkey_col is not None and pkey_col in df.columns:
                metadata.update_column(table_name, pkey_col, sdtype="id")
                if metadata.get_primary_key(table_name) != pkey_col:
                    metadata.set_primary_key(table_name, pkey_col)

            time_col = getattr(db.table_dict[table_name], "time_col", None)
            if time_col is not None and time_col in df.columns:
                metadata.update_column(table_name, time_col, sdtype="datetime")

        for relationship in relationships:
            child_table = relationship["child_table_name"]
            child_foreign_key = relationship["child_foreign_key"]
            if child_foreign_key not in tables[child_table].columns:
                continue
            metadata.update_column(child_table, child_foreign_key, sdtype="id")
            metadata.add_relationship(**relationship)

        for table_name, df in tables.items():
            for column in metadata.get_column_names(table_name, sdtype="numerical"):
                dtype = str(df[column].dtype)
                if dtype.startswith("int") or dtype.startswith("Int"):
                    metadata.update_column(
                        table_name, column, computer_representation="Int64"
                    )
                elif dtype.startswith("float") or dtype.startswith("Float"):
                    metadata.update_column(
                        table_name, column, computer_representation="Float"
                    )

        metadata.validate()
        metadata.validate_data(tables)
        save_tables(tables, str(output_dir), metadata=metadata, save_metadata=True)
        print(f"Wrote Syntherela metadata to {output_dir / 'metadata.json'}")
        return True
    except Exception as exc:
        print(f"Could not create Syntherela metadata.json: {exc}")
        print("CSV files and relbench_schema.json were still written.")
        return False


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    upto_test_timestamp = not args.full_database
    db = load_relbench_db(args.dataset_name, upto_test_timestamp=upto_test_timestamp)
    table_dict = db.table_dict

    customer_table_name = resolve_table_name(table_dict, CUSTOMER_ALIASES)
    product_table_name = resolve_table_name(table_dict, PRODUCT_ALIASES)
    review_table_name = resolve_table_name(table_dict, REVIEW_ALIASES)
    selected_table_names = [customer_table_name, product_table_name, review_table_name]

    source_tables = {
        table_name: table_df(table_dict[table_name]) for table_name in selected_table_names
    }
    print_row_counts(
        f"{args.dataset_name} row counts (upto_test_timestamp={upto_test_timestamp})",
        row_counts(source_tables),
    )

    customer_pk = getattr(table_dict[customer_table_name], "pkey_col", None)
    product_pk = getattr(table_dict[product_table_name], "pkey_col", None)
    if customer_pk is None or product_pk is None:
        raise SystemExit("Customer and product tables must have primary keys.")

    review_customer_fk = find_foreign_key(
        db,
        review_table_name,
        customer_table_name,
        fallback_columns=("customer_id", "user_id", "uid"),
    )
    review_product_fk = find_foreign_key(
        db,
        review_table_name,
        product_table_name,
        fallback_columns=("product_id", "item_id", "asin"),
    )

    customers = source_tables[customer_table_name]
    reviews = source_tables[review_table_name]
    products = source_tables[product_table_name]

    if args.sample_from == "reviewed":
        eligible_customer_ids = reviews[review_customer_fk].dropna().unique()
        eligible_customers = customers[customers[customer_pk].isin(eligible_customer_ids)]
    else:
        eligible_customers = customers

    if eligible_customers.empty:
        raise SystemExit("No eligible customers were found for sampling.")

    sample_size = min(args.num_customers, len(eligible_customers))
    toy_customers = eligible_customers.sample(
        n=sample_size, random_state=args.seed, replace=False
    )
    selected_customer_ids = set(toy_customers[customer_pk].tolist())
    toy_reviews = reviews[reviews[review_customer_fk].isin(selected_customer_ids)]

    selected_product_ids = set(toy_reviews[review_product_fk].dropna().tolist())
    toy_products = products[products[product_pk].isin(selected_product_ids)]

    missing_product_count = len(selected_product_ids - set(toy_products[product_pk].tolist()))
    if missing_product_count:
        print(f"Warning: {missing_product_count:,} reviewed products were not found.")

    toy_tables = {
        customer_table_name: normalize_for_csv(toy_customers),
        product_table_name: normalize_for_csv(toy_products),
        review_table_name: normalize_for_csv(toy_reviews),
    }

    for table_name, df in toy_tables.items():
        df.to_csv(output_dir / f"{table_name}.csv", index=False)

    relationships = build_relationships(db, set(selected_table_names))
    primary_keys = {
        table_name: getattr(table_dict[table_name], "pkey_col", None)
        for table_name in selected_table_names
    }
    time_columns = {
        table_name: getattr(table_dict[table_name], "time_col", None)
        for table_name in selected_table_names
    }
    schema = {
        "source_dataset": args.dataset_name,
        "source_upto_test_timestamp": upto_test_timestamp,
        "output_dir": str(output_dir),
        "num_customers_requested": args.num_customers,
        "num_customers_sampled": len(toy_customers),
        "seed": args.seed,
        "sample_from": args.sample_from,
        "table_names": {
            "customers": customer_table_name,
            "products": product_table_name,
            "reviews": review_table_name,
        },
        "primary_keys": primary_keys,
        "relationships": relationships,
        "time_columns": time_columns,
        "review_customer_foreign_key": review_customer_fk,
        "review_product_foreign_key": review_product_fk,
        "source_row_counts": row_counts(source_tables),
        "toy_row_counts": row_counts(toy_tables),
    }
    with (output_dir / "relbench_schema.json").open("w") as handle:
        json.dump(schema, handle, indent=2)
        handle.write("\n")

    if not args.no_syntherela_metadata:
        maybe_write_syntherela_metadata(output_dir, toy_tables, db, relationships)

    print_row_counts("amazon toy row counts", row_counts(toy_tables))
    print(f"\nWrote toy dataset to {output_dir}")
    print(f"Wrote RelBench schema summary to {output_dir / 'relbench_schema.json'}")


if __name__ == "__main__":
    main()
