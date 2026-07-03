#!/usr/bin/env python3
"""Create a paper-ready event-spine comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


MAIN_METHOD_KEY = "time_biased_local_kernel_main"
MAIN_METHOD_LABEL = "TimeBiasedBlockStubMatching local_kernel + penalized dynamic affinity"

METHOD_LABELS = {
    "static_degree": "StaticDegree",
    "ct_2k_sbm_temporal_kde_stubs": "CT-2K-SBM temporal KDE stubs",
    "time_biased_median_mixture": "TimeBiased median mixture",
    "time_biased_empirical_exact": "TimeBiased empirical exact",
    "time_biased_local_kernel_random_pairing": "TimeBiased local kernel + random pairing",
    MAIN_METHOD_KEY: MAIN_METHOD_LABEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a paper-ready event-spine comparison table.")
    parser.add_argument("--input-json", default="outputs/rel-amazon/event_spine_generator_comparison.json")
    parser.add_argument("--output-csv", default="outputs/rel-amazon/event_spine_paper_table.csv")
    parser.add_argument("--output-md", default="outputs/rel-amazon/event_spine_paper_table.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.input_json).open() as handle:
        comparison = json.load(handle)
    rows = []
    for method_key, metrics in comparison.items():
        rows.append(paper_row(method_key, metrics))
    table = pd.DataFrame(rows)
    write_csv(table, args.output_csv)
    write_markdown(table, args.output_md)
    print(table.to_string(index=False))
    print(f"[done] wrote {args.output_csv}")
    print(f"[done] wrote {args.output_md}")


def paper_row(method_key: str, metrics: Dict[str, Any]) -> Dict[str, str]:
    method = METHOD_LABELS.get(method_key, method_key)
    if method_key == MAIN_METHOD_KEY:
        method = f"{method} (MAIN)"
    return {
        "Method": method,
        "Degree KS C/P": pair(metrics, "customer_degree_ks", "product_degree_ks"),
        "Block-time L1": fmt(metrics.get("block_pair_time_count_l1")),
        "Product first/last/peak corr": triple(metrics, "product_first_time_corr", "product_last_time_corr", "product_peak_time_corr"),
        "Customer first/last/peak corr": triple(metrics, "customer_first_time_corr", "customer_last_time_corr", "customer_peak_time_corr"),
        "Joint coactive window": fmt(metrics.get("joint_coactive_window_rate")),
        "Exact event overlap": sci(metrics.get("exact_event_overlap_rate")),
        "Duplicate ratio": fmt(metrics.get("duplicate_rate_ratio")),
        "Dynamic affinity KS": fmt(metrics.get("dynamic_affinity_distribution_ks")),
        "C2ST AUC": fmt(metrics.get("event_tuple_c2st_auc")),
        "Runtime": runtime(metrics),
    }


def pair(metrics: Dict[str, Any], left: str, right: str) -> str:
    return f"{fmt(metrics.get(left))} / {fmt(metrics.get(right))}"


def triple(metrics: Dict[str, Any], first: str, second: str, third: str) -> str:
    return f"{fmt(metrics.get(first))} / {fmt(metrics.get(second))} / {fmt(metrics.get(third))}"


def runtime(metrics: Dict[str, Any]) -> str:
    seconds = metrics.get("total_seconds")
    eps = metrics.get("events_per_second")
    if seconds is None and eps is None:
        return ""
    if seconds is None:
        return f"{fmt(eps)} ev/s"
    if eps is None:
        return f"{fmt(seconds)} s"
    return f"{fmt(seconds)} s, {fmt(eps)} ev/s"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return ""
    return f"{number:.3f}"


def sci(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return ""
    if number == 0.0:
        return "0"
    return f"{number:.2e}"


def write_csv(table: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)


def write_markdown(table: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    md_table = table.copy()
    md_table["Method"] = md_table["Method"].map(lambda value: f"**{value}**" if "(MAIN)" in value else value)
    columns = list(md_table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in md_table.iterrows():
        lines.append("| " + " | ".join(escape_markdown(row[column]) for column in columns) + " |")
    output.write_text("\n".join(lines) + "\n")


def escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
