#!/usr/bin/env python3
"""Write diagnostics for Text V1 summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from textgen.masked_summary_dataset import TOKEN_RE, normalize_summary_text  # noqa: E402
from textgen.text_eval import distinct_n, evaluate_summary_text_v1, tokens  # noqa: E402
from textgen.text_privacy_metrics import privacy_neighbors_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose temporal summary Text V1 outputs.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--synthetic-reviews", required=True)
    parser.add_argument("--text-col", default="summary")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--verified-col", default="verified")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real = pd.read_csv(args.real_reviews)
    synthetic = pd.read_csv(args.synthetic_reviews)
    metrics = evaluate_summary_text_v1(
        real,
        synthetic,
        text_col=args.text_col,
        rating_col=args.rating_col,
        verified_col=args.verified_col,
        timestamp_col=args.timestamp_col,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        privacy_sample_size=args.sample_size,
        seed=args.seed,
    )
    with (output_dir / "summary_text_v1_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    neighbors = privacy_neighbors_table(real[args.text_col].fillna(""), synthetic, args.text_col)
    neighbors.to_csv(output_dir / "privacy_nearest_neighbors.csv", index=False)
    sample_cols = [
        args.customer_id_col,
        args.product_id_col,
        args.timestamp_col,
        args.rating_col,
        args.verified_col,
        args.text_col,
    ]
    samples = synthetic[[col for col in sample_cols if col in synthetic.columns]].copy()
    samples = samples.rename(columns={args.text_col: "generated_summary"})
    samples["summary_length"] = samples["generated_summary"].fillna("").map(lambda text: len(TOKEN_RE.findall(str(text))))
    merged = samples.reset_index().merge(neighbors, left_on="index", right_on="row_id", how="left")
    merged.head(args.sample_size).to_csv(output_dir / "generated_summary_samples.csv", index=False)
    by_rating(real, synthetic, args.text_col, args.rating_col).to_csv(output_dir / "summary_by_rating.csv", index=False)
    print(f"Wrote diagnostics to {output_dir}")


def by_rating(real: pd.DataFrame, synthetic: pd.DataFrame, text_col: str, rating_col: str) -> pd.DataFrame:
    rows = []
    ratings = sorted(set(real[rating_col].dropna()).union(set(synthetic[rating_col].dropna())))
    for rating in ratings:
        real_text = real.loc[real[rating_col] == rating, text_col].fillna("").astype(str).tolist()
        syn_text = synthetic.loc[synthetic[rating_col] == rating, text_col].fillna("").astype(str).tolist()
        rows.append(
            {
                "rating": rating,
                "real_count": len(real_text),
                "synthetic_count": len(syn_text),
                "real_avg_len": avg_len(real_text),
                "synthetic_avg_len": avg_len(syn_text),
                "real_top_words": " ".join(top_words(real_text)),
                "synthetic_top_words": " ".join(top_words(syn_text)),
                "real_distinct_2": distinct_n([tokens(text) for text in real_text], 2),
                "synthetic_distinct_2": distinct_n([tokens(text) for text in syn_text], 2),
            }
        )
    return pd.DataFrame(rows)


def avg_len(texts) -> float:
    if not texts:
        return 0.0
    return float(sum(len(tokens(text)) for text in texts) / len(texts))


def top_words(texts, n: int = 12):
    counts = Counter()
    for text in texts:
        counts.update(token for token in tokens(normalize_summary_text(text)) if token.isalnum())
    return [word for word, _ in counts.most_common(n)]


if __name__ == "__main__":
    main()
