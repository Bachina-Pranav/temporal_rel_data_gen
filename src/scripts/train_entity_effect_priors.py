#!/usr/bin/env python3
"""Train generative priors over customer/product latent effects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reldiff.attributes.entity_effect_priors import (  # noqa: E402
    fit_customer_product_priors,
    save_customer_product_priors,
)
from reldiff.attributes.entity_latent_effects import (  # noqa: E402
    estimate_entity_latent_effects,
    save_entity_effect_estimate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train entity effect priors.")
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--structure-debug-dir", default=None)
    parser.add_argument("--customer-id-col", default="customer_id")
    parser.add_argument("--product-id-col", default="product_id")
    parser.add_argument("--timestamp-col", default="review_time")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--verified-col", default="verified")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--effect-prior-type", default="conditional_gaussian")
    parser.add_argument("--num-degree-bins", type=int, default=4)
    parser.add_argument("--min-entities-per-cell", type=int, default=20)
    parser.add_argument("--alpha-product-rating", default="auto")
    parser.add_argument("--alpha-customer-rating", default="auto")
    parser.add_argument("--alpha-product-verified", default="auto")
    parser.add_argument("--alpha-customer-verified", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.effect_prior_type != "conditional_gaussian":
        raise ValueError("Only --effect-prior-type conditional_gaussian is supported.")
    reviews = pd.read_csv(args.real_reviews)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimate = estimate_entity_latent_effects(
        reviews,
        structure_debug_dir=args.structure_debug_dir,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        timestamp_col=args.timestamp_col,
        rating_col=args.rating_col,
        verified_col=args.verified_col,
        alpha_product_rating=args.alpha_product_rating,
        alpha_customer_rating=args.alpha_customer_rating,
        alpha_product_verified=args.alpha_product_verified,
        alpha_customer_verified=args.alpha_customer_verified,
    )
    save_entity_effect_estimate(estimate, output_dir)
    customer_prior, product_prior, diagnostics = fit_customer_product_priors(
        estimate.customer_effects,
        estimate.product_effects,
        customer_id_col=args.customer_id_col,
        product_id_col=args.product_id_col,
        num_degree_bins=args.num_degree_bins,
        min_entities_per_cell=args.min_entities_per_cell,
    )
    diagnostics["global_effect_stats"] = estimate.global_stats
    diagnostics["seed"] = int(args.seed)
    save_customer_product_priors(output_dir, customer_prior, product_prior, diagnostics)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
