from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from reldiff.attributes.entity_latent_effects import estimate_entity_latent_effects  # noqa: E402


def test_effect_estimation_shrinks_low_degree_more(tmp_path):
    rows = []
    for index in range(100):
        rows.append((f"c_base_{index}", "p_base", f"2020-01-{1 + index % 28:02d}", 3, True))
    for index in range(20):
        rows.append((f"c_high_{index}", "p_high", f"2020-02-{1 + index % 28:02d}", 5, True))
    rows.append(("c_low", "p_low", "2020-03-01", 5, True))
    reviews = pd.DataFrame(
        rows,
        columns=["customer_id", "product_id", "review_time", "rating", "verified"],
    )
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    pd.DataFrame(
        {
            "customer_id": sorted(reviews["customer_id"].unique()),
            "customer_block": 0,
        }
    ).to_csv(debug_dir / "customer_blocks.csv", index=False)
    pd.DataFrame(
        {
            "product_id": sorted(reviews["product_id"].unique()),
            "product_block": 0,
        }
    ).to_csv(debug_dir / "product_blocks.csv", index=False)

    estimate = estimate_entity_latent_effects(reviews, structure_debug_dir=debug_dir)
    product_effects = estimate.product_effects.set_index("product_id")
    assert product_effects.loc["p_high", "rating_effect"] > 0
    assert product_effects.loc["p_low", "rating_effect"] > 0
    assert abs(product_effects.loc["p_high", "rating_effect"]) > abs(
        product_effects.loc["p_low", "rating_effect"]
    )
    assert "global_mean_rating" in estimate.global_stats
