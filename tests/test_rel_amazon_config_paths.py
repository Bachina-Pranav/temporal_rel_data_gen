from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rel_amazon_configs_use_rel_amazon_paths_only():
    attr = (ROOT / "configs/attribute_generation/conditional_tabdlm_rel_amazon_exp5_3_lstm_length_preserving.yaml").read_text()
    eval_cfg = (ROOT / "configs/evaluation/single_event_table_paper_metrics_rel_amazon.yaml").read_text()

    assert "rel-amazon-toy" not in attr
    assert "amazon-toy" not in attr
    assert "rel-amazon-toy" not in eval_cfg
    assert "amazon-toy" not in eval_cfg
    assert "data/original/rel-amazon/review.csv" in eval_cfg
    assert "outputs/rel-amazon/time_biased_block_stub_matching_kernel_main/synthetic_review.csv" in attr
