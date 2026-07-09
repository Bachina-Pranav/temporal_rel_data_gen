from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_rel_amazon_eval_config_no_toy_paths_and_skips_full_relational_metrics():
    path = ROOT / "configs/evaluation/single_event_table_paper_metrics_rel_amazon.yaml"
    config = yaml.safe_load(path.read_text())

    assert config["dataset_name"] == "rel_amazon"
    assert config["paper_metrics_version"] == "single_event_table_v1.1"
    assert "toy" not in str(config)
    assert config["evaluation_level"] == "single_event_table"
