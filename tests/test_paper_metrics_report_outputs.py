from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics, write_legacy_metrics  # noqa: E402
from evaluation.paper_metrics.reporting import write_markdown_report  # noqa: E402
from evaluation.paper_metrics.utils import write_json  # noqa: E402


def test_paper_metrics_report_outputs_and_separate_legacy_json(tmp_path):
    real_path, syn_path, config = write_tiny_tables(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    metrics = evaluate_paper_metrics(config, output_dir)
    write_json(metrics, output_dir / "metrics.json")
    write_json(metrics, output_dir / "paper_metrics.json")
    write_markdown_report(metrics, output_dir / "metrics.md")
    write_legacy_metrics(config, output_dir)

    assert real_path.exists()
    assert syn_path.exists()
    for name in [
        "metrics.json",
        "paper_metrics.json",
        "metrics.md",
        "per_column_metrics.csv",
        "per_pair_trend_metrics.csv",
        "per_fk_metrics.csv",
        "per_temporal_metrics.csv",
        "c2st_report.json",
        "text_embedding_c2st_report.json",
        "evaluator_warnings.json",
        "legacy_diagnostic_metrics.json",
    ]:
        assert (output_dir / name).exists()


def write_tiny_tables(tmp_path: Path):
    parent = pd.DataFrame({"user_fk": ["u1", "u2"]})
    parent_path = tmp_path / "parent.csv"
    parent.to_csv(parent_path, index=False)
    real = pd.DataFrame(
        {
            "user_fk": ["u1", "u2", "u1", "u2"],
            "event_ts": pd.date_range("2020-01-01", periods=4, freq="D"),
            "label": ["a", "b", "a", "b"],
            "description": ["hello world", "nice item", "hello again", "nice again"],
        }
    )
    synthetic = real.copy()
    real_path = tmp_path / "real.csv"
    syn_path = tmp_path / "synthetic.csv"
    real.to_csv(real_path, index=False)
    synthetic.to_csv(syn_path, index=False)
    config = {
        "dataset_name": "toy",
        "evaluation_level": "single_event_table",
        "real_table_path": str(real_path),
        "synthetic_table_path": str(syn_path),
        "legacy_evaluator": {"enabled": False},
        "table": {
            "name": "events",
            "columns": {
                "user_fk": {
                    "type": "foreign_key",
                    "references": {"table": "users", "column": "user_fk"},
                    "parent_table_path": str(parent_path),
                    "nullable": False,
                },
                "event_ts": {"type": "datetime", "nullable": False},
                "label": {"type": "categorical", "valid_values": ["a", "b"], "nullable": False},
                "description": {"type": "text", "nullable": False},
            },
        },
        "evaluation": {
            "random_seed": 42,
            "max_rows_for_c2st": 20,
            "temporal": {"timestamp_columns": ["event_ts"], "binning": {"modes": ["adaptive"], "adaptive_target_bins": 3}},
            "text": {"embedding_model": "dummy", "text_columns": ["description"], "max_text_rows": 20, "cache_embeddings": False},
            "c2st": {"enabled": True, "classifiers": ["logistic_regression"], "max_rows": 20},
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return real_path, syn_path, config
