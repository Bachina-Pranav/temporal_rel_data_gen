from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "scripts"))

from evaluate_single_event_table_paper_metrics import evaluate_paper_metrics  # noqa: E402
from evaluation.paper_metrics.reporting import write_markdown_report  # noqa: E402
from evaluation.paper_metrics.utils import write_json  # noqa: E402


def test_rel_amazon_tiny_smoke_evaluation_writes_expected_files(tmp_path):
    config = write_tiny_rel_amazon_eval_inputs(tmp_path)
    output_dir = tmp_path / "paper_metrics_single_event_table_smoke_50k_v1_1"

    metrics = evaluate_paper_metrics(config, output_dir)
    write_json(metrics, output_dir / "metrics.json")
    write_markdown_report(metrics, output_dir / "metrics.md")

    assert metrics["skipped_metrics"]["k_hop_relational_correlation"]["status"] == "skipped"
    assert metrics["skipped_metrics"]["c2st_agg"]["status"] == "skipped"
    for name in [
        "metrics.json",
        "metrics.md",
        "per_column_metrics.csv",
        "per_pair_trend_metrics.csv",
        "per_fk_metrics.csv",
        "per_temporal_metrics.csv",
        "c2st_report.json",
        "text_embedding_c2st_report.json",
        "evaluator_warnings.json",
    ]:
        assert (output_dir / name).exists()


def write_tiny_rel_amazon_eval_inputs(tmp_path: Path) -> dict:
    customer = pd.DataFrame({"customer_id": ["c1", "c2"]})
    product = pd.DataFrame({"product_id": ["p1", "p2"]})
    real = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c1", "c2"],
            "product_id": ["p1", "p2", "p1", "p2"],
            "review_time": pd.date_range("2020-01-01", periods=4, freq="D"),
            "rating": [5, 4, 5, 4],
            "verified": [1, 0, 1, 0],
            "summary": ["good", "ok", "great", "fine"],
            "review_text": ["good product", "ok item", "great product", "fine item"],
        }
    )
    synthetic = real.copy()
    paths = {}
    for name, frame in [("customer", customer), ("product", product), ("real", real), ("synthetic", synthetic)]:
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return {
        "dataset_name": "rel_amazon",
        "paper_metrics_version": "single_event_table_v1.1",
        "evaluation_level": "single_event_table",
        "real_table_path": str(paths["real"]),
        "synthetic_table_path": str(paths["synthetic"]),
        "legacy_evaluator": {"enabled": False},
        "table": {
            "name": "review",
            "columns": {
                "customer_id": {
                    "type": "foreign_key",
                    "references": {"table": "customer", "column": "customer_id"},
                    "parent_table_path": str(paths["customer"]),
                    "nullable": False,
                },
                "product_id": {
                    "type": "foreign_key",
                    "references": {"table": "product", "column": "product_id"},
                    "parent_table_path": str(paths["product"]),
                    "nullable": False,
                },
                "review_time": {"type": "datetime", "nullable": False},
                "rating": {"type": "categorical", "dtype": "int", "valid_values": [1, 2, 3, 4, 5], "nullable": False},
                "verified": {"type": "categorical", "dtype": "int", "valid_values": [0, 1], "nullable": False},
                "summary": {"type": "text", "nullable": False},
                "review_text": {"type": "text", "nullable": False},
            },
        },
        "evaluation": {
            "random_seed": 42,
            "max_rows_for_c2st": 20,
            "temporal": {"timestamp_columns": ["review_time"], "binning": {"modes": ["adaptive"], "adaptive_target_bins": 3}},
            "text": {"embedding_model": "dummy", "text_columns": ["summary", "review_text"], "max_text_rows": 20, "cache_embeddings": False},
            "c2st": {"enabled": True, "classifiers": ["logistic_regression"], "max_rows": 20},
        },
    }
