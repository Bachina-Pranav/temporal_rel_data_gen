#!/usr/bin/env python3
"""Evaluate dataset-agnostic paper metrics for one generated event table.

The legacy attribute evaluator is still available for dataset-specific debugging,
but this script writes those old diagnostic metrics to a separate JSON file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.evaluate import evaluate_from_config as legacy_evaluate_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config as load_legacy_config  # noqa: E402
from evaluation.paper_metrics.c2st import single_table_c2st_metrics  # noqa: E402
from evaluation.paper_metrics.fk_cardinality import fk_cardinality_metrics  # noqa: E402
from evaluation.paper_metrics.reporting import write_markdown_report, write_table  # noqa: E402
from evaluation.paper_metrics.schema_validation import constraint_violation_metrics  # noqa: E402
from evaluation.paper_metrics.shape_trend import shape_metrics, trend_metrics  # noqa: E402
from evaluation.paper_metrics.temporal_fidelity import temporal_metrics  # noqa: E402
from evaluation.paper_metrics.text_embedding import text_embedding_c2st_metrics  # noqa: E402
from evaluation.paper_metrics.utils import categorical_canonicalization_diagnostics, ensure_dir, write_json  # noqa: E402


PAPER_METRICS_VERSION = "single_event_table_v1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper-grade metrics for a single generated event table.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-table", default=None)
    parser.add_argument("--synthetic-table", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--legacy-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.real_table:
        config["real_table_path"] = args.real_table
    if args.synthetic_table:
        config["synthetic_table_path"] = args.synthetic_table
    if args.sample_size is not None:
        config.setdefault("evaluation", {})["sample_size"] = int(args.sample_size)
    if args.seed is not None:
        config.setdefault("evaluation", {})["random_seed"] = int(args.seed)
    if args.legacy_config:
        config.setdefault("legacy_evaluator", {})["config_path"] = args.legacy_config
        config.setdefault("legacy_evaluator", {})["enabled"] = True
    output_dir = ensure_dir(args.output_dir)
    metrics = evaluate_paper_metrics(config, output_dir)
    write_json(metrics, output_dir / "metrics.json")
    write_json(metrics, output_dir / "paper_metrics.json")
    write_markdown_report(metrics, output_dir / "metrics.md")
    write_legacy_metrics(config, output_dir)
    print(output_dir / "metrics.json")
    print(output_dir / "legacy_diagnostic_metrics.json")


def evaluate_paper_metrics(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    seed = int((config.get("evaluation") or {}).get("random_seed", 42))
    real = pd.read_csv(config["real_table_path"])
    synthetic = pd.read_csv(config["synthetic_table_path"])
    sample_size = (config.get("evaluation") or {}).get("sample_size")
    if sample_size:
        n_real = min(int(sample_size), len(real))
        n_syn = min(int(sample_size), len(synthetic))
        real = real.sample(n=n_real, random_state=seed).reset_index(drop=True)
        synthetic = synthetic.sample(n=n_syn, random_state=seed + 1).reset_index(drop=True)
    table_cfg = config.get("table") or {}
    row_count_match = len(real) == len(synthetic)
    row_count_ratio = float(len(synthetic) / len(real)) if len(real) else None
    evaluator_warnings = evaluator_warning_records(real, synthetic)

    validity = constraint_violation_metrics(synthetic, table_cfg)
    fk, fk_df = fk_cardinality_metrics(real, synthetic, table_cfg, row_count_match=row_count_match)
    if not row_count_match:
        evaluator_warnings.append(
            {
                "code": "FK_CARDINALITY_ROW_COUNT_CONFOUNDED",
                "message": "Absolute FK cardinality metrics are row-count-confounded; headline FK similarity uses normalized cardinality.",
            }
        )
    temporal, temporal_df = temporal_metrics(real, synthetic, config)
    shape, column_df = shape_metrics(real, synthetic, table_cfg, config)
    trend, pair_df = trend_metrics(real, synthetic, table_cfg, config)
    text_embedding = text_embedding_c2st_metrics(real, synthetic, config, output_dir)
    c2st, feature_importance = single_table_c2st_metrics(real, synthetic, config)
    categorical_diagnostics = categorical_diagnostics_for_table(real, synthetic, table_cfg)

    write_table(column_df, output_dir / "per_column_metrics.csv")
    write_table(pair_df, output_dir / "per_pair_trend_metrics.csv")
    write_table(fk_df, output_dir / "per_fk_metrics.csv")
    write_table(temporal_df, output_dir / "per_temporal_metrics.csv")
    write_table(feature_importance, output_dir / "c2st_feature_importance.csv")
    write_json(c2st, output_dir / "c2st_report.json")
    write_json(text_embedding, output_dir / "text_embedding_c2st_report.json")
    write_json({"warnings": evaluator_warnings}, output_dir / "evaluator_warnings.json")

    summary = {
        "constraint_violation_rate": validity.get("constraint_violation_rate"),
        "fk_cardinality_similarity": fk.get("macro_similarity"),
        "temporal_event_distance": temporal.get("macro_temporal_event_distance"),
        "shape_error": shape.get("macro_non_id_shape_error", shape.get("macro_attribute_shape_error", shape.get("macro_shape_error"))),
        "trend_error": trend.get("macro_headline_trend_error", trend.get("macro_attribute_trend_error", trend.get("macro_trend_error"))),
        "text_embedding_c2st_error": text_embedding.get("macro_error"),
        "single_table_c2st_error": c2st.get("error"),
    }
    skipped = {
        "k_hop_relational_correlation": {
            "status": "skipped",
            "reason": "requires full multi-table relational generation",
        },
        "c2st_agg": {
            "status": "skipped",
            "reason": "requires full multi-table relational generation",
        },
    }
    return {
        "paper_metrics_version": config.get("paper_metrics_version", PAPER_METRICS_VERSION),
        "dataset": {
            "dataset_name": config.get("dataset_name"),
            "evaluation_level": config.get("evaluation_level", "single_event_table"),
            "real_table_path": config.get("real_table_path"),
            "synthetic_table_path": config.get("synthetic_table_path"),
            "num_real_rows": int(len(real)),
            "num_synthetic_rows": int(len(synthetic)),
            "row_count_match": bool(row_count_match),
            "row_count_ratio": row_count_ratio,
        },
        "evaluator_warnings": evaluator_warnings,
        "paper_metrics_summary": summary,
        "internal_overall_score": internal_overall_score(summary),
        "categorical_canonicalization": categorical_diagnostics,
        "validity": validity,
        "fk_cardinality": fk,
        "temporal": temporal,
        "shape": shape,
        "trend": trend,
        "text_embedding_c2st": text_embedding,
        "single_table_c2st": c2st,
        "skipped_metrics": skipped,
    }


def evaluator_warning_records(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[dict[str, str]]:
    if len(real) == len(synthetic):
        return []
    return [
        {
            "code": "ROW_COUNT_MISMATCH",
            "message": (
                "Real and synthetic row counts differ. Some count-based metrics may be biased. "
                "For final paper evaluation, generate synthetic rows equal to real rows or use explicit balanced subsampling."
            ),
        }
    ]


def categorical_diagnostics_for_table(real: pd.DataFrame, synthetic: pd.DataFrame, table_cfg: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for column, cfg in (table_cfg.get("columns", {}) or {}).items():
        if str((cfg or {}).get("type", "")).lower() != "categorical":
            continue
        if column not in real or column not in synthetic:
            continue
        diagnostics[column] = categorical_canonicalization_diagnostics(real[column], synthetic[column], cfg or {})
    return diagnostics


def internal_overall_score(summary: dict[str, Any]) -> dict[str, Any]:
    positive = []
    mappings = {
        "constraint_violation_rate": lambda x: 1.0 - x,
        "fk_cardinality_similarity": lambda x: x,
        "temporal_event_distance": lambda x: 1.0 - x,
        "shape_error": lambda x: 1.0 - x,
        "trend_error": lambda x: 1.0 - x,
        "text_embedding_c2st_error": lambda x: 1.0 - x,
        "single_table_c2st_error": lambda x: 1.0 - x,
    }
    for key, fn in mappings.items():
        value = summary.get(key)
        if value is None:
            continue
        positive.append(max(0.0, min(float(fn(float(value))), 1.0)))
    return {
        "value": float(sum(positive) / len(positive)) if positive else None,
        "label": "not intended as a paper headline metric",
    }


def write_legacy_metrics(config: dict[str, Any], output_dir: Path) -> None:
    legacy_cfg = dict(config.get("legacy_evaluator", {}) or {})
    legacy_path = legacy_cfg.get("config_path")
    output_path = output_dir / "legacy_diagnostic_metrics.json"
    if not bool(legacy_cfg.get("enabled", bool(legacy_path))) or not legacy_path:
        write_json({"status": "skipped", "reason": "legacy_evaluator.config_path not configured"}, output_path)
        return
    try:
        legacy_metrics = legacy_evaluate_from_config(
            load_legacy_config(legacy_path),
            synthetic_reviews_path=config.get("synthetic_table_path"),
            real_reviews_path=config.get("real_table_path"),
            output_path=output_path,
            output_dir=output_dir / "legacy_diagnostic_report",
        )
        write_json(legacy_metrics, output_dir / "legacy_metrics.json")
    except Exception as exc:
        write_json({"status": "failed", "reason": str(exc), "config_path": legacy_path}, output_path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
