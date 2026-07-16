#!/usr/bin/env python3
"""Audit rating-domain handling across interaction data, vocab, checkpoint, and outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.constrained import normalize_rating_value  # noqa: E402
from attribute_generation.conditional_tabdlm.evaluate import rating_domain_from_config, rating_validity_details  # noqa: E402
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402
from attribute_generation.conditional_tabdlm.utils import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit rating domains for a MovieLens-style LSTM run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation-config", default=None)
    parser.add_argument("--raw-ratings", default=None)
    parser.add_argument("--processed-table", default=None)
    parser.add_argument("--train-table", default=None)
    parser.add_argument("--validation-table", default=None)
    parser.add_argument("--test-table", default=None)
    parser.add_argument("--sampled-output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rating-col", default="rating")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(args, config)
    declared_domain = rating_domain_from_config(config)
    rows = []
    for stage, path in paths.items():
        if path is None:
            rows.append(stage_row(stage, None, args.rating_col, declared_domain, mapping="not provided"))
            continue
        rows.append(stage_row(stage, Path(path), args.rating_col, declared_domain, mapping="CSV value -> normalized rating"))
    rows.extend(metadata_rows(args, config, declared_domain))
    payload = {
        "config": args.config,
        "evaluation_config": args.evaluation_config,
        "declared_rating_domain": declared_domain,
        "stages": rows,
        "rating_domain_bug_candidates": find_domain_warnings(rows, declared_domain),
        "commands": {
            "audit": " ".join(sys.argv),
        },
    }
    json_path = output_dir / "rating_domain_audit.json"
    json_path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "rating_domain_audit.csv", index=False)
    write_markdown(payload, output_dir / "rating_domain_audit.md")
    print(json_path)


def resolve_paths(args: argparse.Namespace, config: Any) -> dict[str, str | None]:
    data_dir = config.output_dir / "data"
    return {
        "raw_source": args.raw_ratings,
        "processed_full_subset": args.processed_table or str(config.train_data_path),
        "training_split": args.train_table or str(data_dir / "train.parquet"),
        "validation_split": args.validation_table or str(data_dir / "valid.parquet"),
        "test_split": args.test_table or str(data_dir / "test.parquet"),
        "final_sampled_csv": args.sampled_output,
    }


def stage_row(stage: str, path: Path | None, rating_col: str, domain: list[int | float], *, mapping: str) -> dict[str, Any]:
    if path is None:
        return {"stage": stage, "path": None, "exists": False, "unique_rating_values": [], "dtype": None, "mapping_applied": mapping}
    if not path.exists():
        return {"stage": stage, "path": str(path), "exists": False, "unique_rating_values": [], "dtype": None, "mapping_applied": mapping}
    frame = read_table(path)
    if rating_col not in frame.columns:
        return {"stage": stage, "path": str(path), "exists": True, "unique_rating_values": [], "dtype": None, "mapping_applied": f"{mapping}; missing rating column"}
    values = frame[rating_col]
    normalized = [normalize_rating_value(value) for value in values.dropna().unique().tolist()]
    unique = sorted({value for value in normalized if value is not None}, key=float)
    validity = rating_validity_details(values, domain)
    return {
        "stage": stage,
        "path": str(path),
        "exists": True,
        "row_count": int(len(frame)),
        "unique_rating_values": unique,
        "dtype": str(values.dtype),
        "mapping_applied": mapping,
        "raw_invalid_count": validity["raw_invalid_rating_count"],
        "canonicalized_invalid_count": validity["canonicalized_invalid_rating_count"],
        "invalid_examples": validity["invalid_rating_examples"],
    }


def metadata_rows(args: argparse.Namespace, config: Any, declared_domain: list[int | float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vocab_path = config.output_dir / "data" / "vocab_rating.json"
    if vocab_path.exists():
        vocab = load_json(vocab_path)
        tokens = list((vocab.get("token_to_id") or {}).keys())
        rows.append(
            {
                "stage": "model_vocab",
                "path": str(vocab_path),
                "exists": True,
                "unique_rating_values": sorted({normalize_rating_value(token) for token in tokens if normalize_rating_value(token) is not None}, key=float),
                "dtype": "vocab_token",
                "mapping_applied": "vocab token -> normalized rating",
            }
        )
    rows.append(
        {
            "stage": "model_config_declared_domain",
            "path": args.config,
            "exists": True,
            "unique_rating_values": declared_domain,
            "dtype": "config",
            "mapping_applied": "generated_attributes.rating.valid_domain",
        }
    )
    if args.evaluation_config:
        eval_cfg = load_json_or_yaml(Path(args.evaluation_config))
        eval_domain = (((eval_cfg.get("table") or {}).get("columns") or {}).get("rating") or {}).get("valid_values", [])
        rows.append(
            {
                "stage": "paper_evaluator_domain",
                "path": args.evaluation_config,
                "exists": Path(args.evaluation_config).exists(),
                "unique_rating_values": sorted({normalize_rating_value(value) for value in eval_domain if normalize_rating_value(value) is not None}, key=float),
                "dtype": "evaluation_config",
                "mapping_applied": "table.columns.rating.valid_values",
            }
        )
    if args.checkpoint:
        rows.extend(checkpoint_rows(Path(args.checkpoint)))
    return rows


def checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return [{"stage": "checkpoint", "path": str(path), "exists": False, "unique_rating_values": [], "dtype": None, "mapping_applied": "checkpoint missing"}]
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        return [{"stage": "checkpoint", "path": str(path), "exists": True, "unique_rating_values": [], "dtype": None, "mapping_applied": f"checkpoint unreadable: {exc}"}]
    rows = []
    vocab = ((checkpoint.get("categorical_vocabs") or {}).get("rating") or {}).get("token_to_id") or {}
    rows.append(
        {
            "stage": "checkpoint_rating_vocab",
            "path": str(path),
            "exists": True,
            "unique_rating_values": sorted({normalize_rating_value(token) for token in vocab if normalize_rating_value(token) is not None}, key=float),
            "dtype": "checkpoint_vocab_token",
            "mapping_applied": "checkpoint categorical_vocabs.rating token -> normalized rating",
        }
    )
    raw_domain = (((checkpoint.get("raw_config") or {}).get("generated_attributes") or {}).get("rating") or {}).get("valid_domain", [])
    rows.append(
        {
            "stage": "checkpoint_declared_domain",
            "path": str(path),
            "exists": True,
            "unique_rating_values": sorted({normalize_rating_value(value) for value in raw_domain if normalize_rating_value(value) is not None}, key=float),
            "dtype": "checkpoint_raw_config",
            "mapping_applied": "checkpoint raw_config generated_attributes.rating.valid_domain",
        }
    )
    return rows


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_json_or_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return load_json(path)
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def find_domain_warnings(rows: list[dict[str, Any]], declared_domain: list[int | float]) -> list[str]:
    warnings = []
    declared = set(declared_domain)
    for row in rows:
        values = set(row.get("unique_rating_values") or [])
        if values and not values.issubset(declared):
            warnings.append(f"{row.get('stage')} has values outside declared domain: {sorted(values - declared, key=float)}")
    if any(0.5 in set(row.get("unique_rating_values") or []) for row in rows) and set(declared_domain) == {1, 2, 3, 4, 5}:
        warnings.append("Half-star values observed but declared domain is integer-only.")
    return warnings


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# MovieLens LSTM Rating-Domain Audit",
        "",
        f"Config: `{payload['config']}`",
        "",
        f"Declared domain: `{payload['declared_rating_domain']}`",
        "",
        "| Stage | Unique rating values | Data type | Mapping applied |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["stages"]:
        lines.append(
            f"| {row.get('stage')} | `{row.get('unique_rating_values')}` | `{row.get('dtype')}` | {row.get('mapping_applied')} |"
        )
    if payload["rating_domain_bug_candidates"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in payload["rating_domain_bug_candidates"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
