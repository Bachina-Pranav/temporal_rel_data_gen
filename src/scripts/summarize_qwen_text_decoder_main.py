#!/usr/bin/env python3
"""Create a compact, JSON-safe summary of the main Qwen text experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def read_json(root: Path, relative: str) -> Any:
    path = root / relative
    return json.loads(path.read_text()) if path.is_file() else None


def read_csv(root: Path, relative: str) -> list[dict[str, Any]] | None:
    path = root / relative
    if not path.is_file():
        return None
    return sanitize_json(pd.read_csv(path).to_dict(orient="records"))


def c2st_scores(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    fields = value.get("per_field") or value.get("per_text_column") or {}

    def field(name: str) -> Any:
        item = fields.get(name) or {}
        return item.get("error")

    summary = field("summary")
    review = field("review_text")
    macro = value.get("macro_error")
    if macro is None and summary is not None and review is not None:
        macro = (float(summary) + float(review)) / 2.0
    return {"summary": summary, "review_text": review, "macro": macro}


def build_summary(root: Path) -> dict[str, Any]:
    efficiency = read_json(root, "training/training_efficiency.json") or {}
    oracle_c2st = read_json(root, "oracle_structured/canonical_text_c2st.json")
    comparison = read_csv(root, "comparison/model_comparison.csv") or []
    generated_alignment = read_json(
        root, "generated_structured/row_alignment_audit.json"
    )
    oracle_scores = c2st_scores(oracle_c2st)
    named_scores = {
        str(row.get("model")): {
            "summary": row.get("summary_c2st"),
            "review_text": row.get("review_c2st"),
            "macro": row.get("macro_text_c2st"),
        }
        for row in comparison
    }
    lstm = next(
        (scores for name, scores in named_scores.items() if "lstm" in name.lower()),
        None,
    )
    diffusion = next(
        (
            scores
            for name, scores in named_scores.items()
            if "diffusion" in name.lower()
        ),
        None,
    )
    qualitative = None
    synthetic_path = root / "oracle_structured/synthetic_text.csv"
    if synthetic_path.is_file():
        frame = pd.read_csv(synthetic_path, low_memory=False)
        columns = [
            column
            for column in ("rating", "verified", "summary", "review_text")
            if column in frame
        ]
        qualitative = sanitize_json(frame.loc[:, columns].head(12).to_dict("records"))
    oracle_macro = oracle_scores.get("macro")
    corrected = {
        "oracle_qwen_beats_lstm": bool(
            oracle_macro is not None
            and lstm is not None
            and lstm.get("macro") is not None
            and float(oracle_macro) < float(lstm["macro"])
        ),
        "oracle_qwen_beats_masked_diffusion": bool(
            oracle_macro is not None
            and diffusion is not None
            and diffusion.get("macro") is not None
            and float(oracle_macro) < float(diffusion["macro"])
        ),
        "scope": (
            "These corrected comparisons apply only to the oracle-structured "
            "decoder diagnostic. They are not end-to-end generated-structured results."
        ),
    }
    return sanitize_json(
        {
            "experiment": "Qwen3-0.6B-Base + LoRA Amazon-Toy text decoder",
            "metric_semantics": {
                "text_c2st_error": "Lower is better; zero is chance-level discrimination.",
                "oracle_structured": (
                    "Uses true held-out rating and verified values. This isolates "
                    "text-decoder quality and is not a complete generative result."
                ),
            },
            "model": {
                key: efficiency.get(key)
                for key in (
                    "model_id",
                    "revision",
                    "license",
                    "total_parameters",
                    "trainable_parameters",
                    "trainable_fraction",
                    "lora_rank",
                    "lora_alpha",
                    "target_modules",
                )
            },
            "training": {
                key: efficiency.get(key)
                for key in (
                    "training_seconds",
                    "completed_epochs",
                    "seconds_per_epoch",
                    "examples_per_second",
                    "tokens_per_second",
                    "peak_gpu_memory_bytes",
                    "chosen_max_length",
                )
            }
            | {
                "train_length_statistics": (
                    efficiency.get("length_statistics") or {}
                ).get("train"),
                "validation_log": read_csv(root, "training/validation_log.csv"),
            },
            "canonical_model_comparison": comparison,
            "oracle_structured": {
                "scores": oracle_scores,
                "generation_metrics": read_json(
                    root, "oracle_structured/generation_metrics.json"
                ),
                "canonical_text_c2st": oracle_c2st,
                "text_distribution": read_json(
                    root, "oracle_structured/text_distribution_metrics.json"
                ),
                "semantic_consistency": read_json(
                    root, "oracle_structured/consistency_metrics.json"
                ),
                "memorization": read_json(
                    root, "oracle_structured/memorization_metrics.json"
                ),
                "qualitative_samples": qualitative,
            },
            "generated_structured": {
                "status": "unavailable",
                "reason": (
                    "No frozen generated-structured artifact exactly aligned "
                    "with the held-out test event spine."
                ),
                "alignment_audit": generated_alignment,
            },
            "corrected_oracle_comparison": corrected,
            "interpretation_constraints": [
                "Do not treat oracle conditioning as an end-to-end paper result.",
                "The generated-structured stop is an input-alignment limitation, not a Qwen decoding crash.",
                "The main console's QWEN BEATS LSTM boolean depended on the unavailable generated mode and does not describe the oracle comparison.",
                "Do not select follow-up configurations using held-out test C2ST.",
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root", default="outputs/qwen_text_decoder_06b"
    )
    parser.add_argument(
        "--output",
        default="outputs/qwen_text_decoder_06b/main_experiment_gpt_summary.json",
    )
    args = parser.parse_args()
    root = Path(args.experiment_root)
    output = Path(args.output)
    summary = build_summary(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(text, encoding="utf-8")
    print(output)
    print(text)


if __name__ == "__main__":
    main()

