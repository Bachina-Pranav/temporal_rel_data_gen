#!/usr/bin/env python3
"""Fast LSTM sampler with length-preserving exact-overlap privacy blocking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    sample_lstm_fast_from_config,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_1_lstm_privacy_alignment.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample joint LSTM attributes with length-preserving privacy blocking.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--synthetic-spine", default=None)
    parser.add_argument("--num-rows", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-output", default=None)
    parser.add_argument("--detailed-profile-output", default=None)
    parser.add_argument("--text-field-policy", default=None)
    parser.set_defaults(
        length_preserving_exact_blocking=True,
        auto_batch_size=True,
        mixed_precision=True,
        active_row_masking=True,
        length_bucketed_decoding=True,
        detokenize_after_generation=True,
    )
    parser.add_argument("--length-preserving-exact-blocking", dest="length_preserving_exact_blocking", action="store_true")
    parser.add_argument("--no-length-preserving-exact-blocking", dest="length_preserving_exact_blocking", action="store_false")
    parser.add_argument("--disable-review-text-ngram-blocking", action="store_true")
    parser.add_argument("--auto-batch-size", dest="auto_batch_size", action="store_true")
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--write-chunk-size", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_rows = parse_num_rows(args.num_rows)
    text_policy = load_text_field_policy(args.text_field_policy)
    options = FastSamplerOptions(
        profile=args.profile,
        profile_output=args.profile_output,
        detailed_profile_output=args.detailed_profile_output,
        decode_mode="bucketed",
        auto_batch_size=args.auto_batch_size,
        mixed_precision=args.mixed_precision,
        cache_graph_context=True,
        graph_context_cache_mode="batch",
        cache_condition_embeddings=True,
        active_row_masking=args.active_row_masking,
        length_bucketed_decoding=args.length_bucketed_decoding,
        detokenize_after_generation=args.detokenize_after_generation,
        write_chunk_size=args.write_chunk_size,
        seed=args.seed,
        use_config_privacy_controls=False,
        exact_train_overlap_blocking_enabled=bool(args.length_preserving_exact_blocking),
        length_preserving_exact_blocking_enabled=bool(args.length_preserving_exact_blocking),
        no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_enabled=False,
        review_text_no_repeat_ngram_enabled=False if args.disable_review_text_ngram_blocking else None,
        summary_no_repeat_ngram_size=0,
        review_text_no_repeat_ngram_size=0,
        max_summary_resample_attempts=5,
        max_review_text_resample_attempts=3,
        text_field_policy=text_policy,
    )
    sample_lstm_fast_from_config(
        load_config(args.config),
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_rows=num_rows,
        batch_size=args.batch_size,
        device=args.device,
        synthetic_spine_path=args.synthetic_spine,
        options=options,
    )


def parse_num_rows(value: Any) -> int | str | None:
    if value in (None, "all"):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def load_text_field_policy(path: str | Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("YAML text-field policies require pyyaml") from exc
        payload = yaml.safe_load(text)
    if isinstance(payload, dict):
        payload = payload.get("text_fields", payload)
    if not isinstance(payload, list):
        raise ValueError("--text-field-policy must point to a list or a dict with text_fields")
    return payload


if __name__ == "__main__":
    main()
