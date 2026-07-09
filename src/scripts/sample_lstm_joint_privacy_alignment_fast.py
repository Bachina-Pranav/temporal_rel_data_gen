#!/usr/bin/env python3
"""Fast privacy-aware sampler for v5.1 joint LSTM privacy/alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    sample_lstm_fast_from_config,
)
from attribute_generation.conditional_tabdlm.schema import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_1_lstm_privacy_alignment.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample v5.1 LSTM privacy/alignment attributes.")
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
    parser.add_argument("--decode-mode", choices=["naive", "batched", "bucketed"], default="bucketed")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--write-chunk-size", type=int, default=10000)
    parser.add_argument("--torch-compile", action="store_true")
    parser.set_defaults(
        auto_batch_size=True,
        mixed_precision=True,
        cache_graph_context=True,
        cache_condition_embeddings=True,
        active_row_masking=True,
        length_bucketed_decoding=True,
        detokenize_after_generation=True,
    )
    parser.add_argument("--auto-batch-size", dest="auto_batch_size", action="store_true")
    parser.add_argument("--no-auto-batch-size", dest="auto_batch_size", action="store_false")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--cache-graph-context", dest="cache_graph_context", action="store_true")
    parser.add_argument("--no-cache-graph-context", dest="cache_graph_context", action="store_false")
    parser.add_argument("--active-row-masking", dest="active_row_masking", action="store_true")
    parser.add_argument("--no-active-row-masking", dest="active_row_masking", action="store_false")
    parser.add_argument("--length-bucketed-decoding", dest="length_bucketed_decoding", action="store_true")
    parser.add_argument("--no-length-bucketed-decoding", dest="length_bucketed_decoding", action="store_false")
    parser.add_argument("--detokenize-after-generation", dest="detokenize_after_generation", action="store_true")
    parser.add_argument("--no-detokenize-after-generation", dest="detokenize_after_generation", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sampling = config.raw.get("sampling", {})
    no_repeat = sampling.get("no_repeat_ngram", {})
    overlap = sampling.get("exact_train_overlap_blocking", {})
    attempts = overlap.get("max_resample_attempts", {})
    num_rows = args.num_rows
    if isinstance(num_rows, str) and num_rows.isdigit():
        num_rows = int(num_rows)
    options = FastSamplerOptions(
        profile=args.profile,
        profile_output=args.profile_output,
        detailed_profile_output=args.detailed_profile_output,
        decode_mode=args.decode_mode,
        max_batch_size=args.max_batch_size,
        auto_batch_size=args.auto_batch_size,
        mixed_precision=args.mixed_precision,
        torch_compile=args.torch_compile,
        cache_graph_context=args.cache_graph_context,
        cache_condition_embeddings=True,
        active_row_masking=args.active_row_masking,
        length_bucketed_decoding=args.length_bucketed_decoding,
        detokenize_after_generation=args.detokenize_after_generation,
        write_chunk_size=args.write_chunk_size,
        seed=args.seed,
        no_repeat_ngram_enabled=bool(no_repeat.get("enabled", True)),
        summary_no_repeat_ngram_size=int(no_repeat.get("summary_ngram_size", 3)),
        review_text_no_repeat_ngram_size=int(no_repeat.get("review_text_ngram_size", 4)),
        exact_train_overlap_blocking_enabled=bool(overlap.get("enabled", True)),
        max_summary_resample_attempts=int(attempts.get("summary", 5)),
        max_review_text_resample_attempts=int(attempts.get("review_text", 3)),
    )
    sample_lstm_fast_from_config(
        config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_rows=num_rows,
        batch_size=args.batch_size,
        device=args.device,
        synthetic_spine_path=args.synthetic_spine,
        options=options,
    )


if __name__ == "__main__":
    main()
