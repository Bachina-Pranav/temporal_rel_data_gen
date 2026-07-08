#!/usr/bin/env python3
"""Fast/profiled sampler for the joint LSTM full-review-text generator."""

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


DEFAULT_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_lstm_joint_full_review_text.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast/profiled joint LSTM full-review-text sampling.")
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
    parser.add_argument("--disable-fast-path", action="store_true")
    parser.add_argument("--decode-mode", choices=["naive", "batched", "bucketed"], default="bucketed")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--graph-context-cache-mode", choices=["none", "batch", "full_tensor"], default="batch")
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
    parser.add_argument("--cache-condition-embeddings", dest="cache_condition_embeddings", action="store_true")
    parser.add_argument("--no-cache-condition-embeddings", dest="cache_condition_embeddings", action="store_false")
    parser.add_argument("--active-row-masking", dest="active_row_masking", action="store_true")
    parser.add_argument("--no-active-row-masking", dest="active_row_masking", action="store_false")
    parser.add_argument("--length-bucketed-decoding", dest="length_bucketed_decoding", action="store_true")
    parser.add_argument("--no-length-bucketed-decoding", dest="length_bucketed_decoding", action="store_false")
    parser.add_argument("--detokenize-after-generation", dest="detokenize_after_generation", action="store_true")
    parser.add_argument("--no-detokenize-after-generation", dest="detokenize_after_generation", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_rows = args.num_rows
    if isinstance(num_rows, str) and num_rows.isdigit():
        num_rows = int(num_rows)
    graph_context_cache_mode = "none" if not args.cache_graph_context else args.graph_context_cache_mode
    options = FastSamplerOptions(
        profile=args.profile,
        profile_output=args.profile_output,
        detailed_profile_output=args.detailed_profile_output,
        disable_fast_path=args.disable_fast_path,
        decode_mode=args.decode_mode,
        max_batch_size=args.max_batch_size,
        auto_batch_size=args.auto_batch_size,
        mixed_precision=args.mixed_precision,
        torch_compile=args.torch_compile,
        cache_graph_context=args.cache_graph_context,
        graph_context_cache_mode=graph_context_cache_mode,
        cache_condition_embeddings=args.cache_condition_embeddings,
        active_row_masking=args.active_row_masking,
        length_bucketed_decoding=args.length_bucketed_decoding,
        detokenize_after_generation=args.detokenize_after_generation,
        write_chunk_size=args.write_chunk_size,
        seed=args.seed,
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


if __name__ == "__main__":
    main()
