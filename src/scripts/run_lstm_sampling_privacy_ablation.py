#!/usr/bin/env python3
"""Run v5.2 sampling-only privacy ablations for the joint LSTM generator."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.conditional_tabdlm.evaluate import evaluate_from_config  # noqa: E402
from attribute_generation.conditional_tabdlm.graph_schema import graph_metadata  # noqa: E402
from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    FastSamplerOptions,
    sample_lstm_fast_from_config,
)
from attribute_generation.conditional_tabdlm.schema import ConditionalTABDLMConfig, load_config  # noqa: E402
from compare_lstm_sampling_privacy_ablation import compare_runs  # noqa: E402


V5_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_lstm_joint_full_review_text.yaml"
V5_CHECKPOINT = "outputs/amazon-toy/conditional_tabdlm_exp5_lstm_joint_full_review_text/checkpoints/best.pt"
V51_CONFIG = "configs/attribute_generation/conditional_tabdlm_amazon_toy_exp5_1_lstm_privacy_alignment.yaml"
V51_CHECKPOINT = "outputs/amazon-toy/conditional_tabdlm_exp5_1_lstm_privacy_alignment/checkpoints/best.pt"

DEFAULT_VARIANTS = [
    "v5_exact_block_only",
    "v5_exact_block_summary_ngram_only",
    "v51_exact_block_only_no_ngram",
    "v51_no_block_no_ngram",
    "v5_summary_exact_block_only",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    checkpoint_source: str
    config_path: str
    checkpoint_path: str
    exact_train_overlap_blocking_enabled: bool
    summary_exact_blocking_enabled: bool
    review_text_exact_blocking_enabled: bool
    no_repeat_ngram_enabled: bool
    summary_no_repeat_ngram_enabled: bool
    review_text_no_repeat_ngram_enabled: bool
    summary_no_repeat_ngram_size: int
    review_text_no_repeat_ngram_size: int
    summary_temperature: float
    review_text_temperature: float
    summary_top_p: float
    review_text_top_p: float
    purpose: str


VARIANTS: dict[str, VariantSpec] = {
    "v5_exact_block_only": VariantSpec(
        name="v5_exact_block_only",
        checkpoint_source="v5",
        config_path=V5_CONFIG,
        checkpoint_path=V5_CHECKPOINT,
        exact_train_overlap_blocking_enabled=True,
        summary_exact_blocking_enabled=True,
        review_text_exact_blocking_enabled=True,
        no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_enabled=False,
        review_text_no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_size=0,
        review_text_no_repeat_ngram_size=0,
        summary_temperature=0.90,
        review_text_temperature=0.90,
        summary_top_p=0.95,
        review_text_top_p=0.95,
        purpose="Cheap exact-overlap blocking only on the v5 checkpoint.",
    ),
    "v5_exact_block_summary_ngram_only": VariantSpec(
        name="v5_exact_block_summary_ngram_only",
        checkpoint_source="v5",
        config_path=V5_CONFIG,
        checkpoint_path=V5_CHECKPOINT,
        exact_train_overlap_blocking_enabled=True,
        summary_exact_blocking_enabled=True,
        review_text_exact_blocking_enabled=True,
        no_repeat_ngram_enabled=True,
        summary_no_repeat_ngram_enabled=True,
        review_text_no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_size=3,
        review_text_no_repeat_ngram_size=0,
        summary_temperature=1.05,
        review_text_temperature=0.90,
        summary_top_p=0.97,
        review_text_top_p=0.95,
        purpose="Exact blocking plus summary-only ngram blocking on the v5 checkpoint.",
    ),
    "v51_exact_block_only_no_ngram": VariantSpec(
        name="v51_exact_block_only_no_ngram",
        checkpoint_source="v5.1",
        config_path=V51_CONFIG,
        checkpoint_path=V51_CHECKPOINT,
        exact_train_overlap_blocking_enabled=True,
        summary_exact_blocking_enabled=True,
        review_text_exact_blocking_enabled=True,
        no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_enabled=False,
        review_text_no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_size=0,
        review_text_no_repeat_ngram_size=0,
        summary_temperature=1.05,
        review_text_temperature=0.95,
        summary_top_p=0.97,
        review_text_top_p=0.95,
        purpose="v5.1 checkpoint with expensive ngram blocking disabled.",
    ),
    "v51_no_block_no_ngram": VariantSpec(
        name="v51_no_block_no_ngram",
        checkpoint_source="v5.1",
        config_path=V51_CONFIG,
        checkpoint_path=V51_CHECKPOINT,
        exact_train_overlap_blocking_enabled=False,
        summary_exact_blocking_enabled=False,
        review_text_exact_blocking_enabled=False,
        no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_enabled=False,
        review_text_no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_size=0,
        review_text_no_repeat_ngram_size=0,
        summary_temperature=1.05,
        review_text_temperature=0.95,
        summary_top_p=0.97,
        review_text_top_p=0.95,
        purpose="v5.1 checkpoint without sampling-time privacy controls.",
    ),
    "v5_summary_exact_block_only": VariantSpec(
        name="v5_summary_exact_block_only",
        checkpoint_source="v5",
        config_path=V5_CONFIG,
        checkpoint_path=V5_CHECKPOINT,
        exact_train_overlap_blocking_enabled=True,
        summary_exact_blocking_enabled=True,
        review_text_exact_blocking_enabled=False,
        no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_enabled=False,
        review_text_no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_size=0,
        review_text_no_repeat_ngram_size=0,
        summary_temperature=0.90,
        review_text_temperature=0.90,
        summary_top_p=0.95,
        review_text_top_p=0.95,
        purpose="Minimal v5 summary exact-overlap blocking only.",
    ),
    "v5_exact_block_review_temperature_1": VariantSpec(
        name="v5_exact_block_review_temperature_1",
        checkpoint_source="v5",
        config_path=V5_CONFIG,
        checkpoint_path=V5_CHECKPOINT,
        exact_train_overlap_blocking_enabled=True,
        summary_exact_blocking_enabled=True,
        review_text_exact_blocking_enabled=True,
        no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_enabled=False,
        review_text_no_repeat_ngram_enabled=False,
        summary_no_repeat_ngram_size=0,
        review_text_no_repeat_ngram_size=0,
        summary_temperature=1.05,
        review_text_temperature=1.00,
        summary_top_p=0.97,
        review_text_top_p=0.95,
        purpose="Optional v5 exact blocking with slightly warmer review text sampling.",
    ),
}


@dataclass(frozen=True)
class VariantPaths:
    run_dir: Path
    output_csv: Path
    metadata_dir: Path
    runtime_json: Path
    sampling_config_json: Path
    evaluation_dir: Path
    eval_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v5.2 LSTM sampling privacy ablations without retraining.")
    parser.add_argument("--synthetic-spine", required=True)
    parser.add_argument("--real-reviews", required=True)
    parser.add_argument("--num-rows", default=50000)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--comparison-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [name for name in args.variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}. Available: {', '.join(sorted(VARIANTS))}")
    output_root = Path(args.output_root)
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    (output_root / "comparison").mkdir(parents=True, exist_ok=True)
    if not args.comparison_only:
        for name in args.variants:
            run_variant(VARIANTS[name], args, output_root)
    compare_runs(list(args.variants), output_root / "runs", output_root / "comparison")


def run_variant(spec: VariantSpec, args: argparse.Namespace, output_root: Path) -> None:
    paths = variant_paths(output_root, spec.name)
    paths.metadata_dir.mkdir(parents=True, exist_ok=True)
    paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(spec.config_path)
    write_sampling_config(spec, args, config, paths.sampling_config_json)
    if not args.skip_sampling:
        if args.skip_existing and paths.output_csv.exists() and paths.runtime_json.exists():
            print(f"[skip] sampling exists for {spec.name}")
        else:
            print(f"[sample] {spec.name}")
            sample_lstm_fast_from_config(
                config,
                checkpoint_path=spec.checkpoint_path,
                output_path=paths.output_csv,
                num_rows=parse_num_rows(args.num_rows),
                batch_size=args.batch_size,
                device=args.device,
                synthetic_spine_path=args.synthetic_spine,
                options=fast_options_for_variant(spec, args, paths),
            )
    if not args.skip_evaluation:
        if args.skip_existing and paths.eval_json.exists():
            print(f"[skip] evaluation exists for {spec.name}")
        else:
            print(f"[evaluate] {spec.name}")
            evaluate_from_config(
                config,
                synthetic_reviews_path=paths.output_csv,
                real_reviews_path=args.real_reviews,
                output_path=paths.eval_json,
            )


def variant_paths(output_root: str | Path, variant_name: str) -> VariantPaths:
    run_dir = Path(output_root) / "runs" / variant_name
    metadata_dir = run_dir / "metadata"
    evaluation_dir = run_dir / "evaluation"
    return VariantPaths(
        run_dir=run_dir,
        output_csv=run_dir / "synthetic_review_attrs_fast.csv",
        metadata_dir=metadata_dir,
        runtime_json=metadata_dir / "runtime_sampling_fast.json",
        sampling_config_json=metadata_dir / "sampling_config.json",
        evaluation_dir=evaluation_dir,
        eval_json=evaluation_dir / "eval_metrics_fast_sampler_fixed_decode_normalized.json",
    )


def fast_options_for_variant(spec: VariantSpec, args: argparse.Namespace, paths: VariantPaths) -> FastSamplerOptions:
    return FastSamplerOptions(
        profile=True,
        profile_output=paths.runtime_json,
        detailed_profile_output=paths.metadata_dir / "runtime_sampling_profile_detailed.json",
        decode_mode="bucketed",
        auto_batch_size=True,
        mixed_precision=True,
        cache_graph_context=True,
        graph_context_cache_mode="batch",
        cache_condition_embeddings=True,
        active_row_masking=True,
        length_bucketed_decoding=True,
        detokenize_after_generation=True,
        write_chunk_size=10000,
        seed=args.seed,
        use_config_privacy_controls=False,
        summary_temperature=spec.summary_temperature,
        review_text_temperature=spec.review_text_temperature,
        summary_top_p=spec.summary_top_p,
        review_text_top_p=spec.review_text_top_p,
        no_repeat_ngram_enabled=spec.no_repeat_ngram_enabled,
        summary_no_repeat_ngram_enabled=spec.summary_no_repeat_ngram_enabled,
        review_text_no_repeat_ngram_enabled=spec.review_text_no_repeat_ngram_enabled,
        summary_no_repeat_ngram_size=spec.summary_no_repeat_ngram_size,
        review_text_no_repeat_ngram_size=spec.review_text_no_repeat_ngram_size,
        exact_train_overlap_blocking_enabled=spec.exact_train_overlap_blocking_enabled,
        summary_exact_blocking_enabled=spec.summary_exact_blocking_enabled,
        review_text_exact_blocking_enabled=spec.review_text_exact_blocking_enabled,
        max_summary_resample_attempts=5 if spec.exact_train_overlap_blocking_enabled else 0,
        max_review_text_resample_attempts=3 if spec.exact_train_overlap_blocking_enabled else 0,
    )


def sampling_config_payload(spec: VariantSpec, args: argparse.Namespace | dict[str, Any], config: ConditionalTABDLMConfig) -> dict[str, Any]:
    args_dict = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    graph_flags = graph_metadata(config.raw, real_graph_used_at_sampling=False)
    return {
        "ablation_name": "v5.2_lstm_sampling_privacy_ablation",
        "variant": spec.name,
        "checkpoint_source": spec.checkpoint_source,
        "checkpoint_path": spec.checkpoint_path,
        "config_path": spec.config_path,
        "synthetic_spine": args_dict.get("synthetic_spine"),
        "real_reviews": args_dict.get("real_reviews"),
        "num_rows": args_dict.get("num_rows"),
        "seed": args_dict.get("seed"),
        "purpose": spec.purpose,
        "variant_controls": asdict(spec),
        "sampler_defaults": {
            "decode_mode": "bucketed",
            "mixed_precision": True,
            "auto_batch_size": True,
            "active_row_masking": True,
            "length_bucketed_decoding": True,
            "detokenize_after_generation": True,
            "write_chunk_size": 10000,
            "use_config_privacy_controls": False,
            "max_summary_resample_attempts": 5 if spec.exact_train_overlap_blocking_enabled else 0,
            "max_review_text_resample_attempts": 3 if spec.exact_train_overlap_blocking_enabled else 0,
        },
        "joint_generation": True,
        "review_text_generated_jointly": "review_text" in config.schema.text_targets,
        "review_text_separate_stage": False,
        **graph_flags,
        "real_graph_used_at_sampling": False,
    }


def write_sampling_config(
    spec: VariantSpec,
    args: argparse.Namespace,
    config: ConditionalTABDLMConfig,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sampling_config_payload(spec, args, config), handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_num_rows(value: Any) -> int | str | None:
    if value in (None, "all"):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


if __name__ == "__main__":
    main()
