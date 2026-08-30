#!/usr/bin/env python3
"""Repeat canonical Text C2ST folds across seeds while reusing embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribute_generation.qwen_text_decoder.decoding_sweep import TEXT_FIELDS
from attribute_generation.qwen_text_decoder.experiment import nested_c2st, write_json
from evaluation.text_c2st_audit import EmbeddingStore, TextC2STProtocol, evaluate_protocol


def candidates() -> list[dict[str, Path | str]]:
    values: list[dict[str, Path | str]] = []
    benchmark = Path("outputs/amazon-toy/hierarchical_diffusion_benchmark")
    base = Path("outputs/qwen_text_decoder_06b")
    for mode in ("oracle_structured", "generated_structured"):
        synthetic = base / mode / "synthetic_text.csv"
        if synthetic.is_file():
            values.append({
                "name": f"qwen_main_{mode}",
                "real": benchmark / "test_real.csv",
                "synthetic": synthetic,
            })
    sweep = base / "decoding_sweep"
    for policy in ("D0_t090_p095_r105", "D1_t105_p095_r105", "D2_t115_p098_r110"):
        synthetic = sweep / policy / "synthetic_text.csv"
        if synthetic.is_file() and (sweep / "validation_subset.csv").is_file():
            values.append({
                "name": f"qwen_decoding_{policy}",
                "real": sweep / "validation_subset.csv",
                "synthetic": synthetic,
            })
    relational = base / "relational_prefix_probe"
    for mode in ("R0_no_prefix", "R1_correct_context", "R2_shuffled_context"):
        synthetic = relational / mode / "synthetic_text.csv"
        if synthetic.is_file() and (relational / "data/validation_subset.csv").is_file():
            values.append({
                "name": f"qwen_relational_{mode}",
                "real": relational / "data/validation_subset.csv",
                "synthetic": synthetic,
            })
    return values


def resolve_model_snapshot() -> str:
    for path in (
        Path("outputs/qwen_text_decoder_06b/evaluation_model_source.json"),
        Path("outputs/qwen_text_decoder_06b/decoding_sweep/evaluation_model_source.json"),
        Path("outputs/qwen_text_decoder_06b/phase1/evaluation_model_source.json"),
    ):
        if not path.is_file():
            continue
        value = json.loads(path.read_text()).get("local_snapshot")
        if value and Path(value).is_dir():
            return value
    raise FileNotFoundError("No pinned local MiniLM snapshot was found; refusing an unpinned fallback")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for name in sorted({row["candidate"] for row in rows}):
        selected = [row for row in rows if row["candidate"] == name]
        item: dict[str, Any] = {"candidate": name, "num_seeds": len(selected)}
        for metric in ("summary_c2st", "review_c2st", "macro_c2st"):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
            item[f"{metric}_min"] = float(values.min())
            item[f"{metric}_max"] = float(values.max())
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73, 101, 137])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--output-dir", default="outputs/overnight_8h_session/repeated_seed_text_c2st"
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    final = output / "seed_robustness.json"
    if args.skip_existing and final.is_file():
        print(final)
        return
    rows = []
    discovered = candidates()
    if not discovered:
        write_json(
            final,
            {
                "status": "no_finished_text_artifacts",
                "data_regenerated": False,
                "seeds": args.seeds,
                "per_seed": [],
                "aggregate": [],
            },
        )
        print(final)
        return
    model = resolve_model_snapshot()
    store = EmbeddingStore(output / "embedding_cache", device=args.device)
    for candidate in discovered:
        real = pd.read_csv(candidate["real"], low_memory=False)
        synthetic = pd.read_csv(candidate["synthetic"], low_memory=False)
        maximum = min(len(real), len(synthetic), 5000)
        for seed in args.seeds:
            protocol = TextC2STProtocol(
                name="canonical_text_c2st_seed_robustness_v1",
                embedding_backend="minilm",
                embedding_model=model,
                preprocessing="canonical",
                classifiers=("logistic_regression",),
                max_rows=maximum,
                seed=seed,
                n_splits=5,
            )
            metrics = evaluate_protocol(
                real, synthetic, protocol, store, fields=TEXT_FIELDS,
                label=str(candidate["name"]),
            )
            summary, review, macro = nested_c2st(metrics)
            rows.append({
                "candidate": candidate["name"], "seed": seed,
                "summary_c2st": summary, "review_c2st": review, "macro_c2st": macro,
            })
    result = {
        "metric_semantics": "lower is better; zero is chance-level discrimination",
        "data_regenerated": False,
        "embeddings_reused_across_seeds": True,
        "seeds": args.seeds,
        "candidates_discovered": len(discovered),
        "per_seed": rows,
        "aggregate": summarize(rows),
    }
    write_json(final, result)
    pd.DataFrame(rows).to_csv(output / "per_seed.csv", index=False)
    print(final)


if __name__ == "__main__":
    main()
