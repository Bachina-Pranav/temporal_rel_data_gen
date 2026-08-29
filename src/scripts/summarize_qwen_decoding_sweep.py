#!/usr/bin/env python3
"""Build one compact GPT briefing for the Qwen decoding sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLICIES = (
    "D0_t090_p095_r105",
    "D1_t105_p095_r105",
    "D2_t115_p098_r110",
)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return sanitize(pd.read_csv(path).to_dict("records"))


def pick(source: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: source.get(key) for key in keys}


def policy_diagnostics(root: Path, name: str) -> dict[str, Any]:
    policy_root = root / name
    diversity = read_json(policy_root / "diversity_metrics.json")
    memorization = read_json(policy_root / "memorization_metrics.json")
    conditioning = read_json(policy_root / "conditioning_metrics.json")
    generation = read_json(policy_root / "generation_metrics.json")
    fields = {}
    for field in ("summary", "review_text"):
        distribution = diversity.get(field, {})
        synthetic = distribution.get("synthetic", {})
        memory = memorization.get(field, {}).get("qwen", {})
        fields[field] = {
            "length": pick(
                synthetic,
                "token_length_mean",
                "token_length_median",
                "token_length_p90",
                "token_length_p95",
                "token_length_p99",
            ),
            "length_ks_vs_real": distribution.get("length_ks"),
            "diversity": pick(
                synthetic,
                "unique_text_ratio",
                "exact_duplicate_rate",
                "distinct_1",
                "distinct_2",
                "repeated_unigram_rate",
                "repeated_bigram_rate",
                "repeated_trigram_rate",
                "repeated_sentence_rate",
                "vocabulary_size",
            ),
            "train_likeness": {
                "exact_train_overlap_rate": memory.get(
                    "exact_train_overlap_rate"
                ),
                "nearest_train_tfidf": memory.get("nearest_neighbor"),
            },
        }
    return {
        "text_diagnostics": fields,
        "rating_consistency": conditioning.get("rating"),
        "verified_consistency": conditioning.get("verified"),
        "generation_efficiency": pick(
            generation,
            "rows",
            "seconds",
            "generated_tokens",
            "rows_per_second",
            "tokens_per_second",
            "average_tokens_per_row",
            "peak_vram_mb",
            "parse",
            "random_seed",
        ),
    }


def real_diagnostics(root: Path, first_policy: dict[str, Any]) -> dict[str, Any]:
    diversity = read_json(root / "real_validation_diversity_metrics.json")
    rating = read_json(root / "fixed_probe_metrics.json").get("rating", {})
    result = {
        "rating_consistency": rating.get("real_validation"),
        "text_diagnostics": {},
    }
    for field in ("summary", "review_text"):
        memory = (
            first_policy.get(field, {}).get("real_heldout", {})
        )
        result["text_diagnostics"][field] = {
            "distribution": diversity.get(field),
            "train_likeness": {
                "exact_train_overlap_rate": memory.get(
                    "exact_train_overlap_rate"
                ),
                "nearest_train_tfidf": memory.get("nearest_neighbor"),
            },
        }
    return result


def phase1_context(base: Path) -> dict[str, Any]:
    phase1 = base / "phase1"
    oracle = read_json(phase1 / "oracle_summary/metrics.json")
    generated = read_json(phase1 / "generated_structured_qwen/metrics.json")
    diffusion = read_csv(
        phase1 / "diffusion_oracle_canonical/canonical_results.csv"
    )
    return {
        "oracle_summary_diagnostic": {
            "review_c2st": oracle.get("review_c2st"),
            "oracle_summary_gain": oracle.get("oracle_summary_gain"),
        },
        "generated_structured_diagnostic": generated,
        "diffusion_oracle_comparison": diffusion,
        "interpretation": [
            "Generated-summary propagation was not the dominant remaining failure.",
            "Generated rating/verified caused little additional review-text degradation.",
            "The decoding sweep therefore isolated mode concentration under oracle structured conditioning.",
        ],
    }


def derived_comparisons(
    rows: list[dict[str, Any]], decision: dict[str, Any]
) -> dict[str, Any]:
    indexed = {row["configuration"]: row for row in rows}
    baseline = indexed.get(POLICIES[0], {})
    selected_name = decision.get("selected_configuration")
    selected = indexed.get(selected_name, {})
    higher = indexed.get(POLICIES[2], {})

    def improvement(metric: str, candidate: dict[str, Any]) -> Any:
        if baseline.get(metric) is None or candidate.get(metric) is None:
            return None
        return float(baseline[metric]) - float(candidate[metric])

    macro_gain = improvement("macro_c2st", selected)
    relative = None
    if macro_gain is not None and baseline.get("macro_c2st"):
        relative = macro_gain / float(baseline["macro_c2st"])
    return {
        "selected_vs_D0": {
            "absolute_summary_c2st_reduction": improvement(
                "summary_c2st", selected
            ),
            "absolute_review_c2st_reduction": improvement(
                "review_c2st", selected
            ),
            "absolute_macro_c2st_reduction": macro_gain,
            "relative_macro_c2st_reduction": relative,
            "review_distinct_2_change": (
                float(selected["review_distinct_2"])
                - float(baseline["review_distinct_2"])
                if selected.get("review_distinct_2") is not None
                and baseline.get("review_distinct_2") is not None
                else None
            ),
            "review_repetition_reduction": improvement(
                "review_repeated_ngram_rate", selected
            ),
            "rating_macro_f1_change": (
                float(selected["rating_macro_f1"])
                - float(baseline["rating_macro_f1"])
                if selected.get("rating_macro_f1") is not None
                and baseline.get("rating_macro_f1") is not None
                else None
            ),
        },
        "D2_tradeoff": {
            "macro_c2st_vs_selected": (
                float(higher["macro_c2st"]) - float(selected["macro_c2st"])
                if higher.get("macro_c2st") is not None
                and selected.get("macro_c2st") is not None
                else None
            ),
            "review_distinct_2_vs_selected": (
                float(higher["review_distinct_2"])
                - float(selected["review_distinct_2"])
                if higher.get("review_distinct_2") is not None
                and selected.get("review_distinct_2") is not None
                else None
            ),
            "rating_macro_f1_vs_selected": (
                float(higher["rating_macro_f1"])
                - float(selected["rating_macro_f1"])
                if higher.get("rating_macro_f1") is not None
                and selected.get("rating_macro_f1") is not None
                else None
            ),
            "interpretation": "D2 is the diversity extreme; check whether its extra diversity justifies its C2ST and rating-consistency regressions relative to D1.",
        },
    }


def build_summary(root: Path) -> dict[str, Any]:
    base = root.parent
    preflight = read_json(root / "preflight.json")
    subset = read_json(root / "validation_subset_manifest.json")
    decision = read_json(root / "decoding_decision.json")
    comparison = read_csv(root / "comparison.csv")
    confirmation = read_json(
        root / "test_confirmation/confirmation_decision.json"
    )
    audit = read_json(root / "phase1_metadata_audit.json")
    first_memory = read_json(root / POLICIES[0] / "memorization_metrics.json")
    policies = {
        name: policy_diagnostics(root, name)
        for name in POLICIES
        if (root / name / "synthetic_text.csv").is_file()
    }
    return sanitize(
        {
            "document_purpose": (
                "Self-contained evidence package for brainstorming the next "
                "RelGen text-generator experiment."
            ),
            "metric_semantics": {
                "text_c2st_error": "Lower is better; 0 means chance-level real/synthetic discrimination.",
                "validation_selection": "D0/D1/D2 selection used only a deterministic 2,000-row validation subset.",
                "test_confirmation": "Only the frozen validation winner was evaluated once on the 3,982-row held-out test population.",
                "memorization": "Train-likeness diagnostics are not a privacy guarantee.",
            },
            "experiment_design": {
                "training_performed": False,
                "model_source": preflight.get("model_source"),
                "adapter_sha256": preflight.get("adapter_sha256"),
                "canonical_evaluator": preflight.get("canonical_evaluator"),
                "validation_subset": subset,
                "policies": preflight.get("policies"),
                "random_seed_strategy": preflight.get("random_seed_strategy"),
            },
            "prior_phase1_context": phase1_context(base),
            "validation_comparison": comparison,
            "real_validation_reference": real_diagnostics(root, first_memory),
            "per_policy_diagnostics": policies,
            "selection_decision": decision,
            "derived_comparisons": derived_comparisons(comparison, decision),
            "heldout_test_confirmation": confirmation,
            "phase1_metadata_audit": {
                "verdict": audit.get("verdict"),
                "experimental_outputs_correct": audit.get(
                    "experimental_outputs_correct"
                ),
                "metadata_aggregation_wrong": audit.get(
                    "metadata_aggregation_wrong"
                ),
                "historical_files_modified": audit.get(
                    "historical_files_modified"
                ),
            },
            "claims_supported": [
                "Decoding policy was a meaningful bottleneck for this frozen Qwen adapter.",
                "D1 was the best validation trade-off under the prespecified joint guardrails.",
                "D2 moved lexical diversity closer to real text but weakened semantic conditioning and C2ST relative to D1.",
            ],
            "claims_not_supported": [
                "No statistical-significance claim was made for heuristic differences.",
                "The oracle-structured validation sweep is not an end-to-end generated-database result.",
                "Train-overlap diagnostics do not establish differential privacy.",
                "No conclusion about a larger Qwen model follows without a controlled capacity experiment.",
            ],
            "brainstorming_questions": [
                "Does the held-out confirmation support freezing D1 as the final decoding policy?",
                "How should the difference between oracle-structured validation and generated-structured test results be interpreted?",
                "Is D2's closer-to-real lexical diversity scientifically useful despite its weaker C2ST and rating consistency?",
                "Should the next single experiment target temporal-relational conditioning, model capacity, or the conditioning objective?",
                "Which results are suitable for the main paper, and which belong only in an ablation or appendix?",
            ],
        }
    )


def fmt(value: Any, digits: int = 6) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def build_markdown(summary: dict[str, Any]) -> str:
    rows = summary["validation_comparison"]
    decision = summary["selection_decision"]
    confirmation = summary["heldout_test_confirmation"]
    derived = summary["derived_comparisons"]["selected_vs_D0"]
    real = summary["real_validation_reference"]
    lines = [
        "# GPT Brainstorming Brief: Qwen3-0.6B Decoding Sweep",
        "",
        "## What I Want You To Do",
        "",
        "Act as a critical research collaborator. Analyze whether D1 should be frozen, distinguish oracle diagnostics from end-to-end evidence, identify the most likely remaining failure source, and recommend exactly one next experiment. Do not treat descriptive differences as statistical significance.",
        "",
        "## Metric Semantics",
        "",
        "- Text C2ST error is `2 * abs(AUC - 0.5)`; **lower is better** and 0 is chance.",
        "- D0/D1/D2 selection used validation only (2,000 deterministic rows, seed 42).",
        "- Only the selected validation policy received one held-out test confirmation.",
        "- Memorization metrics are train-likeness diagnostics, not privacy guarantees.",
        "",
        "## Experiment",
        "",
        "No model was retrained. The frozen Qwen3-0.6B LoRA adapter received true validation rating and verified, then generated summary followed by review. The model, subset, batching, token bound, EOS handling, and seed strategy were fixed; only temperature, top-p, and repetition penalty changed.",
        "",
        "| Policy | Temperature | top-p | Repetition penalty |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['configuration']} | {row['temperature']} | {row['top_p']} | {row['repetition_penalty']} |"
        for row in rows
    )
    lines += [
        "",
        "## Validation Results",
        "",
        "| Policy | Summary C2ST | Review C2ST | Macro C2ST | Review distinct-2 | Review repetition | Summary train overlap | Review train overlap | Rating balanced acc. | Rating macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| {configuration} | {summary_c2st:.6f} | {review_c2st:.6f} | {macro_c2st:.6f} | {review_distinct_2:.6f} | {review_repeated_ngram_rate:.6f} | {summary_exact_train_overlap:.6f} | {review_exact_train_overlap:.6f} | {rating_balanced_accuracy:.6f} | {rating_macro_f1:.6f} |".format(
            **row
        )
        for row in rows
    )
    real_review = real["text_diagnostics"]["review_text"]
    real_summary = real["text_diagnostics"]["summary"]
    real_review_distribution = real_review.get("distribution") or {}
    real_rating = real.get("rating_consistency") or {}
    lines += [
        "",
        "### Real Validation Reference",
        "",
        f"- Review distinct-2: `{fmt(real_review_distribution.get('distinct_2'))}`",
        f"- Review repeated bigram rate: `{fmt(real_review_distribution.get('repeated_bigram_rate'))}`",
        f"- Summary exact train overlap: `{fmt(real_summary['train_likeness'].get('exact_train_overlap_rate'))}`",
        f"- Review exact train overlap: `{fmt(real_review['train_likeness'].get('exact_train_overlap_rate'))}`",
        f"- Rating macro-F1: `{fmt(real_rating.get('macro_f1'))}`",
        "",
        "### Selected D1 Versus D0",
        "",
        f"- Absolute macro C2ST reduction: `{fmt(derived.get('absolute_macro_c2st_reduction'))}`",
        f"- Relative macro C2ST reduction: `{fmt(derived.get('relative_macro_c2st_reduction'))}`",
        f"- Review distinct-2 change: `{fmt(derived.get('review_distinct_2_change'))}`",
        f"- Review repetition reduction: `{fmt(derived.get('review_repetition_reduction'))}`",
        f"- Rating macro-F1 change: `{fmt(derived.get('rating_macro_f1_change'))}`",
        "",
        "## Frozen Decision",
        "",
        f"- Selected validation policy: **{decision.get('selected_configuration')}**",
        f"- Decoding-bottleneck classification: **{decision.get('decoding_bottleneck')}**",
        f"- Clearly preferable to D0: `{decision.get('clearly_preferable_to_D0')}`",
        "",
        "## Held-Out Generated-Structured Confirmation",
        "",
        f"- Run: `{confirmation.get('run')}`",
        f"- Frozen policy: `{confirmation.get('frozen_validation_configuration')}`",
        f"- Summary C2ST: `{fmt(confirmation.get('summary_c2st'))}`",
        f"- Review C2ST: `{fmt(confirmation.get('review_c2st'))}`",
        f"- Macro C2ST: `{fmt(confirmation.get('macro_c2st'))}`",
        f"- Macro improvement over fixed generated-structured baseline: `{fmt(confirmation.get('macro_improvement_vs_fixed_baseline'))}`",
        "- This confirmation was run once; no tuning followed test inspection.",
        "",
        "## Phase-1 Context",
        "",
        "- Oracle-summary analysis showed summary propagation was not the dominant remaining failure.",
        "- Generated rating/verified introduced little additional review degradation under the old policy.",
        "- The sweep therefore tested mode concentration directly using oracle structured validation conditions.",
        "- The Phase-1 normal/no-summary metadata problem was an aggregation overwrite; generated CSVs and C2ST outputs were distinct and valid.",
        "",
        "## Interpretation Constraints",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["claims_not_supported"])
    lines += ["", "## Questions To Brainstorm", ""]
    lines.extend(
        f"{index}. {question}"
        for index, question in enumerate(summary["brainstorming_questions"], 1)
    )
    lines += [
        "",
        "## Detailed Evidence",
        "",
        "The companion `gpt_brainstorm_report.json` contains full per-policy length distributions, duplicate/repetition metrics, nearest-training TF-IDF statistics, conditioning probes, generation efficiency, provenance hashes, and the exact decision guardrails.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        default="outputs/qwen_text_decoder_06b/decoding_sweep",
    )
    parser.add_argument("--markdown-output")
    parser.add_argument("--json-output")
    args = parser.parse_args()
    root = Path(args.experiment_root)
    markdown_path = Path(args.markdown_output) if args.markdown_output else root / "gpt_brainstorm_report.md"
    json_path = Path(args.json_output) if args.json_output else root / "gpt_brainstorm_report.json"
    required = [root / "comparison.csv", root / "decoding_decision.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed sweep artifacts:\n- " + "\n- ".join(missing))
    summary = build_summary(root)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(markdown_path)
    print(json_path)


if __name__ == "__main__":
    main()
