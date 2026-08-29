"""Validation-only decoding sweep for the frozen Qwen3-0.6B adapter."""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from evaluation.text_c2st_audit import (
    EmbeddingStore,
    TextC2STProtocol,
    canonical_text,
    evaluate_protocol,
    file_sha256,
)

from .experiment import (
    QwenTextExperiment,
    alignment_audit,
    nested_c2st,
    validate_runtime_dependencies,
    write_json,
)
from .followup import (
    QwenFollowupExperiment,
    dataframe_sha256,
    directory_fingerprint,
    evaluate_text_consistency,
    select_fixed_subset,
)
from .phase1 import (
    EXPECTED_MINILM_REVISION,
    exact_minilm_snapshot,
    matched_memorization_metrics,
)


POLICY_NAMES = (
    "D0_t090_p095_r105",
    "D1_t105_p095_r105",
    "D2_t115_p098_r110",
)
TEXT_FIELDS = ("summary", "review_text")
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def text_tokens(value: Any) -> list[str]:
    return TOKEN_PATTERN.findall(canonical_text(value).lower())


def _repeat_rate(sequences: list[list[str]], n: int) -> float:
    ngrams = [
        tuple(row[index : index + n])
        for row in sequences
        for index in range(max(0, len(row) - n + 1))
    ]
    repeated = sum(max(count - 1, 0) for count in Counter(ngrams).values())
    return float(repeated / max(len(ngrams), 1))


def _text_side_metrics(values: pd.Series) -> dict[str, Any]:
    sequences = [text_tokens(value) for value in values]
    lengths = np.asarray([len(row) for row in sequences], dtype=np.int64)
    flat = [token for row in sequences for token in row]
    bigrams = [pair for row in sequences for pair in zip(row, row[1:])]
    normalized = pd.Series([" ".join(row) for row in sequences], dtype="object")
    sentences = [
        sentence.strip().lower()
        for value in normalized
        for sentence in re.split(r"[.!?]+", value)
        if sentence.strip()
    ]
    return {
        "token_length_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "token_length_median": float(np.median(lengths)) if len(lengths) else 0.0,
        "token_length_p90": float(np.quantile(lengths, 0.90)) if len(lengths) else 0.0,
        "token_length_p95": float(np.quantile(lengths, 0.95)) if len(lengths) else 0.0,
        "token_length_p99": float(np.quantile(lengths, 0.99)) if len(lengths) else 0.0,
        "average_tokens_per_row": float(lengths.mean()) if len(lengths) else 0.0,
        "empty_rate": float(np.mean(lengths == 0)) if len(lengths) else 0.0,
        "unique_text_ratio": float(normalized.nunique() / max(len(normalized), 1)),
        "exact_duplicate_rate": float(normalized.duplicated().mean()),
        "distinct_1": float(len(set(flat)) / max(len(flat), 1)),
        "distinct_2": float(len(set(bigrams)) / max(len(bigrams), 1)),
        "repeated_unigram_rate": _repeat_rate(sequences, 1),
        "repeated_bigram_rate": _repeat_rate(sequences, 2),
        "repeated_trigram_rate": _repeat_rate(sequences, 3),
        "repeated_ngram_rate": _repeat_rate(sequences, 2),
        "repeated_sentence_rate": float(
            1.0 - len(set(sentences)) / max(len(sentences), 1)
        ),
        "vocabulary_size": int(len(set(flat))),
    }


def detailed_diversity_metrics(
    real: pd.DataFrame, synthetic: pd.DataFrame
) -> dict[str, Any]:
    """Extend the project's existing global-repeat definitions consistently."""
    from scipy.stats import ks_2samp

    result: dict[str, Any] = {
        "definitions": {
            "tokenization": "canonical lowercase word tokens (project token regex)",
            "repeated_ngram_rate": "repeated_bigram_rate for backward compatibility",
            "repeat_rate": "sum(max(global_ngram_count - 1, 0)) / total_ngrams",
            "exact_duplicate_rate": "fraction duplicated after canonical token joining",
        }
    }
    for field in TEXT_FIELDS:
        real_metrics = _text_side_metrics(real[field])
        synthetic_metrics = _text_side_metrics(synthetic[field])
        real_lengths = real[field].map(lambda value: len(text_tokens(value)))
        synthetic_lengths = synthetic[field].map(lambda value: len(text_tokens(value)))
        result[field] = {
            "real": real_metrics,
            "synthetic": synthetic_metrics,
            "length_ks": float(
                ks_2samp(real_lengths, synthetic_lengths).statistic
            ),
        }
    return result


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def audit_phase1_metadata(
    base_output: Path, phase1_output: Path, sweep_output: Path
) -> dict[str, Any]:
    """Diagnose the known shared-metadata overwrite without changing old files."""
    normal_csv = phase1_output / "oracle_summary/normal.csv"
    no_summary_csv = phase1_output / "oracle_summary/no_summary.csv"
    oracle_csv = phase1_output / "oracle_summary/oracle_summary.csv"
    shared_metrics_path = phase1_output / "oracle_summary/generation_metrics.json"
    canonical_path = phase1_output / "phase1_canonical_text_c2st.json"
    artifacts = {
        "normal": normal_csv,
        "oracle_summary": oracle_csv,
        "no_summary": no_summary_csv,
    }
    hashes = {
        name: file_sha256(path) if path.is_file() else None
        for name, path in artifacts.items()
    }
    shared = _read_json(shared_metrics_path) or {}
    canonical = _read_json(canonical_path) or {}
    normal_scores = nested_c2st(canonical.get("normal", {}))
    no_summary_scores = nested_c2st(canonical.get("no_summary", {}))
    outputs_distinct = bool(
        hashes["normal"]
        and hashes["no_summary"]
        and hashes["normal"] != hashes["no_summary"]
    )
    c2st_distinct = bool(
        normal_scores[1] is not None
        and no_summary_scores[1] is not None
        and normal_scores[1] != no_summary_scores[1]
    )
    shared_points_to_no_summary = bool(
        shared.get("policy") == "no_summary"
        or shared.get("summary_mode") == "omitted"
        or (
            hashes["no_summary"]
            and shared.get("output_sha256") == hashes["no_summary"]
        )
    )
    result = {
        "historical_files_modified": False,
        "artifact_hashes": hashes,
        "base_normal_hash": (
            file_sha256(base_output / "oracle_structured/synthetic_text.csv")
            if (base_output / "oracle_structured/synthetic_text.csv").is_file()
            else None
        ),
        "shared_generation_metadata": shared,
        "normal_and_no_summary_outputs_distinct": outputs_distinct,
        "normal_and_no_summary_c2st_distinct": c2st_distinct,
        "shared_metadata_points_to_no_summary": shared_points_to_no_summary,
        "experimental_outputs_correct": bool(outputs_distinct and c2st_distinct),
        "metadata_aggregation_wrong": bool(
            outputs_distinct and c2st_distinct and shared_points_to_no_summary
        ),
    }
    if result["metadata_aggregation_wrong"]:
        result["verdict"] = "OUTPUTS CORRECT; SHARED METADATA AGGREGATION WAS WRONG"
    elif result["experimental_outputs_correct"]:
        result["verdict"] = "OUTPUTS CORRECT; NO SHARED-METADATA MISMATCH DETECTED"
    elif not all(path.is_file() for path in artifacts.values()):
        result["verdict"] = "UNRESOLVED; PHASE-1 ARTIFACTS ARE NOT ALL AVAILABLE"
    else:
        result["verdict"] = "DEEPER INCONSISTENCY REQUIRES REVIEW"
    write_json(sweep_output / "phase1_metadata_audit.json", result)
    lines = [
        "# Phase-1 Metadata Audit",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
        "- Historical generation CSVs modified: no",
        f"- Normal and no-summary CSVs distinct: {outputs_distinct}",
        f"- Normal and no-summary C2ST distinct: {c2st_distinct}",
        f"- Shared metadata points to no-summary: {shared_points_to_no_summary}",
        "",
        "The Phase-1 implementation stored all oracle-summary policies in one directory. "
        "Each later policy rewrote that directory's shared `generation_metrics.json`; "
        "the policy-specific CSVs and C2ST files were not overwritten. New code reads "
        "and writes `generation_metrics/<policy>.json` instead.",
    ]
    (sweep_output / "phase1_metadata_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def select_decoding_configuration(
    rows: list[dict[str, Any]], thresholds: dict[str, Any]
) -> dict[str, Any]:
    baseline = next(row for row in rows if row["configuration"] == POLICY_NAMES[0])
    decisions = []
    for row in rows[1:]:
        macro_gain = float(baseline["macro_c2st"] - row["macro_c2st"])
        review_gain = float(baseline["review_c2st"] - row["review_c2st"])
        guards = {
            "rating_macro_f1": row["rating_macro_f1"]
            >= baseline["rating_macro_f1"]
            - float(thresholds["maximum_rating_macro_f1_drop"]),
            "rating_balanced_accuracy": row["rating_balanced_accuracy"]
            >= baseline["rating_balanced_accuracy"]
            - float(thresholds["maximum_rating_balanced_accuracy_drop"]),
            "parse": row["parse_failure_rate"]
            <= max(
                float(thresholds["maximum_parse_failure_rate"]),
                baseline["parse_failure_rate"] + 0.01,
            ),
            "length": row["review_length_ks"]
            <= baseline["review_length_ks"]
            + float(thresholds["maximum_review_length_ks_increase"]),
            "duplicates": row["review_exact_duplicate_rate"]
            <= baseline["review_exact_duplicate_rate"]
            + float(thresholds["maximum_review_duplicate_rate_increase"]),
            "repetition": row["review_repeated_ngram_rate"]
            <= baseline["review_repeated_ngram_rate"]
            + float(thresholds["maximum_review_repetition_rate_increase"]),
            "summary_train_likeness": row["summary_exact_train_overlap"]
            <= baseline["summary_exact_train_overlap"]
            + float(thresholds["maximum_summary_train_overlap_increase"]),
            "review_train_likeness": row["review_exact_train_overlap"]
            <= baseline["review_exact_train_overlap"]
            + float(thresholds["maximum_review_train_overlap_increase"]),
        }
        diversity_improved = bool(
            row["review_distinct_2"] >= baseline["review_distinct_2"] + 0.01
            or row["review_repeated_ngram_rate"]
            <= baseline["review_repeated_ngram_rate"] - 0.01
        )
        clearly_preferable = bool(
            all(guards.values())
            and (
                macro_gain
                >= float(thresholds["substantial_macro_improvement"])
                or (
                    macro_gain
                    >= float(thresholds["minimum_macro_improvement"])
                    and review_gain
                    >= float(
                        thresholds["minimum_review_improvement_for_joint_gate"]
                    )
                    and diversity_improved
                )
            )
        )
        decisions.append(
            {
                "configuration": row["configuration"],
                "macro_improvement_vs_D0": macro_gain,
                "review_improvement_vs_D0": review_gain,
                "diversity_improved": diversity_improved,
                "guardrails": guards,
                "clearly_preferable": clearly_preferable,
            }
        )
    eligible = [item for item in decisions if item["clearly_preferable"]]
    selected = POLICY_NAMES[0]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: next(
                row["macro_c2st"]
                for row in rows
                if row["configuration"] == item["configuration"]
            ),
        )["configuration"]
    best_macro = min(rows, key=lambda row: row["macro_c2st"])
    best_gain = float(baseline["macro_c2st"] - best_macro["macro_c2st"])
    if selected != POLICY_NAMES[0] and best_gain >= 0.06:
        hypothesis = "STRONGLY SUPPORTED"
    elif selected != POLICY_NAMES[0]:
        hypothesis = "MODERATELY SUPPORTED"
    elif any(item["diversity_improved"] for item in decisions) and best_gain < 0.015:
        hypothesis = "WEAKLY SUPPORTED"
    elif best_gain < 0.015:
        hypothesis = "REJECTED"
    else:
        hypothesis = "UNRESOLVED"
    return {
        "selected_configuration": selected,
        "validation_only": True,
        "clearly_preferable_to_D0": selected != POLICY_NAMES[0],
        "decoding_bottleneck": hypothesis,
        "candidate_decisions": decisions,
        "rule": "Macro C2ST primary; review C2ST, diversity, train-likeness, rating conditioning, validity, length, and simplicity are guardrails.",
        "statistical_significance_claim": False,
    }


@dataclass
class QwenDecodingSweepExperiment(QwenFollowupExperiment):
    """Run exactly three decoding policies without fitting any model."""

    config_path: Path

    def __post_init__(self) -> None:
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = Path(self.config["output_dir"])
        self.base_output = Path(self.config["base_output_dir"])
        self.phase1_output = Path(self.config["phase1_output_dir"])
        self.base_config_path = Path(self.config["base_experiment_config"])
        self.base = QwenTextExperiment(
            self.base_config_path, output_dir=self.base_output
        )
        self.seed = int(self.config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def subset_path(self) -> Path:
        return self.output / "validation_subset.csv"

    def _policy_path(self, name: str) -> Path:
        return self.output / name / "synthetic_text.csv"

    def preflight(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        required = [
            self.base_output / "training/model_source.json",
            self.base_output / "training/training_efficiency.json",
            self.base_output / "training/best_adapter/adapter_config.json",
            self.base.benchmark / "benchmark_manifest.json",
            self.base.benchmark / "train_real.csv",
            self.base.benchmark / "validation_real.csv",
            self.base.benchmark / "test_real.csv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required frozen decoding-sweep inputs are missing:\n- "
                + "\n- ".join(missing)
            )
        if tuple(self.config["generation"]["policies"]) != POLICY_NAMES:
            raise ValueError("Decoding sweep must contain exactly D0, D1, and D2")
        validate_runtime_dependencies()
        model_source = json.loads(
            (self.base_output / "training/model_source.json").read_text()
        )
        adapter = self.base_output / "training/best_adapter"
        if not adapter.is_dir() or not Path(model_source["local_snapshot"]).is_dir():
            raise FileNotFoundError("Frozen adapter or pinned Qwen snapshot is missing")
        evaluation = self.config["evaluation"]
        if evaluation["embedding_revision"] != EXPECTED_MINILM_REVISION:
            raise ValueError("Canonical MiniLM revision changed")
        minilm = exact_minilm_snapshot(
            evaluation["embedding_model"], evaluation["embedding_revision"]
        )
        validation = pd.read_csv(
            self.base.benchmark / "validation_real.csv", low_memory=False
        )
        subset_rows = int(self.config["validation_subset"]["rows"])
        expected_subset, expected_manifest = select_fixed_subset(
            validation,
            target_rows=subset_rows,
            maximum_rows=subset_rows,
            seed=self.seed,
        )
        if len(expected_subset) != subset_rows:
            raise RuntimeError(
                f"Expected exactly {subset_rows:,} validation rows, got "
                f"{len(expected_subset):,}"
            )
        manifest_path = self.output / "validation_subset_manifest.json"
        existing_manifest = _read_json(manifest_path)
        reuse_subset = bool(
            self.subset_path.is_file()
            and existing_manifest
            and existing_manifest.get("source_split") == "validation"
            and int(existing_manifest.get("row_count", -1)) == subset_rows
            and int(existing_manifest.get("seed", -1)) == self.seed
            and existing_manifest.get("sha256") == file_sha256(self.subset_path)
        )
        if reuse_subset:
            subset = pd.read_csv(self.subset_path, low_memory=False)
            indices = existing_manifest.get("source_row_indices", [])
            reuse_subset = bool(
                len(subset) == subset_rows
                and len(indices) == subset_rows
                and indices == expected_manifest["source_row_indices"]
                and dataframe_sha256(subset) == dataframe_sha256(expected_subset)
            )
        if reuse_subset:
            manifest = existing_manifest
            manifest["reused_existing_valid_subset"] = True
        else:
            subset, manifest = expected_subset, expected_manifest
            subset.to_csv(self.subset_path, index=False)
            manifest.update(
                {
                    "source_split": "validation",
                    "selection_method": self.config["validation_subset"][
                        "selection_method"
                    ],
                    "seed": self.seed,
                    "sha256": file_sha256(self.subset_path),
                    "reused_existing_valid_subset": False,
                }
            )
        write_json(self.output / "validation_subset_manifest.json", manifest)
        metadata_audit = audit_phase1_metadata(
            self.base_output, self.phase1_output, self.output
        )
        throughput = float(
            self.config["compute_budget"]["measured_rows_per_second"]
        )
        validation_generation = subset_rows * len(POLICY_NAMES) / throughput
        validation_total = validation_generation + 60 * float(
            self.config["compute_budget"]["validation_evaluation_minutes"]
        )
        confirmation_total = (
            int(self.config["confirmation"]["expected_rows"]) / throughput
            + 60
            * float(
                self.config["compute_budget"]["confirmation_evaluation_minutes"]
            )
        )
        projected_total_hours = (validation_total + confirmation_total) / 3600
        budget = {
            "validation_generation_seconds": validation_generation,
            "validation_total_projected_hours": validation_total / 3600,
            "confirmation_incremental_projected_hours": confirmation_total / 3600,
            "total_with_confirmation_projected_hours": projected_total_hours,
            "confirmation_within_hard_budget": projected_total_hours
            <= float(self.config["compute_budget"]["confirmation_skip_hours"]),
        }
        write_json(self.output / "compute_budget.json", budget)
        report = {
            "status": "passed",
            "training_performed": False,
            "model_source": model_source,
            "adapter": str(adapter),
            "adapter_sha256": directory_fingerprint(adapter),
            "validation_subset": manifest,
            "canonical_evaluator": minilm,
            "policies": self.config["generation"]["policies"],
            "random_seed_strategy": "torch.manual_seed(42) and torch.cuda.manual_seed_all(42) are reset immediately before every policy; row order and batching are identical",
            "phase1_metadata_audit": metadata_audit["verdict"],
            "compute_budget": budget,
        }
        write_json(self.output / "preflight.json", report)
        return report

    def _ensure_preflight(self) -> dict[str, Any]:
        path = self.output / "preflight.json"
        return json.loads(path.read_text()) if path.is_file() else self.preflight()

    def generate(
        self, device: str = "cuda", *, skip_existing: bool = True
    ) -> dict[str, Any]:
        self._ensure_preflight()
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        subset = pd.read_csv(self.subset_path, low_memory=False)
        adapter = self.base_output / "training/best_adapter"
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter, local_files_only=True, torch_dtype=dtype
        ).to(device).eval()
        records = {}
        started = time.perf_counter()
        for name in POLICY_NAMES:
            destination = self._policy_path(name)
            metrics_path = destination.parent / "generation_metrics.json"
            if skip_existing and destination.is_file() and metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text())
                policy = self.config["generation"]["policies"][name]
                reusable = bool(
                    metrics.get("output_sha256") == file_sha256(destination)
                    and metrics.get("policy") == name
                    and int(metrics.get("rows", -1)) == len(subset)
                    and int(metrics.get("random_seed", -1)) == self.seed
                    and float(metrics.get("temperature", -1))
                    == float(policy["temperature"])
                    and float(metrics.get("top_p", -1)) == float(policy["top_p"])
                    and float(metrics.get("repetition_penalty", -1))
                    == float(policy["repetition_penalty"])
                )
                if not reusable:
                    raise RuntimeError(f"Refusing stale generation metadata for {name}")
                records[name] = metrics
                print(f"[decoding-sweep] reuse {name}", flush=True)
                continue
            records[name] = self._generate_policy(
                model,
                tokenizer,
                subset,
                name,
                self.config["generation"]["policies"][name],
                destination,
                int(self.config["generation"]["batch_size"]),
                device,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = {
            "training_performed": False,
            "wall_clock_seconds": time.perf_counter() - started,
            "policies": records,
        }
        write_json(self.output / "generation_summary.json", result)
        return result

    def _evaluation_context(
        self, device: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, EmbeddingStore, TextC2STProtocol, str]:
        train = pd.read_csv(self.base.benchmark / "train_real.csv", low_memory=False)
        real = pd.read_csv(self.subset_path, low_memory=False)
        evaluation = self.config["evaluation"]
        source = exact_minilm_snapshot(
            evaluation["embedding_model"], evaluation["embedding_revision"]
        )
        model_path = source["local_snapshot"]
        protocol = TextC2STProtocol(
            name="canonical_paper_text_c2st_v1",
            embedding_backend="minilm",
            embedding_model=model_path,
            preprocessing="canonical",
            classifiers=("logistic_regression",),
            max_rows=int(evaluation["max_rows_per_class"]),
            seed=int(evaluation["seed"]),
            n_splits=int(evaluation["folds"]),
        )
        store = EmbeddingStore(self.output / "embedding_cache", device=device)
        return train, real, store, protocol, model_path

    def evaluate(self, device: str = "cuda") -> list[dict[str, Any]]:
        self._ensure_preflight()
        train, real, store, protocol, model_path = self._evaluation_context(device)
        frames = {
            name: pd.read_csv(self._policy_path(name), low_memory=False)
            for name in POLICY_NAMES
        }
        for name, frame in frames.items():
            audit = alignment_audit(real, frame)
            if not audit["aligned"]:
                raise RuntimeError(f"{name} is not aligned to validation subset")
        c2st = {
            name: evaluate_protocol(
                real,
                frame,
                protocol,
                store,
                fields=TEXT_FIELDS,
                label=f"decoding_sweep_{name}",
            )
            for name, frame in frames.items()
        }
        consistency = evaluate_text_consistency(
            train, real, frames, store, model_path, self.seed
        )
        rows = []
        for name, frame in frames.items():
            out = self.output / name
            diversity = detailed_diversity_metrics(real, frame)
            memorization = matched_memorization_metrics(
                train,
                real,
                frame,
                training_rows=int(
                    self.config["memorization"][
                        "nearest_neighbor_training_rows"
                    ]
                ),
                max_features=int(self.config["memorization"]["max_features"]),
            )
            conditioning = {
                "probe_training_source": "one fixed real chronological training split probe shared by REAL, D0, D1, and D2",
                "rating": {
                    "real_validation": consistency["rating"]["real_validation"],
                    "synthetic": consistency["rating"][name],
                },
                "verified": {
                    "real_validation": consistency["verified"]["real_validation"],
                    "synthetic": consistency["verified"][name],
                },
            }
            write_json(out / "canonical_text_c2st.json", c2st[name])
            write_json(out / "diversity_metrics.json", diversity)
            write_json(out / "memorization_metrics.json", memorization)
            write_json(out / "conditioning_metrics.json", conditioning)
            summary_c2st, review_c2st, macro_c2st = nested_c2st(c2st[name])
            policy = self.config["generation"]["policies"][name]
            generation = json.loads((out / "generation_metrics.json").read_text())
            review_diversity = diversity["review_text"]["synthetic"]
            summary_diversity = diversity["summary"]["synthetic"]
            rating = conditioning["rating"]["synthetic"]
            rows.append(
                {
                    "configuration": name,
                    "temperature": float(policy["temperature"]),
                    "top_p": float(policy["top_p"]),
                    "repetition_penalty": float(policy["repetition_penalty"]),
                    "summary_c2st": summary_c2st,
                    "review_c2st": review_c2st,
                    "macro_c2st": macro_c2st,
                    "summary_distinct_1": summary_diversity["distinct_1"],
                    "summary_distinct_2": summary_diversity["distinct_2"],
                    "summary_exact_duplicate_rate": summary_diversity[
                        "exact_duplicate_rate"
                    ],
                    "review_distinct_1": review_diversity["distinct_1"],
                    "review_distinct_2": review_diversity["distinct_2"],
                    "review_exact_duplicate_rate": review_diversity[
                        "exact_duplicate_rate"
                    ],
                    "review_repeated_ngram_rate": review_diversity[
                        "repeated_ngram_rate"
                    ],
                    "review_length_ks": diversity["review_text"]["length_ks"],
                    "summary_exact_train_overlap": memorization["summary"][
                        "qwen"
                    ]["exact_train_overlap_rate"],
                    "review_exact_train_overlap": memorization["review_text"][
                        "qwen"
                    ]["exact_train_overlap_rate"],
                    "rating_accuracy": rating["accuracy"],
                    "rating_balanced_accuracy": rating["balanced_accuracy"],
                    "rating_macro_f1": rating["macro_f1"],
                    "rating_ordinal_mae": rating["mae"],
                    "generated_tokens": generation["generated_tokens"],
                    "rows_per_second": generation["rows_per_second"],
                    "tokens_per_second": generation["tokens_per_second"],
                    "peak_vram_mb": generation["peak_vram_mb"],
                    "parse_failure_rate": generation["parse"]["parse_failure"],
                    "empty_summary_rate": generation["parse"]["empty_summary"],
                    "empty_review_rate": generation["parse"]["empty_review"],
                }
            )
        pd.DataFrame(rows).to_csv(self.output / "comparison.csv", index=False)
        write_json(self.output / "fixed_probe_metrics.json", consistency)
        write_json(
            self.output / "real_validation_diversity_metrics.json",
            {
                field: _text_side_metrics(real[field])
                for field in TEXT_FIELDS
            },
        )
        decision = select_decoding_configuration(rows, self.config["selection"])
        budget = json.loads((self.output / "compute_budget.json").read_text())
        decision["test_confirmation_run"] = False
        decision["test_confirmation_required"] = bool(
            decision["clearly_preferable_to_D0"]
            and budget["confirmation_within_hard_budget"]
        )
        decision["test_confirmation_skip_reason"] = (
            None
            if decision["test_confirmation_required"]
            else (
                "D0 retained or validation differences were not clearly preferable"
                if not decision["clearly_preferable_to_D0"]
                else "projected total runtime exceeded 2.5 hours"
            )
        )
        write_json(self.output / "decoding_decision.json", decision)
        return rows

    def confirm(self, device: str = "cuda", *, skip_existing: bool = True) -> dict[str, Any]:
        decision = json.loads((self.output / "decoding_decision.json").read_text())
        out = self.output / "test_confirmation"
        out.mkdir(parents=True, exist_ok=True)
        if not decision["test_confirmation_required"]:
            result = {
                "run": False,
                "reason": decision["test_confirmation_skip_reason"],
                "no_test_tuning": True,
            }
            write_json(out / "confirmation_decision.json", result)
            return result
        name = decision["selected_configuration"]
        conditions_path = Path(
            self.config["confirmation"]["generated_structured_conditions"]
        )
        if not conditions_path.is_file():
            raise FileNotFoundError(
                "Selected validation policy requires one test confirmation, but the "
                f"existing generated-structured conditions are missing: {conditions_path}"
            )
        real = pd.read_csv(self.base.benchmark / "test_real.csv", low_memory=False)
        conditions = pd.read_csv(conditions_path, low_memory=False)
        if len(real) != int(self.config["confirmation"]["expected_rows"]):
            raise RuntimeError("Held-out confirmation population is not exactly 3,982 rows")
        audit = alignment_audit(real, conditions)
        if not audit["aligned"]:
            raise RuntimeError("Generated-structured conditions do not match held-out test")
        destination = out / "synthetic_text.csv"
        metrics_path = out / "generation_metrics.json"
        reusable = False
        if skip_existing and destination.is_file() and metrics_path.is_file():
            existing_metrics = json.loads(metrics_path.read_text())
            reusable = bool(
                existing_metrics.get("policy") == name
                and existing_metrics.get("output_sha256") == file_sha256(destination)
                and int(existing_metrics.get("random_seed", -1)) == self.seed
            )
        if not reusable:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer

            adapter = self.base_output / "training/best_adapter"
            tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            )
            model = AutoPeftModelForCausalLM.from_pretrained(
                adapter, local_files_only=True, torch_dtype=dtype
            ).to(device).eval()
            self._generate_policy(
                model,
                tokenizer,
                conditions,
                name,
                self.config["generation"]["policies"][name],
                destination,
                int(self.config["generation"]["batch_size"]),
                device,
            )
            generated_metrics = json.loads(
                (self.output / "generation_metrics" / f"{name}.json").read_text()
            )
            write_json(metrics_path, generated_metrics)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        synthetic = pd.read_csv(destination, low_memory=False)
        evaluation = self.config["evaluation"]
        source = exact_minilm_snapshot(
            evaluation["embedding_model"], evaluation["embedding_revision"]
        )
        protocol = TextC2STProtocol(
            name="canonical_paper_text_c2st_v1_frozen_confirmation",
            embedding_backend="minilm",
            embedding_model=source["local_snapshot"],
            preprocessing="canonical",
            classifiers=("logistic_regression",),
            max_rows=len(real),
            seed=int(evaluation["seed"]),
            n_splits=int(evaluation["folds"]),
        )
        store = EmbeddingStore(out / "embedding_cache", device=device)
        c2st = evaluate_protocol(
            real,
            synthetic,
            protocol,
            store,
            fields=TEXT_FIELDS,
            label="decoding_sweep_test_confirmation",
        )
        write_json(out / "canonical_text_c2st.json", c2st)
        summary, review, macro = nested_c2st(c2st)
        baseline = self.config["confirmation"]["baseline"]
        result = {
            "run": True,
            "frozen_validation_configuration": name,
            "no_test_tuning": True,
            "summary_c2st": summary,
            "review_c2st": review,
            "macro_c2st": macro,
            "fixed_generated_structured_baseline": baseline,
            "macro_improvement_vs_fixed_baseline": float(
                baseline["macro_c2st"] - macro
            ),
        }
        write_json(out / "confirmation_decision.json", result)
        decision["test_confirmation_run"] = True
        decision["test_confirmation"] = result
        write_json(self.output / "decoding_decision.json", decision)
        return result

    def report(self) -> dict[str, Any]:
        rows = pd.read_csv(self.output / "comparison.csv").to_dict("records")
        decision = json.loads((self.output / "decoding_decision.json").read_text())
        real_diversity = json.loads(
            (self.output / "real_validation_diversity_metrics.json").read_text()
        )
        first_mem = json.loads(
            (self.output / POLICY_NAMES[0] / "memorization_metrics.json").read_text()
        )
        fixed_probe = json.loads(
            (self.output / "fixed_probe_metrics.json").read_text()
        )
        confirmation = _read_json(
            self.output / "test_confirmation/confirmation_decision.json"
        )
        selected = decision["selected_configuration"]
        selected_row = next(row for row in rows if row["configuration"] == selected)
        if decision["decoding_bottleneck"] in {"REJECTED", "WEAKLY SUPPORTED"}:
            next_experiment = "Run the Qwen3-0.6B versus Qwen3-1.7B capacity probe."
        elif selected_row["rating_macro_f1"] < rows[0]["rating_macro_f1"] - 0.03:
            next_experiment = "Investigate the training and conditioning objective."
        else:
            next_experiment = "Freeze this decoding policy and test temporal-relational conditioning."
        table = [
            "| Configuration | Summary C2ST | Review C2ST | Macro C2ST | Review distinct-2 | Review repetition | Rating macro-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        table.extend(
            "| {configuration} | {summary_c2st:.6f} | {review_c2st:.6f} | {macro_c2st:.6f} | {review_distinct_2:.6f} | {review_repeated_ngram_rate:.6f} | {rating_macro_f1:.6f} |".format(
                **row
            )
            for row in rows
        )
        report = f"""# Qwen3-0.6B Decoding Sweep

## 1. Executive Result

Decoding bottleneck: **{decision['decoding_bottleneck']}**. Selected validation configuration: **{selected}**.

## 2. Experimental Setup

The frozen Qwen3-0.6B adapter generated summary then review for one deterministic 2,000-row validation subset. True rating and verified were the only structured attributes supplied. D0/D1/D2 used seed 42, identical row order, batching, EOS handling, and data-derived token bounds. No model was trained.

## 3. Canonical Text C2ST

{chr(10).join(table)}

Lower is better. Selection used validation only.

## 4. Diversity and Mode Concentration

Real review distinct-2: `{real_diversity['review_text']['distinct_2']:.6f}`; real review repeated bigram rate: `{real_diversity['review_text']['repeated_ngram_rate']:.6f}`. Full summary/review length, KS, duplication, vocabulary, and n-gram diagnostics are stored per configuration.

## 5. Train-Likeness

Real validation exact train overlap: summary `{first_mem['summary']['real_heldout']['exact_train_overlap_rate']:.6f}`, review `{first_mem['review_text']['real_heldout']['exact_train_overlap_rate']:.6f}`. Per-configuration exact and nearest-neighbor TF-IDF results are in `memorization_metrics.json`. These are diagnostics, not privacy guarantees.

## 6. Structured-Text Consistency

One MiniLM(review)-to-rating probe was trained once on the real chronological training split and reused for real validation and all policies. Real validation macro-F1: `{fixed_probe['rating']['real_validation']['macro_f1']:.6f}`.

## 7. Efficiency

Per-policy generated-token, throughput, runtime, parse, empty-output, and peak-VRAM measurements are in `comparison.csv` and each `generation_metrics.json`.

## 8. Interpretation

**{decision['decoding_bottleneck']}**. The decision jointly applies C2ST improvement and validity, conditioning, repetition, duplicate, and length guardrails; it does not claim statistical significance.

## 9. Selected Validation Configuration

**{selected}**. Exact rule details are in `decoding_decision.json`.

## 10. Held-Out Confirmation

{('Run once with the frozen validation winner.' if confirmation and confirmation.get('run') else 'Not run: ' + str((confirmation or {}).get('reason', decision.get('test_confirmation_skip_reason'))))}

## 11. Next Experiment

{next_experiment}
"""
        (self.output / "decoding_report.md").write_text(report, encoding="utf-8")
        self._print_console(rows, real_diversity, first_mem, fixed_probe, decision)
        return decision

    def _print_console(
        self,
        rows: list[dict[str, Any]],
        real: dict[str, Any],
        memorization: dict[str, Any],
        probe: dict[str, Any],
        decision: dict[str, Any],
    ) -> None:
        print("=" * 60)
        print("QWEN3-0.6B DECODING SWEEP")
        print("=" * 60)
        print("VALIDATION POPULATION: 2,000 deterministic rows; seed 42")
        print("\nTEXT C2ST - LOWER IS BETTER")
        print(f"{'':24s} {'SUMMARY':>9s} {'REVIEW':>9s} {'MACRO':>9s}")
        for row in rows:
            print(
                f"{row['configuration']:24s} {row['summary_c2st']:9.6f} "
                f"{row['review_c2st']:9.6f} {row['macro_c2st']:9.6f}"
            )
        print("\nDIVERSITY")
        print(
            f"REAL review distinct-2={real['review_text']['distinct_2']:.6f}; "
            f"repeated-ngram={real['review_text']['repeated_ngram_rate']:.6f}"
        )
        for row in rows:
            print(
                f"{row['configuration']}: distinct-2={row['review_distinct_2']:.6f}; "
                f"repeated-ngram={row['review_repeated_ngram_rate']:.6f}"
            )
        print("\nTRAIN OVERLAP")
        print(
            "REAL HELDOUT: summary="
            f"{memorization['summary']['real_heldout']['exact_train_overlap_rate']:.6f}; "
            "review="
            f"{memorization['review_text']['real_heldout']['exact_train_overlap_rate']:.6f}"
        )
        for row in rows:
            print(
                f"{row['configuration']}: summary={row['summary_exact_train_overlap']:.6f}; "
                f"review={row['review_exact_train_overlap']:.6f}"
            )
        print("\nRATING CONSISTENCY")
        print(f"REAL macro-F1={probe['rating']['real_validation']['macro_f1']:.6f}")
        for row in rows:
            print(f"{row['configuration']}: macro-F1={row['rating_macro_f1']:.6f}")
        print(f"\nDECODING BOTTLENECK: {decision['decoding_bottleneck']}")
        print(
            "SELECTED VALIDATION CONFIGURATION: "
            f"{decision['selected_configuration']}"
        )
        print(
            "TEST CONFIRMATION RUN: "
            f"{'YES' if decision.get('test_confirmation_run') else 'NO'}"
        )
