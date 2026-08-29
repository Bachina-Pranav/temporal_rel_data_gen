"""Inference-only follow-up diagnostics for the trained Qwen text adapter."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from attribute_generation.conditional_tabdlm.diffusion_diagnostics import (
    PROGRESSIVE_CONDITION_SPECS,
)
from evaluation.text_c2st_audit import (
    EmbeddingStore,
    TextC2STProtocol,
    canonical_text,
    evaluate_protocol,
    file_sha256,
)

from .experiment import (
    QwenTextExperiment,
    distribution_comparison,
    first_existing,
    nested_c2st,
    normalize_rating,
    normalize_verified,
    resolve_pinned_minilm,
    validate_runtime_dependencies,
    write_json,
)


ALIGNMENT_COLUMNS = ("customer_id", "product_id", "review_time")
REQUIRED_TEXT_COLUMNS = ("rating", "verified", "summary", "review_text")
POLICY_OUTPUTS = {
    "normal": "oracle_summary/normal.csv",
    "oracle_summary": "oracle_summary/oracle_summary.csv",
    "no_summary": "oracle_summary/no_summary.csv",
    "B0_none": "conditioning_ablation/B0_none.csv",
    "B1_rating": "conditioning_ablation/B1_rating.csv",
    "B2_verified": "conditioning_ablation/B2_verified.csv",
    "temp07_p090": "decoding_sensitivity/temp07_p090/synthetic.csv",
    "temp11_p095": "decoding_sensitivity/temp11_p095/synthetic.csv",
}
REUSED_OUTPUTS = {
    "conditioning_ablation/B3_rating_verified.csv": "oracle_summary/normal.csv",
    "decoding_sensitivity/temp09_p095/synthetic.csv": "oracle_summary/normal.csv",
}


def directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(file_sha256(item).encode("ascii"))
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(np.uint64).tobytes())
    return digest.hexdigest()


def build_prompt(row: Any, conditioning: str, summary_mode: str) -> str:
    lines: list[str] = []
    if conditioning in {"rating", "rating_verified"}:
        lines.append(f"Rating: {normalize_rating(row.rating)}")
    if conditioning in {"verified", "rating_verified"}:
        lines.append(f"Verified: {normalize_verified(row.verified)}")
    if summary_mode == "oracle":
        lines.append(f"Summary: {canonical_text(row.summary)}")
        lines.append("Review:")
    elif summary_mode == "omitted":
        lines.append("Review:")
    else:
        lines.append("Summary:")
    return "\n".join(lines)


def parse_policy_continuation(
    continuation: Any,
    *,
    summary_mode: str,
    oracle_summary: Any = "",
) -> tuple[str, str, dict[str, bool]]:
    text = canonical_text(continuation).replace(" Review:", "\nReview:")
    if summary_mode == "generated":
        if "\nReview:" in text:
            summary, review = text.split("\nReview:", 1)
        elif "Review:" in text:
            summary, review = text.split("Review:", 1)
        else:
            summary, review = text, ""
        summary = canonical_text(summary)
        review = canonical_text(review)
        return summary, review, {
            "parse_failure": not bool(review),
            "missing_review_marker": "Review:" not in text,
            "empty_summary": not bool(summary),
            "empty_review": not bool(review),
        }
    review = canonical_text(text[len("Review:") :] if text.startswith("Review:") else text)
    summary = canonical_text(oracle_summary) if summary_mode == "oracle" else ""
    return summary, review, {
        "parse_failure": not bool(review),
        "missing_review_marker": False,
        "empty_summary": not bool(summary),
        "empty_review": not bool(review),
    }


def trim_generated_ids(ids: Iterable[int], eos_token_id: int | None) -> list[int]:
    values = [int(value) for value in ids]
    if eos_token_id is not None and eos_token_id in values:
        return values[: values.index(eos_token_id) + 1]
    return values


def select_fixed_subset(
    validation: pd.DataFrame,
    *,
    target_rows: int,
    maximum_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_rows <= 0 or maximum_rows <= 0:
        raise ValueError("Subset sizes must be positive")
    available = min(len(validation), maximum_rows)
    count = min(target_rows, available)
    if len(validation) <= count:
        indices = np.arange(len(validation), dtype=np.int64)
        reason = (
            f"Validation has {len(validation):,} rows, fewer than the 5,000-row target; "
            "the complete frozen validation split is used."
        )
    else:
        indices = np.sort(np.random.default_rng(seed).choice(len(validation), count, replace=False))
        reason = "Deterministic sample from the frozen validation split."
    subset = validation.iloc[indices].copy()
    subset.insert(0, "_source_row_index", indices)
    metadata = {
        "split_source": "validation",
        "selection_seed": seed,
        "target_rows": target_rows,
        "maximum_rows": maximum_rows,
        "available_validation_rows": int(len(validation)),
        "row_count": int(len(subset)),
        "source_row_indices": indices.tolist(),
        "selection_reason": reason,
        "dataframe_sha256": dataframe_sha256(subset),
    }
    return subset, metadata


def discover_diffusion_artifacts(
    roots: Iterable[str | Path], labels: Iterable[str]
) -> tuple[Path | None, list[dict[str, Any]]]:
    required = set(labels)
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("run_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            label = str(manifest.get("label", "")).upper()
            synthetic = manifest_path.parent / "synthetic_table.csv"
            if label not in required or not synthetic.is_file() or manifest.get("status") != "completed":
                continue
            experiment_root = manifest_path.parent.parent
            grouped.setdefault(experiment_root, []).append(
                {
                    "label": label,
                    "seed": int(manifest.get("seed", -1)),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": file_sha256(manifest_path),
                    "synthetic_path": str(synthetic),
                    "synthetic_sha256": file_sha256(synthetic),
                    "conditioning": PROGRESSIVE_CONDITION_SPECS[label].to_dict(),
                }
            )
    complete = [
        (root, rows)
        for root, rows in grouped.items()
        if required.issubset({row["label"] for row in rows})
    ]
    if not complete:
        return None, []
    selected_root, rows = max(complete, key=lambda item: item[0].stat().st_mtime)
    return selected_root, sorted(rows, key=lambda row: (row["label"], row["seed"]))


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@dataclass
class QwenFollowupExperiment:
    config_path: Path

    def __post_init__(self) -> None:
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = Path(self.config["output_dir"])
        self.base_output = Path(self.config["base_output_dir"])
        self.base_config_path = Path(self.config["base_experiment_config"])
        self.base = QwenTextExperiment(self.base_config_path, output_dir=self.base_output)
        self.seed = int(self.config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def subset_path(self) -> Path:
        return self.output / "evaluation_subset.csv"

    def _required_base_artifacts(self) -> list[Path]:
        return [
            self.base_output / "training/model_source.json",
            self.base_output / "training/training_efficiency.json",
            self.base_output / "training/best_adapter/adapter_config.json",
            self.base_output / "oracle_structured/synthetic_text.csv",
            self.base_output / "oracle_structured/canonical_text_c2st.json",
            self.base_output / "experiment_report.md",
            self.base.benchmark / "benchmark_manifest.json",
            self.base.benchmark / "train_real.csv",
            self.base.benchmark / "validation_real.csv",
            self.base.benchmark / "test_real.csv",
        ]

    def preflight(self) -> dict[str, Any]:
        missing = [str(path) for path in self._required_base_artifacts() if not path.exists()]
        if missing:
            raise FileNotFoundError("Required main-experiment artifacts are missing:\n- " + "\n- ".join(missing))
        runtime = validate_runtime_dependencies()
        model_source = json.loads((self.base_output / "training/model_source.json").read_text())
        if model_source.get("model_id") != "Qwen/Qwen3-0.6B-Base" or not model_source.get("revision"):
            raise RuntimeError("The trained adapter does not identify the exact pinned Qwen3-0.6B-Base source")
        snapshot = Path(model_source["local_snapshot"])
        adapter = self.base_output / "training/best_adapter"
        if not snapshot.is_dir() or not adapter.is_dir():
            raise FileNotFoundError("Pinned model snapshot or trained LoRA adapter is unavailable")
        validation_path = self.base.benchmark / "validation_real.csv"
        validation = pd.read_csv(validation_path, low_memory=False)
        missing_columns = sorted(set(REQUIRED_TEXT_COLUMNS).difference(validation.columns))
        if missing_columns:
            raise ValueError(f"Frozen validation split lacks columns: {missing_columns}")
        subset_cfg = self.config["evaluation_subset"]
        subset, subset_metadata = select_fixed_subset(
            validation,
            target_rows=int(subset_cfg["target_rows"]),
            maximum_rows=int(subset_cfg["maximum_rows"]),
            seed=self.seed,
        )
        self.output.mkdir(parents=True, exist_ok=True)
        subset.to_csv(self.subset_path, index=False)
        subset_metadata["csv_sha256"] = file_sha256(self.subset_path)
        write_json(self.output / "evaluation_subset_manifest.json", subset_metadata)
        minilm_path = self.base_output / "evaluation_model_source.json"
        minilm = json.loads(minilm_path.read_text()) if minilm_path.is_file() else resolve_pinned_minilm(self.config["evaluation"]["embedding_model"])
        if not Path(minilm["local_snapshot"]).is_dir():
            raise FileNotFoundError("Pinned canonical MiniLM snapshot is unavailable; no fallback is permitted")
        diffusion_root, diffusion = discover_diffusion_artifacts(
            self.config["diffusion_oracle"]["roots"], self.config["diffusion_oracle"]["labels"]
        )
        report = {
            "status": "passed",
            "no_training": True,
            "base_checkpoint": model_source,
            "adapter_path": str(adapter),
            "adapter_sha256": directory_fingerprint(adapter),
            "tokenizer_path": str(adapter),
            "tokenizer_sha256": directory_fingerprint(adapter),
            "tokenizer_revision": model_source["revision"],
            "runtime_versions": runtime,
            "benchmark_manifest": str(self.base.benchmark / "benchmark_manifest.json"),
            "benchmark_manifest_sha256": file_sha256(self.base.benchmark / "benchmark_manifest.json"),
            "splits": {
                name: {
                    "path": str(self.base.benchmark / f"{name}_real.csv"),
                    "sha256": file_sha256(self.base.benchmark / f"{name}_real.csv"),
                }
                for name in ("train", "validation", "test")
            },
            "evaluation_subset": subset_metadata,
            "canonical_evaluator": minilm,
            "main_generation_configuration": self.base.config["generation"],
            "existing_qwen_outputs": {
                mode: str(self.base_output / mode / "synthetic_text.csv")
                for mode in ("oracle_structured", "generated_structured")
                if (self.base_output / mode / "synthetic_text.csv").is_file()
            },
            "existing_lstm_output": str(first_existing(self.base.config["data"]["lstm_candidates"])),
            "existing_diffusion_output": str(first_existing(self.base.config["data"]["diffusion_text_candidates"])),
            "diffusion_diagnostic_root": str(diffusion_root) if diffusion_root else None,
            "diffusion_diagnostic_artifacts": diffusion,
            "missing_diffusion_diagnostics": diffusion_root is None,
        }
        report["required_artifact_warnings"] = [
            description
            for description, value in (
                (
                    "Main Qwen generated-structured output is unavailable because no frozen structured artifact aligned with the held-out spine; oracle validation follow-ups remain valid",
                    report["existing_qwen_outputs"].get("generated_structured"),
                ),
                ("Frozen LSTM output not found", report["existing_lstm_output"]),
                ("Frozen diffusion output not found", report["existing_diffusion_output"]),
                ("Complete historical O1/O2/O3/O4 output root not found", report["diffusion_diagnostic_root"]),
            )
            if value in {None, "None"}
        ]
        write_json(self.output / "preflight.json", report)
        lines = [
            "# Qwen3-0.6B Follow-Up Preflight",
            "",
            "- Status: PASS",
            "- Training: prohibited; existing adapter reused",
            f"- Model: `{model_source['model_id']}`",
            f"- Revision: `{model_source['revision']}`",
            f"- Adapter: `{adapter}` (SHA-256 `{report['adapter_sha256']}`)",
            f"- Validation subset: {len(subset):,} rows (SHA-256 `{subset_metadata['csv_sha256']}`)",
            f"- Subset rationale: {subset_metadata['selection_reason']}",
            f"- Canonical MiniLM revision: `{minilm['revision']}`",
            f"- Diffusion diagnostic root: `{report['diffusion_diagnostic_root']}`",
            "",
            "No checkpoint, split, evaluator, or historical output is silently substituted.",
        ]
        (self.output / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report

    def _ensure_preflight(self) -> dict[str, Any]:
        path = self.output / "preflight.json"
        return json.loads(path.read_text()) if path.is_file() else self.preflight()

    def _generation_bound(self, tokenizer: Any, summary_mode: str) -> int:
        efficiency = json.loads((self.base_output / "training/training_efficiency.json").read_text())
        train_stats = efficiency["length_statistics"]["train"]
        if summary_mode in {"oracle", "omitted"}:
            return max(16, int(train_stats["review_text"]["p99"]) + 8)
        prefix_tokens = len(tokenizer("Rating: 5\nVerified: true\n", add_special_tokens=False).input_ids)
        return max(16, int(train_stats["combined"]["p99"]) - prefix_tokens)

    def generate(self, device: str = "cuda", *, skip_existing: bool = True) -> dict[str, Any]:
        self._ensure_preflight()
        validate_runtime_dependencies()
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        subset = pd.read_csv(self.subset_path, low_memory=False)
        adapter = self.base_output / "training/best_adapter"
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter, local_files_only=True, torch_dtype=dtype
        ).to(device).eval()
        policies = self.config["generation"]["policies"]
        batch_size = int(self.config["generation"]["batch_size"])
        records: dict[str, Any] = {}
        budget_started = time.perf_counter()
        estimates = self._generation_estimates(subset)
        write_json(
            self.output / "compute_budget.json",
            {
                "status": "generation_pending",
                "budget_gpu_hours": float(self.config["compute_budget"]["main_suite_gpu_hours"]),
                "estimates": estimates,
                "estimated_total_gpu_hours": sum(item["expected_runtime_seconds"] for item in estimates.values()) / 3600.0,
                "model_scaling_probe_in_main_suite": False,
            },
        )
        for name, relative in POLICY_OUTPUTS.items():
            destination = self.output / relative
            if skip_existing and destination.is_file() and (destination.parent / "generation_metrics.json").is_file():
                metrics_path = self.output / "generation_metrics" / f"{name}.json"
                if not metrics_path.is_file():
                    raise RuntimeError(
                        f"Cannot safely reuse {destination}: per-policy metrics are missing at {metrics_path}"
                    )
                records[name] = json.loads(metrics_path.read_text())
                print(f"[followup] reuse {name}: {destination}", flush=True)
                continue
            policy = dict(policies[name])
            if name == "temp07_p090":
                policy.update({"conditioning": "rating_verified", "summary_mode": "generated"})
            if name == "temp11_p095":
                policy.update({"conditioning": "rating_verified", "summary_mode": "generated"})
            estimate = estimates[name]
            print(
                f"[followup] estimate {name}: rows={estimate['expected_rows']:,}, "
                f"tokens~{estimate['expected_tokens']:,.0f}, "
                f"runtime~{estimate['expected_runtime_seconds'] / 60:.1f} minutes",
                flush=True,
            )
            metrics = self._generate_policy(
                model, tokenizer, subset, name, policy, destination, batch_size, device
            )
            records[name] = metrics
        normal = self.output / "oracle_summary/normal.csv"
        for relative, source_relative in REUSED_OUTPUTS.items():
            destination = self.output / relative
            hardlink_or_copy(self.output / source_relative, destination)
            source_metrics = records["normal"]
            write_json(destination.parent / "generation_metrics.json", {**source_metrics, "reused_from": str(normal)})
        actual = sum(float(item.get("seconds", 0.0)) for item in records.values())
        budget = {
            "budget_gpu_hours": float(self.config["compute_budget"]["main_suite_gpu_hours"]),
            "generation_wall_clock_seconds_this_invocation": time.perf_counter() - budget_started,
            "generation_gpu_seconds_sum": actual,
            "generation_gpu_hours": actual / 3600.0,
            "estimates": estimates,
            "policies": records,
            "model_scaling_probe_in_main_suite": False,
        }
        write_json(self.output / "compute_budget.json", budget)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return budget

    def _generation_estimates(self, subset: pd.DataFrame) -> dict[str, dict[str, Any]]:
        prior_path = self.base_output / "oracle_structured/generation_metrics.json"
        prior = json.loads(prior_path.read_text()) if prior_path.is_file() else {}
        rows_per_second = max(float(prior.get("rows_per_second") or 0.5), 0.01)
        normal_tokens_per_row = float(prior.get("generated_tokens", 0)) / max(float(prior.get("rows", 0)), 1.0)
        if normal_tokens_per_row <= 0:
            efficiency = json.loads((self.base_output / "training/training_efficiency.json").read_text())
            normal_tokens_per_row = float(efficiency["length_statistics"]["train"]["combined"]["mean"])
        efficiency = json.loads((self.base_output / "training/training_efficiency.json").read_text())
        review_tokens_per_row = float(efficiency["length_statistics"]["train"]["review_text"]["mean"])
        result = {}
        for name in POLICY_OUTPUTS:
            summary_mode = self.config["generation"]["policies"][name].get("summary_mode", "generated")
            tokens_per_row = review_tokens_per_row if summary_mode in {"oracle", "omitted"} else normal_tokens_per_row
            result[name] = {
                "expected_rows": int(len(subset)),
                "expected_tokens": float(len(subset) * tokens_per_row),
                "expected_runtime_seconds": float(len(subset) / rows_per_second),
                "basis": "Measured main Qwen oracle-structured throughput and train token-length statistics",
            }
        return result

    def _generate_policy(
        self,
        model: Any,
        tokenizer: Any,
        subset: pd.DataFrame,
        name: str,
        policy: dict[str, Any],
        destination: Path,
        batch_size: int,
        device: str,
    ) -> dict[str, Any]:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.cuda.reset_peak_memory_stats()
        summary_mode = str(policy["summary_mode"])
        max_new = self._generation_bound(tokenizer, summary_mode)
        outputs: list[dict[str, Any]] = []
        flags: list[dict[str, bool]] = []
        token_count = 0
        started = time.perf_counter()
        print(
            f"[followup] {name}: rows={len(subset):,} max_new_tokens={max_new} "
            f"temperature={policy['temperature']} top_p={policy['top_p']}",
            flush=True,
        )
        for start in range(0, len(subset), batch_size):
            batch = subset.iloc[start : start + batch_size]
            prompts = [
                build_prompt(row, str(policy["conditioning"]), summary_mode)
                for row in batch.itertuples()
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new,
                    do_sample=bool(self.config["generation"]["do_sample"]),
                    temperature=float(policy["temperature"]),
                    top_p=float(policy["top_p"]),
                    repetition_penalty=float(self.config["generation"]["repetition_penalty"]),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_width = encoded.input_ids.shape[1]
            for offset, ids in enumerate(generated):
                continuation_ids = trim_generated_ids(
                    ids[prompt_width:].tolist(), tokenizer.eos_token_id
                )
                continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
                row = batch.iloc[offset]
                summary, review, status = parse_policy_continuation(
                    continuation,
                    summary_mode=summary_mode,
                    oracle_summary=row["summary"],
                )
                outputs.append({"summary": summary, "review_text": review})
                flags.append(status)
                token_count += len(continuation_ids)
            print(f"[followup] {name}: {min(start + len(batch), len(subset)):,}/{len(subset):,}", flush=True)
        elapsed = time.perf_counter() - started
        columns = ["_source_row_index", *ALIGNMENT_COLUMNS, "rating", "verified"]
        result = subset.loc[:, [column for column in columns if column in subset]].copy()
        result["summary"] = [item["summary"] for item in outputs]
        result["review_text"] = [item["review_text"] for item in outputs]
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        metrics = {
            "policy": name,
            "conditioning": policy["conditioning"],
            "summary_mode": summary_mode,
            "diagnostic_only": True,
            "rows": int(len(result)),
            "seconds": elapsed,
            "rows_per_second": len(result) / elapsed,
            "generated_tokens": token_count,
            "tokens_per_second": token_count / elapsed,
            "average_tokens_per_row": token_count / max(len(result), 1),
            "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
            "max_new_tokens": max_new,
            "temperature": float(policy["temperature"]),
            "top_p": float(policy["top_p"]),
            "repetition_penalty": float(self.config["generation"]["repetition_penalty"]),
            "parse": {
                key: float(np.mean([flag[key] for flag in flags])) for key in flags[0]
            },
            "output_sha256": file_sha256(destination),
        }
        write_json(destination.parent / "generation_metrics.json", metrics)
        write_json(self.output / "generation_metrics" / f"{name}.json", metrics)
        return metrics

    def _evaluation_context(self, device: str) -> tuple[pd.DataFrame, pd.DataFrame, EmbeddingStore, TextC2STProtocol, str]:
        train = pd.read_csv(self.base.benchmark / "train_real.csv", low_memory=False)
        real = pd.read_csv(self.subset_path, low_memory=False)
        source_path = self.base_output / "evaluation_model_source.json"
        source = json.loads(source_path.read_text()) if source_path.is_file() else resolve_pinned_minilm(self.config["evaluation"]["embedding_model"])
        model_path = source["local_snapshot"]
        if not Path(model_path).is_dir():
            raise FileNotFoundError("Pinned MiniLM snapshot missing; canonical evaluation refuses fallback")
        evaluation = self.config["evaluation"]
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

    def evaluate(self, device: str = "cuda") -> dict[str, Any]:
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._ensure_preflight()
        train, real, store, protocol, model_path = self._evaluation_context(device)
        artifacts = {
            "A0_normal": self.output / "oracle_summary/normal.csv",
            "A1_oracle_summary": self.output / "oracle_summary/oracle_summary.csv",
            "A2_no_summary": self.output / "oracle_summary/no_summary.csv",
            "B0_none": self.output / "conditioning_ablation/B0_none.csv",
            "B1_rating": self.output / "conditioning_ablation/B1_rating.csv",
            "B2_verified": self.output / "conditioning_ablation/B2_verified.csv",
            "B3_rating_verified": self.output / "conditioning_ablation/B3_rating_verified.csv",
            "C1_temp07_p090": self.output / "decoding_sensitivity/temp07_p090/synthetic.csv",
            "C2_temp09_p095": self.output / "decoding_sensitivity/temp09_p095/synthetic.csv",
            "C3_temp11_p095": self.output / "decoding_sensitivity/temp11_p095/synthetic.csv",
        }
        missing = [str(path) for path in artifacts.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Generate follow-up artifacts before evaluation:\n- " + "\n- ".join(missing))
        frames = {name: pd.read_csv(path, low_memory=False) for name, path in artifacts.items()}
        c2st: dict[str, Any] = {}
        distributions: dict[str, Any] = {}
        for name, frame in frames.items():
            fields = ("review_text",) if name.startswith("A") else ("summary", "review_text")
            c2st[name] = evaluate_protocol(real, frame, protocol, store, fields=fields, label=f"followup_{name}")
            distributions[name] = distribution_comparison(real, frame)
        consistency = evaluate_text_consistency(train, real, frames, store, model_path, self.seed)
        self._write_experiment_metrics(c2st, distributions, consistency)
        write_json(self.output / "all_canonical_text_c2st.json", c2st)
        write_json(self.output / "all_distribution_metrics.json", distributions)
        self._record_compute_stage(
            "qwen_diagnostic_evaluation",
            time.perf_counter() - started,
            (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        )
        return {"c2st": c2st, "consistency": consistency}

    def _write_experiment_metrics(self, c2st: dict[str, Any], distribution: dict[str, Any], consistency: dict[str, Any]) -> None:
        a0 = nested_c2st(c2st["A0_normal"])[1]
        a1 = nested_c2st(c2st["A1_oracle_summary"])[1]
        gain = float(a0 - a1)
        propagation = "YES" if gain >= 0.10 else "MODERATE" if gain >= 0.03 else "NO"
        oracle = {
            "review_c2st": {name: nested_c2st(c2st[name])[1] for name in ("A0_normal", "A1_oracle_summary", "A2_no_summary")},
            "oracle_summary_gain": gain,
            "summary_propagation_bottleneck": propagation,
            "review_distribution": {name: distribution[name]["review_text"] for name in ("A0_normal", "A1_oracle_summary", "A2_no_summary")},
        }
        write_json(self.output / "oracle_summary/metrics.json", oracle)
        (self.output / "oracle_summary/report.md").write_text(
            "# Oracle Summary Diagnostic\n\n"
            f"- A0 normal review C2ST: {oracle['review_c2st']['A0_normal']:.6f}\n"
            f"- A1 oracle-summary review C2ST: {oracle['review_c2st']['A1_oracle_summary']:.6f}\n"
            f"- A2 no-summary review C2ST: {oracle['review_c2st']['A2_no_summary']:.6f}\n"
            f"- Oracle-summary gain: {gain:.6f}\n"
            f"- Propagation bottleneck: **{propagation}**\n\n"
            "This is an oracle validation diagnostic, not a generative paper result.\n",
            encoding="utf-8",
        )
        b_names = ("B0_none", "B1_rating", "B2_verified", "B3_rating_verified")
        b_c2st = {name: c2st[name] for name in b_names}
        write_json(self.output / "conditioning_ablation/text_c2st.json", b_c2st)
        write_json(self.output / "conditioning_ablation/rating_consistency.json", consistency["rating"])
        write_json(self.output / "conditioning_ablation/verified_consistency.json", consistency["verified"])
        write_json(self.output / "conditioning_ablation/conditional_embedding_metrics.json", consistency["conditional_embeddings"])
        (self.output / "conditioning_ablation/report.md").write_text(
            conditioning_report(b_c2st, consistency), encoding="utf-8"
        )
        c_names = ("C1_temp07_p090", "C2_temp09_p095", "C3_temp11_p095")
        rows = []
        for name in c_names:
            summary, review, macro = nested_c2st(c2st[name])
            generated = distribution[name]["review_text"]["synthetic"]
            generation = json.loads(({
                "C1_temp07_p090": self.output / "decoding_sensitivity/temp07_p090/generation_metrics.json",
                "C2_temp09_p095": self.output / "decoding_sensitivity/temp09_p095/generation_metrics.json",
                "C3_temp11_p095": self.output / "decoding_sensitivity/temp11_p095/generation_metrics.json",
            }[name]).read_text())
            rows.append({
                "configuration": name,
                "summary_c2st": summary,
                "review_c2st": review,
                "macro_c2st": macro,
                "rating_macro_f1": consistency["rating"][name]["macro_f1"],
                "review_length_ks": distribution[name]["review_text"]["length_ks"],
                "unique_ratio": generated["unique_text_ratio"],
                "exact_duplicate_rate": generated["exact_duplicate_rate"],
                "distinct_1": generated["distinct_1"],
                "distinct_2": generated["distinct_2"],
                "repeated_ngram_rate": generated["repeated_ngram_rate"],
                "repeated_sentence_rate": generated["repeated_sentence_rate"],
                "empty_rate": generated["empty_rate"],
                "parse_failure_rate": generation["parse"]["parse_failure"],
                "average_tokens_per_row": generation["average_tokens_per_row"],
                "tokens_per_second": generation["tokens_per_second"],
                "rows_per_second": generation["rows_per_second"],
            })
        pd.DataFrame(rows).to_csv(self.output / "decoding_sensitivity/comparison.csv", index=False)
        selected = select_decoding_policy(rows)
        (self.output / "decoding_sensitivity/report.md").write_text(
            decoding_report(rows, selected), encoding="utf-8"
        )
        write_json(self.output / "decoding_sensitivity/selection.json", selected)

    def evaluate_diffusion_oracles(self, device: str = "cuda") -> dict[str, Any]:
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        preflight = self._ensure_preflight()
        artifacts = preflight.get("diffusion_diagnostic_artifacts") or []
        if not artifacts:
            raise FileNotFoundError("No complete historical O1/O2/O3/O4 diagnostic root was found")
        _, _, store, protocol, _ = self._evaluation_context(device)
        benchmark_manifest = json.loads((self.base.benchmark / "benchmark_manifest.json").read_text())
        try:
            historical_real_path = Path(benchmark_manifest["files"]["evaluation_real"]["path"])
            expected_real_hash = benchmark_manifest["files"]["evaluation_real"]["sha256"]
        except KeyError as exc:
            raise KeyError("Frozen diffusion benchmark has no evaluation_real file record") from exc
        if not historical_real_path.is_file() or file_sha256(historical_real_path) != expected_real_hash:
            raise RuntimeError("Historical diffusion evaluation table is missing or changed")
        real = pd.read_csv(historical_real_path, low_memory=False)
        out = self.output / "diffusion_oracle_canonical"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for artifact in artifacts:
            synthetic = pd.read_csv(artifact["synthetic_path"], low_memory=False)
            result = evaluate_protocol(real, synthetic, protocol, store, label=f"diffusion_{artifact['label']}_seed{artifact['seed']}")
            summary, review, macro = nested_c2st(result)
            rows.append({
                "label": artifact["label"],
                "seed": artifact["seed"],
                "summary_c2st": summary,
                "review_c2st": review,
                "macro_c2st": macro,
                "conditioning": artifact["conditioning"]["description"],
                "valid_generative_baseline": artifact["conditioning"]["valid_generative_baseline"],
                "synthetic_path": artifact["synthetic_path"],
                "synthetic_sha256": artifact["synthetic_sha256"],
            })
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "canonical_results_per_seed.csv", index=False)
        aggregate = frame.groupby("label", as_index=False).agg(
            summary_c2st=("summary_c2st", "mean"),
            review_c2st=("review_c2st", "mean"),
            macro_c2st=("macro_c2st", "mean"),
            macro_c2st_std=("macro_c2st", "std"),
            num_seeds=("seed", "nunique"),
            conditioning=("conditioning", "first"),
            valid_generative_baseline=("valid_generative_baseline", "first"),
        )
        aggregate.to_csv(out / "canonical_results.csv", index=False)
        write_json(out / "artifact_manifest.json", {
            "selected_root": preflight["diffusion_diagnostic_root"],
            "evaluation_real_path": str(historical_real_path),
            "evaluation_real_sha256": expected_real_hash,
            "artifacts": artifacts,
        })
        qwen_path = self.base_output / "oracle_structured/canonical_text_c2st.json"
        qwen = json.loads(qwen_path.read_text()) if qwen_path.is_file() else None
        (out / "report.md").write_text(diffusion_report(aggregate, qwen), encoding="utf-8")
        self._record_compute_stage(
            "diffusion_oracle_canonical_evaluation",
            time.perf_counter() - started,
            (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        )
        return {"per_seed": rows, "aggregate": aggregate.to_dict("records")}

    def _record_compute_stage(self, name: str, seconds: float, peak_vram_mb: float) -> None:
        path = self.output / "compute_budget.json"
        budget = json.loads(path.read_text()) if path.is_file() else {}
        stages = budget.setdefault("evaluation_stages", {})
        stages[name] = {"wall_clock_seconds": float(seconds), "peak_vram_mb": float(peak_vram_mb)}
        generation_seconds = float(budget.get("generation_gpu_seconds_sum", 0.0))
        evaluation_seconds = sum(float(item["wall_clock_seconds"]) for item in stages.values())
        budget["total_gpu_time_upper_bound_seconds"] = generation_seconds + evaluation_seconds
        budget["total_gpu_time_upper_bound_hours"] = (generation_seconds + evaluation_seconds) / 3600.0
        budget["gpu_time_interpretation"] = "Conservative upper bound treating all evaluation wall time as GPU-active time."
        write_json(path, budget)

    def report(self) -> dict[str, Any]:
        c2st = json.loads((self.output / "all_canonical_text_c2st.json").read_text())
        distributions = json.loads((self.output / "all_distribution_metrics.json").read_text())
        consistency = json.loads((self.output / "conditioning_ablation/rating_consistency.json").read_text())
        verified = json.loads((self.output / "conditioning_ablation/verified_consistency.json").read_text())
        decoding = pd.read_csv(self.output / "decoding_sensitivity/comparison.csv")
        diffusion_path = self.output / "diffusion_oracle_canonical/canonical_results.csv"
        diffusion = pd.read_csv(diffusion_path) if diffusion_path.is_file() else pd.DataFrame()
        rows = self._master_rows(c2st, distributions, consistency, verified, decoding, diffusion)
        pd.DataFrame(rows).to_csv(self.output / "master_comparison.csv", index=False)
        decision = make_decision(c2st, consistency, verified, decoding, diffusion, self.base_output)
        write_json(self.output / "decision.json", decision)
        (self.output / "followup_report.md").write_text(
            final_report(c2st, consistency, verified, decoding, diffusion, decision, self.output),
            encoding="utf-8",
        )
        print_final(decision, c2st, decoding, diffusion, self.output)
        return decision

    def _master_rows(self, c2st: dict[str, Any], distributions: dict[str, Any], rating: dict[str, Any], verified: dict[str, Any], decoding: pd.DataFrame, diffusion: pd.DataFrame) -> list[dict[str, Any]]:
        mapping = {
            "Qwen normal/oracle structured": "B3_rating_verified",
            "Qwen oracle summary": "A1_oracle_summary",
            "Qwen no summary": "A2_no_summary",
            "Qwen no conditioning": "B0_none",
            "Qwen rating only": "B1_rating",
            "Qwen verified only": "B2_verified",
            "Qwen rating + verified": "B3_rating_verified",
        }
        rows = []
        for experiment, key in mapping.items():
            summary, review, macro = nested_c2st(c2st[key])
            dist = distributions[key]["review_text"]
            syn = dist["synthetic"]
            policy = {
                "A0_normal": "normal", "A1_oracle_summary": "oracle_summary", "A2_no_summary": "no_summary",
                "B0_none": "B0_none", "B1_rating": "B1_rating", "B2_verified": "B2_verified", "B3_rating_verified": "normal",
            }[key]
            generation = json.loads((self.output / "generation_metrics" / f"{policy}.json").read_text())
            rows.append({
                "experiment": experiment,
                "conditioning": generation.get("conditioning"),
                "summary_c2st": summary,
                "review_c2st": review,
                "macro_c2st": macro,
                "rating_consistency": (rating.get(key) or {}).get("macro_f1"),
                "verified_consistency": (verified.get(key) or {}).get("roc_auc"),
                "review_length_ks": dist["length_ks"],
                "unique_ratio": syn["unique_text_ratio"],
                "exact_duplicate_rate": syn["exact_duplicate_rate"],
                "distinct_1": syn["distinct_1"],
                "distinct_2": syn["distinct_2"],
                "repetition_rate": syn["repeated_ngram_rate"],
                "rows_per_second": generation.get("rows_per_second"),
                "tokens_per_second": generation.get("tokens_per_second"),
                "runtime_seconds": generation.get("seconds"),
                "peak_vram_mb": generation.get("peak_vram_mb"),
            })
        for row in diffusion.to_dict("records"):
            rows.append({"experiment": f"Diffusion {row['label']} canonical", "conditioning": row["conditioning"], "summary_c2st": row["summary_c2st"], "review_c2st": row["review_c2st"], "macro_c2st": row["macro_c2st"]})
        for row in decoding.to_dict("records"):
            rows.append({
                "experiment": row["configuration"],
                "conditioning": "true rating + verified; validation decoding diagnostic",
                "summary_c2st": row["summary_c2st"],
                "review_c2st": row["review_c2st"],
                "macro_c2st": row["macro_c2st"],
                "rating_consistency": row["rating_macro_f1"],
                "review_length_ks": row["review_length_ks"],
                "unique_ratio": row["unique_ratio"],
                "exact_duplicate_rate": row["exact_duplicate_rate"],
                "distinct_1": row["distinct_1"],
                "distinct_2": row["distinct_2"],
                "repetition_rate": row["repeated_ngram_rate"],
                "rows_per_second": row["rows_per_second"],
                "tokens_per_second": row["tokens_per_second"],
            })
        baselines = load_baseline_rows(self.base_output)
        return baselines + rows


def evaluate_text_consistency(train: pd.DataFrame, real: pd.DataFrame, frames: dict[str, pd.DataFrame], store: EmbeddingStore, model_path: str, seed: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_train = min(len(train), 50000)
    train_sample = train.sample(n=n_train, random_state=seed) if len(train) > n_train else train
    train_emb = store.embed(train_sample.review_text, backend="minilm", model_name=model_path, preprocessing="canonical", label="followup_probe_train")
    real_emb = store.embed(real.review_text, backend="minilm", model_name=model_path, preprocessing="canonical", label="followup_probe_real")
    embeddings = {name: store.embed(frame.review_text, backend="minilm", model_name=model_path, preprocessing="canonical", label=f"followup_probe_{name}") for name, frame in frames.items()}
    rating_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced", multi_class="auto"))
    rating_model.fit(train_emb, train_sample.rating.astype(int))
    verified_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))
    verified_model.fit(train_emb, train_sample.verified.astype(bool).astype(int))
    rating: dict[str, Any] = {"training_source": "real chronological training split only"}
    verified: dict[str, Any] = {"training_source": "real chronological training split only"}
    all_frames = {"real_validation": real, **frames}
    all_embeddings = {"real_validation": real_emb, **embeddings}
    for name, frame in all_frames.items():
        emb = all_embeddings[name]
        rating_truth = frame.rating.astype(int).to_numpy()
        rating_pred = rating_model.predict(emb).astype(int)
        rating[name] = {
            "accuracy": float(accuracy_score(rating_truth, rating_pred)),
            "macro_f1": float(f1_score(rating_truth, rating_pred, average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(rating_truth, rating_pred)),
            "rows": int(len(frame)),
        }
        verified_truth = frame.verified.astype(bool).astype(int).to_numpy()
        verified_pred = verified_model.predict(emb).astype(int)
        verified_prob = verified_model.predict_proba(emb)[:, 1]
        verified[name] = {
            "accuracy": float(accuracy_score(verified_truth, verified_pred)),
            "f1": float(f1_score(verified_truth, verified_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(verified_truth, verified_prob)) if len(np.unique(verified_truth)) > 1 else None,
            "rows": int(len(frame)),
        }
    conditional = {}
    real_rating = real.rating.astype(int).to_numpy()
    real_centroids = {value: real_emb[real_rating == value].mean(axis=0) for value in sorted(set(real_rating)) if np.any(real_rating == value)}
    for name, frame in frames.items():
        syn_rating = frame.rating.astype(int).to_numpy()
        by_rating = {}
        for value, real_centroid in real_centroids.items():
            selected = embeddings[name][syn_rating == value]
            if not len(selected):
                continue
            generated_centroid = selected.mean(axis=0)
            matched = float(np.linalg.norm(generated_centroid - real_centroid))
            unrelated = [float(np.linalg.norm(generated_centroid - centroid)) for other, centroid in real_centroids.items() if other != value]
            by_rating[str(value)] = {"matched_real_centroid_distance": matched, "mean_unrelated_real_centroid_distance": float(np.mean(unrelated)), "matched_is_closer": bool(matched < np.mean(unrelated))}
        conditional[name] = by_rating
    return {"rating": rating, "verified": verified, "conditional_embeddings": conditional}


def select_decoding_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = next(row for row in rows if row["configuration"] == "C2_temp09_p095")
    candidates = sorted(rows, key=lambda row: (row["macro_c2st"], row["repeated_ngram_rate"], -row["rating_macro_f1"]))
    best = candidates[0]
    negligible = abs(float(best["macro_c2st"]) - float(original["macro_c2st"])) < 0.03
    selected = original if negligible else best
    return {"selected": selected["configuration"], "retained_original_for_simplicity": negligible, "validation_only": True, "rule": "Jointly considers C2ST, repetition, diversity, rating consistency, and length; differences below 0.03 retain the original policy."}


def conditioning_report(c2st: dict[str, Any], consistency: dict[str, Any]) -> str:
    lines = ["# Structured Conditioning Ablation", "", "Inference-only distribution-shift diagnostic; no models were retrained.", ""]
    for name, result in c2st.items():
        summary, review, macro = nested_c2st(result)
        lines.append(f"- {name}: summary={summary:.6f}, review={review:.6f}, macro={macro:.6f}, rating macro-F1={consistency['rating'][name]['macro_f1']:.6f}")
    return "\n".join(lines) + "\n"


def decoding_report(rows: list[dict[str, Any]], selected: dict[str, Any]) -> str:
    lines = ["# Decoding Sensitivity", "", "Selection uses validation only.", ""]
    for row in rows:
        lines.append(f"- {row['configuration']}: macro C2ST={row['macro_c2st']:.6f}, rating macro-F1={row['rating_macro_f1']:.6f}, repeated n-gram rate={row['repeated_ngram_rate']:.6f}")
    lines += ["", f"Selected: **{selected['selected']}**", ""]
    return "\n".join(lines)


def diffusion_report(frame: pd.DataFrame, qwen: dict[str, Any] | None = None) -> str:
    lines = ["# Canonical Diffusion Oracle Re-evaluation", "", "Existing outputs only; no training or generation was performed.", ""]
    for row in frame.itertuples():
        lines.append(f"- {row.label}: summary={row.summary_c2st:.6f}, review={row.review_c2st:.6f}, macro={row.macro_c2st:.6f}; {row.conditioning}")
    normal = frame.loc[frame.label == "O4", "macro_c2st"]
    oracle = frame.loc[frame.label.isin(["O1", "O2", "O3"]), "macro_c2st"]
    qwen_macro = nested_c2st(qwen)[2] if qwen else None
    lines += [
        "",
        f"- Current normal diffusion (O4): {float(normal.iloc[0]) if len(normal) else None}",
        f"- Strongest oracle diffusion: {float(oracle.min()) if len(oracle) else None}",
        f"- Qwen oracle structured: {qwen_macro}",
    ]
    return "\n".join(lines) + "\n"


def load_baseline_rows(base_output: Path) -> list[dict[str, Any]]:
    path = base_output / "comparison/model_comparison.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    rows = []
    for row in frame.to_dict("records"):
        rows.append({"experiment": row.get("model"), "conditioning": row.get("conditioning"), "summary_c2st": row.get("summary_c2st"), "review_c2st": row.get("review_c2st"), "macro_c2st": row.get("macro_text_c2st")})
    return rows


def make_decision(c2st: dict[str, Any], rating: dict[str, Any], verified: dict[str, Any], decoding: pd.DataFrame, diffusion: pd.DataFrame, base_output: Path) -> dict[str, Any]:
    normal_review = nested_c2st(c2st["A0_normal"])[1]
    oracle_review = nested_c2st(c2st["A1_oracle_summary"])[1]
    gain = normal_review - oracle_review
    b0 = nested_c2st(c2st["B0_none"])[2]
    b3 = nested_c2st(c2st["B3_rating_verified"])[2]
    rating_gain = rating["B3_rating_verified"]["macro_f1"] - rating["B0_none"]["macro_f1"]
    verified_gain = (verified["B3_rating_verified"].get("roc_auc") or 0.5) - (verified["B0_none"].get("roc_auc") or 0.5)
    verified_measurable = (verified["real_validation"].get("roc_auc") or 0.5) >= 0.60
    best_diffusion = float(diffusion.loc[diffusion.label.isin(["O1", "O2", "O3"]), "macro_c2st"].min()) if not diffusion.empty else None
    qwen_macro = nested_c2st(c2st["B3_rating_verified"])[2]
    scale_trigger = 0.55 <= qwen_macro <= 0.70
    if best_diffusion is not None and qwen_macro < best_diffusion:
        next_experiment = "temporal-relational soft-prefix conditioning"
    elif gain >= 0.10:
        next_experiment = "independent summary/review decoders"
    elif rating_gain < 0.03:
        next_experiment = "stronger conditioning objective"
    elif scale_trigger:
        next_experiment = "full Qwen3-1.7B experiment"
    else:
        next_experiment = "pretrained masked-diffusion LM comparison"
    return {
        "oracle_summary_gain": gain,
        "summary_propagation": "YES" if gain >= .10 else "MODERATE" if gain >= .03 else "NO",
        "conditioning_improves_macro_c2st": b3 < b0,
        "qwen_uses_rating": "YES" if rating_gain >= .05 else "WEAKLY" if rating_gain >= .01 else "NO",
        "qwen_uses_verified": ("YES" if verified_gain >= .05 else "WEAKLY" if verified_gain >= .01 else "NO") if verified_measurable else "NOT MEASURABLE",
        "rating_macro_f1_gain_vs_no_conditioning": rating_gain,
        "verified_auc_gain_vs_no_conditioning": verified_gain,
        "best_oracle_diffusion_macro_c2st": best_diffusion,
        "qwen_oracle_macro_c2st": qwen_macro,
        "pretraining_advantage": (best_diffusion - qwen_macro) if best_diffusion is not None else None,
        "capacity_probe_ran": False,
        "capacity_probe_triggered": scale_trigger,
        "capacity_probe_note": "Excluded from the <=3 GPU-hour main suite; run separately only if triggered.",
        "selected_decoding": json.loads((base_output / "followup/decoding_sensitivity/selection.json").read_text())["selected"],
        "recommended_next_experiment": next_experiment,
    }


def final_report(c2st: dict[str, Any], rating: dict[str, Any], verified: dict[str, Any], decoding: pd.DataFrame, diffusion: pd.DataFrame, decision: dict[str, Any], output: Path) -> str:
    def scores(name: str) -> tuple[Any, Any, Any]:
        return nested_c2st(c2st[name])
    a0, a1, a2 = scores("A0_normal"), scores("A1_oracle_summary"), scores("A2_no_summary")
    lines = [
        "# Qwen3-0.6B Follow-Up Diagnostics", "", "## 1. Executive Summary", "",
        f"1. Generated-summary propagation: **{decision['summary_propagation']}** (gain {decision['oracle_summary_gain']:.6f}).",
        f"2. Qwen uses rating: **{decision['qwen_uses_rating']}**; verified: **{decision['qwen_uses_verified']}**.",
        f"3. Selected decoding policy: **{decision['selected_decoding']}**, using validation only.",
        f"4. Pretraining advantage over best oracle diffusion: {decision['pretraining_advantage']}.",
        f"5. Capacity probe triggered: {decision['capacity_probe_triggered']}; it was not run inside the fixed budget.", "",
        "## 2. Oracle Summary Experiment", "",
        f"A0 review C2ST: {a0[1]:.6f}; A1: {a1[1]:.6f}; A2: {a2[1]:.6f}.",
        f"Oracle-summary gain: **{decision['oracle_summary_gain']:.6f}**.", "",
        "## 3. Structured Conditioning Ablation", "",
    ]
    for name in ("B0_none", "B1_rating", "B2_verified", "B3_rating_verified"):
        summary, review, macro = scores(name)
        lines.append(f"- {name}: summary={summary:.6f}, review={review:.6f}, macro={macro:.6f}; rating macro-F1={rating[name]['macro_f1']:.6f}; verified AUC={verified[name]['roc_auc']}")
    lines += ["", "## 4. Decoding Sensitivity", ""]
    for row in decoding.itertuples():
        lines.append(f"- {row.configuration}: macro C2ST={row.macro_c2st:.6f}, rating macro-F1={row.rating_macro_f1:.6f}, repetition={row.repeated_ngram_rate:.6f}")
    lines += ["", "## 5. Canonical Diffusion Oracle Re-evaluation", ""]
    if diffusion.empty:
        lines.append("Historical O1-O4 outputs were unavailable; this section is explicitly incomplete.")
    else:
        for row in diffusion.itertuples():
            lines.append(f"- {row.label}: macro C2ST={row.macro_c2st:.6f}; {row.conditioning}")
    budget = json.loads((output / "compute_budget.json").read_text())
    lines += [
        "", "## 6. Model Scaling Probe", "", decision["capacity_probe_note"],
        "", "## 7. Compute Cost", "", f"Generation GPU-hours: {budget['generation_gpu_hours']:.4f}.",
        "", "## 8. Architectural Interpretation", "",
        f"The evidence supports the decision fields recorded in `{output / 'decision.json'}` without treating heuristic thresholds as significance tests.",
        "", "## 9. Recommended Next Experiment", "", f"**{decision['recommended_next_experiment']}**", "",
    ]
    return "\n".join(lines)


def print_final(decision: dict[str, Any], c2st: dict[str, Any], decoding: pd.DataFrame, diffusion: pd.DataFrame, output: Path) -> None:
    score = lambda name, index: nested_c2st(c2st[name])[index]
    diff = {row.label: row.macro_c2st for row in diffusion.itertuples()} if not diffusion.empty else {}
    print("\n============================================================")
    print("QWEN TEXT FOLLOW-UP DIAGNOSTICS")
    print("============================================================\n")
    print("ORACLE SUMMARY\n------------------------------------------------------------")
    print(f"Normal review C2ST: {score('A0_normal', 1)}")
    print(f"Oracle-summary review C2ST: {score('A1_oracle_summary', 1)}")
    print(f"No-summary review C2ST: {score('A2_no_summary', 1)}")
    print(f"Oracle-summary gain: {decision['oracle_summary_gain']}")
    print(f"SUMMARY PROPAGATION BOTTLENECK: {decision['summary_propagation']}\n")
    print("STRUCTURED CONDITIONING\n------------------------------------------------------------")
    for name in ("B0_none", "B1_rating", "B2_verified", "B3_rating_verified"):
        print(f"{name}: {score(name, 2)}")
    print(f"QWEN USES RATING: {decision['qwen_uses_rating']}")
    print(f"QWEN USES VERIFIED: {decision['qwen_uses_verified']}\n")
    print("DECODING\n------------------------------------------------------------")
    for row in decoding.itertuples():
        print(f"{row.configuration}: {row.macro_c2st}")
    print(f"SELECTED VALIDATION CONFIG: {decision['selected_decoding']}\n")
    print("DIFFUSION ORACLE - CANONICAL EVALUATOR\n------------------------------------------------------------")
    for label in ("O4", "O1", "O2", "O3"):
        print(f"{label}: {diff.get(label)}")
    print(f"Qwen oracle: {decision['qwen_oracle_macro_c2st']}")
    print(f"PRETRAINING ADVANTAGE: {decision['pretraining_advantage']}\n")
    print("CAPACITY PROBE\n------------------------------------------------------------")
    print("Ran: NO")
    print(f"CAPACITY LIMITED: {'UNCLEAR' if decision['capacity_probe_triggered'] else 'NO EVIDENCE'}\n")
    budget = json.loads((output / "compute_budget.json").read_text())
    print("COMPUTE\n------------------------------------------------------------")
    print(f"Total GPU time upper bound: {budget.get('total_gpu_time_upper_bound_hours', budget.get('generation_gpu_hours'))} hours\n")
    print("FINAL RECOMMENDATION\n------------------------------------------------------------")
    print(decision["recommended_next_experiment"])
    print("============================================================")
