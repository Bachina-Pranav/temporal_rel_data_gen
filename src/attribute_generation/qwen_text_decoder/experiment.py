"""Focused Qwen text-decoder experiment with leakage and alignment guards."""

from __future__ import annotations

import csv
import hashlib
try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - compatibility for legacy dev shells
    import importlib_metadata
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


PREFIX_TEMPLATE = "Rating: {rating}\nVerified: {verified}\n"
OUTPUT_TEMPLATE = "Summary: {summary}\nReview: {review_text}"
ALIGNMENT_COLUMNS = ("customer_id", "product_id", "review_time")
REQUIRED_RUNTIME_VERSIONS = {
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
    "peft": "0.15.2",
    "accelerate": "1.6.0",
    "huggingface-hub": "0.30.2",
}


def validate_runtime_dependencies() -> dict[str, str]:
    installed: dict[str, str] = {}
    mismatches = []
    for package, required in REQUIRED_RUNTIME_VERSIONS.items():
        try:
            actual = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            actual = "missing"
        installed[package] = actual
        if actual != required:
            mismatches.append(f"{package}=={required} (installed: {actual})")
    if mismatches:
        command = "pip install --upgrade --force-reinstall " + " ".join(
            f"'{package}=={version}'" for package, version in REQUIRED_RUNTIME_VERSIONS.items()
        )
        raise RuntimeError(
            "Incompatible Qwen experiment runtime:\n- "
            + "\n- ".join(mismatches)
            + f"\nInstall the pinned stack with:\n{command}"
        )
    return installed


def normalize_rating(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def normalize_verified(value: Any) -> str:
    if isinstance(value, str):
        return "true" if value.strip().lower() in {"1", "true", "yes"} else "false"
    return "true" if bool(value) else "false"


def conditioning_prefix(rating: Any, verified: Any) -> str:
    return PREFIX_TEMPLATE.format(
        rating=normalize_rating(rating), verified=normalize_verified(verified)
    )


def serialize_example(row: dict[str, Any] | pd.Series, eos_token: str = "") -> str:
    return conditioning_prefix(row["rating"], row["verified"]) + OUTPUT_TEMPLATE.format(
        summary=canonical_text(row.get("summary")),
        review_text=canonical_text(row.get("review_text")),
    ) + eos_token


def encode_training_example(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    """Encode prefix and target separately so no conditioning token receives loss."""
    prefix_ids = tokenizer(conditioning_prefix(row["rating"], row["verified"]), add_special_tokens=False).input_ids
    target = OUTPUT_TEMPLATE.format(summary=canonical_text(row.get("summary")), review_text=canonical_text(row.get("review_text"))) + (tokenizer.eos_token or "")
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    input_ids = (prefix_ids + target_ids)[:max_length]
    masked = min(len(prefix_ids), len(input_ids))
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": [-100] * masked + input_ids[masked:]}


def parse_generated_text(text: Any) -> tuple[str, str, dict[str, bool]]:
    value = canonical_text(text).replace(" Summary:", "\nSummary:").replace(
        " Review:", "\nReview:"
    )
    summary_match = re.search(r"(?:^|\n)Summary:\s*(.*?)(?=\nReview:|$)", value, re.S)
    review_matches = list(re.finditer(r"(?:^|\n)Review:\s*(.*)", value, re.S))
    summary = canonical_text(summary_match.group(1)) if summary_match else ""
    review = canonical_text(review_matches[0].group(1)) if review_matches else ""
    return summary, review, {
        "parse_failure": not bool(summary_match and review_matches),
        "missing_review_marker": not bool(review_matches),
        "empty_summary": not bool(summary),
        "empty_review": not bool(review),
        "multiple_review_markers": len(review_matches) > 1,
    }


def frame_fingerprint(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for column in columns:
        digest.update(column.encode())
        values = frame[column]
        if column == "review_time":
            values = pd.to_datetime(values, errors="coerce", utc=True).astype(str)
        for value in values:
            encoded = canonical_text(value).encode("utf-8", errors="surrogatepass")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def alignment_audit(real: pd.DataFrame, generated: pd.DataFrame) -> dict[str, Any]:
    missing = [c for c in ALIGNMENT_COLUMNS if c not in real or c not in generated]
    report: dict[str, Any] = {
        "required_columns": list(ALIGNMENT_COLUMNS),
        "missing_columns": missing,
        "real_rows": int(len(real)),
        "generated_rows": int(len(generated)),
    }
    if missing or len(real) != len(generated):
        report.update({"aligned": False, "reason": "missing columns or row-count mismatch"})
        return report
    comparisons = {}
    for column in ALIGNMENT_COLUMNS:
        left, right = real[column], generated[column]
        if column == "review_time":
            left = pd.to_datetime(left, errors="coerce", utc=True)
            right = pd.to_datetime(right, errors="coerce", utc=True)
        else:
            left, right = left.astype(str), right.astype(str)
        comparisons[column] = {
            "equal_rows": int(left.eq(right).sum()),
            "equal_fraction": float(left.eq(right).mean()),
        }
    report.update(
        {
            "column_comparisons": comparisons,
            "real_spine_sha256": frame_fingerprint(real, ALIGNMENT_COLUMNS),
            "generated_spine_sha256": frame_fingerprint(generated, ALIGNMENT_COLUMNS),
        }
    )
    report["aligned"] = bool(
        report["real_spine_sha256"] == report["generated_spine_sha256"]
    )
    report["reason"] = "exact ordered spine match" if report["aligned"] else "ordered spine mismatch"
    return report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def first_existing(paths: Iterable[str | Path]) -> Path | None:
    return next((Path(p) for p in paths if Path(p).is_file()), None)


@dataclass
class QwenTextExperiment:
    config_path: Path
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output_dir = self.output_dir or Path(self.config["output_dir"])
        self.seed = int(self.config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def benchmark(self) -> Path:
        return Path(self.config["data"]["benchmark_dir"])

    def ensure_benchmark(self) -> None:
        required = ["benchmark_manifest.json", "train_real.csv", "validation_real.csv", "test_real.csv"]
        missing = [name for name in required if not (self.benchmark / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing frozen benchmark files {missing}. Run prepare_hierarchical_diffusion_benchmark.py first."
            )

    def resolve_model(self) -> dict[str, Any]:
        """Resolve an immutable HF revision and local snapshot; never use hosted inference."""
        from huggingface_hub import HfApi, snapshot_download

        model_cfg = self.config["model"]
        model_id = model_cfg["model_id"]
        requested = model_cfg.get("revision")
        api = HfApi()
        info = api.model_info(model_id, revision=None if requested == "resolve-and-pin" else requested)
        revision = info.sha
        snapshot = snapshot_download(repo_id=model_id, revision=revision)
        card = info.card_data.to_dict() if info.card_data else {}
        result = {
            "model_id": model_id,
            "requested_revision": requested,
            "revision": revision,
            "local_snapshot": str(Path(snapshot).resolve()),
            "license": card.get("license"),
        }
        if not result["license"]:
            raise RuntimeError("Qwen model license was not exposed by Hugging Face; refusing an unaudited run")
        write_json(self.output_dir / "training/model_source.json", result)
        return result

    def preflight(self, *, resolve_model: bool = True) -> dict[str, Any]:
        started = time.perf_counter()
        self.ensure_benchmark()
        manifest = json.loads((self.benchmark / "benchmark_manifest.json").read_text())
        splits = {
            name: self.benchmark / f"{name}_real.csv"
            for name in ("train", "validation", "test")
        }
        frames = {name: pd.read_csv(path, low_memory=False) for name, path in splits.items()}
        required = {"rating", "verified", "summary", "review_text"}
        for name, frame in frames.items():
            missing = required - set(frame)
            if missing:
                raise ValueError(f"{name} split lacks required columns: {sorted(missing)}")
        model = self.resolve_model() if resolve_model else {"status": "not_resolved"}
        structured = first_existing(self.config["data"]["structured_candidates"])
        lstm = first_existing(self.config["data"]["lstm_candidates"])
        diffusion = first_existing(self.config["data"]["diffusion_text_candidates"])
        report = {
            "manifest": str(self.benchmark / "benchmark_manifest.json"),
            "manifest_sha256": file_sha256(self.benchmark / "benchmark_manifest.json"),
            "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
            "splits": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in splits.items()},
            "full_artifact_rows": manifest.get("row_counts", {}).get("evaluation_real"),
            "full_79663_artifact_semantics": (
                "The 79,663-row artifact is the full-data synthetic experiment; this Qwen experiment uses the frozen chronological test split."
            ),
            "structured_output": str(structured) if structured else None,
            "lstm_output": str(lstm) if lstm else None,
            "diffusion_output": str(diffusion) if diffusion else None,
            "model": model,
            "elapsed_seconds": time.perf_counter() - started,
        }
        lines = ["# Qwen Text Decoder Preflight", "", "## Accepted Split", ""]
        lines += [f"- {name}: {len(frames[name]):,} rows; `{splits[name]}`; SHA-256 `{file_sha256(splits[name])}`" for name in splits]
        lines += ["", "## Semantics", "", report["full_79663_artifact_semantics"], "", "## Frozen Inputs", "", f"- Structured: `{report['structured_output']}`", f"- LSTM: `{report['lstm_output']}`", f"- Diffusion text: `{report['diffusion_output']}`", "", "## Model", "", f"- ID: `{model.get('model_id')}`", f"- Revision: `{model.get('revision')}`", f"- License: `{model.get('license')}`"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "preflight.md").write_text("\n".join(lines) + "\n")
        write_json(self.output_dir / "preflight.json", report)
        print(json.dumps(report, indent=2))
        return report

    def _model_source(self) -> dict[str, Any]:
        path = self.output_dir / "training/model_source.json"
        if not path.is_file():
            return self.resolve_model()
        return json.loads(path.read_text())

    def tokenizer_statistics(self, tokenizer: Any, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        def lengths(frame: pd.DataFrame, kind: str) -> np.ndarray:
            if kind == "summary": texts = frame.summary.map(canonical_text).tolist()
            elif kind == "review_text": texts = frame.review_text.map(canonical_text).tolist()
            else: texts = [serialize_example(row, tokenizer.eos_token or "") for _, row in frame.iterrows()]
            output: list[int] = []
            for start in range(0, len(texts), 2048):
                encoded = tokenizer(texts[start:start + 2048], add_special_tokens=False, truncation=False)
                output.extend(len(ids) for ids in encoded["input_ids"])
            return np.asarray(output, dtype=np.int32)
        result: dict[str, Any] = {}
        arrays = {}
        for split, frame in frames.items():
            result[split] = {}
            arrays[split] = {}
            for kind in ("summary", "review_text", "combined"):
                values = lengths(frame, kind)
                arrays[split][kind] = values
                result[split][kind] = stats(values)
        train = arrays["train"]["combined"]
        coverage = float(self.config["training"].get("max_length_coverage", 0.99))
        cap = int(np.quantile(train, coverage, method="higher"))
        model_cap = int(min(getattr(tokenizer, "model_max_length", cap), 8192))
        cap = min(cap, model_cap)
        result["chosen_max_length"] = cap
        result["target_coverage"] = coverage
        for split in frames:
            values = arrays[split]["combined"]
            result[split]["coverage"] = float(np.mean(values <= cap))
            result[split]["truncation_rate"] = float(np.mean(values > cap))
        return result

    def train(self, device: str = "cuda") -> dict[str, Any]:
        runtime_versions = validate_runtime_dependencies()
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

        self.ensure_benchmark()
        source = self._model_source()
        train = pd.read_csv(self.benchmark / "train_real.csv", low_memory=False)
        validation = pd.read_csv(self.benchmark / "validation_real.csv", low_memory=False)
        started_load = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(source["local_snapshot"], local_files_only=True)
        if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
        if torch.cuda.is_available() and self.config["training"].get("bf16"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(source["local_snapshot"], local_files_only=True, torch_dtype=dtype)
        model_load_seconds = time.perf_counter() - started_load
        target_modules = discover_lora_targets(model)
        lora = self.config["model"]
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=int(lora["lora_rank"]), lora_alpha=int(lora["lora_alpha"]), lora_dropout=float(lora["lora_dropout"]), target_modules=target_modules, bias="none"))
        if self.config["training"].get("gradient_checkpointing"):
            model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        token_started = time.perf_counter()
        length_stats = self.tokenizer_statistics(tokenizer, {"train": train, "validation": validation})
        max_length = int(length_stats["chosen_max_length"])

        class CausalRows(Dataset):
            def __init__(self, frame: pd.DataFrame):
                rows = frame.to_dict("records")
                prefixes = [conditioning_prefix(row["rating"], row["verified"]) for row in rows]
                targets = [OUTPUT_TEMPLATE.format(summary=canonical_text(row.get("summary")), review_text=canonical_text(row.get("review_text"))) + (tokenizer.eos_token or "") for row in rows]
                self.items = []
                for start in range(0, len(rows), 2048):
                    prefix_ids = tokenizer(prefixes[start:start + 2048], add_special_tokens=False)["input_ids"]
                    target_ids = tokenizer(targets[start:start + 2048], add_special_tokens=False)["input_ids"]
                    for prefix, target in zip(prefix_ids, target_ids):
                        input_ids = (prefix + target)[:max_length]
                        masked = min(len(prefix), len(input_ids))
                        self.items.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": [-100] * masked + input_ids[masked:]})
            def __len__(self): return len(self.items)
            def __getitem__(self, index): return self.items[index]

        class Collator:
            def __call__(self, features):
                max_len = max(len(x["input_ids"]) for x in features)
                result = {k: [] for k in ("input_ids", "attention_mask", "labels")}
                for x in features:
                    pad = max_len - len(x["input_ids"])
                    result["input_ids"].append(x["input_ids"] + [tokenizer.pad_token_id] * pad)
                    result["attention_mask"].append(x["attention_mask"] + [0] * pad)
                    result["labels"].append(x["labels"] + [-100] * pad)
                return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}

        epoch_runtime: list[dict[str, Any]] = []
        class EpochRuntimeCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                self.started = time.perf_counter()
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            def on_epoch_end(self, args, state, control, **kwargs):
                epoch_runtime.append({"epoch": int(math.ceil(float(state.epoch or 0))), "wall_clock_seconds": time.perf_counter() - self.started, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0})

        out = self.output_dir / "training"; out.mkdir(parents=True, exist_ok=True)
        cfg = self.config["training"]
        args = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=float(cfg["epochs"]), per_device_train_batch_size=int(cfg["train_batch_size"]), per_device_eval_batch_size=int(cfg["eval_batch_size"]), gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]), learning_rate=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]), warmup_ratio=float(cfg["warmup_ratio"]), eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps", logging_steps=25, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16, dataloader_num_workers=int(cfg["dataloader_num_workers"]), report_to=[], seed=self.seed)
        train_dataset, validation_dataset = CausalRows(train), CausalRows(validation)
        tokenization_seconds = time.perf_counter() - token_started
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=validation_dataset, data_collator=Collator(), callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg["early_stopping_patience"])), EpochRuntimeCallback()])
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        started = time.perf_counter(); result = trainer.train(); training_seconds = time.perf_counter() - started
        checkpoints = sorted((out / "trainer").glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1]))
        if checkpoints:
            final_dir = out / "final_adapter"; final_dir.mkdir(parents=True, exist_ok=True)
            for artifact in checkpoints[-1].glob("adapter_*"):
                if artifact.is_file(): shutil.copy2(artifact, final_dir / artifact.name)
            tokenizer.save_pretrained(out / "final_adapter")
        trainer.save_model(str(out / "best_adapter")); tokenizer.save_pretrained(out / "best_adapter")
        logs = pd.DataFrame(trainer.state.log_history)
        per_epoch = []
        for runtime in epoch_runtime:
            epoch = runtime["epoch"]
            selected = logs[(logs.get("epoch", 0) > epoch - 1) & (logs.get("epoch", 0) <= epoch)]
            train_losses = selected.get("loss", pd.Series(dtype=float)).dropna()
            eval_losses = selected.get("eval_loss", pd.Series(dtype=float)).dropna()
            learning_rates = selected.get("learning_rate", pd.Series(dtype=float)).dropna()
            per_epoch.append({**runtime, "train_loss": float(train_losses.mean()) if len(train_losses) else None, "validation_loss": float(eval_losses.iloc[-1]) if len(eval_losses) else None, "learning_rate": float(learning_rates.iloc[-1]) if len(learning_rates) else None})
        pd.DataFrame(per_epoch).to_csv(out / "train_log.csv", index=False)
        pd.DataFrame([{k: row[k] for k in ("epoch", "validation_loss", "wall_clock_seconds", "peak_vram_bytes")} for row in per_epoch]).to_csv(out / "validation_log.csv", index=False)
        preflight = json.loads((self.output_dir / "preflight.json").read_text()) if (self.output_dir / "preflight.json").is_file() else {}
        completed_epochs = max(float(trainer.state.epoch or 1), 1.0)
        observed_tokens = sum(len(item["input_ids"]) for item in train_dataset.items) * completed_epochs
        peak_vram = max((row["peak_vram_bytes"] for row in epoch_runtime), default=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
        efficiency = {"model_id": source["model_id"], "revision": source["revision"], "license": source["license"], "preprocessing_seconds": preflight.get("elapsed_seconds"), "model_load_seconds": model_load_seconds, "tokenization_seconds": tokenization_seconds, "training_seconds": training_seconds, "completed_epochs": completed_epochs, "seconds_per_epoch": training_seconds / completed_epochs, "examples_per_second": result.metrics.get("train_samples_per_second"), "tokens_per_second": observed_tokens / training_seconds, "peak_gpu_memory_bytes": peak_vram, "gpu_model": torch.cuda.get_device_name() if torch.cuda.is_available() else None, "total_parameters": total, "trainable_parameters": trainable, "trainable_fraction": trainable / total, "lora_rank": int(lora["lora_rank"]), "lora_alpha": int(lora["lora_alpha"]), "target_modules": target_modules, "chosen_max_length": max_length, "length_statistics": length_stats}
        efficiency["runtime_versions"] = runtime_versions
        write_json(out / "training_efficiency.json", efficiency)
        write_json(out / "config.json", self.config)
        return efficiency

    def generate(self, mode: str, device: str = "cuda") -> dict[str, Any]:
        validate_runtime_dependencies()
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        if mode not in {"oracle_structured", "generated_structured"}: raise ValueError(mode)
        real = pd.read_csv(self.benchmark / "test_real.csv", low_memory=False)
        if mode == "oracle_structured": conditions = real.copy()
        else:
            conditions = None; candidate_audits = []
            for candidate in self.config["data"]["structured_candidates"]:
                path = Path(candidate)
                if not path.is_file():
                    candidate_audits.append({"path": str(path), "status": "missing"}); continue
                frame = pd.read_csv(path, low_memory=False)
                audit = alignment_audit(real, frame); audit["path"] = str(path); audit["sha256"] = file_sha256(path)
                candidate_audits.append(audit)
                if audit["aligned"] and {"rating", "verified"}.issubset(frame.columns): conditions = frame; break
            alignment_report = {"selected_path": next((x["path"] for x in candidate_audits if x.get("aligned")), None), "candidates": candidate_audits, "aligned": conditions is not None}
            write_json(self.output_dir / mode / "row_alignment_audit.json", alignment_report)
            if conditions is None: raise RuntimeError("Mode B stopped: no frozen generated-structured artifact exactly aligns with the held-out test spine")
        adapter = self.output_dir / "training/best_adapter"
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        model = AutoPeftModelForCausalLM.from_pretrained(adapter, local_files_only=True, torch_dtype=dtype).to(device).eval()
        max_new = self._generation_bound(tokenizer)
        cfg = self.config["generation"]; batch_size = int(cfg["batch_size"])
        outputs, flags, token_count = [], [], 0
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        started = time.perf_counter()
        for start in range(0, len(conditions), batch_size):
            batch = conditions.iloc[start:start + batch_size]
            prompts = [conditioning_prefix(r.rating, r.verified) + "Summary:" for r in batch.itertuples()]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=max_new, do_sample=bool(cfg["do_sample"]), temperature=float(cfg["temperature"]), top_p=float(cfg["top_p"]), repetition_penalty=float(cfg["repetition_penalty"]), eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            for i, ids in enumerate(generated):
                continuation = tokenizer.decode(ids[encoded.input_ids.shape[1]:], skip_special_tokens=True)
                summary, review, status = parse_generated_text("Summary:" + continuation)
                outputs.append({"summary": summary, "review_text": review}); flags.append(status)
                token_count += int(len(ids) - encoded.input_ids.shape[1])
        elapsed = time.perf_counter() - started
        result = conditions.loc[:, [c for c in conditions.columns if c in (*ALIGNMENT_COLUMNS, "rating", "verified")]].copy()
        result["summary"] = [x["summary"] for x in outputs]; result["review_text"] = [x["review_text"] for x in outputs]
        out = self.output_dir / mode; out.mkdir(parents=True, exist_ok=True); result.to_csv(out / "synthetic_text.csv", index=False)
        metrics = {"mode": mode, "diagnostic_only": mode == "oracle_structured", "rows": len(result), "seconds": elapsed, "rows_per_second": len(result) / elapsed, "generated_tokens": token_count, "tokens_per_second": token_count / elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0, "max_new_tokens": max_new, "generation_settings": cfg, "parsing": {key: float(np.mean([f[key] for f in flags])) for key in flags[0]} if flags else {}}
        write_json(out / "generation_metrics.json", metrics); return metrics

    def _generation_bound(self, tokenizer: Any) -> int:
        efficiency = json.loads((self.output_dir / "training/training_efficiency.json").read_text())
        stats_ = efficiency["length_statistics"]["train"]
        prefix = len(tokenizer(conditioning_prefix(5, True), add_special_tokens=False).input_ids)
        return max(16, int(stats_["combined"]["p99"]) - prefix)

    def evaluate(self, mode: str, device: str = "cuda") -> dict[str, Any]:
        validate_runtime_dependencies()
        real = pd.read_csv(self.benchmark / "test_real.csv", low_memory=False)
        synthetic = pd.read_csv(self.output_dir / mode / "synthetic_text.csv", low_memory=False)
        evaluation = self.config["evaluation"]
        pin_path = self.output_dir / "evaluation_model_source.json"
        if pin_path.is_file():
            embedding_source = json.loads(pin_path.read_text())
            if not Path(embedding_source["local_snapshot"]).is_dir(): raise RuntimeError("Pinned MiniLM snapshot moved or was deleted")
        else:
            embedding_source = resolve_pinned_minilm(evaluation["embedding_model"])
            write_json(pin_path, embedding_source)
        model_source = embedding_source["local_snapshot"]
        protocol = TextC2STProtocol(name="canonical_paper_text_c2st_v1", embedding_backend="minilm", embedding_model=model_source, preprocessing="canonical", classifiers=("logistic_regression",), max_rows=int(evaluation["max_rows_per_class"]), seed=int(evaluation["seed"]), n_splits=int(evaluation["folds"]))
        store = EmbeddingStore(self.output_dir / "embedding_cache", device=device)
        result = evaluate_protocol(real, synthetic, protocol, store, label=f"qwen_{mode}")
        result["embedding_model_metadata"] = store.model_metadata
        result["embedding_model_metadata"].update({"requested_model_id": embedding_source["model_id"], "pinned_snapshot_commit": embedding_source["revision"]})
        write_json(self.output_dir / mode / "canonical_text_c2st.json", result)
        write_json(self.output_dir / mode / "text_distribution_metrics.json", distribution_comparison(real, synthetic))
        write_json(self.output_dir / mode / "memorization_metrics.json", memorization_metrics(pd.read_csv(self.benchmark / "train_real.csv", low_memory=False), synthetic))
        write_json(
            self.output_dir / mode / "consistency_metrics.json",
            consistency_metrics(
                pd.read_csv(self.benchmark / "train_real.csv", low_memory=False),
                real,
                synthetic,
                store,
                model_source,
                self.seed,
            ),
        )
        return result

    def report(self) -> dict[str, Any]:
        modes = {}
        for mode in ("oracle_structured", "generated_structured"):
            c2st_path = self.output_dir / mode / "canonical_text_c2st.json"
            generation_path = self.output_dir / mode / "generation_metrics.json"
            if c2st_path.is_file():
                modes[mode] = {
                    "c2st": json.loads(c2st_path.read_text()),
                    "generation": json.loads(generation_path.read_text()),
                }
        baselines = load_frozen_canonical_baselines()
        rows = []
        for name, item in baselines.items():
            rows.append(comparison_row(name, item, None, conditioning="frozen canonical"))
        for mode, item in modes.items():
            rows.append(comparison_row(
                f"Qwen3-0.6B {mode.upper()}", item["c2st"], item["generation"],
                conditioning="true rating/verified (diagnostic only)" if mode == "oracle_structured" else "frozen generated rating/verified",
            ))
        comparison_dir = self.output_dir / "comparison"; comparison_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(comparison_dir / "model_comparison.csv", index=False)
        efficiency = json.loads((self.output_dir / "training/training_efficiency.json").read_text())
        qwen = modes.get("generated_structured", {})
        qwen_macro = nested_c2st(qwen.get("c2st", {}))[2] if qwen else None
        oracle_macro = nested_c2st(modes.get("oracle_structured", {}).get("c2st", {}))[2] if modes.get("oracle_structured") else None
        lstm_macro = next((r["macro_text_c2st"] for r in rows if r["model"].lower().startswith("current final lstm")), None)
        diffusion_macro = next((r["macro_text_c2st"] for r in rows if "masked-diffusion" in r["model"].lower()), None)
        recommendation = recommendation_for(qwen_macro, lstm_macro)
        decision = {
            "pretraining_helps": bool(oracle_macro is not None and diffusion_macro is not None and oracle_macro < diffusion_macro),
            "oracle_macro_text_c2st": oracle_macro,
            "generated_structured_macro_text_c2st": qwen_macro,
            "qwen_beats_lstm": bool(qwen_macro is not None and lstm_macro is not None and qwen_macro < lstm_macro),
            "qwen_beats_masked_diffusion": bool(qwen_macro is not None and diffusion_macro is not None and qwen_macro < diffusion_macro),
            "recommendation": recommendation,
        }
        write_json(comparison_dir / "decision.json", decision)
        write_efficiency_report(comparison_dir / "model_efficiency_report.md", efficiency, rows)
        write_qualitative_samples(comparison_dir / "qualitative_samples.md", self.output_dir)
        write_experiment_report(self.output_dir / "experiment_report.md", efficiency, rows, decision)
        print_final_summary(efficiency, rows, decision)
        return decision


def discover_lora_targets(model: torch.nn.Module) -> list[str]:
    names = sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and any(token in name.lower() for token in ("q_proj", "k_proj", "v_proj", "o_proj"))})
    if not names: raise RuntimeError("No Qwen attention projection modules discovered for LoRA")
    return names


def stats(values: np.ndarray) -> dict[str, Any]:
    maximum = values.max()
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)), "p99": float(np.quantile(values, .99)), "max": int(maximum) if np.issubdtype(values.dtype, np.integer) else float(maximum)}


def resolve_pinned_minilm(model_id: str) -> dict[str, str]:
    """Require a local immutable HF snapshot; no silent network/fallback during metrics."""
    from huggingface_hub import scan_cache_dir
    matches = [r for r in scan_cache_dir().repos if r.repo_id == model_id]
    revisions = [rev for repo in matches for rev in repo.revisions if rev.snapshot_path.is_dir()]
    if not revisions: raise RuntimeError(f"Pinned local MiniLM snapshot unavailable for {model_id}; canonical evaluation fails loudly")
    revision = sorted(revisions, key=lambda x: x.commit_hash)[-1]
    return {"model_id": model_id, "revision": revision.commit_hash, "local_snapshot": str(revision.snapshot_path.resolve())}


def tokens(text: Any) -> list[str]:
    return re.findall(r"\b\w+\b", canonical_text(text).lower())


def distribution_comparison(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import ks_2samp
    result = {}
    for field in ("summary", "review_text"):
        left, right = real[field].map(tokens), synthetic[field].map(tokens)
        def side(values):
            flat = [token for row in values for token in row]; bigrams = [pair for row in values for pair in zip(row, row[1:])]
            texts = values.map(lambda x: " ".join(x))
            repeated_ngrams = sum(max(count - 1, 0) for count in Counter(bigrams).values())
            sentences = [sentence.strip().lower() for text in texts for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
            return {**stats(values.map(len).to_numpy()), "empty_rate": float(values.map(len).eq(0).mean()), "unique_text_ratio": float(texts.nunique() / max(len(texts), 1)), "exact_duplicate_rate": float(texts.duplicated().mean()), "distinct_1": len(set(flat)) / max(len(flat), 1), "distinct_2": len(set(bigrams)) / max(len(bigrams), 1), "repeated_ngram_rate": repeated_ngrams / max(len(bigrams), 1), "repeated_sentence_rate": 1.0 - len(set(sentences)) / max(len(sentences), 1)}
        result[field] = {"real": side(left), "synthetic": side(right), "length_ks": float(ks_2samp(left.map(len), right.map(len)).statistic)}
    return result


def memorization_metrics(train: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    result = {}
    for field in ("summary", "review_text"):
        train_text = set(train[field].map(canonical_text)) - {""}; generated = synthetic[field].map(canonical_text)
        train_sample = list(sorted(train_text))[:20000]
        generated_sample = generated.head(5000).tolist()
        nearest = None
        if train_sample and generated_sample:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
            train_matrix = vectorizer.fit_transform(train_sample)
            generated_matrix = vectorizer.transform(generated_sample)
            distances, _ = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(train_matrix).kneighbors(generated_matrix)
            similarities = 1.0 - distances[:, 0]
            nearest = {"sample_rows": len(generated_sample), "mean": float(similarities.mean()), "p95": float(np.quantile(similarities, .95)), "max": float(similarities.max()), "metric": "nearest train TF-IDF token unigram+bigram cosine"}
        result[field] = {"exact_train_overlap_rate": float(generated.isin(train_text).mean()), "exact_train_overlap_count": int(generated.isin(train_text).sum()), "nearest_neighbor_token_overlap": nearest}
    result["privacy_claim"] = "No differential privacy claim; this is an obvious-memorization diagnostic only."
    return result


def consistency_metrics(
    train: pd.DataFrame,
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    store: EmbeddingStore,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    """Train fixed real-data probes, then evaluate real and generated text."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    n_train = min(len(train), 50000)
    n_eval = min(len(real), len(synthetic), 50000)
    train_sample = train.sample(n=n_train, random_state=seed) if len(train) > n_train else train
    real_sample = real.sample(n=n_eval, random_state=seed) if len(real) > n_eval else real.head(n_eval)
    syn_sample = synthetic.loc[real_sample.index] if set(real_sample.index).issubset(synthetic.index) else synthetic.head(n_eval)
    train_emb = store.embed(train_sample.review_text, backend="minilm", model_name=model_name, preprocessing="canonical", label="consistency_train_review")
    real_emb = store.embed(real_sample.review_text, backend="minilm", model_name=model_name, preprocessing="canonical", label="consistency_real_review")
    syn_emb = store.embed(syn_sample.review_text, backend="minilm", model_name=model_name, preprocessing="canonical", label="consistency_syn_review")
    probes = {}
    for target in ("rating", "verified"):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"))
        model.fit(train_emb, train_sample[target].astype(str))
        probes[target] = {}
        for label, emb, frame in (("real_heldout", real_emb, real_sample), ("synthetic", syn_emb, syn_sample)):
            pred = model.predict(emb); truth = frame[target].astype(str)
            probes[target][label] = {"accuracy": float(accuracy_score(truth, pred)), "balanced_accuracy": float(balanced_accuracy_score(truth, pred)), "rows": int(len(truth))}
    summary_emb = store.embed(syn_sample.summary, backend="minilm", model_name=model_name, preprocessing="canonical", label="consistency_syn_summary")
    denom = np.linalg.norm(summary_emb, axis=1) * np.linalg.norm(syn_emb, axis=1)
    cosine = np.divide(np.sum(summary_emb * syn_emb, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    return {"probe_training_source": "real chronological training split only", "rating_from_review_text": probes["rating"], "verified_from_review_text": probes["verified"], "summary_review_cosine": stats(cosine), "rating_conditional_embedding_centroid_distance": conditional_centroid_distances(real_sample, real_emb, syn_sample, syn_emb, "rating")}


def conditional_centroid_distances(real: pd.DataFrame, real_emb: np.ndarray, synthetic: pd.DataFrame, syn_emb: np.ndarray, column: str) -> dict[str, Any]:
    result = {}
    for value in sorted(set(real[column].astype(str)) & set(synthetic[column].astype(str))):
        left = real_emb[real[column].astype(str).to_numpy() == value]
        right = syn_emb[synthetic[column].astype(str).to_numpy() == value]
        if len(left) and len(right): result[value] = float(np.linalg.norm(left.mean(0) - right.mean(0)))
    return result


def nested_c2st(value: dict[str, Any]) -> tuple[Any, Any, Any]:
    per = value.get("per_field") or value.get("per_text_column") or {}
    def score(field):
        item = per.get(field, {}); return item.get("error") if isinstance(item, dict) else None
    summary, review = score("summary"), score("review_text")
    macro = value.get("macro_error")
    if macro is None and summary is not None and review is not None: macro = (summary + review) / 2
    return summary, review, macro


def load_frozen_canonical_baselines() -> dict[str, dict[str, Any]]:
    candidates = [Path("outputs/text_c2st_audit/canonical_text_c2st_results.json"), Path("outputs/evaluation_cleanup/canonical_text_c2st_results.json")]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None: return {}
    data = json.loads(path.read_text())
    return {"Current final LSTM": data.get("final_lstm", {}), "Current masked-diffusion": data.get("diffusion", {})}


def comparison_row(name: str, c2st: dict[str, Any], generation: dict[str, Any] | None, *, conditioning: str) -> dict[str, Any]:
    summary, review, macro = nested_c2st(c2st)
    return {"model": name, "conditioning": conditioning, "summary_c2st": summary, "review_c2st": review, "macro_text_c2st": macro, "generation_rows_per_second": (generation or {}).get("rows_per_second"), "tokens_per_second": (generation or {}).get("tokens_per_second"), "peak_vram_bytes": (generation or {}).get("peak_gpu_memory_bytes")}


def recommendation_for(score: float | None, lstm: float | None) -> str:
    if score is None: return "Incomplete: run both generation modes and canonical evaluation."
    if score <= .50: return "A. Pretrained Qwen clearly solves much of the text problem; proceed to temporal-relational conditioning."
    if score <= .60: return "B. Pretraining helps but conditioning needs improvement; proceed to relational prefix/context experiment."
    if score < .70: return "B. Moderate gain; proceed to a controlled relational prefix/context experiment."
    if lstm is not None and score >= lstm: return "C. Pretraining produces little gain; investigate training/data objective before increasing model size."
    return "D. Qwen3-0.6B may be capacity limited; test Qwen3-1.7B-Base next only after objective checks."


def write_efficiency_report(path: Path, efficiency: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.write_text("# Model Efficiency Report\n\n## Qwen3-0.6B-Base\n\n" + "\n".join([f"- Total parameters: {efficiency.get('total_parameters')}", f"- LoRA trainable parameters: {efficiency.get('trainable_parameters')}", f"- Trainable fraction: {efficiency.get('trainable_fraction')}", f"- Training seconds: {efficiency.get('training_seconds')}", f"- Peak VRAM bytes: {efficiency.get('peak_gpu_memory_bytes')}", "", "Trainable parameter count and wall-clock speed are distinct quantities; no proportional-speedup claim is made."]) + "\n", encoding="utf-8")


def write_qualitative_samples(path: Path, output: Path) -> None:
    source = output / "generated_structured/synthetic_text.csv"
    if not source.is_file(): return
    frame = pd.read_csv(source, low_memory=False); frame["length"] = frame.review_text.map(lambda x: len(tokens(x)))
    frame["length_group"] = pd.qcut(frame["length"].rank(method="first"), 3, labels=["short", "medium", "long"])
    picked = frame.sort_values(["rating", "verified", "length_group", "length"]).groupby(["rating", "verified", "length_group"], observed=True).head(1)
    lines = ["# Deterministic Qualitative Samples", "", "No real held-out text is included.", ""]
    for row in picked.itertuples(): lines += [f"## Rating {row.rating}; verified {row.verified}; {row.length_group}", "", f"**Summary:** {canonical_text(row.summary)}", "", f"**Review:** {canonical_text(row.review_text)}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_experiment_report(path: Path, efficiency: dict[str, Any], rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    if rows:
        columns = list(rows[0])
        table_lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        table_lines.extend("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows)
        table = "\n".join(table_lines)
    else:
        table = "No completed evaluations."
    text = f"""# Qwen3-0.6B Text Decoder Experiment

## 1. Executive Result

{decision['recommendation']}

## 2. Data and Split

The exact frozen chronological benchmark and hashes are recorded in `preflight.md`.

## 3. Qwen Model

`{efficiency.get('model_id')}` at immutable revision `{efficiency.get('revision')}`, license `{efficiency.get('license')}`.

## 4. Fine-Tuning Setup

LoRA rank {efficiency.get('lora_rank')}, alpha {efficiency.get('lora_alpha')}; conditioning-prefix labels are masked.

## 5. Training Efficiency

Training seconds: {efficiency.get('training_seconds')}; peak VRAM bytes: {efficiency.get('peak_gpu_memory_bytes')}.

## 6. Oracle Structured Results

Oracle conditioning is diagnostic only and is not an end-to-end generative result.

## 7. Generated Structured Results

Mode B runs only after an exact ordered-spine alignment gate passes.

## 8. Comparison to Current Decoders

{table}

## 9. Structured-Text Consistency

See each mode's `consistency_metrics.json`.

## 10. Memorization

See each mode's `memorization_metrics.json`; no differential privacy claim is made.

## 11. Failure Analysis

Distribution, parsing, repetition, and consistency artifacts identify remaining separability.

## 12. Recommendation

{decision['recommendation']}
"""
    path.write_text(text, encoding="utf-8")


def print_final_summary(efficiency: dict[str, Any], rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    print("=" * 60); print("QWEN3-0.6B TEXT DECODER EXPERIMENT"); print("=" * 60)
    print(f"MODEL: {efficiency.get('model_id')}\nREVISION: {efficiency.get('revision')}\nTOTAL PARAMETERS: {efficiency.get('total_parameters')}\nTRAINABLE PARAMETERS: {efficiency.get('trainable_parameters')}\nTRAINABLE FRACTION: {efficiency.get('trainable_fraction')}\nTRAINING METHOD: LoRA\nTRAINING TIME: {efficiency.get('training_seconds')}\nPEAK VRAM: {efficiency.get('peak_gpu_memory_bytes')}")
    print("\nCANONICAL TEXT C2ST")
    for row in rows: print(f"{row['model']}: summary={row['summary_c2st']} review={row['review_c2st']} macro={row['macro_text_c2st']}")
    print(f"\nPRETRAINING HELPS: {'YES' if decision['pretraining_helps'] else 'NO'}\nQWEN BEATS LSTM: {'YES' if decision['qwen_beats_lstm'] else 'NO'}\nQWEN BEATS MASKED DIFFUSION: {'YES' if decision['qwen_beats_masked_diffusion'] else 'NO'}\nRECOMMENDATION: {decision['recommendation']}"); print("=" * 60)
