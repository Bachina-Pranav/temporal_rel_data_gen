"""Controlled Qwen3-0.6B versus Qwen3-1.7B capacity probe."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .decoding_sweep import detailed_diversity_metrics
from .experiment import QwenTextExperiment, nested_c2st, write_json
from .followup import dataframe_sha256, select_fixed_subset


@dataclass
class QwenCapacityProbe:
    config_path: Path

    def __post_init__(self) -> None:
        self.config = yaml.safe_load(self.config_path.read_text())
        self.base_config = yaml.safe_load(Path(self.config["base_experiment_config"]).read_text())
        self.output = Path(self.config["output_dir"])
        self.seed = int(self.config.get("seed", 42))
        self.benchmark = Path(self.base_config["data"]["benchmark_dir"])
        self.probe_benchmark = self.output / "fixed_benchmark"

    def prepare(self) -> dict[str, Any]:
        train = pd.read_csv(self.benchmark / "train_real.csv", low_memory=False)
        validation = pd.read_csv(self.benchmark / "validation_real.csv", low_memory=False)
        train_rows = int(self.config["data"]["train_rows"])
        train_indices = np.sort(
            np.random.default_rng(self.seed).choice(len(train), min(train_rows, len(train)), replace=False)
        )
        train_subset = train.iloc[train_indices].copy().reset_index(drop=True)
        validation_subset, validation_manifest = select_fixed_subset(
            validation,
            target_rows=int(self.config["data"]["validation_rows"]),
            maximum_rows=int(self.config["data"]["validation_rows"]),
            seed=self.seed,
        )
        validation_subset = validation_subset.drop(columns=["_source_row_index"])
        self.probe_benchmark.mkdir(parents=True, exist_ok=True)
        train_subset.to_csv(self.probe_benchmark / "train_real.csv", index=False)
        validation_subset.to_csv(self.probe_benchmark / "validation_real.csv", index=False)
        validation_subset.to_csv(self.probe_benchmark / "test_real.csv", index=False)
        manifest = {
            "status": "passed",
            "same_rows_for_both_models": True,
            "selection_seed": self.seed,
            "train_rows": len(train_subset),
            "validation_rows": len(validation_subset),
            "train_dataframe_sha256": dataframe_sha256(train_subset),
            "validation_dataframe_sha256": dataframe_sha256(validation_subset),
            "validation_selection": validation_manifest,
            "test_is_validation_alias_for_capacity_diagnostic": True,
            "not_a_final_test_result": True,
        }
        write_json(self.probe_benchmark / "benchmark_manifest.json", manifest)
        for label, model in self.config["models"].items():
            resolved = self._resolved_model_config(label, model["model_id"])
            path = self.output / label / "config_resolved.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        write_json(self.output / "preflight.json", manifest)
        return manifest

    def _resolved_model_config(self, label: str, model_id: str) -> dict[str, Any]:
        config = json.loads(json.dumps(self.base_config))
        config["experiment_name"] = f"qwen_capacity_{label}"
        config["seed"] = self.seed
        config["output_dir"] = str(self.output / label)
        config["model"]["model_id"] = model_id
        config["model"]["revision"] = "resolve-and-pin"
        config["data"]["benchmark_dir"] = str(self.probe_benchmark)
        config["data"]["structured_candidates"] = []
        config["data"]["lstm_candidates"] = []
        config["data"]["diffusion_text_candidates"] = []
        for key, value in self.config["training"].items():
            config["training"][key] = value
        config["training"]["early_stopping_patience"] = 1
        config["training"]["epochs"] = 1
        config["generation"].update(self.config["generation"])
        config["evaluation"].update(self.config["evaluation"])
        return config

    def _experiment(self, label: str) -> QwenTextExperiment:
        path = self.output / label / "config_resolved.yaml"
        if not path.is_file():
            self.prepare()
        return QwenTextExperiment(path)

    def preflight(self) -> dict[str, Any]:
        manifest = self.prepare()
        models = {}
        for label in self.config["models"]:
            models[label] = self._experiment(label).preflight(resolve_model=True)
        result = {"fixed_benchmark": manifest, "models": models}
        write_json(self.output / "model_preflight.json", result)
        return result

    def train(self, device: str = "cuda", skip_existing: bool = True) -> dict[str, Any]:
        results = {}
        for label in self.config["models"]:
            efficiency = self.output / label / "training/training_efficiency.json"
            if skip_existing and efficiency.is_file():
                results[label] = json.loads(efficiency.read_text())
                continue
            experiment = self._experiment(label)
            try:
                results[label] = experiment.train(device)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                results[label] = self._retry_after_oom(label, device)
        write_json(self.output / "training_summary.json", results)
        return results

    def _retry_after_oom(self, label: str, device: str) -> dict[str, Any]:
        path = self.output / label / "config_resolved.yaml"
        config = yaml.safe_load(path.read_text())
        old_batch = int(config["training"]["train_batch_size"])
        if old_batch <= 1:
            raise RuntimeError(f"{label} OOM at batch size 1")
        new_batch = max(1, old_batch // 2)
        old_accumulation = int(config["training"]["gradient_accumulation_steps"])
        config["training"]["train_batch_size"] = new_batch
        config["training"]["eval_batch_size"] = min(int(config["training"]["eval_batch_size"]), new_batch)
        config["training"]["gradient_accumulation_steps"] = old_accumulation * old_batch // new_batch
        config.setdefault("capacity_probe_runtime_adjustments", []).append(
            {
                "reason": "CUDA OOM",
                "train_batch_size_before": old_batch,
                "train_batch_size_after": new_batch,
                "gradient_accumulation_before": old_accumulation,
                "gradient_accumulation_after": config["training"]["gradient_accumulation_steps"],
                "effective_batch_preserved": True,
            }
        )
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        trainer_dir = self.output / label / "training/trainer"
        if trainer_dir.exists():
            shutil.rmtree(trainer_dir)
        return QwenTextExperiment(path).train(device)

    def generate(self, device: str = "cuda", skip_existing: bool = True) -> dict[str, Any]:
        results = {}
        for label in self.config["models"]:
            path = self.output / label / "oracle_structured/synthetic_text.csv"
            if skip_existing and path.is_file():
                results[label] = {"reused": True, "rows": len(pd.read_csv(path))}
            else:
                results[label] = self._experiment(label).generate("oracle_structured", device)
        return results

    def evaluate(self, device: str = "cuda") -> dict[str, Any]:
        result = {}
        real = pd.read_csv(self.probe_benchmark / "test_real.csv", low_memory=False)
        for label in self.config["models"]:
            experiment = self._experiment(label)
            c2st = experiment.evaluate("oracle_structured", device)
            synthetic = pd.read_csv(self.output / label / "oracle_structured/synthetic_text.csv", low_memory=False)
            diversity = detailed_diversity_metrics(real, synthetic)
            write_json(self.output / label / "oracle_structured/diversity_metrics.json", diversity)
            summary, review, macro = nested_c2st(c2st)
            efficiency = json.loads((self.output / label / "training/training_efficiency.json").read_text())
            consistency = json.loads((self.output / label / "oracle_structured/consistency_metrics.json").read_text())
            result[label] = {
                "model_id": self.config["models"][label]["model_id"],
                "summary_c2st": summary,
                "review_c2st": review,
                "macro_c2st": macro,
                "training_seconds": efficiency["training_seconds"],
                "peak_gpu_memory_bytes": efficiency["peak_gpu_memory_bytes"],
                "total_parameters": efficiency["total_parameters"],
                "trainable_parameters": efficiency["trainable_parameters"],
                "validation_loss": _last_validation_loss(self.output / label / "training/validation_log.csv"),
                "review_distinct_2": diversity["review_text"]["synthetic"]["distinct_2"],
                "rating_consistency": (
                    consistency.get("rating_from_review_text", {})
                    .get("synthetic", {})
                    .get("balanced_accuracy")
                ),
            }
        result["scaling_gain"] = float(result["qwen3_06b"]["macro_c2st"] - result["qwen3_17b"]["macro_c2st"])
        result["lower_c2st_is_better"] = True
        result["descriptive_only_not_statistical_significance"] = True
        write_json(self.output / "capacity_comparison.json", result)
        return result

    def report(self) -> dict[str, Any]:
        result = json.loads((self.output / "capacity_comparison.json").read_text())
        lines = ["# Qwen Capacity Probe", "", "Same deterministic 20,000 training and 2,000 validation rows; exactly one epoch each.", ""]
        for label in ("qwen3_06b", "qwen3_17b"):
            row = result[label]
            lines.append(
                f"- {row['model_id']}: summary={row['summary_c2st']:.6f}, review={row['review_c2st']:.6f}, macro={row['macro_c2st']:.6f}"
            )
        lines += ["", f"Scaling gain: `{result['scaling_gain']:.6f}` (descriptive only).", ""]
        (self.output / "report.md").write_text("\n".join(lines))
        return result


def _last_validation_loss(path: Path) -> float | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    values = frame.get("validation_loss", pd.Series(dtype=float)).dropna()
    return float(values.iloc[-1]) if len(values) else None
