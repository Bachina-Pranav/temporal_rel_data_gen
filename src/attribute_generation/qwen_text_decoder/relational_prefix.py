"""Frozen-Qwen temporal-relational soft-prefix experiments."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from attribute_generation.conditional_tabdlm.graph_dataset import (
    build_temporal_history_index,
)
from attribute_generation.conditional_tabdlm.graph_schema import graph_mode
from attribute_generation.conditional_tabdlm.lstm_joint import load_lstm_checkpoint
from evaluation.text_c2st_audit import (
    EmbeddingStore,
    TextC2STProtocol,
    canonical_text,
    evaluate_protocol,
    file_sha256,
)

from .decoding_sweep import (
    TEXT_FIELDS,
    detailed_diversity_metrics,
)
from .experiment import (
    ALIGNMENT_COLUMNS,
    OUTPUT_TEMPLATE,
    alignment_audit,
    conditioning_prefix,
    nested_c2st,
    parse_generated_text,
    validate_runtime_dependencies,
    write_json,
)
from .followup import (
    dataframe_sha256,
    evaluate_text_consistency,
    select_fixed_subset,
)
from .phase1 import exact_minilm_snapshot


MODES = ("R0_no_prefix", "R1_correct_context", "R2_shuffled_context")


class RelationalSoftPrefix(nn.Module):
    """Map a frozen relational representation to gated Qwen soft tokens."""

    def __init__(
        self,
        context_dim: int,
        language_dim: int,
        num_tokens: int,
        hidden_dim: int,
        dropout: float = 0.05,
        gate_initial_value: float = 0.10,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.language_dim = int(language_dim)
        self.num_tokens = int(num_tokens)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_tokens * self.language_dim),
        )
        gate = min(max(float(gate_initial_value), 1e-5), 1 - 1e-5)
        self.raw_gate = nn.Parameter(torch.tensor(math.log(gate / (1 - gate))))
        self.output_norm = nn.LayerNorm(self.language_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        projected = self.projector(context.float()).view(
            len(context), self.num_tokens, self.language_dim
        )
        return torch.sigmoid(self.raw_gate) * self.output_norm(projected)

    def metadata(self) -> dict[str, Any]:
        return {
            "context_dim": self.context_dim,
            "language_dim": self.language_dim,
            "num_tokens": self.num_tokens,
            "gate": float(torch.sigmoid(self.raw_gate.detach()).cpu()),
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
        }


def freeze_language_model(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def assert_only_prefix_trainable(model: nn.Module, prefix: nn.Module) -> dict[str, Any]:
    language_trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    prefix_trainable = [name for name, value in prefix.named_parameters() if value.requires_grad]
    if language_trainable:
        raise RuntimeError(f"Frozen Qwen/LoRA parameters became trainable: {language_trainable[:10]}")
    if not prefix_trainable:
        raise RuntimeError("Relational prefix has no trainable parameters")
    return {
        "language_model_trainable_parameters": 0,
        "prefix_trainable_parameter_names": prefix_trainable,
        "prefix_trainable_parameters": sum(p.numel() for p in prefix.parameters() if p.requires_grad),
    }


def deterministic_training_subset(frame: pd.DataFrame, rows: int | str, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if rows == "all" or int(rows) >= len(frame):
        indices = np.arange(len(frame), dtype=np.int64)
    else:
        indices = np.sort(np.random.default_rng(seed).choice(len(frame), int(rows), replace=False))
    subset = frame.iloc[indices].copy().reset_index(drop=True)
    subset.insert(0, "_source_row_index", indices)
    return subset, indices


def combined_query_indices(prefix_frames: Iterable[pd.DataFrame], target_indices: Iterable[int]) -> tuple[pd.DataFrame, np.ndarray]:
    frames = [frame.reset_index(drop=True) for frame in prefix_frames]
    if not frames:
        raise ValueError("At least one temporal-history frame is required")
    offset = sum(len(frame) for frame in frames[:-1])
    return pd.concat(frames, ignore_index=True), offset + np.asarray(list(target_indices), dtype=np.int64)


class EncodedRows(Dataset):
    def __init__(self, frame: pd.DataFrame, contexts: torch.Tensor, tokenizer: Any, max_length: int):
        if len(frame) != len(contexts):
            raise ValueError("Text rows and relational contexts are misaligned")
        self.contexts = contexts.float()
        self.items = []
        rows = frame.to_dict("records")
        prefixes = [conditioning_prefix(row["rating"], row["verified"]) for row in rows]
        targets = [
            OUTPUT_TEMPLATE.format(
                summary=canonical_text(row.get("summary")),
                review_text=canonical_text(row.get("review_text")),
            )
            + (tokenizer.eos_token or "")
            for row in rows
        ]
        for start in range(0, len(rows), 2048):
            prefix_ids = tokenizer(prefixes[start : start + 2048], add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(targets[start : start + 2048], add_special_tokens=False)["input_ids"]
            for prefix, target in zip(prefix_ids, target_ids):
                ids = (prefix + target)[: int(max_length)]
                masked = min(len(prefix), len(ids))
                self.items.append(
                    {
                        "input_ids": ids,
                        "attention_mask": [1] * len(ids),
                        "labels": [-100] * masked + ids[masked:],
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**self.items[index], "context": self.contexts[index]}


class PrefixCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        result: dict[str, Any] = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            pad = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * pad)
            result["attention_mask"].append(row["attention_mask"] + [0] * pad)
            result["labels"].append(row["labels"] + [-100] * pad)
        tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
        tensors["context"] = torch.stack([row["context"] for row in rows])
        return tensors


@dataclass
class QwenRelationalPrefixExperiment:
    config_path: Path

    def __post_init__(self) -> None:
        self.config = yaml.safe_load(self.config_path.read_text())
        self.output = Path(self.config["output_dir"])
        self.base_config = yaml.safe_load(Path(self.config["base_experiment_config"]).read_text())
        self.decoding_config = yaml.safe_load(Path(self.config["decoding_config"]).read_text())
        self.benchmark = Path(self.base_config["data"]["benchmark_dir"])
        self.base_output = Path(self.base_config["output_dir"])
        self.decoding_output = Path(self.decoding_config["output_dir"])
        self.seed = int(self.config.get("seed", 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @property
    def train_subset_path(self) -> Path:
        return self.output / "data/train_subset.csv"

    @property
    def validation_subset_path(self) -> Path:
        return self.output / "data/validation_subset.csv"

    def preflight(self) -> dict[str, Any]:
        required = [
            self.benchmark / "train_real.csv",
            self.benchmark / "validation_real.csv",
            self.benchmark / "test_real.csv",
            self.base_output / "training/best_adapter/adapter_config.json",
            Path(self.config["graph"]["checkpoint"]),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Relational-prefix inputs are missing:\n- " + "\n- ".join(missing))
        probe_compatibility = self._verify_probe_compatibility()
        train = pd.read_csv(required[0], low_memory=False)
        validation = pd.read_csv(required[1], low_memory=False)
        subset, indices = deterministic_training_subset(
            train, self.config["training"]["train_rows"], self.seed
        )
        validation_subset, validation_manifest = select_fixed_subset(
            validation,
            target_rows=int(self.config["training"]["validation_rows"]),
            maximum_rows=int(self.config["training"]["validation_rows"]),
            seed=self.seed,
        )
        self.train_subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(self.train_subset_path, index=False)
        validation_subset.to_csv(self.validation_subset_path, index=False)
        report = {
            "status": "passed",
            "model": "Qwen/Qwen3-0.6B-Base with frozen existing LoRA",
            "adapter": str(self.base_output / "training/best_adapter"),
            "adapter_config_sha256": file_sha256(required[3]),
            "graph_checkpoint": str(required[4]),
            "graph_checkpoint_sha256": file_sha256(required[4]),
            "graph_expected_mode": self.config["graph"]["expected_mode"],
            "train_rows": int(len(subset)),
            "train_source_indices_sha256": _array_sha256(indices),
            "validation": validation_manifest,
            "leakage_contract": {
                "history": "strictly earlier timestamps only",
                "same_timestamp_events": "excluded",
                "target_attributes_in_h_i": False,
                "future_events_in_h_i": False,
                "qwen_and_lora_frozen": True,
            },
            "decoding": self.config["generation"],
            "probe_compatibility": probe_compatibility,
        }
        write_json(self.output / "preflight.json", report)
        (self.output / "preflight.md").write_text(
            "# Relational Prefix Preflight\n\n"
            f"- Train rows: {len(subset):,}\n"
            f"- Validation rows: {len(validation_subset):,}\n"
            f"- Graph checkpoint: `{required[4]}`\n"
            "- Qwen and LoRA are frozen.\n"
            "- Histories are strict past-only and exclude equal timestamps.\n"
        )
        return report

    def _verify_probe_compatibility(self) -> dict[str, Any] | None:
        probe_dir_value = self.config.get("probe_output_dir")
        if not probe_dir_value:
            return None
        probe_dir = Path(probe_dir_value)
        decision_path = probe_dir / "relational_decision.json"
        checkpoint_path = probe_dir / "training/best_prefix.pt"
        missing = [str(path) for path in (decision_path, checkpoint_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Full relational-prefix training requires the completed probe:\n- "
                + "\n- ".join(missing)
            )
        decision = json.loads(decision_path.read_text())
        if decision.get("classification") not in {"strongly_supported", "moderately_supported"}:
            raise RuntimeError(
                "Full relational-prefix training is not authorized by the validation-only "
                f"probe decision: {decision.get('classification')!r}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        probe_config = checkpoint.get("experiment_config") or {}
        comparisons = {
            "prefix": probe_config.get("prefix") == self.config.get("prefix"),
            "generation": probe_config.get("generation") == self.config.get("generation"),
            "graph": probe_config.get("graph") == self.config.get("graph"),
        }
        if not all(comparisons.values()):
            raise RuntimeError(
                "Full-data configuration differs from the architecture selected by the probe: "
                + json.dumps(comparisons, sort_keys=True)
            )
        return {
            "probe_output_dir": str(probe_dir),
            "probe_classification": decision["classification"],
            "same_prefix_generation_and_graph_config": True,
            "probe_checkpoint_sha256": file_sha256(checkpoint_path),
        }

    def _ensure_preflight(self) -> None:
        if not (self.output / "preflight.json").is_file():
            self.preflight()

    def _context_path(self, split: str) -> Path:
        return self.output / f"context_cache/{split}.pt"

    def build_context_cache(self, device: str = "cuda") -> dict[str, Any]:
        self._ensure_preflight()
        train = pd.read_csv(self.benchmark / "train_real.csv", low_memory=False)
        validation = pd.read_csv(self.benchmark / "validation_real.csv", low_memory=False)
        selected_train = pd.read_csv(self.train_subset_path, low_memory=False)
        selected_validation = pd.read_csv(self.validation_subset_path, low_memory=False)
        requests = {
            "train": (train, selected_train["_source_row_index"].to_numpy(dtype=np.int64)),
            "validation": (
                pd.concat([train, validation], ignore_index=True),
                len(train) + selected_validation["_source_row_index"].to_numpy(dtype=np.int64),
            ),
        }
        contexts, metadata = self._encode_context_requests(requests, device)
        for name, tensor in contexts.items():
            path = self._context_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_torch_save(tensor.to(dtype=torch.float16), path)
        write_json(self.output / "context_cache/manifest.json", metadata)
        return metadata

    def _encode_context_requests(
        self,
        requests: dict[str, tuple[pd.DataFrame, np.ndarray]],
        device: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        checkpoint = Path(self.config["graph"]["checkpoint"])
        lstm, graph_config, _, _, encoder = load_lstm_checkpoint(
            checkpoint, device=device, include_graph=True
        )
        del lstm
        if encoder is None:
            raise RuntimeError("Confirmed graph checkpoint does not contain a graph encoder")
        actual_mode = graph_mode(graph_config.raw)
        if actual_mode != str(self.config["graph"]["expected_mode"]):
            raise RuntimeError(f"Expected graph mode {self.config['graph']['expected_mode']!r}, found {actual_mode!r}")
        forbidden = set(graph_config.raw.get("graph_conditioning", {}).get("forbidden_node_features", []))
        required_forbidden = {"rating", "verified", "summary", "review_text"}
        if not required_forbidden.issubset(forbidden):
            raise RuntimeError("Graph checkpoint does not prove target-attribute exclusion")
        encoder.eval()
        output: dict[str, torch.Tensor] = {}
        reports: dict[str, Any] = {}
        batch_size = int(self.config["graph"].get("batch_size", 1024))
        with torch.inference_mode():
            for name, (history_frame, query_indices) in requests.items():
                index = build_temporal_history_index(history_frame, graph_config, seed=self.seed)
                chunks = []
                for start in range(0, len(query_indices), batch_size):
                    rows = query_indices[start : start + batch_size]
                    with _autocast(device):
                        chunk = encoder(index.build_batch(rows, device=device, deterministic=True))
                    chunks.append(chunk.detach().float().cpu())
                output[name] = torch.cat(chunks) if chunks else torch.empty((0, encoder.output_dim))
                stats = index.diagnostics_for_rows(query_indices).to_dict()
                reports[name] = {
                    "history_rows": int(len(history_frame)),
                    "query_rows": int(len(query_indices)),
                    "query_indices_sha256": _array_sha256(query_indices),
                    "strict_past_only": True,
                    "same_timestamp_excluded": True,
                    **stats,
                }
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output, {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "graph_mode": actual_mode,
            "target_attributes_excluded": True,
            "splits": reports,
        }

    def _load_qwen(self, device: str) -> tuple[Any, Any, torch.dtype]:
        validate_runtime_dependencies()
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        adapter = self.base_output / "training/best_adapter"
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = (
            torch.bfloat16
            if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if device.startswith("cuda") and torch.cuda.is_available()
            else torch.float32
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter, local_files_only=True, torch_dtype=dtype
        ).to(device)
        freeze_language_model(model)
        model.config.use_cache = False
        return model, tokenizer, dtype

    def _new_prefix(self, context_dim: int, language_dim: int) -> RelationalSoftPrefix:
        cfg = self.config["prefix"]
        return RelationalSoftPrefix(
            context_dim=context_dim,
            language_dim=language_dim,
            num_tokens=int(cfg["num_tokens"]),
            hidden_dim=int(cfg["projector_hidden_dim"]),
            dropout=float(cfg["dropout"]),
            gate_initial_value=float(cfg["gate_initial_value"]),
        )

    def train(self, device: str = "cuda", max_epochs: int | None = None) -> dict[str, Any]:
        self._ensure_preflight()
        if not self._context_path("train").is_file():
            self.build_context_cache(device)
        model, tokenizer, dtype = self._load_qwen(device)
        train_frame = pd.read_csv(self.train_subset_path, low_memory=False)
        validation_frame = pd.read_csv(self.validation_subset_path, low_memory=False)
        train_context = torch.load(self._context_path("train"), map_location="cpu").float()
        validation_context = torch.load(self._context_path("validation"), map_location="cpu").float()
        efficiency_path = self.base_output / "training/training_efficiency.json"
        efficiency = json.loads(efficiency_path.read_text())
        max_length = int(efficiency["chosen_max_length"])
        prefix = self._new_prefix(
            int(train_context.shape[1]), int(model.get_input_embeddings().embedding_dim)
        ).to(device)
        freeze_audit = assert_only_prefix_trainable(model, prefix)
        cfg = self.config["training"]
        collator = PrefixCollator(tokenizer.pad_token_id)
        train_loader = DataLoader(
            EncodedRows(train_frame, train_context, tokenizer, max_length),
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            collate_fn=collator,
            num_workers=int(cfg.get("dataloader_num_workers", 0)),
        )
        validation_loader = DataLoader(
            EncodedRows(validation_frame, validation_context, tokenizer, max_length),
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            collate_fn=collator,
            num_workers=int(cfg.get("dataloader_num_workers", 0)),
        )
        optimizer = torch.optim.AdamW(
            prefix.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
        )
        accumulation = int(cfg["gradient_accumulation_steps"])
        epochs = min(int(max_epochs or cfg["epochs"]), int(cfg["epochs"]))
        best_loss = float("inf")
        patience = 0
        logs = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            prefix.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for step, batch in enumerate(train_loader, start=1):
                loss = self._prefix_loss(model, prefix, batch, device, dtype) / accumulation
                loss.backward()
                train_loss += float(loss.detach()) * accumulation
                if step % accumulation == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(prefix.parameters(), float(cfg["gradient_clip_norm"]))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            valid_loss = self._validation_loss(model, prefix, validation_loader, device, dtype)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / max(len(train_loader), 1),
                "validation_loss": valid_loss,
                "gate": float(torch.sigmoid(prefix.raw_gate.detach()).cpu()),
            }
            logs.append(row)
            print(json.dumps(row), flush=True)
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience = 0
                self._save_prefix(prefix, epoch, best_loss, freeze_audit)
            else:
                patience += 1
                if patience >= int(cfg["early_stopping_patience"]):
                    break
        elapsed = time.perf_counter() - started
        pd.DataFrame(logs).to_csv(self.output / "training/train_log.csv", index=False)
        result = {
            "training_seconds": elapsed,
            "completed_epochs": len(logs),
            "best_validation_loss": best_loss,
            "qwen_and_lora_frozen": True,
            "trainable": freeze_audit,
            "prefix": prefix.metadata(),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }
        write_json(self.output / "training/training_efficiency.json", result)
        del model, prefix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def _prefix_loss(self, model: Any, prefix: RelationalSoftPrefix, batch: dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> torch.Tensor:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        context = batch["context"].to(device)
        with _autocast(device, dtype):
            token_embeddings = model.get_input_embeddings()(ids)
            soft = prefix(context).to(token_embeddings.dtype)
            embeddings = torch.cat([soft, token_embeddings], dim=1)
            prefix_mask = torch.ones((len(ids), soft.shape[1]), dtype=mask.dtype, device=device)
            prefix_labels = torch.full((len(ids), soft.shape[1]), -100, dtype=labels.dtype, device=device)
            return model(
                inputs_embeds=embeddings,
                attention_mask=torch.cat([prefix_mask, mask], dim=1),
                labels=torch.cat([prefix_labels, labels], dim=1),
                use_cache=False,
            ).loss

    def _validation_loss(self, model: Any, prefix: RelationalSoftPrefix, loader: DataLoader, device: str, dtype: torch.dtype) -> float:
        prefix.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                losses.append(float(self._prefix_loss(model, prefix, batch, device, dtype)))
        return float(np.mean(losses))

    def _save_prefix(self, prefix: RelationalSoftPrefix, epoch: int, loss: float, audit: dict[str, Any]) -> None:
        path = self.output / "training/best_prefix.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": prefix.state_dict(),
            "prefix_config": prefix.metadata(),
            "epoch": epoch,
            "validation_loss": loss,
            "freeze_audit": audit,
            "experiment_config": self.config,
        }
        _atomic_torch_save(payload, path)

    def _load_prefix(self, model: Any, device: str) -> RelationalSoftPrefix:
        checkpoint = torch.load(self.output / "training/best_prefix.pt", map_location="cpu")
        meta = checkpoint["prefix_config"]
        prefix = self._new_prefix(int(meta["context_dim"]), int(meta["language_dim"]))
        prefix.load_state_dict(checkpoint["state_dict"])
        return prefix.to(device).eval()

    def generate_validation(self, device: str = "cuda", skip_existing: bool = True) -> dict[str, Any]:
        self._ensure_preflight()
        if not (self.output / "training/best_prefix.pt").is_file():
            raise FileNotFoundError("Train the relational prefix before generation")
        subset = pd.read_csv(self.validation_subset_path, low_memory=False)
        context = torch.load(self._context_path("validation"), map_location="cpu").float()
        model, tokenizer, dtype = self._load_qwen(device)
        prefix = self._load_prefix(model, device)
        results = {}
        r0_source = self.decoding_output / "D1_t105_p095_r105/synthetic_text.csv"
        r0_destination = self.output / "R0_no_prefix/synthetic_text.csv"
        r0_destination.parent.mkdir(parents=True, exist_ok=True)
        if not r0_source.is_file():
            raise FileNotFoundError(f"Frozen D1 validation output is missing: {r0_source}")
        r0 = pd.read_csv(r0_source, low_memory=False)
        if not alignment_audit(subset, r0)["aligned"]:
            raise RuntimeError("Frozen R0 D1 output does not match the relational validation subset")
        if not r0_destination.is_file():
            try:
                r0_destination.hardlink_to(r0_source)
            except OSError:
                shutil.copy2(r0_source, r0_destination)
        results["R0_no_prefix"] = {"reused_frozen_D1": True, "rows": len(r0)}
        permutation = np.random.default_rng(self.seed).permutation(len(context))
        for mode, mode_context in (
            ("R1_correct_context", context),
            ("R2_shuffled_context", context[permutation]),
        ):
            destination = self.output / mode / "synthetic_text.csv"
            if skip_existing and destination.is_file():
                existing = pd.read_csv(destination, low_memory=False)
                if alignment_audit(subset, existing)["aligned"]:
                    results[mode] = {"reused": True, "rows": len(existing)}
                    continue
            results[mode] = self._generate_with_prefix(
                model, tokenizer, prefix, subset, mode_context, destination, device, dtype, mode
            )
        write_json(
            self.output / "generation_summary.json",
            {"modes": results, "shuffle_permutation_sha256": _array_sha256(permutation)},
        )
        del model, prefix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def _generate_with_prefix(
        self,
        model: Any,
        tokenizer: Any,
        prefix: RelationalSoftPrefix,
        conditions: pd.DataFrame,
        contexts: torch.Tensor,
        destination: Path,
        device: str,
        dtype: torch.dtype,
        mode: str,
    ) -> dict[str, Any]:
        cfg = self.config["generation"]
        batch_size = int(cfg["batch_size"])
        max_new = self._generation_bound(tokenizer)
        outputs, flags = [], []
        generated_tokens = 0
        started = time.perf_counter()
        model.config.use_cache = True
        try:
            for start in range(0, len(conditions), batch_size):
                batch = conditions.iloc[start : start + batch_size]
                prompts = [conditioning_prefix(row.rating, row.verified) + "Summary:" for row in batch.itertuples()]
                encoded = tokenizer(prompts, add_special_tokens=False)["input_ids"]
                with torch.inference_mode(), _autocast(device, dtype):
                    soft = prefix(contexts[start : start + len(batch)].to(device))
                    embeddings, attention = _left_padded_prefix_embeddings(
                        model.get_input_embeddings(), encoded, soft, device
                    )
                    generated = model.generate(
                        inputs_embeds=embeddings,
                        attention_mask=attention,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=float(cfg["temperature"]),
                        top_p=float(cfg["top_p"]),
                        repetition_penalty=float(cfg["repetition_penalty"]),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                for ids in generated:
                    values = ids.detach().cpu().tolist()
                    continuation = tokenizer.decode(values, skip_special_tokens=True)
                    summary, review, status = parse_generated_text("Summary:" + continuation)
                    outputs.append({"summary": summary, "review_text": review})
                    flags.append(status)
                    generated_tokens += len(values)
        finally:
            model.config.use_cache = False
        elapsed = time.perf_counter() - started
        result = conditions.loc[:, [c for c in conditions.columns if c in (*ALIGNMENT_COLUMNS, "rating", "verified")]].copy()
        result["summary"] = [row["summary"] for row in outputs]
        result["review_text"] = [row["review_text"] for row in outputs]
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        metrics = {
            "mode": mode,
            "rows": len(result),
            "seconds": elapsed,
            "rows_per_second": len(result) / max(elapsed, 1e-9),
            "generated_tokens": generated_tokens,
            "max_new_tokens": max_new,
            "output_sha256": file_sha256(destination),
            "parse": {
                key: float(np.mean([bool(item[key]) for item in flags]))
                for key in flags[0]
            } if flags else {},
        }
        write_json(destination.parent / "generation_metrics.json", metrics)
        return metrics

    def _generation_bound(self, tokenizer: Any) -> int:
        efficiency = json.loads(
            (self.base_output / "training/training_efficiency.json").read_text()
        )
        stats = efficiency["length_statistics"]["train"]
        prefix_tokens = len(
            tokenizer(
                conditioning_prefix(5, True), add_special_tokens=False
            ).input_ids
        )
        return max(16, int(stats["combined"]["p99"]) - prefix_tokens)

    def evaluate_validation(self, device: str = "cuda") -> dict[str, Any]:
        train = pd.read_csv(self.benchmark / "train_real.csv", low_memory=False)
        validation = pd.read_csv(self.benchmark / "validation_real.csv", low_memory=False)
        real = pd.read_csv(self.validation_subset_path, low_memory=False)
        frames = {
            mode: pd.read_csv(self.output / mode / "synthetic_text.csv", low_memory=False)
            for mode in MODES
        }
        for mode, frame in frames.items():
            if not alignment_audit(real, frame)["aligned"]:
                raise RuntimeError(f"{mode} output is not aligned")
        eval_cfg = self.config["evaluation"]
        source = exact_minilm_snapshot(eval_cfg["embedding_model"], eval_cfg["embedding_revision"])
        protocol = TextC2STProtocol(
            name="canonical_relational_prefix_validation_v1",
            embedding_backend="minilm",
            embedding_model=source["local_snapshot"],
            preprocessing="canonical",
            classifiers=("logistic_regression",),
            max_rows=int(eval_cfg["max_rows_per_class"]),
            seed=self.seed,
            n_splits=int(eval_cfg["folds"]),
        )
        store = EmbeddingStore(self.output / "embedding_cache", device=device)
        c2st, diversity = {}, {}
        for mode, frame in frames.items():
            c2st[mode] = evaluate_protocol(
                real, frame, protocol, store, fields=TEXT_FIELDS, label=f"rel_prefix_{mode}"
            )
            diversity[mode] = detailed_diversity_metrics(real, frame)
            write_json(self.output / mode / "canonical_text_c2st.json", c2st[mode])
            write_json(self.output / mode / "diversity_metrics.json", diversity[mode])
        consistency = evaluate_text_consistency(
            train, real, frames, store, source["local_snapshot"], self.seed
        )
        history = source_history_text_consistency(
            train,
            validation,
            real,
            frames,
            store,
            source["local_snapshot"],
            max_history=int(eval_cfg["source_history_max_events"]),
        )
        write_json(self.output / "structured_consistency.json", consistency)
        write_json(self.output / "source_history_consistency.json", history)
        rows = []
        for mode in MODES:
            summary, review, macro = nested_c2st(c2st[mode])
            rows.append(
                {
                    "mode": mode,
                    "summary_c2st": summary,
                    "review_c2st": review,
                    "macro_c2st": macro,
                    "rating_macro_f1": consistency["rating"][mode]["macro_f1"],
                    "verified_f1": consistency["verified"][mode]["f1"],
                    "source_history_cosine": history[mode]["mean_cosine_similarity"],
                    "review_distinct_2": diversity[mode]["review_text"]["synthetic"]["distinct_2"],
                }
            )
        pd.DataFrame(rows).to_csv(self.output / "comparison.csv", index=False)
        decision = classify_relational_support(rows, self.config["selection"])
        write_json(self.output / "relational_decision.json", decision)
        return decision

    def confirm(self, device: str = "cuda", force: bool = False) -> dict[str, Any]:
        decision_path = self.output / "relational_decision.json"
        if decision_path.is_file() and not force:
            decision = json.loads(decision_path.read_text())
            if decision["classification"] != "strongly_supported":
                result = {"run": False, "reason": "Validation did not strongly support R1", "no_test_tuning": True}
                write_json(self.output / "test_confirmation/confirmation_decision.json", result)
                return result
        conditions_path = Path(self.config["confirmation"]["generated_structured_conditions"])
        real = pd.read_csv(self.benchmark / "test_real.csv", low_memory=False)
        conditions = pd.read_csv(conditions_path, low_memory=False)
        if len(real) != int(self.config["confirmation"]["expected_rows"]):
            raise RuntimeError("Held-out confirmation population changed")
        if not alignment_audit(real, conditions)["aligned"]:
            raise RuntimeError("Generated structured conditions do not align with held-out test")
        test_context = self._build_test_context(device)
        model, tokenizer, dtype = self._load_qwen(device)
        prefix = self._load_prefix(model, device)
        destination = self.output / "test_confirmation/synthetic_text.csv"
        generation = self._generate_with_prefix(
            model, tokenizer, prefix, conditions, test_context, destination, device, dtype, "test_R1"
        )
        eval_cfg = self.config["evaluation"]
        source = exact_minilm_snapshot(eval_cfg["embedding_model"], eval_cfg["embedding_revision"])
        store = EmbeddingStore(self.output / "test_confirmation/embedding_cache", device=device)
        protocol = TextC2STProtocol(
            name="canonical_relational_prefix_test_v1",
            embedding_backend="minilm",
            embedding_model=source["local_snapshot"],
            preprocessing="canonical",
            classifiers=("logistic_regression",),
            max_rows=len(real),
            seed=self.seed,
            n_splits=int(eval_cfg["folds"]),
        )
        synthetic = pd.read_csv(destination, low_memory=False)
        c2st = evaluate_protocol(real, synthetic, protocol, store, fields=TEXT_FIELDS, label="rel_prefix_test")
        summary, review, macro = nested_c2st(c2st)
        result = {
            "run": True,
            "no_test_tuning": True,
            "summary_c2st": summary,
            "review_c2st": review,
            "macro_c2st": macro,
            "generation": generation,
        }
        write_json(self.output / "test_confirmation/canonical_text_c2st.json", c2st)
        write_json(self.output / "test_confirmation/confirmation_decision.json", result)
        return result

    def _build_test_context(self, device: str) -> torch.Tensor:
        path = self._context_path("test")
        if path.is_file():
            return torch.load(path, map_location="cpu").float()
        train = pd.read_csv(self.benchmark / "train_real.csv", low_memory=False)
        validation = pd.read_csv(self.benchmark / "validation_real.csv", low_memory=False)
        test = pd.read_csv(self.benchmark / "test_real.csv", low_memory=False)
        history = pd.concat([train, validation, test], ignore_index=True)
        indices = len(train) + len(validation) + np.arange(len(test), dtype=np.int64)
        contexts, metadata = self._encode_context_requests({"test": (history, indices)}, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(contexts["test"].half(), path)
        write_json(self.output / "context_cache/test_manifest.json", metadata)
        return contexts["test"]

    def report(self) -> dict[str, Any]:
        decision_path = self.output / "relational_decision.json"
        decision = json.loads(decision_path.read_text()) if decision_path.is_file() else {"classification": "full_training_completed"}
        comparison = pd.read_csv(self.output / "comparison.csv").to_dict("records") if (self.output / "comparison.csv").is_file() else []
        lines = ["# Qwen Temporal-Relational Prefix", "", f"Decision: **{decision['classification']}**", ""]
        for row in comparison:
            lines.append(
                f"- {row['mode']}: summary={row['summary_c2st']:.6f}, review={row['review_c2st']:.6f}, macro={row['macro_c2st']:.6f}"
            )
        lines += ["", "Qwen and the existing LoRA adapter remained frozen. Only projector/prefix/gate/norm parameters were trained.", ""]
        (self.output / "report.md").write_text("\n".join(lines))
        return {"decision": decision, "comparison": comparison}


def classify_relational_support(rows: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    by_mode = {row["mode"]: row for row in rows}
    r0 = float(by_mode["R0_no_prefix"]["macro_c2st"])
    r1 = float(by_mode["R1_correct_context"]["macro_c2st"])
    r2 = float(by_mode["R2_shuffled_context"]["macro_c2st"])
    gain_r0 = r0 - r1
    gain_r2 = r2 - r1
    strong = float(selection["strong_improvement"])
    moderate = float(selection["moderate_improvement"])
    if gain_r0 >= strong and gain_r2 >= strong:
        classification = "strongly_supported"
    elif gain_r0 >= moderate and gain_r2 >= moderate:
        classification = "moderately_supported"
    elif gain_r0 > 0 and gain_r2 > 0:
        classification = "weakly_supported"
    elif gain_r0 <= 0 and gain_r2 <= 0:
        classification = "rejected"
    else:
        classification = "unresolved"
    return {
        "classification": classification,
        "validation_only_decision": True,
        "lower_c2st_is_better": True,
        "macro_c2st": {"R0": r0, "R1": r1, "R2": r2},
        "R1_improvement_over_R0": gain_r0,
        "R1_improvement_over_R2": gain_r2,
        "thresholds": {"strong": strong, "moderate": moderate},
        "shuffled_context_control_mandatory_and_present": True,
    }


def source_history_text_consistency(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selected: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    store: EmbeddingStore,
    model_path: str,
    max_history: int = 20,
) -> dict[str, Any]:
    full = pd.concat([train, validation], ignore_index=True)
    selected_indices = len(train) + selected["_source_row_index"].to_numpy(dtype=np.int64)
    timestamps = pd.to_datetime(full.review_time, errors="coerce", utc=True)
    customer_rows: dict[str, list[int]] = {}
    for index, customer in enumerate(full.customer_id.astype(str)):
        customer_rows.setdefault(customer, []).append(index)
    history_rows = []
    for target in selected_indices:
        customer = str(full.iloc[target].customer_id)
        target_time = timestamps.iloc[target]
        candidates = [
            row for row in customer_rows.get(customer, [])
            if timestamps.iloc[row] < target_time
        ][-int(max_history):]
        history_rows.append(candidates)
    all_history = sorted({row for values in history_rows for row in values})
    if not all_history:
        return {name: {"mean_cosine_similarity": None, "rows_with_history": 0} for name in ["real_validation", *frames]}
    history_emb = store.embed(
        full.iloc[all_history].review_text,
        backend="minilm",
        model_name=model_path,
        preprocessing="canonical",
        label="rel_prefix_source_history_pool",
    )
    lookup = {row: history_emb[index] for index, row in enumerate(all_history)}
    centroids = [np.mean([lookup[row] for row in rows], axis=0) if rows else None for rows in history_rows]
    output = {}
    for name, frame in {"real_validation": selected, **frames}.items():
        embeddings = store.embed(
            frame.review_text,
            backend="minilm",
            model_name=model_path,
            preprocessing="canonical",
            label=f"rel_prefix_source_consistency_{name}",
        )
        similarities = []
        for embedding, centroid in zip(embeddings, centroids):
            if centroid is None:
                continue
            denom = np.linalg.norm(embedding) * np.linalg.norm(centroid)
            similarities.append(float(np.dot(embedding, centroid) / max(denom, 1e-12)))
        output[name] = {
            "mean_cosine_similarity": float(np.mean(similarities)) if similarities else None,
            "median_cosine_similarity": float(np.median(similarities)) if similarities else None,
            "rows_with_history": len(similarities),
            "history_is_strictly_earlier": True,
        }
    return output


def _left_padded_prefix_embeddings(
    embedding_layer: nn.Module,
    token_ids: list[list[int]],
    soft_prefix: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    for index, ids in enumerate(token_ids):
        token_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        token_embeddings = embedding_layer(token_tensor)
        sequences.append(torch.cat([soft_prefix[index].to(token_embeddings.dtype), token_embeddings], dim=0))
    width = max(len(sequence) for sequence in sequences)
    dimension = sequences[0].shape[-1]
    output = torch.zeros((len(sequences), width, dimension), dtype=sequences[0].dtype, device=device)
    attention = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        output[index, -len(sequence) :] = sequence
        attention[index, -len(sequence) :] = 1
    return output, attention


def _autocast(device: str, dtype: torch.dtype | None = None):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=dtype or torch.bfloat16)
    return nullcontext()


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _array_sha256(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()
