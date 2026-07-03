"""Sampling utilities for Conditional TABDLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .model import ConditionalTABDLM
from .schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema
from .tokenization import CategoryVocab, SimpleTextTokenizer, stable_hash_bucket
from .train import build_model, resolve_device
from .utils import ensure_dir, set_seed


try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def sample_from_config(
    config: ConditionalTABDLMConfig,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    num_rows: int | str | None = None,
    batch_size: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    device: str | None = None,
    seed: int | None = None,
) -> Path:
    sampling = config.raw.get("sampling", {})
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else config.checkpoint_dir / "best.pt"
    output_path = Path(output_path) if output_path else config.output_dir / "synthetic_review_attrs.csv"
    num_rows = num_rows if num_rows is not None else sampling.get("num_rows", 100000)
    batch_size = int(batch_size or sampling.get("batch_size", 128))
    temperature = float(temperature if temperature is not None else sampling.get("temperature", 1.0))
    top_p = float(top_p if top_p is not None else sampling.get("top_p", 0.95))
    seed = int(seed if seed is not None else sampling.get("seed", 42))
    device = resolve_device(device or str(sampling.get("device", "auto")))

    set_seed(seed)
    model, ckpt_config, vocabs, tokenizer = load_model_checkpoint(checkpoint_path, device)
    spine = pd.read_csv(config.synthetic_spine_path)
    validate_spine(spine, ckpt_config.schema)
    if num_rows not in (None, "all"):
        spine = spine.head(int(num_rows)).copy()
    attrs = sample_attributes(
        model,
        ckpt_config.schema,
        vocabs,
        tokenizer,
        spine,
        config=ckpt_config,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    output = spine.loc[:, list(ckpt_config.schema.condition_columns)].copy()
    for column in ckpt_config.schema.categorical_targets:
        output[column] = attrs[column]
    for column in ckpt_config.schema.text_targets:
        output[column] = attrs[column]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "synthetic_spine_path": str(config.synthetic_spine_path),
        "output_path": str(output_path),
        "num_rows": int(len(output)),
        "batch_size": batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "target_columns": list(ckpt_config.schema.target_columns),
    }
    with (output_path.parent / "sample_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {output_path}")
    return output_path


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[ConditionalTABDLM, ConditionalTABDLMConfig, dict[str, CategoryVocab], SimpleTextTokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_config = checkpoint["raw_config"]
    schema = ConditionalTABDLMSchema.from_config_dict(raw_config)
    config = ConditionalTABDLMConfig(raw=raw_config, schema=schema, config_path=None)
    vocabs = {
        column: CategoryVocab.from_dict(data)
        for column, data in checkpoint["categorical_vocabs"].items()
    }
    tokenizer = SimpleTextTokenizer.from_dict(checkpoint["tokenizer_metadata"])
    model = build_model(config, vocabs, tokenizer).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, vocabs, tokenizer


@torch.no_grad()
def sample_attributes(
    model: ConditionalTABDLM,
    schema: ConditionalTABDLMSchema,
    categorical_vocabs: dict[str, CategoryVocab],
    text_tokenizer: SimpleTextTokenizer,
    spine: pd.DataFrame,
    config: ConditionalTABDLMConfig,
    batch_size: int,
    temperature: float,
    top_p: float,
    device: str,
) -> dict[str, list[str]]:
    id_cfg = config.raw.get("id_encoding", {})
    diffusion = config.raw.get("diffusion", {})
    num_hash_buckets = int(id_cfg.get("num_buckets", 262144))
    timesteps = int(diffusion.get("timesteps", 50))
    result: dict[str, list[str]] = {column: [] for column in schema.target_columns}
    iterator = range(0, len(spine), int(batch_size))
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(spine) + int(batch_size) - 1) // int(batch_size), desc="sample")
    for start in iterator:
        batch_frame = spine.iloc[start : start + int(batch_size)]
        foreign_key_ids, datetime_values = encode_conditions(batch_frame, schema, num_hash_buckets, device)
        cat_input = torch.empty((len(batch_frame), len(schema.categorical_targets)), dtype=torch.long, device=device)
        for idx, column in enumerate(schema.categorical_targets):
            cat_input[:, idx] = categorical_vocabs[column].mask_id
        cat_remaining = torch.ones_like(cat_input, dtype=torch.bool)

        text_input: dict[str, torch.Tensor] = {}
        text_attention: dict[str, torch.Tensor] = {}
        text_remaining: dict[str, torch.Tensor] = {}
        for column in schema.text_targets:
            length = int(schema.text_max_lengths[column])
            text_input[column] = torch.full((len(batch_frame), length), text_tokenizer.mask_id, dtype=torch.long, device=device)
            text_attention[column] = torch.ones((len(batch_frame), length), dtype=torch.long, device=device)
            text_remaining[column] = torch.ones((len(batch_frame), length), dtype=torch.bool, device=device)

        for step in range(timesteps, 0, -1):
            t = torch.full((len(batch_frame),), step / max(timesteps, 1), dtype=torch.float32, device=device)
            logits = model(
                foreign_key_ids=foreign_key_ids,
                datetime_values=datetime_values,
                categorical_input_ids=cat_input,
                text_input_ids=text_input,
                text_attention=text_attention,
                diffusion_t=t,
            )
            reveal_prob = 1.0 if step == 1 else 1.0 / float(step)
            for idx, column in enumerate(schema.categorical_targets):
                remaining = cat_remaining[:, idx]
                if not bool(remaining.any()):
                    continue
                sampled = sample_logits(logits["categorical"][column], temperature=temperature, top_p=1.0)
                reveal = remaining & (torch.rand_like(remaining.float()) < reveal_prob)
                if step == 1:
                    reveal = remaining
                cat_input[reveal, idx] = sampled[reveal]
                cat_remaining[reveal, idx] = False
            for column in schema.text_targets:
                remaining = text_remaining[column]
                if not bool(remaining.any()):
                    continue
                flat_logits = logits["text"][column].reshape(-1, logits["text"][column].shape[-1])
                sampled = sample_logits(flat_logits, temperature=temperature, top_p=top_p).view_as(text_input[column])
                reveal = remaining & (torch.rand_like(remaining.float()) < reveal_prob)
                if step == 1:
                    reveal = remaining
                text_input[column][reveal] = sampled[reveal]
                text_remaining[column][reveal] = False

        for idx, column in enumerate(schema.categorical_targets):
            decoded = [categorical_vocabs[column].decode(value) for value in cat_input[:, idx].detach().cpu().tolist()]
            result[column].extend(decoded)
        for column in schema.text_targets:
            decoded = [text_tokenizer.decode(row) for row in text_input[column].detach().cpu().tolist()]
            result[column].extend(decoded)
    return result


def encode_conditions(
    frame: pd.DataFrame,
    schema: ConditionalTABDLMSchema,
    num_hash_buckets: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    foreign_keys = []
    for _, row in frame.iterrows():
        foreign_keys.append([
            stable_hash_bucket(column, row[column], num_hash_buckets)
            for column in schema.foreign_key_columns
        ])
    datetimes = []
    for _, row in frame.iterrows():
        datetimes.append([
            pd.Timestamp(row[column]).timestamp()
            for column in schema.datetime_columns
        ])
    return (
        torch.tensor(foreign_keys, dtype=torch.long, device=device),
        torch.tensor(datetimes, dtype=torch.float32, device=device),
    )


def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0)
    logits = logits / max(float(temperature), 1e-6)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        logits = filtered
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(row_sums > 0, probs / row_sums.clamp_min(1e-12), torch.full_like(probs, 1.0 / probs.shape[-1]))
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def validate_spine(spine: pd.DataFrame, schema: ConditionalTABDLMSchema) -> None:
    missing = [column for column in schema.condition_columns if column not in spine.columns]
    if missing:
        raise ValueError(f"Synthetic spine is missing condition columns: {missing}")

