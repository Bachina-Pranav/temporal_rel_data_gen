"""Schema and config handling for conditional TABDLM experiments."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .tokenization import normalize_text
from .utils import load_yaml


FORBIDDEN_ENGINEERED_FEATURES = {
    "customer_block",
    "product_block",
    "customer_degree_bucket",
    "product_degree_bucket",
    "customer_relative_age",
    "product_relative_age",
    "dynamic_affinity",
}


@dataclass(frozen=True)
class ConditionalTABDLMSchema:
    """Column-level definition of fixed conditions and generated targets."""

    foreign_key_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    categorical_targets: tuple[str, ...]
    numerical_targets: tuple[str, ...] = ()
    text_targets: tuple[str, ...] = ()
    auxiliary_categorical_targets: tuple[str, ...] = ()
    text_max_lengths: dict[str, int] = field(default_factory=dict)
    summary_length_buckets: dict[str, tuple[int, int]] = field(default_factory=dict)
    review_text_length_buckets: dict[str, tuple[int, int]] = field(default_factory=dict)
    summary_length_enabled: bool = False
    use_length_bucket_in_sampling: bool = False
    force_eos_after_sampled_length: bool | str = False
    force_pad_after_eos: bool = False

    @property
    def condition_columns(self) -> tuple[str, ...]:
        return self.foreign_key_columns + self.datetime_columns

    @property
    def target_columns(self) -> tuple[str, ...]:
        return self.categorical_targets + self.numerical_targets + self.text_targets

    @property
    def model_categorical_targets(self) -> tuple[str, ...]:
        return self.categorical_targets + self.auxiliary_categorical_targets

    @property
    def model_target_columns(self) -> tuple[str, ...]:
        return self.model_categorical_targets + self.numerical_targets + self.text_targets

    @property
    def length_bucket_targets(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in self.model_categorical_targets
            if column in {"summary_length_bucket", "review_text_length_bucket"}
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self.condition_columns + self.target_columns

    def text_column_for_length_bucket(self, bucket_column: str) -> str:
        if bucket_column == "summary_length_bucket":
            return "summary" if "summary" in self.text_targets else self.text_targets[0]
        if bucket_column == "review_text_length_bucket":
            return "review_text"
        raise KeyError(f"Unsupported length bucket target: {bucket_column}")

    def buckets_for_length_bucket(self, bucket_column: str) -> dict[str, tuple[int, int]]:
        if bucket_column == "summary_length_bucket":
            return self.summary_length_buckets
        if bucket_column == "review_text_length_bucket":
            return self.review_text_length_buckets
        raise KeyError(f"Unsupported length bucket target: {bucket_column}")

    def validate(self) -> None:
        all_columns = list(self.condition_columns) + list(self.target_columns) + list(self.auxiliary_categorical_targets)
        repeated = sorted({col for col in all_columns if all_columns.count(col) > 1})
        if repeated:
            raise ValueError(f"Columns cannot appear in more than one schema role: {repeated}")
        forbidden = sorted(FORBIDDEN_ENGINEERED_FEATURES.intersection(all_columns))
        if forbidden:
            raise ValueError(f"Engineered feature columns are forbidden in Conditional TABDLM: {forbidden}")
        if not self.foreign_key_columns:
            raise ValueError("At least one foreign key condition column is required")
        if not self.datetime_columns:
            raise ValueError("At least one datetime condition column is required")
        if not self.target_columns:
            raise ValueError("At least one generated target column is required")
        for column in self.text_targets:
            if int(self.text_max_lengths.get(column, 0)) <= 0:
                raise ValueError(f"Missing positive text_max_length for text target {column!r}")
        if self.summary_length_enabled:
            if "summary_length_bucket" not in self.auxiliary_categorical_targets:
                raise ValueError("summary_length.enabled requires auxiliary categorical target summary_length_bucket")
            if not self.summary_length_buckets:
                raise ValueError("summary_length.enabled requires non-empty buckets")
        if "review_text_length_bucket" in self.auxiliary_categorical_targets:
            if "review_text" not in self.text_targets:
                raise ValueError("review_text_length_bucket requires review_text in target text columns")
            if not self.review_text_length_buckets:
                raise ValueError("review_text_length_bucket requires non-empty review_text_length.buckets")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_columns": {
                "foreign_keys": list(self.foreign_key_columns),
                "datetimes": list(self.datetime_columns),
            },
            "target_columns": {
                "categorical": list(self.categorical_targets),
                "numerical": list(self.numerical_targets),
                "text": list(self.text_targets),
            },
            "auxiliary_targets": {
                "categorical": list(self.auxiliary_categorical_targets),
            },
            "text_max_lengths": dict(self.text_max_lengths),
            "summary_length": {
                "enabled": self.summary_length_enabled,
                "use_length_bucket_in_sampling": self.use_length_bucket_in_sampling,
                "force_eos_after_sampled_length": self.force_eos_after_sampled_length,
                "force_pad_after_eos": self.force_pad_after_eos,
                "buckets": {name: list(bounds) for name, bounds in self.summary_length_buckets.items()},
            },
            "review_text_length": {
                "enabled": "review_text_length_bucket" in self.auxiliary_categorical_targets,
                "use_length_bucket_in_sampling": self.use_length_bucket_in_sampling,
                "force_eos_after_sampled_length": self.force_eos_after_sampled_length,
                "force_pad_after_eos": self.force_pad_after_eos,
                "buckets": {name: list(bounds) for name, bounds in self.review_text_length_buckets.items()},
            },
            "forbidden_engineered_features": sorted(FORBIDDEN_ENGINEERED_FEATURES),
        }

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> "ConditionalTABDLMSchema":
        columns = config.get("columns", {})
        conditions = columns.get("condition", config.get("condition_columns", {}))
        targets = columns.get("target", config.get("target_columns", {}))
        auxiliary = config.get("auxiliary_targets", {})
        text_cfg = config.get("text", {})
        text_max = text_cfg.get("max_length", text_cfg.get("text_max_length", config.get("text_max_length", {})))
        summary_length_cfg = config.get("summary_length", {})
        buckets = summary_length_cfg.get("buckets", {})
        parsed_buckets = {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in buckets.items()
        }
        review_text_length_cfg = config.get("review_text_length", {})
        review_text_buckets = review_text_length_cfg.get("buckets", {})
        parsed_review_text_buckets = {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in review_text_buckets.items()
        }
        schema = cls(
            foreign_key_columns=tuple(str(col) for col in conditions.get("foreign_keys", [])),
            datetime_columns=tuple(str(col) for col in conditions.get("datetimes", [])),
            categorical_targets=tuple(str(col) for col in targets.get("categorical", [])),
            numerical_targets=tuple(str(col) for col in targets.get("numerical", [])),
            text_targets=tuple(str(col) for col in targets.get("text", [])),
            auxiliary_categorical_targets=tuple(str(col) for col in auxiliary.get("categorical", [])),
            text_max_lengths={str(k): int(v) for k, v in text_max.items()},
            summary_length_buckets=parsed_buckets,
            review_text_length_buckets=parsed_review_text_buckets,
            summary_length_enabled=bool(summary_length_cfg.get("enabled", False)),
            use_length_bucket_in_sampling=bool(summary_length_cfg.get("use_length_bucket_in_sampling", False)),
            force_eos_after_sampled_length=summary_length_cfg.get("force_eos_after_sampled_length", False),
            force_pad_after_eos=bool(summary_length_cfg.get("force_pad_after_eos", False)),
        )
        schema.validate()
        return schema


@dataclass
class ConditionalTABDLMConfig:
    """Resolved experiment config plus parsed schema."""

    raw: dict[str, Any]
    schema: ConditionalTABDLMSchema
    config_path: Path | None = None

    @property
    def train_data_path(self) -> Path:
        return Path(self.raw["paths"]["train_data_path"])

    @property
    def synthetic_spine_path(self) -> Path:
        return Path(self.raw["paths"]["synthetic_spine_path"])

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["paths"]["output_dir"])

    @property
    def data_dir(self) -> Path:
        return self.output_dir / "data"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.raw)
        data["schema_resolved"] = self.schema.to_dict()
        if self.config_path is not None:
            data["config_path"] = str(self.config_path)
        return data


def load_config(path: str | Path) -> ConditionalTABDLMConfig:
    path = Path(path)
    raw = resolve_auto_review_text_config(load_yaml(path))
    schema = ConditionalTABDLMSchema.from_config_dict(raw)
    return ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=path)


def resolve_auto_review_text_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve v4 review_text max length and quantile buckets from train data."""

    raw = copy.deepcopy(raw_config)
    targets = raw.get("columns", {}).get("target", raw.get("target_columns", {}))
    text_targets = {str(column) for column in targets.get("text", [])}
    auxiliary = {str(column) for column in raw.get("auxiliary_targets", {}).get("categorical", [])}
    review_cfg = dict(raw.get("review_text", {}) or {})
    text_cfg = raw.setdefault("text", {})
    max_lengths = text_cfg.setdefault("max_length", text_cfg.get("text_max_length", raw.get("text_max_length", {})) or {})
    wants_auto = str(review_cfg.get("max_tokens", "")).lower() == "auto" or str(max_lengths.get("review_text", "")).lower() == "auto"
    if not wants_auto:
        return raw
    if "review_text" not in text_targets:
        raise ValueError("review_text.max_tokens=auto requires review_text in target text columns")
    if "review_text_length_bucket" not in auxiliary:
        raise ValueError("review_text.max_tokens=auto requires auxiliary target review_text_length_bucket")
    train_path = Path(raw.get("paths", {}).get("train_data_path", ""))
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot resolve review_text max_tokens=auto; missing train_data_path: {train_path}")
    content_lengths = review_text_content_lengths_from_csv(
        train_path,
        chunk_size=int(raw.get("training", {}).get("auto_text_length_chunk_size", 500_000)),
    )
    if content_lengths.size == 0:
        raise ValueError("Cannot resolve review_text max_tokens=auto from empty review_text column")
    token_lengths = content_lengths + 2
    strategy = str(review_cfg.get("max_tokens_strategy", "max_if_feasible_else_p99"))
    max_feasible = int(review_cfg.get("max_feasible_tokens", 512))
    min_coverage = float(review_cfg.get("min_coverage_rate", 0.99))
    observed_max = int(token_lengths.max())
    sorted_lengths = np.sort(token_lengths)
    if observed_max <= max_feasible:
        cap = observed_max
        cap_source = "max"
    else:
        index = int(np.ceil(min_coverage * len(sorted_lengths))) - 1
        index = int(np.clip(index, 0, len(sorted_lengths) - 1))
        quantile_cap = int(sorted_lengths[index])
        if strategy == "max":
            cap = observed_max
            cap_source = "max_explicit"
        elif quantile_cap <= max_feasible:
            cap = quantile_cap
            cap_source = f"p{int(round(min_coverage * 100))}"
        else:
            cap = max_feasible
            cap_source = f"max_feasible_under_p{int(round(min_coverage * 100))}"
    cap = max(3, int(cap))
    max_content = max(1, cap - 2)
    clipped_content_lengths = np.minimum(content_lengths, max_content)
    coverage = float(np.mean(token_lengths <= cap))
    truncation = float(np.mean(token_lengths > cap))
    length_stats = review_text_length_stats(content_lengths, token_lengths)
    buckets, distribution = quantile_length_buckets(clipped_content_lengths, max_content)

    max_lengths["review_text"] = int(cap)
    review_length_cfg = raw.setdefault("review_text_length", {})
    review_length_cfg["enabled"] = True
    review_length_cfg.setdefault("use_length_bucket_in_sampling", True)
    review_length_cfg.setdefault("calibrate_length_bucket_sampling", True)
    review_length_cfg["buckets"] = {name: [int(low), int(high)] for name, (low, high) in buckets.items()}
    review_length_cfg["bucket_distribution_real"] = distribution
    review_cfg.update(
        {
            "max_tokens": "auto",
            "resolved_max_tokens": int(cap),
            "max_tokens_strategy": strategy,
            "max_feasible_tokens": int(max_feasible),
            "min_coverage_rate": float(min_coverage),
            "length_cap_source": cap_source,
            "truncation_rate_train": truncation,
            "coverage_rate_train": coverage,
            "length_stats_real": length_stats,
        }
    )
    raw["review_text"] = review_cfg
    raw.setdefault("_auto_text_length_metadata", {})["review_text"] = {
        "review_text_max_tokens": int(cap),
        "review_text_max_content_tokens": int(max_content),
        "review_text_max_tokens_strategy": strategy,
        "review_text_length_cap_source": cap_source,
        "review_text_length_stats_real": length_stats,
        "review_text_truncation_rate_train": truncation,
        "review_text_coverage_rate_train": coverage,
        "review_text_length_bucket_edges": {name: list(bounds) for name, bounds in buckets.items()},
        "review_text_length_bucket_distribution_real": distribution,
    }
    return raw


def review_text_content_lengths_from_csv(train_path: Path, chunk_size: int = 500_000) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for chunk in pd.read_csv(train_path, usecols=["review_text"], chunksize=int(chunk_size), low_memory=False):
        normalized = chunk["review_text"].map(normalize_text)
        normalized = normalized[normalized.str.len() > 0]
        if len(normalized):
            pieces.append(normalized.map(lambda text: len(str(text).split())).to_numpy(dtype=np.int64))
    if not pieces:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(pieces).astype(np.int64, copy=False)


def review_text_length_stats(content_lengths: np.ndarray, token_lengths: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for prefix, values in [("content", content_lengths), ("token", token_lengths)]:
        stats[f"{prefix}_mean"] = float(np.mean(values))
        stats[f"{prefix}_median"] = float(np.median(values))
        stats[f"{prefix}_p90"] = float(np.quantile(values, 0.90))
        stats[f"{prefix}_p95"] = float(np.quantile(values, 0.95))
        stats[f"{prefix}_p99"] = float(np.quantile(values, 0.99))
        stats[f"{prefix}_max"] = int(np.max(values))
    return stats


def quantile_length_buckets(
    content_lengths: np.ndarray,
    max_content_tokens: int,
) -> tuple[dict[str, tuple[int, int]], dict[str, float]]:
    names = ["q0_q20", "q20_q40", "q40_q60", "q60_q80", "q80_q90", "q90_q95", "q95_q99", "q99_max"]
    quantiles = [0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99, 1.0]
    values = np.asarray(content_lengths, dtype=np.int64)
    buckets: dict[str, tuple[int, int]] = {}
    previous_high = -1
    for name, q in zip(names, quantiles):
        if q >= 1.0:
            high = int(max_content_tokens)
        else:
            high = int(np.ceil(float(np.quantile(values, q))))
            high = min(int(max_content_tokens), max(high, previous_high))
        low = max(0, previous_high + 1)
        if high < low:
            high = low
        buckets[name] = (int(low), int(high))
        previous_high = int(high)
    buckets[names[-1]] = (buckets[names[-1]][0], int(max_content_tokens))
    assignments = [bucket_name_for_length(int(length), buckets) for length in values]
    counts = pd.Series(assignments).value_counts(normalize=True).reindex(names, fill_value=0.0)
    return buckets, {str(name): float(counts.loc[name]) for name in names}


def bucket_name_for_length(content_length: int, buckets: dict[str, tuple[int, int]]) -> str:
    for name, (low, high) in buckets.items():
        if int(low) <= int(content_length) <= int(high):
            return str(name)
    return list(buckets)[-1]
