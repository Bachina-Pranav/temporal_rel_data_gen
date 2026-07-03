"""Schema and config handling for conditional TABDLM experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    summary_length_enabled: bool = False
    use_length_bucket_in_sampling: bool = False
    force_eos_after_sampled_length: bool = False
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
    def required_columns(self) -> tuple[str, ...]:
        return self.condition_columns + self.target_columns

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
        schema = cls(
            foreign_key_columns=tuple(str(col) for col in conditions.get("foreign_keys", [])),
            datetime_columns=tuple(str(col) for col in conditions.get("datetimes", [])),
            categorical_targets=tuple(str(col) for col in targets.get("categorical", [])),
            numerical_targets=tuple(str(col) for col in targets.get("numerical", [])),
            text_targets=tuple(str(col) for col in targets.get("text", [])),
            auxiliary_categorical_targets=tuple(str(col) for col in auxiliary.get("categorical", [])),
            text_max_lengths={str(k): int(v) for k, v in text_max.items()},
            summary_length_buckets=parsed_buckets,
            summary_length_enabled=bool(summary_length_cfg.get("enabled", False)),
            use_length_bucket_in_sampling=bool(summary_length_cfg.get("use_length_bucket_in_sampling", False)),
            force_eos_after_sampled_length=bool(summary_length_cfg.get("force_eos_after_sampled_length", False)),
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
    raw = load_yaml(path)
    schema = ConditionalTABDLMSchema.from_config_dict(raw)
    return ConditionalTABDLMConfig(raw=raw, schema=schema, config_path=path)
