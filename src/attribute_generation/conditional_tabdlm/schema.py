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
    text_max_lengths: dict[str, int] = field(default_factory=dict)

    @property
    def condition_columns(self) -> tuple[str, ...]:
        return self.foreign_key_columns + self.datetime_columns

    @property
    def target_columns(self) -> tuple[str, ...]:
        return self.categorical_targets + self.numerical_targets + self.text_targets

    @property
    def required_columns(self) -> tuple[str, ...]:
        return self.condition_columns + self.target_columns

    def validate(self) -> None:
        all_columns = list(self.condition_columns) + list(self.target_columns)
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
            "text_max_lengths": dict(self.text_max_lengths),
            "forbidden_engineered_features": sorted(FORBIDDEN_ENGINEERED_FEATURES),
        }

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> "ConditionalTABDLMSchema":
        columns = config.get("columns", {})
        conditions = columns.get("condition", {})
        targets = columns.get("target", {})
        text_cfg = config.get("text", {})
        text_max = text_cfg.get("max_length", {})
        schema = cls(
            foreign_key_columns=tuple(str(col) for col in conditions.get("foreign_keys", [])),
            datetime_columns=tuple(str(col) for col in conditions.get("datetimes", [])),
            categorical_targets=tuple(str(col) for col in targets.get("categorical", [])),
            numerical_targets=tuple(str(col) for col in targets.get("numerical", [])),
            text_targets=tuple(str(col) for col in targets.get("text", [])),
            text_max_lengths={str(k): int(v) for k, v in text_max.items()},
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

