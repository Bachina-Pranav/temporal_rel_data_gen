"""Configuration schema for temporal interaction tables adapted to RelDiff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RelDiffDatasetConfig:
    key: str
    display_name: str
    interaction_path: Path
    source_entity_path: Path
    destination_entity_path: Path
    source_table: str
    destination_table: str
    interaction_table: str
    source_pk: str
    destination_pk: str
    source_fk: str
    destination_fk: str
    event_id: str
    timestamp: str
    numerical_attributes: tuple[str, ...]
    categorical_attributes: tuple[str, ...]
    ignored_attributes: tuple[str, ...]
    evaluation_config: Path
    timestamp_unit: str = "seconds"
    split_policy: str = "explicit_or_chronological_90_5_5"

    @property
    def semantic_columns(self) -> tuple[str, ...]:
        return (
            self.source_fk,
            self.destination_fk,
            self.timestamp,
            *self.numerical_attributes,
            *self.categorical_attributes,
        )

    @property
    def structured_attributes(self) -> tuple[str, ...]:
        return (*self.numerical_attributes, *self.categorical_attributes)

    def validate(self) -> None:
        if self.timestamp_unit != "seconds":
            raise ValueError("The baseline currently fixes timestamp_unit to 'seconds'")
        if self.split_policy != "explicit_or_chronological_90_5_5":
            raise ValueError(f"Unsupported split policy: {self.split_policy}")
        roles = [
            self.source_fk,
            self.destination_fk,
            self.timestamp,
            *self.numerical_attributes,
            *self.categorical_attributes,
            *self.ignored_attributes,
        ]
        duplicates = sorted({name for name in roles if roles.count(name) > 1})
        if duplicates:
            raise ValueError(f"Columns assigned to multiple roles: {duplicates}")
        if self.source_table == self.destination_table:
            raise ValueError("Source and destination tables must have distinct names")
        if self.interaction_table in {self.source_table, self.destination_table}:
            raise ValueError("Interaction table must have a distinct name")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "paths": {
                "interaction": str(self.interaction_path),
                "source_entity": str(self.source_entity_path),
                "destination_entity": str(self.destination_entity_path),
                "evaluation_config": str(self.evaluation_config),
            },
            "tables": {
                "source": self.source_table,
                "destination": self.destination_table,
                "interaction": self.interaction_table,
            },
            "columns": {
                "source_pk": self.source_pk,
                "destination_pk": self.destination_pk,
                "source_fk": self.source_fk,
                "destination_fk": self.destination_fk,
                "event_id": self.event_id,
                "timestamp": self.timestamp,
                "numerical_attributes": list(self.numerical_attributes),
                "categorical_attributes": list(self.categorical_attributes),
                "ignored_attributes": list(self.ignored_attributes),
            },
            "timestamp_encoding": {
                "unit": self.timestamp_unit,
                "origin": "training minimum only",
                "clipping": False,
            },
            "split_policy": self.split_policy,
        }


def load_dataset_config(path: str | Path) -> RelDiffDatasetConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paths = raw["paths"]
    tables = raw["tables"]
    columns = raw["columns"]
    config = RelDiffDatasetConfig(
        key=str(raw["key"]),
        display_name=str(raw["display_name"]),
        interaction_path=Path(paths["interaction"]),
        source_entity_path=Path(paths["source_entity"]),
        destination_entity_path=Path(paths["destination_entity"]),
        source_table=str(tables["source"]),
        destination_table=str(tables["destination"]),
        interaction_table=str(tables["interaction"]),
        source_pk=str(columns.get("source_pk", columns["source_fk"])),
        destination_pk=str(columns.get("destination_pk", columns["destination_fk"])),
        source_fk=str(columns["source_fk"]),
        destination_fk=str(columns["destination_fk"]),
        event_id=str(columns.get("event_id", "event_id")),
        timestamp=str(columns["timestamp"]),
        numerical_attributes=tuple(columns.get("numerical_attributes") or ()),
        categorical_attributes=tuple(columns.get("categorical_attributes") or ()),
        ignored_attributes=tuple(columns.get("ignored_attributes") or ()),
        evaluation_config=Path(paths["evaluation_config"]),
        timestamp_unit=str(raw.get("timestamp", {}).get("unit", "seconds")),
        split_policy=str(raw.get("split_policy", "explicit_or_chronological_90_5_5")),
    )
    config.validate()
    return config

