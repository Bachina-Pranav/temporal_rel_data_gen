"""Schema metadata helpers for interaction benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SEMANTIC_TYPES = {
    "categorical",
    "ordinal_categorical",
    "boolean",
    "continuous_numerical",
    "count_numerical",
    "text",
    "datetime",
    "foreign_key",
}


def validate_schema_metadata(schema: dict[str, Any]) -> None:
    fields = schema.get("fields", {})
    if not isinstance(fields, dict) or not fields:
        raise ValueError("schema metadata requires non-empty fields mapping")
    for column, meta in fields.items():
        semantic = meta.get("semantic_type")
        if semantic not in SEMANTIC_TYPES:
            raise ValueError(f"Unsupported semantic_type for {column!r}: {semantic!r}")
    generated = set(schema.get("generated_attributes", []))
    missing = sorted(generated.difference(fields))
    if missing:
        raise ValueError(f"Generated fields missing field metadata: {missing}")


def write_schema_yaml(schema: dict[str, Any], path: str | Path) -> Path:
    validate_schema_metadata(schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(schema, handle, sort_keys=False)
    return path
