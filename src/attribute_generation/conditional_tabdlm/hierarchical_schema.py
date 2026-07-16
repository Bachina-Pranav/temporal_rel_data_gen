"""Schema-driven generation stages for hierarchical Conditional TABDLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import ConditionalTABDLMSchema


RESERVED_DEPENDENCIES = {"event_context", "graph_context"}


@dataclass(frozen=True)
class GenerationStage:
    name: str
    fields: tuple[str, ...]
    condition_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationPlan:
    stages: tuple[GenerationStage, ...]

    @property
    def structured_fields(self) -> tuple[str, ...]:
        return self.stage_fields_by_role("structured")

    @property
    def text_fields(self) -> tuple[str, ...]:
        return tuple(
            field
            for stage in self.stages
            for field in stage.fields
            if stage.name == "text"
        )

    def stage_fields_by_role(self, role: str) -> tuple[str, ...]:
        return tuple(
            field
            for stage in self.stages
            for field in stage.fields
            if stage.name == role
        )

    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [
                {"name": stage.name, "fields": list(stage.fields), "condition_on": list(stage.condition_on)}
                for stage in self.stages
            ]
        }


def generation_plan_from_config(raw_config: dict[str, Any], schema: ConditionalTABDLMSchema) -> GenerationPlan:
    generation = raw_config.get("generation") or {}
    stage_cfg = generation.get("stages")
    if not stage_cfg:
        stage_cfg = [
            {"name": "structured", "fields": list(schema.model_categorical_targets), "condition_on": ["event_context", "graph_context"]},
            {"name": "text", "fields": list(schema.text_targets), "condition_on": ["structured", "event_context", "graph_context"]},
        ]
    stages = tuple(
        GenerationStage(
            name=str(item["name"]),
            fields=tuple(str(field) for field in item.get("fields", [])),
            condition_on=tuple(str(dep) for dep in item.get("condition_on", [])),
        )
        for item in stage_cfg
    )
    plan = GenerationPlan(stages=stages)
    validate_generation_plan(plan, raw_config, schema)
    return plan


def validate_generation_plan(plan: GenerationPlan, raw_config: dict[str, Any], schema: ConditionalTABDLMSchema) -> None:
    if not plan.stages:
        raise ValueError("generation.stages must contain at least one stage")
    generated_fields = set(schema.model_categorical_targets + schema.numerical_targets + schema.text_targets)
    stage_names = [stage.name for stage in plan.stages]
    if len(stage_names) != len(set(stage_names)):
        raise ValueError(f"generation.stages contains duplicate stage names: {stage_names}")

    assigned: list[str] = [field for stage in plan.stages for field in stage.fields]
    duplicates = sorted({field for field in assigned if assigned.count(field) > 1})
    if duplicates:
        raise ValueError(f"Generated fields must belong to exactly one stage; duplicated fields: {duplicates}")
    missing = sorted(generated_fields.difference(assigned))
    extra = sorted(set(assigned).difference(generated_fields))
    if missing:
        raise ValueError(f"Generated fields missing from generation.stages: {missing}")
    if extra:
        raise ValueError(f"generation.stages contains non-generated or conditioning fields: {extra}")

    seen = set(RESERVED_DEPENDENCIES)
    for stage in plan.stages:
        for dep in stage.condition_on:
            if dep not in seen:
                raise ValueError(f"Stage {stage.name!r} depends on {dep!r}, which is not an earlier stage or reserved context")
        seen.add(stage.name)

    field_meta = ((raw_config.get("schema") or {}).get("fields") or {})
    for column in schema.text_targets:
        length_field = length_field_for_text(schema, column, field_meta)
        if length_field is None:
            raise ValueError(f"Text field {column!r} must declare or infer a length mechanism")
        if length_field not in schema.model_categorical_targets:
            raise ValueError(f"Text field {column!r} length field {length_field!r} is not a model categorical target")


def length_field_for_text(schema: ConditionalTABDLMSchema, text_column: str, field_meta: dict[str, Any] | None = None) -> str | None:
    field_meta = field_meta or {}
    configured = (field_meta.get(text_column) or {}).get("length_field")
    if configured:
        return str(configured)
    for bucket_column in schema.length_bucket_targets:
        try:
            if schema.text_column_for_length_bucket(bucket_column) == text_column:
                return bucket_column
        except KeyError:
            continue
    return None

