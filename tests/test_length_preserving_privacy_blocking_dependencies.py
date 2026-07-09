from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_fast_sampler import (  # noqa: E402
    TextFieldPolicy,
    topological_text_field_order,
)


def test_dependency_order_finalizes_parent_before_dependent():
    policies = [
        TextFieldPolicy(name="body", target_column="body", conditions_on=("title",)),
        TextFieldPolicy(name="title", target_column="title", downstream_dependents=("body",)),
    ]

    ordered = topological_text_field_order(policies)

    assert [policy.target_column for policy in ordered] == ["title", "body"]


def test_dependency_order_detects_cycles():
    policies = [
        TextFieldPolicy(name="a", target_column="a", conditions_on=("b",)),
        TextFieldPolicy(name="b", target_column="b", conditions_on=("a",)),
    ]

    try:
        topological_text_field_order(policies)
    except ValueError as exc:
        assert "Cycle" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected dependency cycle to raise")
