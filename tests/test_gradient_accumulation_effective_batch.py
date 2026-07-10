from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.lstm_joint import gradient_accumulation_steps_for  # noqa: E402


def test_gradient_accumulation_preserves_target_effective_batch():
    training = {"target_effective_batch_size": 512}

    steps = gradient_accumulation_steps_for(training, physical_batch_size=64)

    assert steps == 8
    assert steps * 64 == 512


def test_explicit_gradient_accumulation_wins():
    training = {"target_effective_batch_size": 512, "gradient_accumulation_steps": 4}

    assert gradient_accumulation_steps_for(training, physical_batch_size=64) == 4
