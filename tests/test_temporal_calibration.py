from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from reldiff.attributes.temporal_calibration import (  # noqa: E402
    calibrate_rating_logits_np,
    js_divergence_probs,
    softmax_np,
)


def test_temporal_calibration_moves_distribution_toward_target():
    logits = np.tile(np.asarray([[3.0, 0.0, -2.0]]), (100, 1))
    target = np.asarray([0.1, 0.2, 0.7])
    before = softmax_np(logits).mean(axis=0)
    calibrated, correction = calibrate_rating_logits_np(logits, target, strength=0.75)
    after = softmax_np(calibrated).mean(axis=0)
    assert np.linalg.norm(after - target) < np.linalg.norm(before - target)
    assert js_divergence_probs(after, target) < js_divergence_probs(before, target)
    assert correction.shape == (3,)
