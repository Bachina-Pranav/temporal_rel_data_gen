from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from profile_lstm_joint_training_step import bottleneck_guess  # noqa: E402


def test_profiler_bottleneck_guess_uses_largest_timing_component():
    runtime = {
        "avg_batch_load_seconds": 0.1,
        "avg_h2d_seconds": 0.01,
        "avg_graph_context_seconds": 0.4,
        "avg_forward_seconds": 0.2,
        "avg_backward_seconds": 0.3,
        "avg_optimizer_seconds": 0.05,
    }

    assert bottleneck_guess(runtime) == "graph_context"
