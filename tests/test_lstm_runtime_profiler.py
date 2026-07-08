from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.runtime_profiler import RuntimeProfiler  # noqa: E402


def test_lstm_runtime_profiler_records_timings_and_rates():
    profiler = RuntimeProfiler(enabled=True)
    profiler.start_total()
    with profiler.timer("review_text_decoding_seconds"):
        time.sleep(0.001)
    profiler.stop_total()

    summary = profiler.summary(
        rows_generated=10,
        num_batches=2,
        batch_size_requested=8,
        batch_size_used=5,
        auto_batch_size_enabled=True,
        summary_lengths=[1, 2],
        review_text_lengths=[3, 4, 10],
        device="cpu",
        mixed_precision_used=False,
        dtype_used="float32",
        torch_compile_used=False,
    )

    assert summary["total_sampling_seconds"] > 0
    assert summary["review_text_decoding_seconds"] > 0
    assert summary["rows_per_second"] > 0
    assert summary["p95_review_text_tokens_generated"] is not None
    assert profiler.events
