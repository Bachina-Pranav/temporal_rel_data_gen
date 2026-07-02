from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from textgen.text_privacy_metrics import compute_text_privacy_metrics  # noqa: E402


def test_text_privacy_metrics_detect_exact_and_normalized_copies():
    real = ["Great product!", "Works very well", "Terrible item"]
    synthetic = ["Great product!", "great product", "A new generated summary", "A new generated summary"]

    metrics = compute_text_privacy_metrics(real, synthetic, sample_size=10)

    assert metrics["exact_copy_rate"] == 0.25
    assert metrics["normalized_exact_copy_rate"] == 0.5
    assert metrics["duplicate_synthetic_rate"] == 0.5
    assert metrics["top_duplicate_summaries"][0]["summary"] == "A new generated summary"
