from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attribute_generation.conditional_tabdlm.schema import bucket_name_for_length, quantile_length_buckets  # noqa: E402


def test_review_text_quantile_buckets_cover_all_examples():
    lengths = np.asarray([0, 1, 2, 3, 5, 8, 13, 21, 34, 55], dtype=np.int64)
    buckets, distribution = quantile_length_buckets(lengths, max_content_tokens=60)

    assert list(buckets) == ["q0_q20", "q20_q40", "q40_q60", "q60_q80", "q80_q90", "q90_q95", "q95_q99", "q99_max"]
    for low, high in buckets.values():
        assert low <= high
    assigned = [bucket_name_for_length(int(length), buckets) for length in lengths]
    assert len(assigned) == len(lengths)
    assert abs(sum(distribution.values()) - 1.0) < 1e-9

