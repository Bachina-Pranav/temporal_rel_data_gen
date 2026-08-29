from __future__ import annotations

import math

import numpy as np

from scripts.summarize_qwen_text_decoder_main import sanitize_json


def test_sanitize_json_converts_all_nonfinite_values_to_null():
    value = {
        "nan": float("nan"),
        "positive_infinity": float("inf"),
        "negative_infinity": np.float64(-math.inf),
        "nested": [1.0, np.float32("nan")],
    }
    assert sanitize_json(value) == {
        "nan": None,
        "positive_infinity": None,
        "negative_infinity": None,
        "nested": [1.0, None],
    }
