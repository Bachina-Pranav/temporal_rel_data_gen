"""Chronological sampling helpers for temporal attribute generation."""

from __future__ import annotations

from typing import Iterator, Tuple

import pandas as pd


def chronological_groups(
    df: pd.DataFrame,
    timestamp_col: str,
    mode: str = "date",
    window_days: float = 1.0,
) -> Iterator[Tuple[object, pd.DataFrame]]:
    """Yield groups whose rows should not condition on each other."""
    if mode not in {"date", "exact", "window"}:
        raise ValueError("mode must be date, exact, or window.")
    frame = df.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if mode == "date":
        key = frame[timestamp_col].dt.floor("D")
    elif mode == "exact":
        key = frame[timestamp_col]
    else:
        start = frame[timestamp_col].min()
        day_offsets = (frame[timestamp_col] - start).dt.total_seconds() / 86400.0
        key = (day_offsets // max(float(window_days), 1e-6)).astype(int)
    grouped = frame.assign(_sampling_time_group=key).sort_values(
        timestamp_col, kind="mergesort"
    )
    for group_key, group in grouped.groupby("_sampling_time_group", sort=True):
        yield group_key, group.drop(columns=["_sampling_time_group"])
