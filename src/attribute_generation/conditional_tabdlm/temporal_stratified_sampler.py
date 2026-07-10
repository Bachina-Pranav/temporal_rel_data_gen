"""Reusable row samplers for fixed-step temporal attribute training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


@dataclass
class TemporalSamplerDiagnostics:
    num_rows: int
    timestamp_column: str | None
    num_time_bins: int
    bin_counts: list[int]
    min_bin_count: int
    max_bin_count: int
    sampled_bin_histogram_after_training: list[int] = field(default_factory=list)
    sampled_time_range: dict[str, int | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "num_rows": int(self.num_rows),
            "timestamp_column": self.timestamp_column,
            "num_time_bins": int(self.num_time_bins),
            "bin_counts": [int(value) for value in self.bin_counts],
            "min_bin_count": int(self.min_bin_count),
            "max_bin_count": int(self.max_bin_count),
            "sampled_bin_histogram_after_training": [
                int(value) for value in self.sampled_bin_histogram_after_training
            ],
            "sampled_time_range": dict(self.sampled_time_range),
        }


class TemporalStratifiedSampler(Sampler[int]):
    """Sample row indices from timestamp bins for fixed-step training.

    The sampler is intentionally data-agnostic: it only needs an array of
    timestamps aligned with the dataset rows. It can produce a finite stream for
    a fixed number of optimizer steps, or an effectively infinite stream when
    ``num_samples`` is omitted.
    """

    def __init__(
        self,
        timestamps_ns: np.ndarray,
        *,
        mode: str = "temporal_stratified",
        num_time_bins: int = 128,
        binning: str = "quantile",
        replacement: bool = True,
        seed: int = 42,
        num_samples: int | None = None,
        timestamp_column: str | None = None,
    ):
        timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        if timestamps.ndim != 1:
            raise ValueError("timestamps_ns must be a 1D array")
        if timestamps.size == 0:
            raise ValueError("TemporalStratifiedSampler needs at least one row")
        self.timestamps_ns = timestamps
        self.mode = str(mode)
        self.num_time_bins = max(1, int(num_time_bins))
        self.binning = str(binning)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.num_samples = None if num_samples is None else max(0, int(num_samples))
        self.timestamp_column = timestamp_column
        self.bin_ids = assign_time_bins(timestamps, self.num_time_bins, self.binning)
        self.nonempty_bins = np.asarray(sorted(np.unique(self.bin_ids)), dtype=np.int64)
        self.indices_by_bin = {
            int(bin_id): np.flatnonzero(self.bin_ids == int(bin_id)).astype(np.int64)
            for bin_id in self.nonempty_bins
        }
        self.bin_counts = np.asarray(
            [len(self.indices_by_bin[int(bin_id)]) for bin_id in self.nonempty_bins],
            dtype=np.int64,
        )
        self.sampled_bin_histogram = np.zeros(int(self.nonempty_bins.size), dtype=np.int64)
        self._bin_to_hist_pos = {int(bin_id): idx for idx, bin_id in enumerate(self.nonempty_bins.tolist())}
        self._sampled_min_ts: int | None = None
        self._sampled_max_ts: int | None = None

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        produced = 0
        while self.num_samples is None or produced < self.num_samples:
            if self.mode == "uniform":
                row = int(rng.integers(0, len(self.timestamps_ns)))
                bin_id = int(self.bin_ids[row])
            elif self.mode == "temporal_weighted":
                pos = int(rng.choice(len(self.nonempty_bins), p=self.bin_counts / self.bin_counts.sum()))
                bin_id = int(self.nonempty_bins[pos])
                row = int(rng.choice(self.indices_by_bin[bin_id]))
            elif self.mode == "hybrid":
                if float(rng.random()) < 0.5:
                    row = int(rng.integers(0, len(self.timestamps_ns)))
                    bin_id = int(self.bin_ids[row])
                else:
                    bin_id = int(rng.choice(self.nonempty_bins))
                    row = int(rng.choice(self.indices_by_bin[bin_id]))
            elif self.mode == "temporal_stratified":
                bin_id = int(rng.choice(self.nonempty_bins))
                row = int(rng.choice(self.indices_by_bin[bin_id]))
            else:
                raise ValueError(f"Unknown sampling mode: {self.mode!r}")
            pos = self._bin_to_hist_pos[int(bin_id)]
            self.sampled_bin_histogram[pos] += 1
            ts = int(self.timestamps_ns[row])
            self._sampled_min_ts = ts if self._sampled_min_ts is None else min(self._sampled_min_ts, ts)
            self._sampled_max_ts = ts if self._sampled_max_ts is None else max(self._sampled_max_ts, ts)
            produced += 1
            yield row

    def __len__(self) -> int:
        if self.num_samples is None:
            return len(self.timestamps_ns)
        return int(self.num_samples)

    def diagnostics(self) -> TemporalSamplerDiagnostics:
        return TemporalSamplerDiagnostics(
            num_rows=int(len(self.timestamps_ns)),
            timestamp_column=self.timestamp_column,
            num_time_bins=int(len(self.nonempty_bins)),
            bin_counts=[int(value) for value in self.bin_counts.tolist()],
            min_bin_count=int(self.bin_counts.min()) if self.bin_counts.size else 0,
            max_bin_count=int(self.bin_counts.max()) if self.bin_counts.size else 0,
            sampled_bin_histogram_after_training=[
                int(value) for value in self.sampled_bin_histogram.tolist()
            ],
            sampled_time_range={
                "min_timestamp_ns": self._sampled_min_ts,
                "max_timestamp_ns": self._sampled_max_ts,
            },
        )


def assign_time_bins(timestamps_ns: np.ndarray, num_time_bins: int, binning: str = "quantile") -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    bins = max(1, min(int(num_time_bins), int(timestamps.size)))
    if bins == 1 or np.all(timestamps == timestamps[0]):
        return np.zeros(len(timestamps), dtype=np.int64)
    if str(binning) == "quantile":
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(timestamps, quantiles)
        edges = np.unique(edges.astype(np.int64))
        if edges.size <= 2:
            return equal_width_bins(timestamps, bins)
        return np.searchsorted(edges[1:-1], timestamps, side="right").astype(np.int64)
    if str(binning) == "equal_width":
        return equal_width_bins(timestamps, bins)
    raise ValueError(f"Unknown temporal binning strategy: {binning!r}")


def equal_width_bins(timestamps: np.ndarray, bins: int) -> np.ndarray:
    low = int(np.min(timestamps))
    high = int(np.max(timestamps))
    if high <= low:
        return np.zeros(len(timestamps), dtype=np.int64)
    edges = np.linspace(low, high, int(bins) + 1)
    return np.searchsorted(edges[1:-1], timestamps, side="right").astype(np.int64)
