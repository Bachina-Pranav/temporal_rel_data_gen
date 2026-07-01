"""Temporal KDE-stub generator for review event spines.

This generator is the generative timestamp sibling of
``ct_2k_sbm_temporal_stubs``. It preserves customer/product endpoint stubs and
SBM block-pair event counts exactly, but samples timestamps from learned
block-pair temporal intensity models instead of reusing exact timestamp stubs.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .block_diagnostics import (
    block_pair_counts_frame,
    compute_all_block_diagnostics,
)
from .continuous_time_2k_sbm_plus import (
    count_correlation,
    total_variation,
)
from .continuous_time_temporal_sbm import (
    ContinuousKDESampler,
    ContinuousTimeTemporalSBMGenerator,
    DEFAULT_FALLBACK_BANDWIDTH,
    duplicate_pair_rate,
    empirical_ks_statistic,
    empirical_wasserstein_1d,
    estimate_bandwidth,
)


GENERATOR_NAME = "ct_2k_sbm_temporal_kde_stubs"

OccurrenceStub = Tuple[Any, float]
TimestampSample = Tuple[pd.Timestamp, float]


@dataclass
class TimestampGranularity:
    mode: str
    fraction_midnight: float


@dataclass
class TimestampSamplingResult:
    samples: list[TimestampSample]
    model_used: str
    source_level: str
    alpha: float | None = None
    bandwidth: float | None = None


class ContinuousTime2KSBMTemporalKDEStubsGenerator(ContinuousTimeTemporalSBMGenerator):
    """2K/SBM structural generator with generated timestamps."""

    def __init__(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        reviews: pd.DataFrame,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        seed: int = 42,
        sbm_block_level: Any = "auto",
        timestamp_model: str = "auto",
        timestamp_smoothing_alpha: Any = "auto",
        timestamp_bandwidth: Any = "scott",
        timestamp_min_block_count: int = 20,
        pairing_mode: str = "temporal_window_shuffle",
        temporal_window_size: int | None = None,
        avoid_real_edge_prob: float = 0.95,
        max_swap_attempts: int = 20,
    ):
        if timestamp_model not in {
            "auto",
            "smoothed_date_pmf",
            "block_pair_kde",
            "global_kde",
            "bootstrap_jitter",
        }:
            raise ValueError(
                "timestamp_model must be auto, smoothed_date_pmf, block_pair_kde, "
                "global_kde, or bootstrap_jitter."
            )
        if pairing_mode not in {"random", "temporal_sorted", "temporal_window_shuffle"}:
            raise ValueError(
                "pairing_mode must be random, temporal_sorted, or temporal_window_shuffle."
            )
        if temporal_window_size is not None and temporal_window_size <= 0:
            raise ValueError("temporal_window_size must be positive when provided.")
        if not 0.0 <= avoid_real_edge_prob <= 1.0:
            raise ValueError("avoid_real_edge_prob must be in [0, 1].")
        if max_swap_attempts <= 0:
            raise ValueError("max_swap_attempts must be positive.")

        self.timestamp_model_requested = timestamp_model
        self.timestamp_smoothing_alpha = timestamp_smoothing_alpha
        self.timestamp_bandwidth = timestamp_bandwidth
        self.timestamp_min_block_count = int(timestamp_min_block_count)
        self.pairing_mode = pairing_mode
        self.temporal_window_size = temporal_window_size
        self.avoid_real_edge_prob = float(avoid_real_edge_prob)
        self.max_swap_attempts = int(max_swap_attempts)

        super().__init__(
            customers=customers,
            products=products,
            reviews=reviews,
            customer_id_col=customer_id_col,
            product_id_col=product_id_col,
            timestamp_col=timestamp_col,
            seed=seed,
            sbm_block_level=sbm_block_level,
        )

        self.timestamp_granularity = detect_timestamp_granularity(
            self.reviews[self.timestamp_col]
        )
        self.timestamp_model_resolved = self._resolve_timestamp_model()
        self.date_grid = pd.date_range(
            self.min_time.floor("D"), self.max_time.floor("D"), freq="D"
        )
        self.timestamp_model_usage: Counter = Counter()
        self.timestamp_alpha_values: list[float] = []
        self.timestamp_bandwidth_values: list[float] = []
        self.edge_avoidance_stats: Counter = Counter()
        self.block_pair_timestamp_rows: list[dict[str, Any]] = []

    @classmethod
    def from_csv(
        cls,
        customers_path: str | Path,
        products_path: str | Path,
        reviews_path: str | Path,
        **kwargs: Any,
    ) -> "ContinuousTime2KSBMTemporalKDEStubsGenerator":
        return cls(
            customers=pd.read_csv(customers_path),
            products=pd.read_csv(products_path),
            reviews=pd.read_csv(reviews_path),
            **kwargs,
        )

    @classmethod
    def from_reviews(
        cls,
        reviews: pd.DataFrame,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        **kwargs: Any,
    ) -> "ContinuousTime2KSBMTemporalKDEStubsGenerator":
        customers = pd.DataFrame(
            {customer_id_col: pd.unique(reviews[customer_id_col].dropna())}
        )
        products = pd.DataFrame(
            {product_id_col: pd.unique(reviews[product_id_col].dropna())}
        )
        return cls(
            customers=customers,
            products=products,
            reviews=reviews,
            customer_id_col=customer_id_col,
            product_id_col=product_id_col,
            **kwargs,
        )

    def _resolve_timestamp_model(self) -> str:
        if self.timestamp_model_requested != "auto":
            return self.timestamp_model_requested
        if self.timestamp_granularity.mode == "date_only":
            return "smoothed_date_pmf"
        return "block_pair_kde"

    def _annotated_reviews(self) -> pd.DataFrame:
        assert self.sbm_result is not None
        reviews = self.reviews.copy()
        reviews["_customer_block"] = reviews[self.customer_id_col].map(
            self.sbm_result.customer_blocks
        )
        reviews["_product_block"] = reviews[self.product_id_col].map(
            self.sbm_result.product_blocks
        )
        return reviews

    def generate(
        self,
        output_path: str | Path | None = None,
        debug_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        if self.sbm_result is None:
            self.fit()

        self.timestamp_model_usage = Counter()
        self.timestamp_alpha_values = []
        self.timestamp_bandwidth_values = []
        self.edge_avoidance_stats = Counter()
        self.block_pair_timestamp_rows = []

        annotated = self._annotated_reviews()
        groups = {
            (int(customer_block), int(product_block)): group.copy()
            for (customer_block, product_block), group in annotated.groupby(
                ["_customer_block", "_product_block"], sort=True
            )
        }

        records: list[dict[str, Any]] = []
        with tqdm(
            total=len(self.reviews),
            desc="Generating ct_2k_sbm_temporal_kde_stubs events",
            unit="event",
        ) as pbar:
            for block_pair, group in sorted(groups.items()):
                block_records = self._generate_block_pair_records(block_pair, group)
                records.extend(block_records)
                pbar.update(len(block_records))

        synthetic_full = pd.DataFrame.from_records(records)
        synthetic_full = self._avoid_exact_timestamp_multiset(synthetic_full)
        synthetic_full = synthetic_full.sort_values(self.timestamp_col, kind="mergesort")
        output = synthetic_full[
            [self.customer_id_col, self.product_id_col, self.timestamp_col]
        ].reset_index(drop=True)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                "Writing ct_2k_sbm_temporal_kde_stubs synthetic event spine "
                f"to {output_path}..."
            )
            output.to_csv(output_path, index=False)

        if debug_dir is not None:
            self.write_kde_stubs_debug_outputs(Path(debug_dir), output)
        return output

    def _generate_block_pair_records(
        self, block_pair: tuple[int, int], group: pd.DataFrame
    ) -> list[dict[str, Any]]:
        count = len(group)
        customer_stubs = self._occurrence_stubs(group, self.customer_id_col)
        product_stubs = self._occurrence_stubs(group, self.product_id_col)
        timestamp_result = self._sample_timestamps_for_block_pair(block_pair, group)
        timestamp_samples = timestamp_result.samples

        if self.pairing_mode == "random":
            self.rng.shuffle(customer_stubs)
            self.rng.shuffle(product_stubs)
            self.rng.shuffle(timestamp_samples)
            window_size = count
        else:
            customer_stubs.sort(key=lambda stub: stub[1])
            product_stubs.sort(key=lambda stub: stub[1])
            timestamp_samples.sort(key=lambda sample: sample[1])
            window_size = self._effective_window_size(count)
            if self.pairing_mode == "temporal_window_shuffle":
                customer_stubs = self._shuffle_occurrences_in_windows(
                    customer_stubs, window_size
                )
                product_stubs = self._shuffle_occurrences_in_windows(
                    product_stubs, window_size
                )

        triples = [
            [customer_stub[0], product_stub[0], timestamp_sample]
            for customer_stub, product_stub, timestamp_sample in zip(
                customer_stubs, product_stubs, timestamp_samples
            )
        ]
        real_edges = set(
            map(
                tuple,
                group[[self.customer_id_col, self.product_id_col]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        self._avoid_real_edges_in_windows(triples, real_edges, window_size)
        self._record_block_pair_timestamp_diagnostics(
            block_pair, group, triples, timestamp_result
        )

        return [
            {
                self.customer_id_col: customer_id,
                self.product_id_col: product_id,
                self.timestamp_col: timestamp,
                "_customer_block": block_pair[0],
                "_product_block": block_pair[1],
                "_time_x": float(x),
            }
            for customer_id, product_id, (timestamp, x) in triples
        ]

    def _occurrence_stubs(self, group: pd.DataFrame, column: str) -> list[OccurrenceStub]:
        return [
            (entity_id, float(x))
            for entity_id, x in zip(group[column].to_numpy(dtype=object), group["_time_x"])
        ]

    def _sample_timestamps_for_block_pair(
        self, block_pair: tuple[int, int], group: pd.DataFrame
    ) -> TimestampSamplingResult:
        if self.timestamp_model_resolved == "smoothed_date_pmf":
            return self._sample_smoothed_dates(block_pair, group)
        return self._sample_kde_timestamps(block_pair, group)

    def _sample_smoothed_dates(
        self, block_pair: tuple[int, int], group: pd.DataFrame
    ) -> TimestampSamplingResult:
        count = len(group)
        alpha = self._resolve_alpha(count)
        parent_probs, source_level = self._date_parent_distribution(
            block_pair, prefer_parent=count < self.timestamp_min_block_count
        )
        real_counts = self._date_counts(group)
        probs = (real_counts + alpha * parent_probs) / (count + alpha)
        probs = normalize_probs(probs)
        date_values = self.date_grid.to_numpy(dtype="datetime64[ns]")
        sampled = self.rng.choice(date_values, size=count, replace=True, p=probs)
        sampled_timestamps = [pd.Timestamp(value).floor("D") for value in sampled]
        sampled_timestamps = self._avoid_exact_group_dates(
            group[self.timestamp_col], sampled_timestamps
        )
        samples = [
            (timestamp, self._normalize_timestamp(timestamp))
            for timestamp in sampled_timestamps
        ]

        model_source = "block_pair" if count >= self.timestamp_min_block_count else source_level
        self.timestamp_model_usage[model_source] += 1
        self.timestamp_alpha_values.append(float(alpha))
        return TimestampSamplingResult(
            samples=samples,
            model_used="smoothed_date_pmf",
            source_level=model_source,
            alpha=float(alpha),
        )

    def _sample_kde_timestamps(
        self, block_pair: tuple[int, int], group: pd.DataFrame
    ) -> TimestampSamplingResult:
        count = len(group)
        collection, source_level = self._select_kde_collection(block_pair)
        bandwidth = self._resolve_bandwidth(collection.times)
        sampler = ContinuousKDESampler(collection.times, bandwidth)
        samples = []
        for _ in range(count):
            x = sampler.sample(self.rng)
            if self.timestamp_model_resolved == "bootstrap_jitter":
                x = self._bootstrap_jitter_sample(collection.times, bandwidth)
            timestamp = self.denormalize_time(float(x))
            if self.timestamp_granularity.mode == "date_only":
                timestamp = timestamp.floor("D")
                x = self._normalize_timestamp(timestamp)
            samples.append((pd.Timestamp(timestamp), float(x)))

        self.timestamp_model_usage[source_level] += 1
        self.timestamp_bandwidth_values.append(float(bandwidth))
        return TimestampSamplingResult(
            samples=samples,
            model_used=self.timestamp_model_resolved,
            source_level=source_level,
            bandwidth=float(bandwidth),
        )

    def _date_parent_distribution(
        self, block_pair: tuple[int, int], prefer_parent: bool
    ) -> tuple[np.ndarray, str]:
        annotated = self._annotated_reviews()
        customer_block, product_block = block_pair
        customer_group = annotated[annotated["_customer_block"] == customer_block]
        product_group = annotated[annotated["_product_block"] == product_block]
        if prefer_parent and len(customer_group) >= self.timestamp_min_block_count:
            return self._date_probs(customer_group), "customer_block"
        if prefer_parent and len(product_group) >= self.timestamp_min_block_count:
            return self._date_probs(product_group), "product_block"
        if prefer_parent:
            return self._date_probs(annotated), "global"
        if len(customer_group) > 0:
            return self._date_probs(customer_group), "customer_block"
        if len(product_group) > 0:
            return self._date_probs(product_group), "product_block"
        return self._date_probs(annotated), "global"

    def _date_counts(self, df: pd.DataFrame) -> np.ndarray:
        dates = pd.to_datetime(df[self.timestamp_col]).dt.floor("D")
        counts = dates.value_counts().reindex(self.date_grid, fill_value=0)
        return counts.to_numpy(dtype=float)

    def _date_probs(self, df: pd.DataFrame) -> np.ndarray:
        return normalize_probs(self._date_counts(df))

    def _select_kde_collection(self, block_pair: tuple[int, int]) -> tuple[Any, str]:
        assert self.global_events is not None
        if self.timestamp_model_resolved == "global_kde":
            return self.global_events, "global"

        customer_block, product_block = block_pair
        candidates = [
            (self.block_pair_events.get(block_pair), "block_pair"),
            (self.customer_block_events.get(customer_block), "customer_block"),
            (self.product_block_events.get(product_block), "product_block"),
            (self.global_events, "global"),
        ]
        for collection, source_level in candidates:
            if collection is not None and len(collection) >= self.timestamp_min_block_count:
                return collection, source_level
        for collection, source_level in candidates:
            if collection is not None and len(collection) > 0:
                return collection, source_level
        return self.global_events, "global"

    def _resolve_alpha(self, count: int) -> float:
        if self.timestamp_smoothing_alpha == "auto":
            base = max(5.0, float(np.sqrt(max(count, 1))))
            if count < self.timestamp_min_block_count:
                return max(20.0, float(count), base)
            return base
        return float(self.timestamp_smoothing_alpha)

    def _resolve_bandwidth(self, times: np.ndarray) -> float:
        if isinstance(self.timestamp_bandwidth, (int, float)):
            return max(float(self.timestamp_bandwidth), 1e-4)
        mode = str(self.timestamp_bandwidth)
        if mode not in {"scott", "silverman"}:
            return max(float(mode), 1e-4)
        times = np.asarray(times, dtype=float)
        if len(times) < 3:
            return DEFAULT_FALLBACK_BANDWIDTH
        std = float(np.std(times))
        if std <= 1e-4:
            return DEFAULT_FALLBACK_BANDWIDTH
        if mode == "silverman":
            return max(0.9 * std * len(times) ** (-1 / 5), 1e-4)
        return estimate_bandwidth(times, DEFAULT_FALLBACK_BANDWIDTH)

    def _bootstrap_jitter_sample(self, times: np.ndarray, bandwidth: float) -> float:
        center = float(times[int(self.rng.integers(0, len(times)))])
        jitter = max(float(bandwidth) * 0.5, 1e-4)
        return ContinuousKDESampler(np.asarray([center]), jitter).sample(self.rng)

    def _avoid_exact_group_dates(
        self, real_times: pd.Series, sampled_timestamps: list[pd.Timestamp]
    ) -> list[pd.Timestamp]:
        if len(self.date_grid) <= 1:
            return sampled_timestamps
        real_counts = timestamp_counter(real_times, date_only=True)
        sample_counts = timestamp_counter(pd.Series(sampled_timestamps), date_only=True)
        if real_counts != sample_counts:
            return sampled_timestamps
        sampled_timestamps = list(sampled_timestamps)
        for index, timestamp in enumerate(sampled_timestamps):
            shifted = timestamp + pd.Timedelta(days=1)
            if shifted > self.max_time.floor("D"):
                shifted = timestamp - pd.Timedelta(days=1)
            if self.min_time.floor("D") <= shifted <= self.max_time.floor("D"):
                sampled_timestamps[index] = shifted
                break
        return sampled_timestamps

    def _avoid_exact_timestamp_multiset(self, synthetic: pd.DataFrame) -> pd.DataFrame:
        synthetic = synthetic.copy()
        date_only = self.timestamp_granularity.mode == "date_only"
        real_counts = timestamp_counter(self.reviews[self.timestamp_col], date_only=date_only)
        synthetic_counts = timestamp_counter(synthetic[self.timestamp_col], date_only=date_only)
        if real_counts != synthetic_counts or len(synthetic) == 0:
            return synthetic

        index = synthetic.index[0]
        timestamp = pd.Timestamp(synthetic.at[index, self.timestamp_col])
        if date_only:
            shifted = timestamp.floor("D") + pd.Timedelta(days=1)
            if shifted > self.max_time.floor("D"):
                shifted = timestamp.floor("D") - pd.Timedelta(days=1)
            if self.min_time.floor("D") <= shifted <= self.max_time.floor("D"):
                synthetic.at[index, self.timestamp_col] = shifted
                synthetic.at[index, "_time_x"] = self._normalize_timestamp(shifted)
        else:
            span_seconds = max(self.time_span.total_seconds(), 1.0)
            shifted = timestamp + pd.Timedelta(seconds=max(1.0, span_seconds * 1e-4))
            if shifted > self.max_time:
                shifted = timestamp - pd.Timedelta(seconds=max(1.0, span_seconds * 1e-4))
            shifted = min(max(shifted, self.min_time), self.max_time)
            synthetic.at[index, self.timestamp_col] = shifted
            synthetic.at[index, "_time_x"] = self._normalize_timestamp(shifted)
        return synthetic

    def _effective_window_size(self, count: int) -> int:
        if count <= 1:
            return 1
        if self.temporal_window_size is not None:
            return min(int(self.temporal_window_size), count)
        return min(count, max(10, int(np.sqrt(count))))

    def _shuffle_occurrences_in_windows(
        self, stubs: list[OccurrenceStub], window_size: int
    ) -> list[OccurrenceStub]:
        shuffled: list[OccurrenceStub] = []
        for start in range(0, len(stubs), max(window_size, 1)):
            window = list(stubs[start : start + max(window_size, 1)])
            self.rng.shuffle(window)
            shuffled.extend(window)
        return shuffled

    def _avoid_real_edges_in_windows(
        self,
        triples: list[list[Any]],
        real_edges: set[tuple[Any, Any]],
        window_size: int,
    ) -> None:
        before_overlap = sum(
            1 for customer_id, product_id, _ in triples if (customer_id, product_id) in real_edges
        )
        self.edge_avoidance_stats["overlap_before"] += before_overlap
        self.edge_avoidance_stats["total_pairs"] += len(triples)
        if self.avoid_real_edge_prob <= 0.0 or len(triples) < 2:
            self.edge_avoidance_stats["overlap_after"] += before_overlap
            return

        for start in range(0, len(triples), max(window_size, 1)):
            end = min(start + max(window_size, 1), len(triples))
            indices = list(range(start, end))
            if len(indices) < 2:
                continue
            for index in indices:
                customer_id, product_id, _ = triples[index]
                if (customer_id, product_id) not in real_edges:
                    continue
                if self.rng.random() >= self.avoid_real_edge_prob:
                    continue
                if self._try_local_swap(triples, real_edges, index, indices):
                    self.edge_avoidance_stats["successful_swaps"] += 1
                else:
                    self.edge_avoidance_stats["failed_swaps"] += 1

        after_overlap = sum(
            1 for customer_id, product_id, _ in triples if (customer_id, product_id) in real_edges
        )
        self.edge_avoidance_stats["overlap_after"] += after_overlap

    def _try_local_swap(
        self,
        triples: list[list[Any]],
        real_edges: set[tuple[Any, Any]],
        index: int,
        candidate_indices: list[int],
    ) -> bool:
        customer_id, product_id, _ = triples[index]
        candidates = list(candidate_indices)
        self.rng.shuffle(candidates)
        attempts = 0
        for candidate in candidates:
            if candidate == index:
                continue
            attempts += 1
            other_customer_id, other_product_id, _ = triples[candidate]
            if (
                (customer_id, other_product_id) not in real_edges
                and (other_customer_id, product_id) not in real_edges
            ):
                triples[index][1], triples[candidate][1] = (
                    triples[candidate][1],
                    triples[index][1],
                )
                return True
            if (
                (other_customer_id, product_id) not in real_edges
                and (customer_id, other_product_id) not in real_edges
            ):
                triples[index][0], triples[candidate][0] = (
                    triples[candidate][0],
                    triples[index][0],
                )
                return True
            if attempts >= self.max_swap_attempts:
                break
        return False

    def _record_block_pair_timestamp_diagnostics(
        self,
        block_pair: tuple[int, int],
        group: pd.DataFrame,
        triples: list[list[Any]],
        timestamp_result: TimestampSamplingResult,
    ) -> None:
        synthetic_times = pd.Series([sample[0] for _, _, sample in triples])
        real_times = pd.to_datetime(group[self.timestamp_col])
        real_days = timestamp_days(real_times)
        synthetic_days = timestamp_days(synthetic_times)
        date_only = self.timestamp_granularity.mode == "date_only"
        row = {
            "customer_block": block_pair[0],
            "product_block": block_pair[1],
            "real_event_count": int(len(group)),
            "synthetic_event_count": int(len(triples)),
            "timestamp_model_used": timestamp_result.model_used,
            "timestamp_source_level": timestamp_result.source_level,
            "timestamp_ks": empirical_ks_statistic(real_days, synthetic_days),
            "timestamp_wasserstein_days": empirical_wasserstein_1d(
                real_days, synthetic_days
            ),
            "real_min_timestamp": str(real_times.min()),
            "real_max_timestamp": str(real_times.max()),
            "synthetic_min_timestamp": str(synthetic_times.min()),
            "synthetic_max_timestamp": str(synthetic_times.max()),
            "real_unique_timestamp_count": int(real_times.nunique()),
            "synthetic_unique_timestamp_count": int(synthetic_times.nunique()),
            "timestamp_count_l1_by_date": timestamp_count_l1_by_date(
                real_times, synthetic_times
            )
            if date_only
            else None,
            "timestamp_count_corr_by_date": timestamp_count_correlation_by_date(
                real_times, synthetic_times
            )
            if date_only
            else None,
        }
        self.block_pair_timestamp_rows.append(row)

    def _normalize_timestamp(self, timestamp: pd.Timestamp) -> float:
        if self.time_span.total_seconds() <= 0:
            return 0.5
        x = (pd.Timestamp(timestamp) - self.min_time).total_seconds()
        return float(np.clip(x / self.time_span.total_seconds(), 0.0, 1.0))

    def write_kde_stubs_debug_outputs(
        self, debug_dir: Path, synthetic: pd.DataFrame
    ) -> None:
        assert self.sbm_result is not None
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing ct_2k_sbm_temporal_kde_stubs debug outputs to {debug_dir}...")

        block_diagnostics = compute_all_block_diagnostics(
            self.reviews,
            synthetic,
            self.sbm_result.customer_blocks,
            self.sbm_result.product_blocks,
            self.customer_id_col,
            self.product_id_col,
            self.timestamp_col,
            min_count=5,
        )
        timestamp_metrics = timestamp_generation_metrics(
            self.reviews[self.timestamp_col],
            synthetic[self.timestamp_col],
            date_only=self.timestamp_granularity.mode == "date_only",
        )
        summary = {
            "generator": GENERATOR_NAME,
            "num_real_reviews": int(len(self.reviews)),
            "num_synthetic_reviews": int(len(synthetic)),
            "num_active_customers_real": int(self.reviews[self.customer_id_col].nunique()),
            "num_active_customers_synthetic": int(synthetic[self.customer_id_col].nunique()),
            "num_active_products_real": int(self.reviews[self.product_id_col].nunique()),
            "num_active_products_synthetic": int(synthetic[self.product_id_col].nunique()),
            "sbm_block_level_requested": self.sbm_result.sbm_block_level_requested,
            "sbm_block_level_resolved": self.sbm_result.sbm_block_level_resolved,
            "sbm_block_level_recommended": self.sbm_result.sbm_block_level_recommended,
            "num_customer_blocks": int(self.sbm_result.num_customer_blocks),
            "num_product_blocks": int(self.sbm_result.num_product_blocks),
            "num_nonzero_block_pairs_real": int(
                block_diagnostics["num_nonzero_block_pairs_real"]
            ),
            "num_nonzero_block_pairs_synthetic": int(
                block_diagnostics["num_nonzero_block_pairs_synthetic"]
            ),
            "block_pair_count_exact_match_rate": block_diagnostics[
                "block_pair_count_exact_match_rate"
            ],
            "timestamp_model": self.timestamp_model_resolved,
            "timestamp_model_requested": self.timestamp_model_requested,
            "timestamp_smoothing_alpha_resolved": summarize_numeric(
                self.timestamp_alpha_values
            ),
            "timestamp_granularity_mode": self.timestamp_granularity.mode,
            "timestamp_midnight_fraction": self.timestamp_granularity.fraction_midnight,
            "timestamp_multiset_preserved_exactly": bool(
                timestamp_metrics["timestamp_multiset_exact_match"]
            ),
            "reuses_exact_timestamp_stubs": False,
            "pairing_mode": self.pairing_mode,
            "temporal_window_size": self.temporal_window_size,
            "avoid_real_edge_prob": self.avoid_real_edge_prob,
            "max_swap_attempts": self.max_swap_attempts,
            "edge_overlap_rate_before_avoidance": self._edge_overlap_rate("before"),
            "edge_overlap_rate_after_avoidance": self._edge_overlap_rate("after"),
            "num_successful_swaps": int(self.edge_avoidance_stats["successful_swaps"]),
            "num_failed_swaps": int(self.edge_avoidance_stats["failed_swaps"]),
            "seed": self.seed,
            **block_diagnostics,
            **timestamp_metrics,
        }
        self._write_json(debug_dir / "summary.json", summary)
        self._write_json(debug_dir / "timestamp_model_summary.json", self._timestamp_model_summary())
        self.write_sbm_summary(debug_dir / "sbm_summary.json")
        self.write_assignment_debug(debug_dir, synthetic)
        self.write_canonical_block_pair_counts(debug_dir, synthetic)
        pd.DataFrame(self.block_pair_timestamp_rows).to_csv(
            debug_dir / "block_pair_timestamp_diagnostics.csv", index=False
        )
        self._write_degree_checks(debug_dir, synthetic)

    def _timestamp_model_summary(self) -> dict[str, Any]:
        num_block_pairs = sum(
            self.timestamp_model_usage[level]
            for level in ("block_pair", "customer_block", "product_block", "global")
        )
        return {
            "timestamp_model": self.timestamp_model_resolved,
            "timestamp_granularity_mode": self.timestamp_granularity.mode,
            "num_block_pairs": int(num_block_pairs),
            "num_block_pairs_using_block_pair_model": int(
                self.timestamp_model_usage["block_pair"]
            ),
            "num_block_pairs_using_customer_block_fallback": int(
                self.timestamp_model_usage["customer_block"]
            ),
            "num_block_pairs_using_product_block_fallback": int(
                self.timestamp_model_usage["product_block"]
            ),
            "num_block_pairs_using_global_fallback": int(
                self.timestamp_model_usage["global"]
            ),
            "timestamp_smoothing_alpha_mode": self.timestamp_smoothing_alpha,
            "timestamp_smoothing_alpha_resolved": summarize_numeric(
                self.timestamp_alpha_values
            ),
            "timestamp_bandwidth_mode": self.timestamp_bandwidth,
            "timestamp_bandwidth_resolved": summarize_numeric(
                self.timestamp_bandwidth_values
            ),
            "timestamp_multiset_preserved_exactly": False,
        }

    def _edge_overlap_rate(self, stage: str) -> float:
        total = int(self.edge_avoidance_stats["total_pairs"])
        if total == 0:
            return 0.0
        key = "overlap_before" if stage == "before" else "overlap_after"
        return float(self.edge_avoidance_stats[key] / total)

    def _write_degree_checks(self, debug_dir: Path, synthetic: pd.DataFrame) -> None:
        for entity, column, block_map, filename in (
            (
                "customer",
                self.customer_id_col,
                self.sbm_result.customer_blocks,
                "ct_2k_sbm_temporal_kde_stubs_customer_degree_check.csv",
            ),
            (
                "product",
                self.product_id_col,
                self.sbm_result.product_blocks,
                "ct_2k_sbm_temporal_kde_stubs_product_degree_check.csv",
            ),
        ):
            real_counts = Counter(self.reviews[column])
            synthetic_counts = Counter(synthetic[column])
            rows = []
            for entity_id in sorted(block_map, key=str):
                real_count = int(real_counts.get(entity_id, 0))
                synthetic_count = int(synthetic_counts.get(entity_id, 0))
                rows.append(
                    {
                        f"{entity}_id": entity_id,
                        f"{entity}_block": block_map[entity_id],
                        "real_event_count": real_count,
                        "synthetic_event_count": synthetic_count,
                        "abs_error": abs(real_count - synthetic_count),
                    }
                )
            pd.DataFrame(rows).to_csv(debug_dir / filename, index=False)


def detect_timestamp_granularity(timestamps: pd.Series) -> TimestampGranularity:
    timestamps = pd.to_datetime(timestamps, errors="coerce").dropna()
    if timestamps.empty:
        return TimestampGranularity("datetime", 0.0)
    offsets = (timestamps - timestamps.dt.floor("D")).dt.total_seconds()
    fraction_midnight = float((offsets == 0).mean())
    mode = "date_only" if fraction_midnight >= 0.99 else "datetime"
    return TimestampGranularity(mode, fraction_midnight)


def normalize_probs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        return np.ones(len(values), dtype=float) / max(len(values), 1)
    return values / total


def timestamp_days(timestamps: pd.Series) -> np.ndarray:
    return pd.to_datetime(timestamps).astype("int64").to_numpy(dtype=float) / 1e9 / 86400.0


def timestamp_counter(timestamps: pd.Series, date_only: bool) -> Counter:
    values = pd.to_datetime(timestamps, errors="coerce").dropna()
    if date_only:
        values = values.dt.floor("D")
    return Counter(values.tolist())


def timestamp_count_l1_by_date(real_times: pd.Series, synthetic_times: pd.Series) -> float | None:
    real_counts = pd.to_datetime(real_times).dt.floor("D").value_counts()
    synthetic_counts = pd.to_datetime(synthetic_times).dt.floor("D").value_counts()
    index = real_counts.index.union(synthetic_counts.index)
    if len(index) == 0:
        return None
    total = int(real_counts.sum())
    if total == 0:
        return None
    diff = (
        real_counts.reindex(index, fill_value=0)
        - synthetic_counts.reindex(index, fill_value=0)
    ).abs()
    return float(diff.sum() / total)


def timestamp_count_correlation_by_date(
    real_times: pd.Series, synthetic_times: pd.Series
) -> float | None:
    real_counts = pd.to_datetime(real_times).dt.floor("D").value_counts()
    synthetic_counts = pd.to_datetime(synthetic_times).dt.floor("D").value_counts()
    index = real_counts.index.union(synthetic_counts.index)
    if len(index) < 2:
        return None
    real_values = real_counts.reindex(index, fill_value=0).to_numpy(dtype=float)
    synthetic_values = synthetic_counts.reindex(index, fill_value=0).to_numpy(dtype=float)
    if real_values.std() == 0 or synthetic_values.std() == 0:
        return None
    return float(np.corrcoef(real_values, synthetic_values)[0, 1])


def timestamp_multiset_overlap_rate(
    real_times: pd.Series, synthetic_times: pd.Series, date_only: bool
) -> float:
    real_counts = timestamp_counter(real_times, date_only=date_only)
    synthetic_counts = timestamp_counter(synthetic_times, date_only=date_only)
    total = sum(synthetic_counts.values())
    if total == 0:
        return 0.0
    overlap = 0
    for value, count in synthetic_counts.items():
        overlap += min(int(count), int(real_counts.get(value, 0)))
    return float(overlap / total)


def timestamp_generation_metrics(
    real_times: pd.Series, synthetic_times: pd.Series, date_only: bool
) -> dict[str, Any]:
    real_counts = timestamp_counter(real_times, date_only=date_only)
    synthetic_counts = timestamp_counter(synthetic_times, date_only=date_only)
    return {
        "timestamp_multiset_exact_match": real_counts == synthetic_counts,
        "timestamp_multiset_overlap_rate": timestamp_multiset_overlap_rate(
            real_times, synthetic_times, date_only=date_only
        ),
        "timestamp_count_l1_by_date": timestamp_count_l1_by_date(
            real_times, synthetic_times
        ),
        "timestamp_count_correlation_by_date": timestamp_count_correlation_by_date(
            real_times, synthetic_times
        ),
    }


def summarize_numeric(values: list[float]) -> Any:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }
