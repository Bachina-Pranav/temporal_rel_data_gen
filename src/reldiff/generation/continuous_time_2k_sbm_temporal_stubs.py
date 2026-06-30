"""Three-way temporal-stub generator for temporal review event spines.

This is a microcanonical sibling of ``ct_2k_sbm_plus``. It keeps RelDiff's
aggregate SBM block inference, but within each customer/product block pair it
preserves three stub multisets:

    customer stubs, product stubs, timestamp stubs

The output remains structural only:

    customer_id, product_id, review_time
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .continuous_time_2k_sbm_plus import (
    TimestampGranularityModel,
    count_correlation,
    pair_multiplicity_distribution,
    total_variation,
)
from .continuous_time_temporal_sbm import (
    DEFAULT_FALLBACK_BANDWIDTH,
    ContinuousTimeTemporalSBMGenerator,
    duplicate_pair_rate,
    empirical_ks_statistic,
    empirical_wasserstein_1d,
    estimate_bandwidth,
    reflect_unit_interval,
    scale_counts_largest_remainder,
)


GENERATOR_NAME = "ct_2k_sbm_temporal_stubs"

OccurrenceStub = Tuple[Any, float]
TimestampStub = Tuple[pd.Timestamp, float]


class ContinuousTime2KSBMTemporalStubsGenerator(ContinuousTimeTemporalSBMGenerator):
    """Temporal 2K+SBM generator with customer/product/timestamp stubs."""

    def __init__(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        reviews: pd.DataFrame,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        seed: int = 42,
        stub_pairing: str = "temporal_window_shuffle",
        timestamp_stub_mode: str = "reuse_block_pair_timestamps",
        temporal_window_size: int | None = None,
        avoid_real_edge_prob: float = 0.95,
        pair_multiplicity_mode: str = "none",
    ):
        if stub_pairing not in {
            "random",
            "temporal_sorted",
            "temporal_window_shuffle",
        }:
            raise ValueError(
                "stub_pairing must be one of: random, temporal_sorted, "
                "temporal_window_shuffle."
            )
        if timestamp_stub_mode not in {"reuse_block_pair_timestamps", "kde_jitter"}:
            raise ValueError(
                "timestamp_stub_mode must be 'reuse_block_pair_timestamps' or "
                "'kde_jitter'."
            )
        if pair_multiplicity_mode not in {"none", "empirical_block_pair"}:
            raise ValueError(
                "pair_multiplicity_mode must be 'none' or 'empirical_block_pair'."
            )
        if temporal_window_size is not None and temporal_window_size <= 0:
            raise ValueError("temporal_window_size must be positive when provided.")
        if not 0.0 <= avoid_real_edge_prob <= 1.0:
            raise ValueError("avoid_real_edge_prob must be in [0, 1].")

        self.stub_pairing = stub_pairing
        self.timestamp_stub_mode = timestamp_stub_mode
        self.temporal_window_size = temporal_window_size
        self.avoid_real_edge_prob = float(avoid_real_edge_prob)
        self.pair_multiplicity_mode = pair_multiplicity_mode

        super().__init__(
            customers=customers,
            products=products,
            reviews=reviews,
            customer_id_col=customer_id_col,
            product_id_col=product_id_col,
            timestamp_col=timestamp_col,
            seed=seed,
        )
        self.granularity_model = TimestampGranularityModel.fit(
            self.reviews[self.timestamp_col]
        )

    @classmethod
    def from_csv(
        cls,
        customers_path: str | Path,
        products_path: str | Path,
        reviews_path: str | Path,
        **kwargs: Any,
    ) -> "ContinuousTime2KSBMTemporalStubsGenerator":
        return cls(
            customers=pd.read_csv(customers_path),
            products=pd.read_csv(products_path),
            reviews=pd.read_csv(reviews_path),
            **kwargs,
        )

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
        num_events: int | None = None,
        output_path: str | Path | None = None,
        debug_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        if self.sbm_result is None:
            self.fit()

        target_counts = scale_counts_largest_remainder(
            self.block_pair_event_count,
            total=len(self.reviews) if num_events is None else int(num_events),
        )
        annotated = self._annotated_reviews()
        groups = {
            (int(customer_block), int(product_block)): group
            for (customer_block, product_block), group in annotated.groupby(
                ["_customer_block", "_product_block"], sort=True
            )
        }

        records: list[dict[str, Any]] = []
        synthetic_times_by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)

        with tqdm(
            total=sum(target_counts.values()),
            desc="Generating ct_2k_sbm_temporal_stubs events",
            unit="event",
        ) as pbar:
            for block_pair, count in sorted(target_counts.items()):
                group = groups[block_pair]
                block_bandwidth = estimate_bandwidth(
                    group["_time_x"].to_numpy(dtype=float),
                    DEFAULT_FALLBACK_BANDWIDTH,
                )
                triples = self._generate_block_pair_temporal_stubs(
                    group, count, block_pair
                )
                for customer_id, product_id, timestamp_stub in triples:
                    timestamp = self._materialize_timestamp(
                        timestamp_stub, block_bandwidth
                    )
                    x = self._normalize_generated_timestamp(timestamp)
                    synthetic_times_by_pair[block_pair].append(x)
                    records.append(
                        {
                            self.customer_id_col: customer_id,
                            self.product_id_col: product_id,
                            self.timestamp_col: timestamp,
                            "_customer_block": block_pair[0],
                            "_product_block": block_pair[1],
                            "_time_x": x,
                        }
                    )
                    pbar.update(1)

        synthetic_full = pd.DataFrame.from_records(records)
        synthetic_full = synthetic_full.sort_values(
            self.timestamp_col, kind="mergesort"
        )
        output = synthetic_full[
            [self.customer_id_col, self.product_id_col, self.timestamp_col]
        ].reset_index(drop=True)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                "Writing ct_2k_sbm_temporal_stubs synthetic event spine "
                f"to {output_path}..."
            )
            output.to_csv(output_path, index=False)

        if debug_dir is not None:
            self.write_temporal_stubs_debug_outputs(
                Path(debug_dir), output, target_counts, synthetic_times_by_pair
            )
        return output

    def _generate_block_pair_temporal_stubs(
        self, group: pd.DataFrame, count: int, block_pair: tuple[int, int]
    ) -> list[tuple[Any, Any, TimestampStub]]:
        customer_stubs = self._scaled_occurrence_stubs(group, self.customer_id_col, count)
        product_stubs = self._scaled_occurrence_stubs(group, self.product_id_col, count)
        timestamp_stubs = self._scaled_timestamp_stubs(group, count)

        if count == 0:
            return []

        if self.stub_pairing == "random":
            self.rng.shuffle(customer_stubs)
            self.rng.shuffle(product_stubs)
            self.rng.shuffle(timestamp_stubs)
            window_size = count
        else:
            customer_stubs.sort(key=lambda stub: stub[1])
            product_stubs.sort(key=lambda stub: stub[1])
            timestamp_stubs.sort(key=lambda stub: stub[1])
            window_size = self._effective_window_size(count)
            if self.stub_pairing == "temporal_window_shuffle":
                customer_stubs = self._shuffle_in_windows(customer_stubs, window_size)
                product_stubs = self._shuffle_in_windows(product_stubs, window_size)

        real_edges = set(
            map(
                tuple,
                group[[self.customer_id_col, self.product_id_col]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        self._avoid_real_edges_in_windows(customer_stubs, product_stubs, real_edges, window_size)

        return [
            (customer_stub[0], product_stub[0], timestamp_stub)
            for customer_stub, product_stub, timestamp_stub in zip(
                customer_stubs, product_stubs, timestamp_stubs
            )
        ]

    def _scaled_occurrence_stubs(
        self, group: pd.DataFrame, column: str, count: int
    ) -> list[OccurrenceStub]:
        values = group[column].to_numpy(dtype=object)
        times = group["_time_x"].to_numpy(dtype=float)
        indices = self._scaled_indices(len(group), count)
        return [(values[index], float(times[index])) for index in indices]

    def _scaled_timestamp_stubs(
        self, group: pd.DataFrame, count: int
    ) -> list[TimestampStub]:
        timestamps = pd.to_datetime(group[self.timestamp_col]).to_numpy()
        times = group["_time_x"].to_numpy(dtype=float)
        indices = self._scaled_indices(len(group), count)
        return [
            (pd.Timestamp(timestamps[index]), float(times[index]))
            for index in indices
        ]

    def _scaled_indices(self, length: int, count: int) -> np.ndarray:
        if count == length:
            return np.arange(length, dtype=int)
        if length == 0:
            return np.asarray([], dtype=int)
        return self.rng.choice(np.arange(length), size=count, replace=True)

    def _effective_window_size(self, count: int) -> int:
        if count <= 1:
            return 1
        if self.temporal_window_size is not None:
            return min(int(self.temporal_window_size), count)
        window_size = max(10, int(np.sqrt(count)))
        return count if count <= window_size else window_size

    def _shuffle_in_windows(
        self, stubs: list[OccurrenceStub], window_size: int
    ) -> list[OccurrenceStub]:
        shuffled: list[OccurrenceStub] = []
        for start in range(0, len(stubs), window_size):
            window = list(stubs[start : start + window_size])
            self.rng.shuffle(window)
            shuffled.extend(window)
        return shuffled

    def _avoid_real_edges_in_windows(
        self,
        customer_stubs: list[OccurrenceStub],
        product_stubs: list[OccurrenceStub],
        real_edges: set[tuple[Any, Any]],
        window_size: int,
    ) -> None:
        if self.avoid_real_edge_prob <= 0.0 or len(product_stubs) < 2:
            return

        max_attempts_per_edge = 20
        for start in range(0, len(product_stubs), max(window_size, 1)):
            end = min(start + max(window_size, 1), len(product_stubs))
            indices = list(range(start, end))
            if len(indices) < 2:
                continue
            for index in indices:
                customer_id = customer_stubs[index][0]
                product_id = product_stubs[index][0]
                if (customer_id, product_id) not in real_edges:
                    continue
                if self.rng.random() >= self.avoid_real_edge_prob:
                    continue

                candidates = list(indices)
                self.rng.shuffle(candidates)
                attempts = 0
                for candidate in candidates:
                    if candidate == index:
                        continue
                    attempts += 1
                    other_customer_id = customer_stubs[candidate][0]
                    other_product_id = product_stubs[candidate][0]
                    if (
                        (customer_id, other_product_id) not in real_edges
                        and (other_customer_id, product_id) not in real_edges
                    ):
                        product_stubs[index], product_stubs[candidate] = (
                            product_stubs[candidate],
                            product_stubs[index],
                        )
                        break
                    if attempts >= max_attempts_per_edge:
                        break

    def _materialize_timestamp(
        self, timestamp_stub: TimestampStub, block_bandwidth: float
    ) -> pd.Timestamp:
        timestamp, x = timestamp_stub
        if self.timestamp_stub_mode == "reuse_block_pair_timestamps":
            adjusted = pd.Timestamp(timestamp)
            if self.granularity_model.mode == "date_only":
                adjusted = adjusted.floor("D")
            return pd.Timestamp(min(max(adjusted, self.min_time), self.max_time))

        jitter = max(block_bandwidth * 0.25, 1e-4)
        jittered_x = reflect_unit_interval(float(x) + self.rng.normal(0.0, jitter))
        raw = super().denormalize_time(jittered_x)
        return self.granularity_model.apply(raw, self.rng, self.min_time, self.max_time)

    def _normalize_generated_timestamp(self, timestamp: pd.Timestamp) -> float:
        if self.time_span.total_seconds() <= 0:
            return 0.5
        x = (pd.Timestamp(timestamp) - self.min_time).total_seconds()
        x = x / self.time_span.total_seconds()
        return float(np.clip(x, 0.0, 1.0))

    def _annotate_synthetic(self, synthetic: pd.DataFrame) -> pd.DataFrame:
        assert self.sbm_result is not None
        annotated = synthetic.copy()
        annotated[self.timestamp_col] = pd.to_datetime(annotated[self.timestamp_col])
        annotated["_customer_block"] = annotated[self.customer_id_col].map(
            self.sbm_result.customer_blocks
        )
        annotated["_product_block"] = annotated[self.product_id_col].map(
            self.sbm_result.product_blocks
        )
        annotated["_time_x"] = [
            self._normalize_generated_timestamp(timestamp)
            for timestamp in annotated[self.timestamp_col]
        ]
        return annotated

    def write_temporal_stubs_debug_outputs(
        self,
        debug_dir: Path,
        synthetic: pd.DataFrame,
        target_counts: dict[tuple[int, int], int],
        synthetic_times_by_pair: dict[tuple[int, int], list[float]],
    ) -> None:
        assert self.sbm_result is not None
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing ct_2k_sbm_temporal_stubs debug outputs to {debug_dir}...")

        real_times = pd.to_datetime(self.reviews[self.timestamp_col])
        synthetic_times = pd.to_datetime(synthetic[self.timestamp_col])
        real_x = self.reviews["_time_x"].to_numpy(dtype=float)
        synthetic_x = np.asarray(
            [self._normalize_generated_timestamp(timestamp) for timestamp in synthetic_times],
            dtype=float,
        )

        annotated_real = self._annotated_reviews()
        annotated_synthetic = self._annotate_synthetic(synthetic)

        summary = {
            "generator": GENERATOR_NAME,
            "num_real_reviews": int(len(self.reviews)),
            "num_synthetic_reviews": int(len(synthetic)),
            "min_time": str(self.min_time),
            "max_time": str(self.max_time),
            "num_active_customers_real": int(self.reviews[self.customer_id_col].nunique()),
            "num_active_customers_synthetic": int(synthetic[self.customer_id_col].nunique()),
            "num_active_products_real": int(self.reviews[self.product_id_col].nunique()),
            "num_active_products_synthetic": int(synthetic[self.product_id_col].nunique()),
            "num_customer_blocks": int(self.sbm_result.num_customer_blocks),
            "num_product_blocks": int(self.sbm_result.num_product_blocks),
            "num_nonzero_block_pairs": int(len(self.block_pair_event_count)),
            "stub_pairing": self.stub_pairing,
            "timestamp_stub_mode": self.timestamp_stub_mode,
            "temporal_window_size": self.temporal_window_size,
            "avoid_real_edge_prob": self.avoid_real_edge_prob,
            "pair_multiplicity_mode": self.pair_multiplicity_mode,
            "seed": self.seed,
            **self.granularity_model.to_debug_dict(),
        }
        self._write_json(
            debug_dir / "ct_2k_sbm_temporal_stubs_summary.json", summary
        )
        self.write_sbm_summary(debug_dir / "sbm_summary.json")
        self._write_assignment_debug(debug_dir)
        self._write_block_pair_debug(
            debug_dir,
            annotated_real,
            annotated_synthetic,
            target_counts,
        )
        self._write_degree_checks(debug_dir, synthetic)
        self._write_timestamp_diagnostics(
            debug_dir,
            real_times,
            synthetic_times,
            real_x,
            synthetic_x,
            synthetic_times_by_pair,
        )
        self._write_pair_multiplicity(debug_dir, synthetic)

    def _write_assignment_debug(self, debug_dir: Path) -> None:
        assert self.sbm_result is not None
        pd.DataFrame(
            [
                {self.customer_id_col: customer_id, "customer_block": block}
                for customer_id, block in self.sbm_result.customer_blocks.items()
            ]
        ).to_csv(
            debug_dir / "ct_2k_sbm_temporal_stubs_customer_assignments.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {self.product_id_col: product_id, "product_block": block}
                for product_id, block in self.sbm_result.product_blocks.items()
            ]
        ).to_csv(
            debug_dir / "ct_2k_sbm_temporal_stubs_product_assignments.csv",
            index=False,
        )

    def _write_block_pair_debug(
        self,
        debug_dir: Path,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_counts: dict[tuple[int, int], int],
    ) -> None:
        rows = []
        synthetic_groups = {
            (int(a), int(b)): group
            for (a, b), group in synthetic.groupby(["_customer_block", "_product_block"])
        }
        for (customer_block, product_block), group in real.groupby(
            ["_customer_block", "_product_block"], sort=True
        ):
            key = (int(customer_block), int(product_block))
            syn_group = synthetic_groups.get(key, pd.DataFrame())
            ks = None
            if not syn_group.empty:
                ks = empirical_ks_statistic(
                    group["_time_x"].to_numpy(dtype=float),
                    syn_group["_time_x"].to_numpy(dtype=float),
                )
            rows.append(
                {
                    "customer_block": key[0],
                    "product_block": key[1],
                    "real_event_count": int(len(group)),
                    "synthetic_event_count": int(target_counts.get(key, len(syn_group))),
                    "real_unique_customers": int(group[self.customer_id_col].nunique()),
                    "synthetic_unique_customers": int(
                        syn_group[self.customer_id_col].nunique()
                    )
                    if not syn_group.empty
                    else 0,
                    "real_unique_products": int(group[self.product_id_col].nunique()),
                    "synthetic_unique_products": int(
                        syn_group[self.product_id_col].nunique()
                    )
                    if not syn_group.empty
                    else 0,
                    "real_timestamp_min": str(group[self.timestamp_col].min()),
                    "real_timestamp_max": str(group[self.timestamp_col].max()),
                    "synthetic_timestamp_min": str(syn_group[self.timestamp_col].min())
                    if not syn_group.empty
                    else None,
                    "synthetic_timestamp_max": str(syn_group[self.timestamp_col].max())
                    if not syn_group.empty
                    else None,
                    "block_pair_timestamp_ks": ks,
                    "block_pair_edge_overlap_rate": self._edge_overlap_rate(
                        group, syn_group
                    )
                    if not syn_group.empty
                    else 0.0,
                }
            )
        pd.DataFrame(rows).to_csv(
            debug_dir / "ct_2k_sbm_temporal_stubs_block_pairs.csv", index=False
        )

    def _write_degree_checks(self, debug_dir: Path, synthetic: pd.DataFrame) -> None:
        assert self.sbm_result is not None
        for entity, column, block_map, filename in (
            (
                "customer",
                self.customer_id_col,
                self.sbm_result.customer_blocks,
                "ct_2k_sbm_temporal_stubs_customer_degree_check.csv",
            ),
            (
                "product",
                self.product_id_col,
                self.sbm_result.product_blocks,
                "ct_2k_sbm_temporal_stubs_product_degree_check.csv",
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

    def _write_timestamp_diagnostics(
        self,
        debug_dir: Path,
        real_times: pd.Series,
        synthetic_times: pd.Series,
        real_x: np.ndarray,
        synthetic_x: np.ndarray,
        synthetic_times_by_pair: dict[tuple[int, int], list[float]],
    ) -> None:
        real_days = real_times.astype("int64").to_numpy(dtype=float) / 1e9 / 86400.0
        synthetic_days = (
            synthetic_times.astype("int64").to_numpy(dtype=float) / 1e9 / 86400.0
        )

        per_pair_ks = []
        for block_pair, real_count in self.block_pair_event_count.items():
            if real_count == 0:
                continue
            real_pair_times = self.block_pair_events[block_pair].times
            synthetic_pair_times = np.asarray(
                synthetic_times_by_pair.get(block_pair, []), dtype=float
            )
            ks = empirical_ks_statistic(real_pair_times, synthetic_pair_times)
            if ks is not None:
                per_pair_ks.append(ks)

        diagnostics = {
            "global_timestamp_ks": empirical_ks_statistic(real_x, synthetic_x),
            "global_timestamp_wasserstein_days": empirical_wasserstein_1d(
                real_days, synthetic_days
            ),
            "hour_of_day_total_variation": total_variation(
                real_times.dt.hour, synthetic_times.dt.hour
            ),
            "day_of_week_total_variation": total_variation(
                real_times.dt.dayofweek, synthetic_times.dt.dayofweek
            ),
            "monthly_or_daily_count_correlation": count_correlation(
                pd.DataFrame({self.timestamp_col: real_times}),
                pd.DataFrame({self.timestamp_col: synthetic_times}),
                self.timestamp_col,
                "M",
            ),
            "per_block_pair_timestamp_ks_mean": float(np.mean(per_pair_ks))
            if per_pair_ks
            else None,
            "per_block_pair_timestamp_ks_median": float(np.median(per_pair_ks))
            if per_pair_ks
            else None,
            "per_block_pair_timestamp_ks_num_pairs": len(per_pair_ks),
        }
        self._write_json(
            debug_dir / "ct_2k_sbm_temporal_stubs_timestamp_diagnostics.json",
            diagnostics,
        )

    def _write_pair_multiplicity(
        self, debug_dir: Path, synthetic: pd.DataFrame
    ) -> None:
        data = {
            "real_duplicate_customer_product_rate": duplicate_pair_rate(
                self.reviews, self.customer_id_col, self.product_id_col
            ),
            "synthetic_duplicate_customer_product_rate": duplicate_pair_rate(
                synthetic, self.customer_id_col, self.product_id_col
            ),
            "real_pair_multiplicity_distribution": pair_multiplicity_distribution(
                self.reviews, self.customer_id_col, self.product_id_col
            ),
            "synthetic_pair_multiplicity_distribution": pair_multiplicity_distribution(
                synthetic, self.customer_id_col, self.product_id_col
            ),
            "edge_overlap_rate": self._edge_overlap_rate(self.reviews, synthetic),
        }
        self._write_json(
            debug_dir / "ct_2k_sbm_temporal_stubs_pair_multiplicity.json", data
        )

    def _edge_overlap_rate(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        if synthetic.empty:
            return 0.0
        real_edges = set(
            map(
                tuple,
                real[[self.customer_id_col, self.product_id_col]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        synthetic_edges = set(
            map(
                tuple,
                synthetic[[self.customer_id_col, self.product_id_col]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        if not synthetic_edges:
            return 0.0
        return float(len(real_edges & synthetic_edges) / len(synthetic_edges))

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with path.open("w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
