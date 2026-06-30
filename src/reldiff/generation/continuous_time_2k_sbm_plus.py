"""Continuous-time 2K+SBM+ generator for temporal review event spines.

This is a stricter, more microcanonical sibling of
ContinuousTimeTemporalSBMGenerator. It keeps the old generator intact while
preserving customer/product endpoint stubs inside each SBM block pair.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .continuous_time_temporal_sbm import (
    ContinuousKDESampler,
    ContinuousTimeTemporalSBMGenerator,
    DEFAULT_FALLBACK_BANDWIDTH,
    EventCollection,
    duplicate_pair_rate,
    empirical_ks_statistic,
    empirical_wasserstein_1d,
    estimate_bandwidth,
    reflect_unit_interval,
    scale_counts_largest_remainder,
)


GENERATOR_NAME = "ct_2k_sbm_plus"


@dataclass
class TimestampGranularityModel:
    mode: str
    fraction_midnight: float
    offset_counts: Counter

    @classmethod
    def fit(cls, timestamps: pd.Series) -> "TimestampGranularityModel":
        timestamps = pd.to_datetime(timestamps, errors="coerce").dropna()
        date_floor = timestamps.dt.floor("D")
        offsets = (timestamps - date_floor).dt.total_seconds().astype(int)
        offset_counts = Counter(offsets.tolist())
        total = sum(offset_counts.values())
        fraction_midnight = offset_counts.get(0, 0) / total if total else 0.0
        num_unique_offsets = len(offset_counts)

        if fraction_midnight >= 0.99:
            mode = "date_only"
        elif num_unique_offsets <= max(24, int(0.01 * total)):
            mode = "empirical_offsets"
        else:
            mode = "continuous_time_of_day"
        return cls(mode, fraction_midnight, offset_counts)

    @property
    def num_unique_time_offsets(self) -> int:
        return len(self.offset_counts)

    @property
    def top_10_offsets_seconds(self) -> list[int]:
        return [int(offset) for offset, _ in self.offset_counts.most_common(10)]

    def apply(
        self,
        timestamp: pd.Timestamp,
        rng: np.random.Generator,
        min_time: pd.Timestamp,
        max_time: pd.Timestamp,
    ) -> pd.Timestamp:
        timestamp = pd.Timestamp(timestamp)
        if self.mode == "date_only":
            adjusted = timestamp.floor("D")
        elif self.mode == "empirical_offsets":
            offsets = np.asarray(list(self.offset_counts.keys()), dtype=float)
            weights = np.asarray(list(self.offset_counts.values()), dtype=float)
            weights = weights / weights.sum()
            offset_seconds = float(rng.choice(offsets, p=weights))
            adjusted = timestamp.floor("D") + pd.to_timedelta(offset_seconds, unit="s")
        else:
            adjusted = timestamp
        return pd.Timestamp(min(max(adjusted, min_time), max_time))

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "timestamp_granularity_mode": self.mode,
            "fraction_midnight": self.fraction_midnight,
            "num_unique_time_offsets": self.num_unique_time_offsets,
            "top_10_offsets_seconds": self.top_10_offsets_seconds,
        }


class ContinuousTime2KSBMPlusGenerator(ContinuousTimeTemporalSBMGenerator):
    """Temporal 2K+SBM+ generator with block-pair endpoint stub constraints."""

    def __init__(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        reviews: pd.DataFrame,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        seed: int = 42,
        stub_pairing: str = "time_sorted",
        pair_multiplicity_mode: str = "none",
    ):
        if stub_pairing not in {"random", "time_sorted"}:
            raise ValueError("stub_pairing must be 'random' or 'time_sorted'.")
        if pair_multiplicity_mode not in {"none", "block_pair_empirical"}:
            raise ValueError(
                "pair_multiplicity_mode must be 'none' or 'block_pair_empirical'."
            )
        self.stub_pairing = stub_pairing
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
        self.customer_time_collections: dict[Any, EventCollection] = {}
        self.product_time_collections: dict[Any, EventCollection] = {}

    @classmethod
    def from_csv(
        cls,
        customers_path: str | Path,
        products_path: str | Path,
        reviews_path: str | Path,
        **kwargs: Any,
    ) -> "ContinuousTime2KSBMPlusGenerator":
        return cls(
            customers=pd.read_csv(customers_path),
            products=pd.read_csv(products_path),
            reviews=pd.read_csv(reviews_path),
            **kwargs,
        )

    def fit(self) -> "ContinuousTime2KSBMPlusGenerator":
        super().fit()
        self._build_entity_time_collections()
        return self

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

    def _build_entity_time_collections(self) -> None:
        print("Building entity-level temporal occurrence collections...")
        fallback = (
            self.global_events.bandwidth
            if self.global_events is not None
            else DEFAULT_FALLBACK_BANDWIDTH
        )
        self.customer_time_collections = {}
        for customer_id, group in tqdm(
            self.reviews.groupby(self.customer_id_col, sort=False),
            desc="Building customer time collections",
            unit="customer",
        ):
            self.customer_time_collections[customer_id] = EventCollection.from_records(
                group["_time_x"], customer_ids=group[self.customer_id_col], fallback_bandwidth=fallback
            )
        self.product_time_collections = {}
        for product_id, group in tqdm(
            self.reviews.groupby(self.product_id_col, sort=False),
            desc="Building product time collections",
            unit="product",
        ):
            self.product_time_collections[product_id] = EventCollection.from_records(
                group["_time_x"], product_ids=group[self.product_id_col], fallback_bandwidth=fallback
            )

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
        records: list[dict[str, Any]] = []
        synthetic_times_by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)

        groups = {
            (int(customer_block), int(product_block)): group
            for (customer_block, product_block), group in annotated.groupby(
                ["_customer_block", "_product_block"], sort=True
            )
        }

        with tqdm(
            total=sum(target_counts.values()),
            desc="Generating ct_2k_sbm_plus events",
            unit="event",
        ) as pbar:
            for block_pair, count in sorted(target_counts.items()):
                group = groups[block_pair]
                pairs = self._generate_block_pair_endpoint_pairs(group, count, block_pair)
                times_x = self._generate_block_pair_times(block_pair, pairs, group)
                for (customer_id, product_id), x in zip(pairs, times_x):
                    raw_timestamp = super(
                        ContinuousTime2KSBMPlusGenerator, self
                    ).denormalize_time(float(x))
                    timestamp = self.granularity_model.apply(
                        raw_timestamp,
                        self.rng,
                        self.min_time,
                        self.max_time,
                    )
                    synthetic_times_by_pair[block_pair].append(float(x))
                    records.append(
                        {
                            self.customer_id_col: customer_id,
                            self.product_id_col: product_id,
                            self.timestamp_col: timestamp,
                            "_customer_block": block_pair[0],
                            "_product_block": block_pair[1],
                            "_time_x": float(x),
                        }
                    )
                    pbar.update(1)

        synthetic = pd.DataFrame.from_records(records)
        synthetic = synthetic.sort_values(self.timestamp_col, kind="mergesort")
        output = synthetic[
            [self.customer_id_col, self.product_id_col, self.timestamp_col]
        ].reset_index(drop=True)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing ct_2k_sbm_plus synthetic event spine to {output_path}...")
            output.to_csv(output_path, index=False)

        if debug_dir is not None:
            self.write_plus_debug_outputs(
                Path(debug_dir), output, target_counts, synthetic_times_by_pair
            )
        return output

    def _generate_block_pair_endpoint_pairs(
        self, group: pd.DataFrame, count: int, block_pair: tuple[int, int]
    ) -> list[tuple[Any, Any]]:
        customer_stubs = self._scaled_stubs(group[self.customer_id_col], count)
        product_stubs = self._scaled_stubs(group[self.product_id_col], count)

        reserved_pairs: list[tuple[Any, Any]] = []
        if self.pair_multiplicity_mode == "block_pair_empirical":
            reserved_pairs, customer_stubs, product_stubs = self._reserve_repeated_pairs(
                group, customer_stubs, product_stubs
            )

        if self.stub_pairing == "random":
            self.rng.shuffle(customer_stubs)
            self.rng.shuffle(product_stubs)
            pairs = list(zip(customer_stubs, product_stubs))
        else:
            customer_stubs = self._sort_stubs_by_tentative_time(
                customer_stubs, endpoint="customer", block_pair=block_pair
            )
            product_stubs = self._sort_stubs_by_tentative_time(
                product_stubs, endpoint="product", block_pair=block_pair
            )
            pairs = list(zip(customer_stubs, product_stubs))
        return reserved_pairs + pairs

    def _scaled_stubs(self, values: pd.Series, count: int) -> list[Any]:
        values = values.to_numpy(dtype=object)
        if count == len(values):
            return values.tolist()
        if len(values) == 0:
            return []
        return self.rng.choice(values, size=count, replace=True).tolist()

    def _reserve_repeated_pairs(
        self,
        group: pd.DataFrame,
        customer_stubs: list[Any],
        product_stubs: list[Any],
    ) -> tuple[list[tuple[Any, Any]], list[Any], list[Any]]:
        customer_remaining = Counter(customer_stubs)
        product_remaining = Counter(product_stubs)
        pair_counts = Counter(
            map(tuple, group[[self.customer_id_col, self.product_id_col]].to_numpy())
        )
        reserved: list[tuple[Any, Any]] = []
        for (customer_id, product_id), multiplicity in sorted(
            pair_counts.items(), key=lambda item: item[1], reverse=True
        ):
            if multiplicity < 2:
                continue
            copies = min(
                multiplicity,
                customer_remaining.get(customer_id, 0),
                product_remaining.get(product_id, 0),
            )
            if copies < 2:
                continue
            reserved.extend([(customer_id, product_id)] * copies)
            customer_remaining[customer_id] -= copies
            product_remaining[product_id] -= copies

        customers = []
        for customer_id, count in customer_remaining.items():
            customers.extend([customer_id] * count)
        products = []
        for product_id, count in product_remaining.items():
            products.extend([product_id] * count)
        return reserved, customers, products

    def _sort_stubs_by_tentative_time(
        self,
        stubs: list[Any],
        endpoint: str,
        block_pair: tuple[int, int],
    ) -> list[Any]:
        tentative = []
        for entity_id in stubs:
            x = self._sample_entity_time(entity_id, endpoint, block_pair)
            tentative.append((x, entity_id))
        tentative.sort(key=lambda item: item[0])
        return [entity_id for _, entity_id in tentative]

    def _sample_entity_time(
        self, entity_id: Any, endpoint: str, block_pair: tuple[int, int]
    ) -> float:
        if endpoint == "customer":
            collection = self.customer_time_collections.get(entity_id)
            block_collection = self.customer_block_events.get(block_pair[0])
        else:
            collection = self.product_time_collections.get(entity_id)
            block_collection = self.product_block_events.get(block_pair[1])
        for candidate in (collection, block_collection, self.block_pair_events.get(block_pair), self.global_events):
            if candidate is not None and len(candidate) > 0:
                return ContinuousKDESampler(candidate.times, candidate.bandwidth).sample(
                    self.rng
                )
        return 0.5

    def _generate_block_pair_times(
        self,
        block_pair: tuple[int, int],
        pairs: list[tuple[Any, Any]],
        group: pd.DataFrame,
    ) -> np.ndarray:
        if not pairs:
            return np.asarray([], dtype=float)
        if self.stub_pairing == "random":
            return np.asarray(
                [self.sample_timestamp_for_block_pair(block_pair)[0] for _ in pairs],
                dtype=float,
            )

        customer_times = [
            self._sample_entity_time(customer_id, "customer", block_pair)
            for customer_id, _ in pairs
        ]
        product_times = [
            self._sample_entity_time(product_id, "product", block_pair)
            for _, product_id in pairs
        ]
        midpoint_times = np.asarray(customer_times) * 0.5 + np.asarray(product_times) * 0.5
        block_bandwidth = max(
            estimate_bandwidth(group["_time_x"].to_numpy(dtype=float), DEFAULT_FALLBACK_BANDWIDTH),
            1e-4,
        )
        jittered = [
            reflect_unit_interval(x + self.rng.normal(0.0, block_bandwidth))
            for x in midpoint_times
        ]
        return np.asarray(jittered, dtype=float)

    def denormalize_time(self, x: float) -> pd.Timestamp:
        raw = super().denormalize_time(x)
        return self.granularity_model.apply(raw, self.rng, self.min_time, self.max_time)

    def write_plus_debug_outputs(
        self,
        debug_dir: Path,
        synthetic: pd.DataFrame,
        target_counts: dict[tuple[int, int], int],
        synthetic_times_by_pair: dict[tuple[int, int], list[float]],
    ) -> None:
        assert self.sbm_result is not None
        assert self.global_events is not None
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing ct_2k_sbm_plus debug outputs to {debug_dir}...")

        real_times = pd.to_datetime(self.reviews[self.timestamp_col])
        synthetic_times = pd.to_datetime(synthetic[self.timestamp_col])
        real_x = self.reviews["_time_x"].to_numpy(dtype=float)
        synthetic_x = (
            (synthetic_times - self.min_time).dt.total_seconds()
            / max(self.time_span.total_seconds(), 1.0)
        ).to_numpy(dtype=float)

        annotated_real = self._annotated_reviews()
        annotated_synthetic = self._annotate_synthetic(synthetic)

        summary = {
            "generator": GENERATOR_NAME,
            "num_real_reviews": int(len(self.reviews)),
            "num_synthetic_reviews": int(len(synthetic)),
            "min_time": str(self.min_time),
            "max_time": str(self.max_time),
            "num_active_customers_real": int(self.reviews[self.customer_id_col].nunique()),
            "num_active_products_real": int(self.reviews[self.product_id_col].nunique()),
            "num_active_customers_synthetic": int(synthetic[self.customer_id_col].nunique()),
            "num_active_products_synthetic": int(synthetic[self.product_id_col].nunique()),
            "num_customer_blocks": int(self.sbm_result.num_customer_blocks),
            "num_product_blocks": int(self.sbm_result.num_product_blocks),
            "num_nonzero_block_pairs": int(len(self.block_pair_event_count)),
            "stub_pairing": self.stub_pairing,
            "pair_multiplicity_mode": self.pair_multiplicity_mode,
            "global_timestamp_bandwidth": self.global_events.bandwidth,
            "seed": self.seed,
            **self.granularity_model.to_debug_dict(),
        }
        self._write_json(debug_dir / "ct_2k_sbm_plus_summary.json", summary)
        self.write_sbm_summary(debug_dir / "sbm_summary.json")
        self._write_block_pair_debug(debug_dir, annotated_real, annotated_synthetic, target_counts)
        self._write_degree_checks(debug_dir, synthetic)
        self._write_timestamp_diagnostics(debug_dir, real_times, synthetic_times, real_x, synthetic_x)
        self._write_pair_multiplicity(debug_dir, synthetic)

    def _annotate_synthetic(self, synthetic: pd.DataFrame) -> pd.DataFrame:
        assert self.sbm_result is not None
        annotated = synthetic.copy()
        annotated["_customer_block"] = annotated[self.customer_id_col].map(
            self.sbm_result.customer_blocks
        )
        annotated["_product_block"] = annotated[self.product_id_col].map(
            self.sbm_result.product_blocks
        )
        return annotated

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
            rows.append(
                {
                    "customer_block": key[0],
                    "product_block": key[1],
                    "real_event_count": len(group),
                    "synthetic_event_count": target_counts.get(key, len(syn_group)),
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
                    "timestamp_bandwidth": self.block_pair_events[key].bandwidth,
                }
            )
        pd.DataFrame(rows).to_csv(
            debug_dir / "ct_2k_sbm_plus_block_pairs.csv", index=False
        )

    def _write_degree_checks(self, debug_dir: Path, synthetic: pd.DataFrame) -> None:
        assert self.sbm_result is not None
        for entity, column, block_map, filename in (
            ("customer", self.customer_id_col, self.sbm_result.customer_blocks, "ct_2k_sbm_plus_customer_degree_check.csv"),
            ("product", self.product_id_col, self.sbm_result.product_blocks, "ct_2k_sbm_plus_product_degree_check.csv"),
        ):
            real_counts = Counter(self.reviews[column])
            syn_counts = Counter(synthetic[column])
            rows = []
            for entity_id in sorted(block_map, key=str):
                real_count = int(real_counts.get(entity_id, 0))
                syn_count = int(syn_counts.get(entity_id, 0))
                rows.append(
                    {
                        f"{entity}_id": entity_id,
                        f"{entity}_block": block_map[entity_id],
                        "real_event_count": real_count,
                        "synthetic_event_count": syn_count,
                        "abs_error": abs(real_count - syn_count),
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
    ) -> None:
        real_days = real_times.astype("int64").to_numpy(dtype=float) / 1e9 / 86400.0
        synthetic_days = (
            synthetic_times.astype("int64").to_numpy(dtype=float) / 1e9 / 86400.0
        )
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
        }
        self._write_json(debug_dir / "ct_2k_sbm_plus_timestamp_diagnostics.json", diagnostics)

    def _write_pair_multiplicity(self, debug_dir: Path, synthetic: pd.DataFrame) -> None:
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
        }
        self._write_json(debug_dir / "ct_2k_sbm_plus_pair_multiplicity.json", data)


def pair_multiplicity_distribution(
    df: pd.DataFrame, customer_col: str, product_col: str
) -> dict[str, int]:
    counts = df.groupby([customer_col, product_col]).size()
    distribution = counts.value_counts().sort_index()
    return {str(int(multiplicity)): int(count) for multiplicity, count in distribution.items()}


def total_variation(real: pd.Series, synthetic: pd.Series) -> float | None:
    real_counts = real.value_counts(normalize=True)
    synthetic_counts = synthetic.value_counts(normalize=True)
    index = real_counts.index.union(synthetic_counts.index)
    if len(index) == 0:
        return None
    return float(
        0.5
        * np.abs(
            real_counts.reindex(index, fill_value=0)
            - synthetic_counts.reindex(index, fill_value=0)
        ).sum()
    )


def count_correlation(
    real: pd.DataFrame, synthetic: pd.DataFrame, timestamp_col: str, freq: str
) -> float | None:
    real_counts = real.set_index(timestamp_col).resample(freq).size()
    synthetic_counts = synthetic.set_index(timestamp_col).resample(freq).size()
    index = real_counts.index.union(synthetic_counts.index)
    if len(index) < 2:
        return None
    real_values = real_counts.reindex(index, fill_value=0).to_numpy(dtype=float)
    synthetic_values = synthetic_counts.reindex(index, fill_value=0).to_numpy(dtype=float)
    if real_values.std() == 0 or synthetic_values.std() == 0:
        return None
    return float(np.corrcoef(real_values, synthetic_values)[0, 1])
