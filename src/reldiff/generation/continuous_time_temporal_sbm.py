"""Continuous-time temporal SBM generator for review event spines.

This module generates only structural temporal review events:

    customer_id, product_id, review_time

It extends RelDiff's static type-constrained SBM idea by fitting customer and
product blocks on the aggregate customer-product graph, then sampling review
events from continuous KDE-style timestamp intensities and local temporal
degree-corrected endpoint distributions.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


GENERATOR_NAME = "continuous_time_temporal_sbm"
MIN_BANDWIDTH = 1e-4
DEFAULT_FALLBACK_BANDWIDTH = 0.05
LOCAL_WINDOW_BANDWIDTHS = 3.0
MIN_KDE_GROUP_SIZE = 3


@dataclass
class SBMBlockResult:
    customer_blocks: dict[Any, int]
    product_blocks: dict[Any, int]
    num_customer_blocks: int
    num_product_blocks: int
    used_existing_reldiff_sbm: bool
    description_length: float | None = None


@dataclass
class EventCollection:
    times: np.ndarray
    customer_ids: np.ndarray | None = None
    product_ids: np.ndarray | None = None
    bandwidth: float = DEFAULT_FALLBACK_BANDWIDTH

    @classmethod
    def from_records(
        cls,
        times: Iterable[float],
        customer_ids: Iterable[Any] | None = None,
        product_ids: Iterable[Any] | None = None,
        fallback_bandwidth: float = DEFAULT_FALLBACK_BANDWIDTH,
    ) -> "EventCollection":
        times_array = np.asarray(list(times), dtype=float)
        customer_array = (
            np.asarray(list(customer_ids), dtype=object)
            if customer_ids is not None
            else None
        )
        product_array = (
            np.asarray(list(product_ids), dtype=object)
            if product_ids is not None
            else None
        )

        if len(times_array) > 0:
            order = np.argsort(times_array, kind="mergesort")
            times_array = times_array[order]
            if customer_array is not None:
                customer_array = customer_array[order]
            if product_array is not None:
                product_array = product_array[order]

        bandwidth = estimate_bandwidth(times_array, fallback_bandwidth)
        return cls(times_array, customer_array, product_array, bandwidth)

    def __len__(self) -> int:
        return len(self.times)


class ContinuousKDESampler:
    """Sample timestamps from a reflected Gaussian KDE on [0, 1]."""

    def __init__(self, times: np.ndarray, bandwidth: float):
        if len(times) == 0:
            raise ValueError("ContinuousKDESampler requires at least one timestamp.")
        self.times = np.asarray(times, dtype=float)
        self.bandwidth = max(float(bandwidth), MIN_BANDWIDTH)

    def sample(self, rng: np.random.Generator) -> float:
        center = self.times[rng.integers(0, len(self.times))]
        value = center + rng.normal(0.0, self.bandwidth)
        return reflect_unit_interval(value)


def reflect_unit_interval(value: float) -> float:
    """Reflect an arbitrary real value into [0, 1]."""
    if not np.isfinite(value):
        return 0.5
    while value < 0.0 or value > 1.0:
        if value < 0.0:
            value = -value
        if value > 1.0:
            value = 2.0 - value
    return float(np.clip(value, 0.0, 1.0))


def estimate_bandwidth(
    times: np.ndarray,
    fallback_bandwidth: float,
    tiny: float = MIN_BANDWIDTH,
) -> float:
    """Scott-style bandwidth for normalized timestamps."""
    times = np.asarray(times, dtype=float)
    if len(times) < MIN_KDE_GROUP_SIZE:
        return max(float(fallback_bandwidth), tiny)
    std = float(np.std(times))
    if std <= tiny:
        return max(float(fallback_bandwidth), tiny)
    return max(std * len(times) ** (-1 / 5), tiny)


def empirical_ks_statistic(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None
    values = np.sort(np.unique(np.concatenate([x, y])))
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    x_cdf = np.searchsorted(x_sorted, values, side="right") / len(x_sorted)
    y_cdf = np.searchsorted(y_sorted, values, side="right") / len(y_sorted)
    return float(np.max(np.abs(x_cdf - y_cdf)))


def empirical_wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None
    q = np.linspace(0.0, 1.0, max(len(x), len(y)))
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def duplicate_pair_rate(df: pd.DataFrame, customer_col: str, product_col: str) -> float:
    if len(df) == 0:
        return 0.0
    unique_pairs = df[[customer_col, product_col]].drop_duplicates()
    return float(1.0 - len(unique_pairs) / len(df))


def compact_labels(values: Iterable[Any]) -> tuple[dict[Any, int], int]:
    mapping: dict[Any, int] = {}
    compact: dict[Any, int] = {}
    for key, value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
        compact[key] = mapping[value]
    return compact, len(mapping)


def scale_counts_largest_remainder(
    counts: dict[tuple[int, int], int], total: int
) -> dict[tuple[int, int], int]:
    current_total = sum(counts.values())
    if current_total == 0:
        raise ValueError("Cannot scale empty block-pair counts.")
    if total == current_total:
        return dict(counts)

    quotas = {key: value * total / current_total for key, value in counts.items()}
    scaled = {key: int(np.floor(quota)) for key, quota in quotas.items()}
    remaining = total - sum(scaled.values())
    if remaining > 0:
        order = sorted(
            quotas,
            key=lambda key: (quotas[key] - scaled[key], counts[key]),
            reverse=True,
        )
        for key in order[:remaining]:
            scaled[key] += 1
    return scaled


def fit_type_constrained_sbm_blocks(
    customer_ids: Iterable[Any],
    product_ids: Iterable[Any],
    unique_pairs: pd.DataFrame,
    customer_col: str,
    product_col: str,
    seed: int,
) -> SBMBlockResult:
    """Fit the same graph-tool nested degree-corrected SBM family RelDiff uses.

    If graph-tool is unavailable, falls back to one block per entity type. The
    fallback is intentionally not a separate clustering method; it only keeps the
    generator runnable in lightweight environments.
    """
    customer_ids = list(customer_ids)
    product_ids = list(product_ids)

    try:
        import graph_tool.all as gt
    except ImportError:
        return type_only_blocks(customer_ids, product_ids)

    try:
        gt.seed_rng(seed)
        graph = gt.Graph(directed=True)
        vertex_map: dict[tuple[str, Any], Any] = {}
        type_label = graph.new_vertex_property("int")

        for customer_id in customer_ids:
            vertex = graph.add_vertex()
            vertex_map[("customer", customer_id)] = vertex
            type_label[vertex] = 0
        for product_id in product_ids:
            vertex = graph.add_vertex()
            vertex_map[("product", product_id)] = vertex
            type_label[vertex] = 1
        graph.vertex_properties["block"] = type_label

        for row in unique_pairs[[customer_col, product_col]].itertuples(index=False):
            customer_id, product_id = row
            graph.add_edge(
                vertex_map[("customer", customer_id)],
                vertex_map[("product", product_id)],
            )

        if graph.num_edges() == 0:
            return type_only_blocks(customer_ids, product_ids)

        state = gt.minimize_nested_blockmodel_dl(
            graph, state_args={"deg_corr": True, "clabel": graph.vp["block"]}
        )
        bottom_state = state.levels[0]
        block_array = bottom_state.b.a

        customer_raw = []
        product_raw = []
        for customer_id in customer_ids:
            vertex = vertex_map[("customer", customer_id)]
            customer_raw.append((customer_id, int(block_array[int(vertex)])))
        for product_id in product_ids:
            vertex = vertex_map[("product", product_id)]
            product_raw.append((product_id, int(block_array[int(vertex)])))

        customer_blocks, num_customer_blocks = compact_labels(customer_raw)
        product_blocks, num_product_blocks = compact_labels(product_raw)
        try:
            description_length = float(state.entropy())
        except Exception:
            description_length = None

        return SBMBlockResult(
            customer_blocks=customer_blocks,
            product_blocks=product_blocks,
            num_customer_blocks=num_customer_blocks,
            num_product_blocks=num_product_blocks,
            used_existing_reldiff_sbm=True,
            description_length=description_length,
        )
    except Exception as exc:
        print(f"RelDiff graph-tool SBM fitting failed; using type-only blocks: {exc}")
        return type_only_blocks(customer_ids, product_ids)


def type_only_blocks(customer_ids: list[Any], product_ids: list[Any]) -> SBMBlockResult:
    return SBMBlockResult(
        customer_blocks={customer_id: 0 for customer_id in customer_ids},
        product_blocks={product_id: 0 for product_id in product_ids},
        num_customer_blocks=1 if customer_ids else 0,
        num_product_blocks=1 if product_ids else 0,
        used_existing_reldiff_sbm=False,
        description_length=None,
    )


class ContinuousTimeTemporalSBMGenerator:
    """Temporal structural generator for Amazon-style review tables."""

    def __init__(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        reviews: pd.DataFrame,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        seed: int = 42,
    ):
        self.customers = customers.copy()
        self.products = products.copy()
        self.raw_reviews = reviews.copy()
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.reviews = self._preprocess_reviews()
        self.min_time = self.reviews[timestamp_col].min()
        self.max_time = self.reviews[timestamp_col].max()
        self.time_span = self.max_time - self.min_time
        self.sbm_result: SBMBlockResult | None = None

        self.block_pair_events: dict[tuple[int, int], EventCollection] = {}
        self.customer_block_events: dict[int, EventCollection] = {}
        self.product_block_events: dict[int, EventCollection] = {}
        self.global_events: EventCollection | None = None
        self.block_pair_event_count: dict[tuple[int, int], int] = {}
        self.customer_event_count: Counter = Counter()
        self.product_event_count: Counter = Counter()
        self.customers_by_block: dict[int, np.ndarray] = {}
        self.products_by_block: dict[int, np.ndarray] = {}
        self.timestamp_fallback_counts: dict[tuple[int, int], Counter] = {}

    @classmethod
    def from_csv(
        cls,
        customers_path: str | Path,
        products_path: str | Path,
        reviews_path: str | Path,
        **kwargs: Any,
    ) -> "ContinuousTimeTemporalSBMGenerator":
        return cls(
            customers=pd.read_csv(customers_path),
            products=pd.read_csv(products_path),
            reviews=pd.read_csv(reviews_path),
            **kwargs,
        )

    def _preprocess_reviews(self) -> pd.DataFrame:
        reviews = self.raw_reviews.copy()
        required = [self.customer_id_col, self.product_id_col, self.timestamp_col]
        missing = [column for column in required if column not in reviews.columns]
        if missing:
            raise ValueError(f"Reviews table is missing required columns: {missing}")
        for name, df, column in (
            ("customers", self.customers, self.customer_id_col),
            ("products", self.products, self.product_id_col),
        ):
            if column not in df.columns:
                raise ValueError(f"{name} table is missing required column {column!r}.")

        reviews[self.timestamp_col] = pd.to_datetime(
            reviews[self.timestamp_col], errors="coerce"
        )
        reviews = reviews.dropna(
            subset=[self.customer_id_col, self.product_id_col, self.timestamp_col]
        ).copy()
        reviews = reviews[
            reviews[self.customer_id_col].isin(self.customers[self.customer_id_col])
        ]
        reviews = reviews[
            reviews[self.product_id_col].isin(self.products[self.product_id_col])
        ]
        reviews = reviews.sort_values(self.timestamp_col, kind="mergesort").reset_index(
            drop=True
        )
        if reviews.empty:
            raise ValueError("No valid review rows remain after preprocessing.")

        min_time = reviews[self.timestamp_col].min()
        max_time = reviews[self.timestamp_col].max()
        span = max_time - min_time
        if span.total_seconds() <= 0:
            reviews["_time_x"] = 0.5
        else:
            reviews["_time_x"] = (
                (reviews[self.timestamp_col] - min_time).dt.total_seconds()
                / span.total_seconds()
            )
        return reviews

    def fit(self) -> "ContinuousTimeTemporalSBMGenerator":
        active_customer_ids = pd.unique(self.reviews[self.customer_id_col])
        active_product_ids = pd.unique(self.reviews[self.product_id_col])
        unique_pairs = self.reviews[
            [self.customer_id_col, self.product_id_col]
        ].drop_duplicates()

        self.sbm_result = fit_type_constrained_sbm_blocks(
            customer_ids=active_customer_ids,
            product_ids=active_product_ids,
            unique_pairs=unique_pairs,
            customer_col=self.customer_id_col,
            product_col=self.product_id_col,
            seed=self.seed,
        )
        self._build_event_collections()
        return self

    def _build_event_collections(self) -> None:
        assert self.sbm_result is not None
        reviews = self.reviews.copy()
        reviews["_customer_block"] = reviews[self.customer_id_col].map(
            self.sbm_result.customer_blocks
        )
        reviews["_product_block"] = reviews[self.product_id_col].map(
            self.sbm_result.product_blocks
        )

        global_bandwidth = estimate_bandwidth(
            reviews["_time_x"].to_numpy(dtype=float), DEFAULT_FALLBACK_BANDWIDTH
        )
        self.global_events = EventCollection.from_records(
            reviews["_time_x"],
            reviews[self.customer_id_col],
            reviews[self.product_id_col],
            fallback_bandwidth=global_bandwidth,
        )

        self.customer_event_count = Counter(reviews[self.customer_id_col])
        self.product_event_count = Counter(reviews[self.product_id_col])

        self.block_pair_events = {}
        self.block_pair_event_count = {}
        for (customer_block, product_block), group in reviews.groupby(
            ["_customer_block", "_product_block"], sort=True
        ):
            key = (int(customer_block), int(product_block))
            collection = EventCollection.from_records(
                group["_time_x"],
                group[self.customer_id_col],
                group[self.product_id_col],
                fallback_bandwidth=global_bandwidth,
            )
            self.block_pair_events[key] = collection
            self.block_pair_event_count[key] = len(group)

        self.customer_block_events = {}
        for customer_block, group in reviews.groupby("_customer_block", sort=True):
            self.customer_block_events[int(customer_block)] = EventCollection.from_records(
                group["_time_x"],
                group[self.customer_id_col],
                fallback_bandwidth=global_bandwidth,
            )

        self.product_block_events = {}
        for product_block, group in reviews.groupby("_product_block", sort=True):
            self.product_block_events[int(product_block)] = EventCollection.from_records(
                group["_time_x"],
                product_ids=group[self.product_id_col],
                fallback_bandwidth=global_bandwidth,
            )

        self.customers_by_block = defaultdict(list)
        for customer_id, block in self.sbm_result.customer_blocks.items():
            self.customers_by_block[block].append(customer_id)
        self.customers_by_block = {
            block: np.asarray(ids, dtype=object)
            for block, ids in self.customers_by_block.items()
        }

        self.products_by_block = defaultdict(list)
        for product_id, block in self.sbm_result.product_blocks.items():
            self.products_by_block[block].append(product_id)
        self.products_by_block = {
            block: np.asarray(ids, dtype=object)
            for block, ids in self.products_by_block.items()
        }

    def generate(
        self,
        num_events: int | None = None,
        output_path: str | Path | None = None,
        debug_dir: str | Path | None = None,
        avoid_duplicate_pairs_same_time_neighborhood: bool = False,
    ) -> pd.DataFrame:
        if self.sbm_result is None:
            self.fit()
        if avoid_duplicate_pairs_same_time_neighborhood:
            print(
                "avoid_duplicate_pairs_same_time_neighborhood is not implemented in "
                "the first version; duplicate customer-product pairs are allowed."
            )

        target_counts = scale_counts_largest_remainder(
            self.block_pair_event_count,
            total=len(self.reviews) if num_events is None else int(num_events),
        )
        self.timestamp_fallback_counts = defaultdict(Counter)

        records = []
        synthetic_times_by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
        for block_pair, count in sorted(target_counts.items()):
            customer_block, product_block = block_pair
            for _ in range(count):
                x, fallback_level = self.sample_timestamp_for_block_pair(block_pair)
                self.timestamp_fallback_counts[block_pair][fallback_level] += 1
                customer_id = self.sample_customer_given_block_time(
                    customer_block, product_block, x
                )
                product_id = self.sample_product_given_block_time(
                    customer_block, product_block, x
                )
                synthetic_times_by_pair[block_pair].append(x)
                records.append(
                    {
                        self.customer_id_col: customer_id,
                        self.product_id_col: product_id,
                        self.timestamp_col: self.denormalize_time(x),
                        "_customer_block": customer_block,
                        "_product_block": product_block,
                        "_time_x": x,
                    }
                )

        synthetic = pd.DataFrame.from_records(records)
        synthetic = synthetic.sort_values(self.timestamp_col, kind="mergesort")
        output = synthetic[
            [self.customer_id_col, self.product_id_col, self.timestamp_col]
        ].reset_index(drop=True)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(output_path, index=False)

        if debug_dir is not None:
            self.write_debug_outputs(
                Path(debug_dir), output, target_counts, synthetic_times_by_pair
            )
        return output

    def sample_timestamp_for_block_pair(self, block_pair: tuple[int, int]) -> tuple[float, str]:
        assert self.global_events is not None
        customer_block, product_block = block_pair
        hierarchy = [
            ("block_pair", self.block_pair_events.get(block_pair)),
            ("customer_block", self.customer_block_events.get(customer_block)),
            ("product_block", self.product_block_events.get(product_block)),
            ("global", self.global_events),
        ]
        for level, collection in hierarchy:
            if collection is None:
                continue
            if level != "global" and len(collection) < MIN_KDE_GROUP_SIZE:
                continue
            sampler = ContinuousKDESampler(collection.times, collection.bandwidth)
            return sampler.sample(self.rng), level
        sampler = ContinuousKDESampler(self.global_events.times, self.global_events.bandwidth)
        return sampler.sample(self.rng), "global"

    def sample_customer_given_block_time(
        self, customer_block: int, product_block: int, x: float
    ) -> Any:
        block_pair = (customer_block, product_block)
        local = self._sample_endpoint_from_collection(
            self.block_pair_events.get(block_pair), x, endpoint="customer"
        )
        if local is not None:
            return local
        block_local = self._sample_endpoint_from_collection(
            self.customer_block_events.get(customer_block), x, endpoint="customer"
        )
        if block_local is not None:
            return block_local
        return self._sample_from_degree_or_uniform(
            self.customers_by_block[customer_block], self.customer_event_count
        )

    def sample_product_given_block_time(
        self, customer_block: int, product_block: int, x: float
    ) -> Any:
        block_pair = (customer_block, product_block)
        local = self._sample_endpoint_from_collection(
            self.block_pair_events.get(block_pair), x, endpoint="product"
        )
        if local is not None:
            return local
        block_local = self._sample_endpoint_from_collection(
            self.product_block_events.get(product_block), x, endpoint="product"
        )
        if block_local is not None:
            return block_local
        return self._sample_from_degree_or_uniform(
            self.products_by_block[product_block], self.product_event_count
        )

    def _sample_endpoint_from_collection(
        self, collection: EventCollection | None, x: float, endpoint: str
    ) -> Any | None:
        if collection is None or len(collection) == 0:
            return None
        ids = collection.customer_ids if endpoint == "customer" else collection.product_ids
        if ids is None or len(ids) == 0:
            return None

        bandwidth = max(collection.bandwidth, MIN_BANDWIDTH)
        left = np.searchsorted(
            collection.times, x - LOCAL_WINDOW_BANDWIDTHS * bandwidth, side="left"
        )
        right = np.searchsorted(
            collection.times, x + LOCAL_WINDOW_BANDWIDTHS * bandwidth, side="right"
        )
        if right <= left:
            return None
        window_times = collection.times[left:right]
        weights = np.exp(-0.5 * ((window_times - x) / bandwidth) ** 2)
        weight_sum = weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            return None
        weights = weights / weight_sum
        choice = self.rng.choice(np.arange(left, right), p=weights)
        return ids[int(choice)]

    def _sample_from_degree_or_uniform(self, ids: np.ndarray, counts: Counter) -> Any:
        if len(ids) == 0:
            raise ValueError("Cannot sample endpoint from an empty block.")
        weights = np.asarray([counts.get(entity_id, 0) for entity_id in ids], dtype=float)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            return ids[int(self.rng.choice(np.arange(len(ids)), p=weights))]
        return ids[int(self.rng.integers(0, len(ids)))]

    def denormalize_time(self, x: float) -> pd.Timestamp:
        if self.time_span.total_seconds() <= 0:
            return pd.Timestamp(self.min_time)
        return pd.Timestamp(self.min_time + x * self.time_span)

    def write_debug_outputs(
        self,
        debug_dir: Path,
        synthetic: pd.DataFrame,
        target_counts: dict[tuple[int, int], int],
        synthetic_times_by_pair: dict[tuple[int, int], list[float]],
    ) -> None:
        assert self.sbm_result is not None
        assert self.global_events is not None
        debug_dir.mkdir(parents=True, exist_ok=True)

        real_times = self.reviews["_time_x"].to_numpy(dtype=float)
        synthetic_times = (
            (pd.to_datetime(synthetic[self.timestamp_col]) - self.min_time).dt.total_seconds()
            / max(self.time_span.total_seconds(), 1.0)
        ).to_numpy(dtype=float)

        summary = {
            "generator": GENERATOR_NAME,
            "num_real_reviews": int(len(self.reviews)),
            "num_synthetic_reviews": int(len(synthetic)),
            "min_time": str(self.min_time),
            "max_time": str(self.max_time),
            "num_active_customers": int(len(self.sbm_result.customer_blocks)),
            "num_active_products": int(len(self.sbm_result.product_blocks)),
            "num_customer_blocks": int(self.sbm_result.num_customer_blocks),
            "num_product_blocks": int(self.sbm_result.num_product_blocks),
            "num_nonzero_block_pairs": int(len(self.block_pair_event_count)),
            "used_existing_reldiff_sbm": self.sbm_result.used_existing_reldiff_sbm,
            "sbm_description_length": self.sbm_result.description_length,
            "global_timestamp_bandwidth": self.global_events.bandwidth,
            "seed": self.seed,
            "real_duplicate_pair_rate": duplicate_pair_rate(
                self.reviews, self.customer_id_col, self.product_id_col
            ),
            "synthetic_duplicate_pair_rate": duplicate_pair_rate(
                synthetic, self.customer_id_col, self.product_id_col
            ),
        }
        self._write_json(debug_dir / "temporal_sbm_summary.json", summary)

        self.write_sbm_summary(debug_dir / "sbm_summary.json")
        self.write_block_pair_debug(debug_dir, target_counts)
        self.write_block_debug(debug_dir)
        self.write_assignment_debug(debug_dir)

        per_pair_ks = []
        for block_pair, real_count in self.block_pair_event_count.items():
            if real_count == 0:
                continue
            real_pair_times = self.block_pair_events[block_pair].times
            syn_pair_times = np.asarray(synthetic_times_by_pair.get(block_pair, []))
            ks = empirical_ks_statistic(real_pair_times, syn_pair_times)
            if ks is not None:
                per_pair_ks.append(ks)

        diagnostics = {
            "global_timestamp_ks": empirical_ks_statistic(real_times, synthetic_times),
            "global_timestamp_wasserstein": empirical_wasserstein_1d(
                real_times, synthetic_times
            ),
            "per_block_pair_timestamp_ks_mean": (
                float(np.mean(per_pair_ks)) if per_pair_ks else None
            ),
            "per_block_pair_timestamp_ks_median": (
                float(np.median(per_pair_ks)) if per_pair_ks else None
            ),
            "per_block_pair_timestamp_ks_num_pairs": len(per_pair_ks),
        }
        self._write_json(debug_dir / "temporal_sbm_timestamp_diagnostics.json", diagnostics)

    def write_sbm_summary(self, path: Path) -> None:
        assert self.sbm_result is not None
        summary = {
            "num_active_customers": int(len(self.sbm_result.customer_blocks)),
            "num_active_products": int(len(self.sbm_result.product_blocks)),
            "num_review_rows": int(len(self.reviews)),
            "num_unique_customer_product_pairs": int(
                len(self.reviews[[self.customer_id_col, self.product_id_col]].drop_duplicates())
            ),
            "num_customer_blocks": int(self.sbm_result.num_customer_blocks),
            "num_product_blocks": int(self.sbm_result.num_product_blocks),
            "total_blocks": int(
                self.sbm_result.num_customer_blocks + self.sbm_result.num_product_blocks
            ),
            "description_length": self.sbm_result.description_length,
            "seed": self.seed,
        }
        self._write_json(path, summary)

    def write_block_pair_debug(
        self, debug_dir: Path, target_counts: dict[tuple[int, int], int]
    ) -> None:
        rows = []
        total_real = sum(self.block_pair_event_count.values())
        total_syn = sum(target_counts.values())
        for (customer_block, product_block), real_count in sorted(
            self.block_pair_event_count.items()
        ):
            syn_count = target_counts.get((customer_block, product_block), 0)
            collection = self.block_pair_events[(customer_block, product_block)]
            rows.append(
                {
                    "customer_block": customer_block,
                    "product_block": product_block,
                    "real_event_count": real_count,
                    "synthetic_event_count": syn_count,
                    "real_event_share": real_count / total_real if total_real else 0.0,
                    "synthetic_event_share": syn_count / total_syn if total_syn else 0.0,
                    "timestamp_bandwidth": collection.bandwidth,
                    "fallback_level_used_count": json.dumps(
                        dict(self.timestamp_fallback_counts.get((customer_block, product_block), {})),
                        sort_keys=True,
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(debug_dir / "temporal_sbm_block_pairs.csv", index=False)

    def write_block_debug(self, debug_dir: Path) -> None:
        customer_rows = []
        for block, ids in sorted(self.customers_by_block.items()):
            degrees = np.asarray([self.customer_event_count.get(entity_id, 0) for entity_id in ids])
            customer_rows.append(
                {
                    "customer_block": block,
                    "num_customers": len(ids),
                    "real_event_count": int(degrees.sum()),
                    "mean_customer_degree": float(degrees.mean()) if len(degrees) else 0.0,
                    "max_customer_degree": int(degrees.max()) if len(degrees) else 0,
                }
            )
        pd.DataFrame(customer_rows).to_csv(
            debug_dir / "temporal_sbm_customer_blocks.csv", index=False
        )

        product_rows = []
        for block, ids in sorted(self.products_by_block.items()):
            degrees = np.asarray([self.product_event_count.get(entity_id, 0) for entity_id in ids])
            product_rows.append(
                {
                    "product_block": block,
                    "num_products": len(ids),
                    "real_event_count": int(degrees.sum()),
                    "mean_product_degree": float(degrees.mean()) if len(degrees) else 0.0,
                    "max_product_degree": int(degrees.max()) if len(degrees) else 0,
                }
            )
        pd.DataFrame(product_rows).to_csv(
            debug_dir / "temporal_sbm_product_blocks.csv", index=False
        )

    def write_assignment_debug(self, debug_dir: Path) -> None:
        assert self.sbm_result is not None
        pd.DataFrame(
            [
                {self.customer_id_col: customer_id, "customer_block": block}
                for customer_id, block in self.sbm_result.customer_blocks.items()
            ]
        ).to_csv(debug_dir / "temporal_sbm_customer_assignments.csv", index=False)
        pd.DataFrame(
            [
                {self.product_id_col: product_id, "product_block": block}
                for product_id, block in self.sbm_result.product_blocks.items()
            ]
        ).to_csv(debug_dir / "temporal_sbm_product_assignments.csv", index=False)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with path.open("w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
