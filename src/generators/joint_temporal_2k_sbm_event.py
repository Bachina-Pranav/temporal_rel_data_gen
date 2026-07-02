"""Joint temporal 2K-SBM event-spine generator."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .event_affinity import (
    EventAffinityScorer,
    EventScoreWeights,
    ProductAgeAffinity,
    StaticCustomerProductAffinity,
    product_lifecycle_table,
    stable_softmax,
)
from .temporal_activity_models import TemporalActivityModel, canonical_day_bucket


METHOD_NAME = "joint_temporal_2k_sbm_event"
METHOD_ALIAS = "joint_temporal_event_spine"


class JointTemporal2KSBMEventGenerator:
    """Approximate p(customer_id, product_id, review_time) using a decomposed event intensity.

    This generator does not estimate a dense probability for every
    (customer, product, time) triple. Instead, it combines block-pair-time
    counts, customer-time activity, product-time activity, static
    customer-product affinity, and product lifecycle compatibility into a
    time-dependent event score F_{u,i,t}.

    F_static(u,i) is not the final pairing function. Pairing uses
    event_score(u,i,t), which is time-dependent.
    """

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        structure_debug_dir: Optional[str | Path] = None,
        time_granularity: str = "day",
        alpha_customer_time: float = 10.0,
        alpha_product_time: float = 5.0,
        block_time_smoothing: float = 5.0,
        age_smoothing: float = 5.0,
        mf_rank: int = 32,
        lambda_static: float = 1.0,
        lambda_ut: float = 1.0,
        lambda_it: float = 1.0,
        lambda_age: float = 0.5,
        lambda_deg: float = 0.1,
        lambda_dup: float = 1.0,
        lambda_mem: float = 2.0,
        sampling_temperature: float = 1.0,
        customer_candidate_pool_size: int = 256,
        product_candidate_pool_size: int = 256,
        allow_degree_slack: bool = False,
        seed: int = 42,
    ):
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.structure_debug_dir = Path(structure_debug_dir) if structure_debug_dir else None
        self.time_granularity = time_granularity
        self.alpha_customer_time = float(alpha_customer_time)
        self.alpha_product_time = float(alpha_product_time)
        self.block_time_smoothing = float(block_time_smoothing)
        self.age_smoothing = float(age_smoothing)
        self.mf_rank = int(mf_rank)
        self.weights = EventScoreWeights(
            lambda_static=float(lambda_static),
            lambda_ut=float(lambda_ut),
            lambda_it=float(lambda_it),
            lambda_age=float(lambda_age),
            lambda_deg=float(lambda_deg),
            lambda_dup=float(lambda_dup),
            lambda_mem=float(lambda_mem),
        )
        self.sampling_temperature = float(sampling_temperature)
        self.customer_candidate_pool_size = int(customer_candidate_pool_size)
        self.product_candidate_pool_size = int(product_candidate_pool_size)
        self.allow_degree_slack = bool(allow_degree_slack)
        self.seed = int(seed)
        self.real_df: Optional[pd.DataFrame] = None
        self.customer_blocks: Dict[Any, int] = {}
        self.product_blocks: Dict[Any, int] = {}
        self.block_pair_time_counts = pd.DataFrame()
        self.customer_activity: Optional[TemporalActivityModel] = None
        self.product_activity: Optional[TemporalActivityModel] = None
        self.product_lifecycle = pd.DataFrame()
        self.age_affinity: Optional[ProductAgeAffinity] = None
        self.static_affinity: Optional[StaticCustomerProductAffinity] = None
        self.scorer: Optional[EventAffinityScorer] = None
        self.customer_degrees: Dict[Any, int] = {}
        self.product_degrees: Dict[Any, int] = {}
        self.real_event_set: set[tuple[Any, Any, str]] = set()
        self.sampling_failures: List[Dict[str, Any]] = []
        self.synthetic_df: Optional[pd.DataFrame] = None

    def fit(self, real_df: pd.DataFrame) -> "JointTemporal2KSBMEventGenerator":
        required = [self.customer_id_col, self.product_id_col, self.timestamp_col]
        missing = [col for col in required if col not in real_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        frame = real_df[required].copy()
        frame["_time_bucket"] = canonical_day_bucket(frame[self.timestamp_col])
        self.real_df = frame
        self.customer_blocks, self.product_blocks = self._load_or_fallback_blocks(frame)
        frame["_customer_block"] = frame[self.customer_id_col].map(self.customer_blocks).fillna(0).astype(int)
        frame["_product_block"] = frame[self.product_id_col].map(self.product_blocks).fillna(0).astype(int)
        self.customer_degrees = frame[self.customer_id_col].value_counts().astype(int).to_dict()
        self.product_degrees = frame[self.product_id_col].value_counts().astype(int).to_dict()
        self.real_event_set = {
            (row[self.customer_id_col], row[self.product_id_col], row["_time_bucket"])
            for _, row in frame.iterrows()
        }
        self.block_pair_time_counts = (
            frame.groupby(["_customer_block", "_product_block", "_time_bucket"])
            .size()
            .reset_index(name="count")
            .rename(
                columns={
                    "_customer_block": "customer_block",
                    "_product_block": "product_block",
                    "_time_bucket": "time_bucket",
                }
            )
            .sort_values(["time_bucket", "customer_block", "product_block"])
            .reset_index(drop=True)
        )
        self.customer_activity = TemporalActivityModel.fit_customer_activity(
            frame,
            self.customer_id_col,
            "_time_bucket",
            self.customer_blocks,
            alpha_customer_time=self.alpha_customer_time,
            block_time_smoothing=self.block_time_smoothing,
        )
        self.product_activity = TemporalActivityModel.fit_product_activity(
            frame,
            self.product_id_col,
            "_time_bucket",
            self.product_blocks,
            alpha_product_time=self.alpha_product_time,
            block_time_smoothing=self.block_time_smoothing,
        )
        self.product_lifecycle = product_lifecycle_table(
            frame,
            self.product_id_col,
            "_time_bucket",
            self.product_blocks,
        )
        self.age_affinity = ProductAgeAffinity(age_smoothing=self.age_smoothing).fit(
            frame,
            self.customer_id_col,
            self.product_id_col,
            "_time_bucket",
            self.customer_blocks,
            self.product_lifecycle,
        )
        self.static_affinity = StaticCustomerProductAffinity(rank=self.mf_rank, seed=self.seed).fit(
            frame, self.customer_id_col, self.product_id_col
        )
        self.scorer = EventAffinityScorer(
            self.static_affinity,
            self.customer_activity,
            self.product_activity,
            self.age_affinity,
            self.customer_blocks,
            self.real_event_set,
            self.weights,
        )
        return self

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        if self.real_df is None or self.scorer is None:
            raise RuntimeError("Call fit before sample.")
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        slots = self._expanded_slots()
        slots["_slot_order"] = rng.permutation(len(slots))
        slots = slots.sort_values(["_slot_order"]).reset_index(drop=True)
        slots[self.customer_id_col] = None
        slots[self.product_id_col] = None
        rem_customer = Counter(self.customer_degrees)
        rem_product = Counter(self.product_degrees)
        self.sampling_failures = []
        self._assign_customers(slots, rem_customer, rng)
        self._assign_products(slots, rem_product, rng)
        residual_customers = {k: int(v) for k, v in rem_customer.items() if v != 0}
        residual_products = {k: int(v) for k, v in rem_product.items() if v != 0}
        if residual_customers or residual_products:
            message = {
                "type": "residual_degree_quota",
                "residual_customers": residual_customers,
                "residual_products": residual_products,
            }
            self.sampling_failures.append(message)
            if not self.allow_degree_slack:
                raise RuntimeError(f"Degree quota repair failed: {message}")
        synthetic = slots[[self.customer_id_col, self.product_id_col, "time_bucket"]].rename(
            columns={"time_bucket": self.timestamp_col}
        )
        synthetic = synthetic.sort_values([self.timestamp_col, self.customer_id_col, self.product_id_col]).reset_index(drop=True)
        self.synthetic_df = synthetic
        return synthetic

    def save_debug(self, debug_dir: str | Path) -> None:
        if self.real_df is None:
            raise RuntimeError("Call fit before save_debug.")
        debug = Path(debug_dir)
        debug.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                self.customer_id_col: list(self.customer_blocks.keys()),
                "customer_block": list(self.customer_blocks.values()),
            }
        ).to_csv(debug / "customer_blocks.csv", index=False)
        pd.DataFrame(
            {
                self.product_id_col: list(self.product_blocks.keys()),
                "product_block": list(self.product_blocks.values()),
            }
        ).to_csv(debug / "product_blocks.csv", index=False)
        self.block_pair_time_counts.to_csv(debug / "block_pair_time_counts.csv", index=False)
        with (debug / "block_pair_time_tensor_summary.json").open("w") as handle:
            json.dump(self.block_pair_time_summary(), handle, indent=2)
            handle.write("\n")
        self.customer_activity.save_summary(debug / "customer_time_activity_summary.json")
        self.product_activity.save_summary(debug / "product_time_activity_summary.json")
        self.product_lifecycle.to_csv(debug / "product_lifecycle.csv", index=False)
        self.age_affinity.save(debug / "product_age_affinity.csv")
        self.static_affinity.save_summary(debug / "affinity_model_summary.json")
        with (debug / "sampling_failures.json").open("w") as handle:
            json.dump(self.sampling_failures, handle, indent=2)
            handle.write("\n")

    def save_metadata(self, path: str | Path) -> None:
        if self.real_df is None:
            raise RuntimeError("Call fit before save_metadata.")
        metadata = self.metadata()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

    def metadata(self) -> Dict[str, Any]:
        num_events = int(len(self.real_df)) if self.real_df is not None else 0
        return {
            "method": METHOD_NAME,
            "alias": METHOD_ALIAS,
            "generates_joint_events": True,
            "event_tuple": [self.customer_id_col, self.product_id_col, self.timestamp_col],
            "uses_dense_F_u_i_t": False,
            "uses_static_affinity_component": True,
            "uses_time_dependent_pairing_score": True,
            "preserves_total_num_events": True,
            "preserves_daily_counts": True,
            "preserves_customer_degrees": True,
            "preserves_product_degrees": True,
            "preserves_block_pair_time_counts": True,
            "seed": int(self.seed),
            "timestamp_col": self.timestamp_col,
            "time_granularity": self.time_granularity,
            "num_customers": int(len(self.customer_degrees)),
            "num_products": int(len(self.product_degrees)),
            "num_events": num_events,
            "num_time_buckets": int(self.real_df["_time_bucket"].nunique()) if self.real_df is not None else 0,
            "num_customer_blocks": int(len(set(self.customer_blocks.values()))),
            "num_product_blocks": int(len(set(self.product_blocks.values()))),
            "event_score_formula": (
                "lambda_static * F_static(u,i) + lambda_ut * log P_u(t) + "
                "lambda_it * log P_i(t) + lambda_age * A[b_u, age_bin_i(t)] + "
                "lambda_deg * log1p(rem_i) - lambda_dup * duplicate_count(u,i) - "
                "lambda_mem * real_event_overlap(u,i,t)"
            ),
            "weights": asdict(self.weights),
            "sampling_temperature": float(self.sampling_temperature),
            "customer_candidate_pool_size": int(self.customer_candidate_pool_size),
            "product_candidate_pool_size": int(self.product_candidate_pool_size),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

    def block_pair_time_summary(self) -> Dict[str, Any]:
        counts = self.block_pair_time_counts["count"].to_numpy(dtype=float)
        return {
            "num_customer_blocks": int(len(set(self.customer_blocks.values()))),
            "num_product_blocks": int(len(set(self.product_blocks.values()))),
            "num_time_buckets": int(self.block_pair_time_counts["time_bucket"].nunique()),
            "num_nonzero_cells": int(len(self.block_pair_time_counts)),
            "total_count": int(counts.sum()) if len(counts) else 0,
            "max_cell_count": int(counts.max()) if len(counts) else 0,
            "mean_nonzero_cell_count": float(counts.mean()) if len(counts) else 0.0,
            "preserves_block_pair_time_counts_default": True,
        }

    def _load_or_fallback_blocks(self, frame: pd.DataFrame) -> tuple[Dict[Any, int], Dict[Any, int]]:
        customer_blocks = load_blocks(
            self.structure_debug_dir,
            "customer_blocks.csv",
            [self.customer_id_col, "id", "customer_id"],
            ["customer_block", "block"],
        )
        product_blocks = load_blocks(
            self.structure_debug_dir,
            "product_blocks.csv",
            [self.product_id_col, "id", "product_id"],
            ["product_block", "block"],
        )
        if not customer_blocks:
            customer_blocks = {entity: 0 for entity in frame[self.customer_id_col].unique()}
        if not product_blocks:
            product_blocks = {entity: 0 for entity in frame[self.product_id_col].unique()}
        return customer_blocks, product_blocks

    def _expanded_slots(self) -> pd.DataFrame:
        parts = []
        for _, row in self.block_pair_time_counts.iterrows():
            count = int(row["count"])
            if count <= 0:
                continue
            parts.append(
                pd.DataFrame(
                    {
                        "customer_block": [int(row["customer_block"])] * count,
                        "product_block": [int(row["product_block"])] * count,
                        "time_bucket": [row["time_bucket"]] * count,
                    }
                )
            )
        if not parts:
            return pd.DataFrame(columns=["customer_block", "product_block", "time_bucket"])
        return pd.concat(parts, ignore_index=True)

    def _assign_customers(self, slots: pd.DataFrame, rem_customer: Counter, rng: np.random.Generator) -> None:
        block_customers: Dict[int, List[Any]] = defaultdict(list)
        for customer, block in self.customer_blocks.items():
            if customer in self.customer_degrees:
                block_customers[int(block)].append(customer)
        grouped = list(slots.groupby(["customer_block", "time_bucket"]).groups.items())
        rng.shuffle(grouped)
        for (block, time_bucket), indices in grouped:
            for idx in rng.permutation(list(indices)):
                candidates = [u for u in block_customers[int(block)] if rem_customer[u] > 0]
                if not candidates:
                    self.sampling_failures.append({"type": "customer_block_exhausted", "block": int(block), "time_bucket": time_bucket})
                    if not self.allow_degree_slack:
                        raise RuntimeError(f"No remaining customers in block {block}")
                    continue
                weights = np.asarray(
                    [rem_customer[u] * self.customer_activity.probability(u, time_bucket) for u in candidates],
                    dtype=float,
                )
                chosen = weighted_choice(candidates, weights, rng)
                slots.at[int(idx), self.customer_id_col] = chosen
                rem_customer[chosen] -= 1

    def _assign_products(self, slots: pd.DataFrame, rem_product: Counter, rng: np.random.Generator) -> None:
        block_products: Dict[int, List[Any]] = defaultdict(list)
        for product, block in self.product_blocks.items():
            if product in self.product_degrees:
                block_products[int(block)].append(product)
        duplicate_counts: Dict[tuple[Any, Any], int] = defaultdict(int)
        for idx in rng.permutation(slots.index.to_numpy()):
            product_block = int(slots.at[int(idx), "product_block"])
            time_bucket = slots.at[int(idx), "time_bucket"]
            customer = slots.at[int(idx), self.customer_id_col]
            candidates = [i for i in block_products[product_block] if rem_product[i] > 0]
            if not candidates:
                self.sampling_failures.append(
                    {"type": "product_block_exhausted", "block": product_block, "time_bucket": time_bucket}
                )
                if not self.allow_degree_slack:
                    raise RuntimeError(f"No remaining products in block {product_block}")
                continue
            pool = self._candidate_pool(candidates, rem_product, self.product_activity, time_bucket, self.product_candidate_pool_size, rng)
            rem_values = [rem_product[i] for i in pool]
            scores = self.scorer.event_score(customer, pool, time_bucket, rem_values, duplicate_counts)
            probs = stable_softmax(scores, self.sampling_temperature)
            chosen = pool[int(rng.choice(len(pool), p=probs))]
            slots.at[int(idx), self.product_id_col] = chosen
            rem_product[chosen] -= 1
            duplicate_counts[(customer, chosen)] += 1

    def _candidate_pool(
        self,
        candidates: List[Any],
        remaining: Counter,
        activity: TemporalActivityModel,
        time_bucket: Any,
        pool_size: int,
        rng: np.random.Generator,
    ) -> List[Any]:
        if len(candidates) <= int(pool_size):
            return list(candidates)
        weights = np.asarray([remaining[item] * activity.probability(item, time_bucket) for item in candidates], dtype=float)
        probs = weights / np.clip(weights.sum(), 1e-12, None)
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = None
        indices = rng.choice(len(candidates), size=int(pool_size), replace=False, p=probs)
        return [candidates[int(index)] for index in indices]


def load_blocks(
    structure_debug_dir: Optional[Path],
    filename: str,
    id_candidates: List[str],
    block_candidates: List[str],
) -> Dict[Any, int]:
    if structure_debug_dir is None:
        return {}
    path = structure_debug_dir / filename
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    id_col = next((col for col in id_candidates if col in frame.columns), None)
    block_col = next((col for col in block_candidates if col in frame.columns), None)
    if id_col is None or block_col is None:
        return {}
    return dict(zip(frame[id_col], pd.to_numeric(frame[block_col], errors="coerce").fillna(0).astype(int)))


def weighted_choice(candidates: List[Any], weights: np.ndarray, rng: np.random.Generator) -> Any:
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    if weights.sum() <= 0:
        return candidates[int(rng.integers(0, len(candidates)))]
    return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
