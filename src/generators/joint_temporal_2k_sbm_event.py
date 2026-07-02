"""Joint temporal 2K-SBM event-spine generator."""

from __future__ import annotations

import json
import time
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
        sampling_mode: str = "fast_time_conditioned",
        use_static_affinity: bool = False,
        use_product_lifecycle_affinity: bool = False,
        enable_fast_repair: bool = True,
        fast_repair_attempts: int = 10,
        preserve_block_pair_time_counts: bool = True,
        preserve_degrees: bool = True,
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
        if sampling_mode not in {"fast_time_conditioned", "candidate_affinity"}:
            raise ValueError("sampling_mode must be 'fast_time_conditioned' or 'candidate_affinity'")
        self.sampling_mode = sampling_mode
        self.use_static_affinity = bool(use_static_affinity)
        self.use_product_lifecycle_affinity = bool(use_product_lifecycle_affinity)
        self.enable_fast_repair = bool(enable_fast_repair)
        self.fast_repair_attempts = int(fast_repair_attempts)
        self.preserve_block_pair_time_counts = bool(preserve_block_pair_time_counts)
        self.preserve_degrees = bool(preserve_degrees)
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
        self.runtime_metadata: Dict[str, Any] = {}

    def fit(self, real_df: pd.DataFrame) -> "JointTemporal2KSBMEventGenerator":
        fit_start = time.time()
        print("[fit] canonicalizing input and building constraints", flush=True)
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
        self.block_pair_time_counts = self._build_block_pair_time_counts(frame)
        print("[fit] fitting customer/product time activity models", flush=True)
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
        if self.use_product_lifecycle_affinity and self.weights.lambda_age != 0.0:
            print("[fit] fitting optional product lifecycle affinity", flush=True)
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
        else:
            self.product_lifecycle = pd.DataFrame(
                columns=["product_id", "product_block", "degree", "first_time", "last_time", "peak_time", "active_span_days", "activity_entropy"]
            )
            self.age_affinity = ZeroProductAgeAffinity()
        fit_static = self.weights.lambda_static != 0.0 and (
            self.sampling_mode == "candidate_affinity" or self.use_static_affinity
        )
        if fit_static:
            print("[fit] fitting optional static customer-product affinity", flush=True)
            self.static_affinity = StaticCustomerProductAffinity(rank=self.mf_rank, seed=self.seed).fit(
                frame, self.customer_id_col, self.product_id_col
            )
        else:
            self.static_affinity = ZeroStaticCustomerProductAffinity()
        self.scorer = EventAffinityScorer(
            self.static_affinity,
            self.customer_activity,
            self.product_activity,
            self.age_affinity,
            self.customer_blocks,
            self.real_event_set,
            self.weights,
        )
        self.runtime_metadata["fit_seconds"] = float(time.time() - fit_start)
        print(f"[fit] done in {self.runtime_metadata['fit_seconds']:.2f}s", flush=True)
        return self

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        if self.real_df is None or self.scorer is None:
            raise RuntimeError("Call fit before sample.")
        if self.sampling_mode == "fast_time_conditioned":
            return self.sample_fast_time_conditioned(seed=seed)
        return self.sample_candidate_affinity(seed=seed)

    def sample_candidate_affinity(self, seed: Optional[int] = None) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        sample_start = time.time()
        print("[sample-candidate] using expensive candidate-affinity ablation sampler", flush=True)
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
        self._finish_runtime(sample_start, len(synthetic), len(self.block_pair_time_counts))
        return synthetic

    def sample_fast_time_conditioned(self, seed: Optional[int] = None) -> pd.DataFrame:
        """Fast O(N)-ish sampler using time-conditioned block/entity stub draws."""

        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        sample_start = time.time()
        print("[sample-fast] using scalable time-conditioned sampler", flush=True)
        rem_customer = Counter(self.customer_degrees)
        rem_product = Counter(self.product_degrees)
        self.sampling_failures = []
        duplicate_counts: Dict[tuple[Any, Any], int] = defaultdict(int)
        events: List[tuple[Any, Any, str]] = []
        cells = self.block_pair_time_counts.sort_values(
            ["time_bucket", "count", "customer_block", "product_block"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
        total_cells = len(cells)
        total_events = int(cells["count"].sum()) if total_cells else 0
        progress_every = max(1, total_cells // 20)
        for cell_index, row in cells.iterrows():
            customer_block = int(row["customer_block"])
            product_block = int(row["product_block"])
            time_bucket = row["time_bucket"]
            n = int(row["count"])
            customer_ids, customer_probs = self.customer_activity.probabilities_for_block_time(customer_block, time_bucket)
            product_ids, product_probs = self.product_activity.probabilities_for_block_time(product_block, time_bucket)
            sampled_customers, customer_status = sample_entities_with_quotas(
                customer_ids,
                rem_customer,
                customer_probs,
                n,
                rng,
            )
            sampled_products, product_status = sample_entities_with_quotas(
                product_ids,
                rem_product,
                product_probs,
                n,
                rng,
            )
            if not customer_status["success"] or not product_status["success"]:
                sampled_customers, sampled_products = self._fast_cell_fallback(
                    customer_block,
                    product_block,
                    time_bucket,
                    n,
                    rem_customer,
                    rem_product,
                    rng,
                    customer_status,
                    product_status,
                )
            rng.shuffle(sampled_products)
            if self.enable_fast_repair and self.fast_repair_attempts > 0:
                sampled_products = self._repair_fast_cell_pairs(
                    sampled_customers,
                    sampled_products,
                    time_bucket,
                    duplicate_counts,
                    rng,
                )
            for customer, product in zip(sampled_customers, sampled_products):
                events.append((customer, product, time_bucket))
                rem_customer[customer] -= 1
                rem_product[product] -= 1
                duplicate_counts[(customer, product)] += 1
            if (cell_index + 1) % progress_every == 0 or cell_index + 1 == total_cells:
                elapsed = time.time() - sample_start
                print(
                    f"[sample-fast] processed {cell_index + 1}/{total_cells} cells, "
                    f"events generated {len(events)}/{total_events}, elapsed {elapsed:.1f}s",
                    flush=True,
                )
        residual_customers = {k: int(v) for k, v in rem_customer.items() if v != 0}
        residual_products = {k: int(v) for k, v in rem_product.items() if v != 0}
        if residual_customers or residual_products:
            message = {
                "type": "residual_degree_quota",
                "residual_customers": residual_customers,
                "residual_products": residual_products,
                "suggestion": (
                    "Try --preserve-block-pair-time-counts false or "
                    "--allow-degree-slack if exact BPT cells are infeasible."
                ),
            }
            self.sampling_failures.append(message)
            if not self.allow_degree_slack:
                raise RuntimeError(f"Degree quota repair failed: {message}")
        synthetic = pd.DataFrame(events, columns=[self.customer_id_col, self.product_id_col, self.timestamp_col])
        synthetic = synthetic.sort_values([self.timestamp_col, self.customer_id_col, self.product_id_col]).reset_index(drop=True)
        self.synthetic_df = synthetic
        self._finish_runtime(sample_start, len(synthetic), total_cells)
        return synthetic

    def _fast_cell_fallback(
        self,
        customer_block: int,
        product_block: int,
        time_bucket: str,
        n: int,
        rem_customer: Counter,
        rem_product: Counter,
        rng: np.random.Generator,
        customer_status: Dict[str, Any],
        product_status: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        failure = {
            "type": "fast_cell_quota_fallback",
            "customer_block": int(customer_block),
            "product_block": int(product_block),
            "time_bucket": time_bucket,
            "count": int(n),
            "customer_status": customer_status,
            "product_status": product_status,
        }
        customer_ids, _ = self.customer_activity.probabilities_for_block_time(customer_block, time_bucket)
        product_ids, _ = self.product_activity.probabilities_for_block_time(product_block, time_bucket)
        sampled_customers, customer_degree_status = sample_entities_with_quotas(
            customer_ids,
            rem_customer,
            np.ones(len(customer_ids), dtype=float),
            n,
            rng,
        )
        sampled_products, product_degree_status = sample_entities_with_quotas(
            product_ids,
            rem_product,
            np.ones(len(product_ids), dtype=float),
            n,
            rng,
        )
        failure["fallback_customer_status"] = customer_degree_status
        failure["fallback_product_status"] = product_degree_status
        self.sampling_failures.append(failure)
        if customer_degree_status["success"] and product_degree_status["success"]:
            return sampled_customers, sampled_products
        message = (
            f"Cannot fill strict block-pair-time cell ({customer_block}, {product_block}, {time_bucket}) "
            f"with n={n}. Try --preserve-block-pair-time-counts false or --allow-degree-slack."
        )
        if not self.allow_degree_slack:
            raise RuntimeError(message)
        return sampled_customers, sampled_products

    def _repair_fast_cell_pairs(
        self,
        customers: np.ndarray,
        products: np.ndarray,
        time_bucket: str,
        duplicate_counts: Mapping[tuple[Any, Any], int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        products = np.asarray(products, dtype=object).copy()
        if len(products) <= 1:
            return products
        for idx in range(len(products)):
            current_badness = self._pair_badness(customers[idx], products[idx], time_bucket, duplicate_counts)
            if current_badness <= 0:
                continue
            for _ in range(self.fast_repair_attempts):
                other = int(rng.integers(0, len(products)))
                if other == idx:
                    continue
                before = current_badness + self._pair_badness(customers[other], products[other], time_bucket, duplicate_counts)
                after = (
                    self._pair_badness(customers[idx], products[other], time_bucket, duplicate_counts)
                    + self._pair_badness(customers[other], products[idx], time_bucket, duplicate_counts)
                )
                if after < before:
                    products[idx], products[other] = products[other], products[idx]
                    current_badness = self._pair_badness(customers[idx], products[idx], time_bucket, duplicate_counts)
                    if current_badness <= 0:
                        break
        return products

    def _pair_badness(
        self,
        customer: Any,
        product: Any,
        time_bucket: str,
        duplicate_counts: Mapping[tuple[Any, Any], int],
    ) -> float:
        badness = 1000.0 * int((customer, product, time_bucket) in self.real_event_set)
        badness += 100.0 * int(duplicate_counts.get((customer, product), 0))
        if self.use_static_affinity and self.weights.lambda_static != 0.0:
            badness -= self.weights.lambda_static * float(self.static_affinity.score_one(customer, product))
        if self.use_product_lifecycle_affinity and self.weights.lambda_age != 0.0:
            customer_block = int(self.customer_blocks.get(customer, 0))
            badness -= self.weights.lambda_age * float(self.age_affinity.score(customer_block, product, time_bucket))
        return badness

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
        metadata = {
            "method": METHOD_NAME,
            "alias": METHOD_ALIAS,
            "generates_joint_events": True,
            "event_tuple": [self.customer_id_col, self.product_id_col, self.timestamp_col],
            "uses_dense_F_u_i_t": False,
            "uses_static_affinity_component": bool(
                self.weights.lambda_static != 0.0
                and (self.sampling_mode == "candidate_affinity" or self.use_static_affinity)
            ),
            "uses_time_dependent_pairing_score": True,
            "sampling_mode": self.sampling_mode,
            "scalable_default": self.sampling_mode == "fast_time_conditioned",
            "uses_candidate_pool_scoring": self.sampling_mode == "candidate_affinity",
            "uses_static_affinity_in_default_sampler": bool(
                self.sampling_mode == "fast_time_conditioned" and self.use_static_affinity
            ),
            "uses_time_conditioned_customer_sampling": True,
            "uses_time_conditioned_product_sampling": True,
            "use_static_affinity": bool(self.use_static_affinity),
            "use_product_lifecycle_affinity": bool(self.use_product_lifecycle_affinity),
            "enable_fast_repair": bool(self.enable_fast_repair),
            "fast_repair_attempts": int(self.fast_repair_attempts),
            "preserves_total_num_events": True,
            "preserves_daily_counts": True,
            "preserves_customer_degrees": bool(self.preserve_degrees),
            "preserves_product_degrees": bool(self.preserve_degrees),
            "preserves_block_pair_time_counts": bool(self.preserve_block_pair_time_counts),
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
        metadata.update(self.runtime_metadata)
        return metadata

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
            "preserves_block_pair_time_counts_default": bool(self.preserve_block_pair_time_counts),
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

    def _build_block_pair_time_counts(self, frame: pd.DataFrame) -> pd.DataFrame:
        exact = (
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
        )
        if self.preserve_block_pair_time_counts:
            return exact.sort_values(["time_bucket", "customer_block", "product_block"]).reset_index(drop=True)
        rng = np.random.default_rng(self.seed)
        global_pairs = exact.groupby(["customer_block", "product_block"])["count"].sum()
        pair_index = list(global_pairs.index)
        global_probs = global_pairs.to_numpy(dtype=float)
        global_probs = global_probs / np.clip(global_probs.sum(), 1e-12, None)
        rows = []
        for time_bucket, group in frame.groupby("_time_bucket"):
            n = len(group)
            day_pairs = (
                group.groupby(["_customer_block", "_product_block"])
                .size()
                .reindex(pair_index, fill_value=0)
                .to_numpy(dtype=float)
            )
            probs = day_pairs + 5.0 * global_probs
            probs = probs / np.clip(probs.sum(), 1e-12, None)
            draws = rng.choice(len(pair_index), size=n, replace=True, p=probs)
            counts = Counter(draws)
            for pair_pos, count in counts.items():
                cblock, pblock = pair_index[int(pair_pos)]
                rows.append(
                    {
                        "customer_block": int(cblock),
                        "product_block": int(pblock),
                        "time_bucket": time_bucket,
                        "count": int(count),
                    }
                )
        return pd.DataFrame(rows).sort_values(["time_bucket", "customer_block", "product_block"]).reset_index(drop=True)

    def _finish_runtime(self, sample_start: float, num_events: int, num_cells: int) -> None:
        sample_seconds = float(time.time() - sample_start)
        fit_seconds = float(self.runtime_metadata.get("fit_seconds", 0.0))
        total_seconds = fit_seconds + sample_seconds
        self.runtime_metadata.update(
            {
                "sample_seconds": sample_seconds,
                "total_seconds": total_seconds,
                "events_per_second": float(num_events / max(sample_seconds, 1e-9)),
                "num_cells_processed": int(num_cells),
                "average_cell_count": float(num_events / max(num_cells, 1)),
                "sampling_mode": self.sampling_mode,
            }
        )
        print(
            f"[done] generated {num_events:,} events in {sample_seconds:.2f}s "
            f"({self.runtime_metadata['events_per_second']:.1f} events/s)",
            flush=True,
        )

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


def sample_entities_with_quotas(
    entity_ids: np.ndarray,
    remaining_degrees: Mapping[Any, int],
    activity_probs_at_t: np.ndarray,
    n: int,
    rng: np.random.Generator,
    max_attempts: int = 5,
    eps: float = 1e-12,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Sample n entity stubs with replacement-like weights but exact local quotas."""

    entity_ids = np.asarray(entity_ids, dtype=object)
    activity_probs_at_t = np.asarray(activity_probs_at_t, dtype=float)
    if len(entity_ids) != len(activity_probs_at_t):
        raise ValueError("entity_ids and activity_probs_at_t must have the same length")
    remaining = np.asarray([int(remaining_degrees.get(entity, 0)) for entity in entity_ids], dtype=int)
    available_mask = remaining > 0
    total_remaining = int(remaining[available_mask].sum())
    if total_remaining < int(n):
        return np.asarray([], dtype=object), {
            "success": False,
            "reason": "insufficient_remaining_quota",
            "requested": int(n),
            "total_remaining": total_remaining,
        }
    if int(n) == 0:
        return np.asarray([], dtype=object), {"success": True, "requested": 0, "sampled": 0}
    available_ids = entity_ids[available_mask]
    available_remaining = remaining[available_mask]
    probs = np.nan_to_num(activity_probs_at_t[available_mask], nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    weights = available_remaining.astype(float) * np.clip(probs, eps, None)
    if not np.isfinite(weights).all() or float(weights.sum()) <= eps:
        weights = available_remaining.astype(float)
    samples: List[Any] = []
    local_counts = np.zeros(len(available_ids), dtype=int)
    needed = int(n)
    for _ in range(max_attempts):
        if needed <= 0:
            break
        residual_capacity = available_remaining - local_counts
        valid = residual_capacity > 0
        if not bool(valid.any()):
            break
        valid_weights = weights.copy()
        valid_weights[~valid] = 0.0
        if valid_weights.sum() <= eps:
            valid_weights = residual_capacity.astype(float)
            valid_weights[~valid] = 0.0
        draw_size = min(max(needed * 2, needed), int(residual_capacity.sum()))
        draw_probs = valid_weights / np.clip(valid_weights.sum(), eps, None)
        draw_indices = rng.choice(len(available_ids), size=draw_size, replace=True, p=draw_probs)
        for draw_index in draw_indices:
            if local_counts[draw_index] >= available_remaining[draw_index]:
                continue
            samples.append(available_ids[draw_index])
            local_counts[draw_index] += 1
            needed -= 1
            if needed <= 0:
                break
    if needed > 0:
        residual_capacity = available_remaining - local_counts
        order = np.argsort(-(weights + residual_capacity * eps))
        for index in order:
            while residual_capacity[index] > 0 and needed > 0:
                samples.append(available_ids[index])
                residual_capacity[index] -= 1
                needed -= 1
            if needed <= 0:
                break
    success = len(samples) == int(n)
    status = {
        "success": bool(success),
        "requested": int(n),
        "sampled": int(len(samples)),
        "total_remaining": total_remaining,
    }
    if not success:
        status["reason"] = "quota_sampling_exhausted_after_attempts"
    return np.asarray(samples, dtype=object), status


class ZeroStaticCustomerProductAffinity:
    """Disabled F_static component used by scalable default sampler."""

    fallback_used = False
    summary = {
        "method": "disabled",
        "rank": 0,
        "num_customers": 0,
        "num_products": 0,
        "nnz": 0,
        "score_mean": 0.0,
        "score_std": 1.0,
        "fallback_used": False,
    }

    def score(self, customer_ids: List[Any], product_ids: List[Any]) -> np.ndarray:
        return np.zeros(len(product_ids), dtype=float)

    def score_one(self, customer_id: Any, product_id: Any) -> float:
        return 0.0

    def save_summary(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.summary, handle, indent=2)
            handle.write("\n")


class ZeroProductAgeAffinity:
    """Disabled product lifecycle compatibility component."""

    table = pd.DataFrame(
        columns=["customer_block", "age_bin", "probability", "global_probability", "log_residual"]
    )

    def score(self, customer_block: int, product_id: Any, time_bucket: Any) -> float:
        return 0.0

    def age_bin(self, product_id: Any, time_bucket: Any) -> str:
        return "disabled"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(path, index=False)
