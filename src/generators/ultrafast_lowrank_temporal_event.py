"""Ultrafast slot-based low-rank temporal event-spine generator."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .fast_event_spine_metrics import evaluate_fast_event_spine
from .fast_lowrank_temporal_event import load_entity_blocks
from .fast_temporal_activity import FastTemporalActivityModel, canonical_time_bucket, normalize_probs
from .lowrank_time_gated_affinity import LowRankTimeGatedAffinity
from .slot_assignment_repair import repair_entity_degrees_by_replacement
from .ultrafast_event_pairing import (
    reorder_products_for_cell,
    repair_cell_pairs_by_swaps,
    update_pair_counts,
)


METHOD_NAME = "ultrafast_lowrank_temporal_event"
METHOD_ALIAS = "uflte_event_spine"


class UltraFastLowRankTemporalEventGenerator:
    """UltraFastLowRankTemporalEventGenerator approximates the joint event
    distribution p(customer_id, product_id, review_time) using a scalable
    slot-assignment approach. It first constructs block-pair-time event
    slots, assigns customers and products to those slots using
    time-conditioned activity models, repairs degree constraints, and then
    pairs products to customers inside each temporal cell using a low-rank
    time-gated dynamic affinity F_{u,i,t} = (z_u * g_t)^T z_i. This avoids
    dense customer-product-time tensors, per-event candidate-pool scoring,
    and cell-level quota rejection sampling while preserving time-dependent
    customer-product compatibility.
    """

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        structure_debug_dir: Optional[str | Path] = None,
        time_granularity: str = "day",
        time_gate_granularity: str = "month",
        block_pair_time_mode: str = "exact",
        preserve_degrees: bool = True,
        rank: int = 32,
        alpha_customer_time: Any = "auto",
        alpha_product_time: Any = "auto",
        alpha_time_gate: Any = "auto",
        block_time_smoothing: float = 5.0,
        pairing_mode: str = "dynamic_projection",
        max_exact_affinity_cell_size: int = 128,
        enable_degree_repair: bool = True,
        enable_fast_overlap_repair: bool = True,
        repair_max_passes: int = 3,
        fast_repair_attempts: int = 10,
        allow_degree_slack: bool = False,
        seed: int = 42,
    ):
        if time_granularity != "day":
            raise ValueError("UltraFastLowRankTemporalEventGenerator currently supports day event buckets only")
        if block_pair_time_mode not in {"exact", "sampled", "none"}:
            raise ValueError("block_pair_time_mode must be exact, sampled, or none")
        if pairing_mode not in {"random", "static_projection", "dynamic_projection", "dynamic_exact_small"}:
            raise ValueError("unsupported pairing_mode")
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.structure_debug_dir = Path(structure_debug_dir) if structure_debug_dir else None
        self.time_granularity = time_granularity
        self.time_gate_granularity = time_gate_granularity
        self.block_pair_time_mode = block_pair_time_mode
        self.preserve_degrees = bool(preserve_degrees)
        self.rank = int(rank)
        self.alpha_customer_time = alpha_customer_time
        self.alpha_product_time = alpha_product_time
        self.alpha_time_gate = alpha_time_gate
        self.block_time_smoothing = float(block_time_smoothing)
        self.pairing_mode = pairing_mode
        self.max_exact_affinity_cell_size = int(max_exact_affinity_cell_size)
        self.enable_degree_repair = bool(enable_degree_repair)
        self.enable_fast_overlap_repair = bool(enable_fast_overlap_repair)
        self.repair_max_passes = int(repair_max_passes)
        self.fast_repair_attempts = int(fast_repair_attempts)
        self.allow_degree_slack = bool(allow_degree_slack)
        self.seed = int(seed)
        self.real_df: Optional[pd.DataFrame] = None
        self.synthetic_df: Optional[pd.DataFrame] = None
        self.customer_blocks: Dict[Any, int] = {}
        self.product_blocks: Dict[Any, int] = {}
        self.block_warnings: List[str] = []
        self.block_pair_time_counts = pd.DataFrame()
        self.customer_degrees: Dict[Any, int] = {}
        self.product_degrees: Dict[Any, int] = {}
        self.real_event_set: set[tuple[Any, Any, str]] = set()
        self.customer_activity: Optional[FastTemporalActivityModel] = None
        self.product_activity: Optional[FastTemporalActivityModel] = None
        self.affinity_model: Optional[LowRankTimeGatedAffinity] = None
        self.slot_customer_block = np.asarray([], dtype=int)
        self.slot_product_block = np.asarray([], dtype=int)
        self.slot_time_bucket = np.asarray([], dtype=object)
        self.slot_time_gate_bucket = np.asarray([], dtype=object)
        self.slot_customer_id = np.asarray([], dtype=object)
        self.slot_product_id = np.asarray([], dtype=object)
        self.slot_summary: Dict[str, Any] = {}
        self.customer_assignment_summary: Dict[str, Any] = {}
        self.product_assignment_summary: Dict[str, Any] = {}
        self.customer_repair_summary: Dict[str, Any] = {}
        self.product_repair_summary: Dict[str, Any] = {}
        self.overlap_repair_summary: Dict[str, Any] = {}
        self.runtime_metadata: Dict[str, Any] = {}

    def fit(self, real_df: pd.DataFrame) -> "UltraFastLowRankTemporalEventGenerator":
        fit_start = time.time()
        print("[fit] loading data", flush=True)
        required = [self.customer_id_col, self.product_id_col, self.timestamp_col]
        missing = [col for col in required if col not in real_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        frame = real_df[required].copy()
        frame["_time_bucket"] = canonical_time_bucket(frame[self.timestamp_col], self.time_granularity)
        frame[self.timestamp_col] = frame["_time_bucket"]
        self.customer_blocks, self.product_blocks, self.block_warnings = self._load_or_fallback_blocks(frame)
        frame["_customer_block"] = frame[self.customer_id_col].map(self.customer_blocks).fillna(0).astype(int)
        frame["_product_block"] = frame[self.product_id_col].map(self.product_blocks).fillna(0).astype(int)
        self.real_df = frame
        self.customer_degrees = frame[self.customer_id_col].value_counts().astype(int).to_dict()
        self.product_degrees = frame[self.product_id_col].value_counts().astype(int).to_dict()
        self.real_event_set = {
            (row[self.customer_id_col], row[self.product_id_col], row["_time_bucket"])
            for _, row in frame.iterrows()
        }
        self.block_pair_time_counts = self._build_block_pair_time_counts(frame)
        print("[fit] fitting activity models", flush=True)
        self.customer_activity = FastTemporalActivityModel(
            alpha=self.alpha_customer_time,
            block_time_smoothing=self.block_time_smoothing,
            granularity=self.time_granularity,
            entity_kind="customer",
        ).fit(frame, self.customer_id_col, "_time_bucket", self.customer_blocks)
        self.product_activity = FastTemporalActivityModel(
            alpha=self.alpha_product_time,
            block_time_smoothing=self.block_time_smoothing,
            granularity=self.time_granularity,
            entity_kind="product",
        ).fit(frame, self.product_id_col, "_time_bucket", self.product_blocks)
        print("[fit] fitting low-rank time-gated affinity", flush=True)
        self.affinity_model = LowRankTimeGatedAffinity(
            rank=self.rank,
            alpha_time_gate=self.alpha_time_gate,
            time_gate_granularity=self.time_gate_granularity,
            seed=self.seed,
        ).fit(frame, self.customer_id_col, self.product_id_col, "_time_bucket")
        self.runtime_metadata["fit_seconds"] = float(time.time() - fit_start)
        print(f"[fit] done in {self.runtime_metadata['fit_seconds']:.2f}s", flush=True)
        return self

    def sample(self, seed: Optional[int] = None) -> pd.DataFrame:
        if self.real_df is None or self.customer_activity is None or self.product_activity is None or self.affinity_model is None:
            raise RuntimeError("Call fit before sample.")
        total_start = time.time()
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        self._build_slots()
        self.slot_customer_id, self.customer_assignment_summary = assign_entities_to_slots_vectorized(
            self.slot_customer_block,
            self.slot_time_bucket,
            self.customer_degrees,
            self.customer_activity,
            rng,
            label="customer",
        )
        print(
            f"[assign-customers] processing {self.customer_assignment_summary['num_block_time_groups']:,} customer-block-time groups",
            flush=True,
        )
        print(f"[assign-customers] done in {self.customer_assignment_summary['assignment_seconds']:.2f}s", flush=True)
        if self.enable_degree_repair:
            self.slot_customer_id, self.customer_repair_summary = repair_entity_degrees_by_replacement(
                self.slot_customer_id,
                self.slot_customer_block,
                self.slot_time_bucket,
                self.customer_degrees,
                self.customer_blocks,
                self.customer_activity,
                rng,
                max_passes=self.repair_max_passes,
                allow_degree_slack=self.allow_degree_slack,
            )
        else:
            self.customer_repair_summary = no_repair_summary(self.slot_customer_id, self.customer_degrees)
        print(
            "[repair-customers] "
            f"L1 before={self.customer_repair_summary['l1_error_before']}, "
            f"after={self.customer_repair_summary['l1_error_after']}, "
            f"replacements={self.customer_repair_summary['num_replacements']}, "
            f"seconds={self.customer_repair_summary['repair_seconds']:.2f}",
            flush=True,
        )
        self.slot_product_id, self.product_assignment_summary = assign_entities_to_slots_vectorized(
            self.slot_product_block,
            self.slot_time_bucket,
            self.product_degrees,
            self.product_activity,
            rng,
            label="product",
        )
        print(
            f"[assign-products] processing {self.product_assignment_summary['num_block_time_groups']:,} product-block-time groups",
            flush=True,
        )
        print(f"[assign-products] done in {self.product_assignment_summary['assignment_seconds']:.2f}s", flush=True)
        if self.enable_degree_repair:
            self.slot_product_id, self.product_repair_summary = repair_entity_degrees_by_replacement(
                self.slot_product_id,
                self.slot_product_block,
                self.slot_time_bucket,
                self.product_degrees,
                self.product_blocks,
                self.product_activity,
                rng,
                max_passes=self.repair_max_passes,
                allow_degree_slack=self.allow_degree_slack,
            )
        else:
            self.product_repair_summary = no_repair_summary(self.slot_product_id, self.product_degrees)
        print(
            "[repair-products] "
            f"L1 before={self.product_repair_summary['l1_error_before']}, "
            f"after={self.product_repair_summary['l1_error_after']}, "
            f"replacements={self.product_repair_summary['num_replacements']}, "
            f"seconds={self.product_repair_summary['repair_seconds']:.2f}",
            flush=True,
        )
        pairing_start = time.time()
        print(f"[pair] processing {len(self.block_pair_time_counts):,} cells with {self.pairing_mode}", flush=True)
        self._pair_slots(rng)
        pairing_seconds = float(time.time() - pairing_start)
        print(f"[pair] done in {pairing_seconds:.2f}s", flush=True)
        synthetic = pd.DataFrame(
            {
                self.customer_id_col: self.slot_customer_id,
                self.product_id_col: self.slot_product_id,
                self.timestamp_col: self.slot_time_bucket,
            }
        ).sort_values([self.timestamp_col, self.customer_id_col, self.product_id_col]).reset_index(drop=True)
        self.synthetic_df = synthetic
        self.runtime_metadata.update(
            {
                "slot_build_seconds": float(self.slot_summary.get("slot_build_seconds", 0.0)),
                "customer_assignment_seconds": float(self.customer_assignment_summary.get("assignment_seconds", 0.0)),
                "customer_repair_seconds": float(self.customer_repair_summary.get("repair_seconds", 0.0)),
                "product_assignment_seconds": float(self.product_assignment_summary.get("assignment_seconds", 0.0)),
                "product_repair_seconds": float(self.product_repair_summary.get("repair_seconds", 0.0)),
                "assignment_seconds": float(
                    self.customer_assignment_summary.get("assignment_seconds", 0.0)
                    + self.product_assignment_summary.get("assignment_seconds", 0.0)
                ),
                "repair_seconds": float(
                    self.customer_repair_summary.get("repair_seconds", 0.0)
                    + self.product_repair_summary.get("repair_seconds", 0.0)
                ),
                "pairing_seconds": pairing_seconds,
                "total_seconds": float(time.time() - total_start + self.runtime_metadata.get("fit_seconds", 0.0)),
                "events_per_second": float(len(synthetic) / max(time.time() - total_start, 1e-9)),
            }
        )
        print(
            f"[done] total_seconds={self.runtime_metadata['total_seconds']:.2f}, "
            f"events_per_second={self.runtime_metadata['events_per_second']:.1f}",
            flush=True,
        )
        return synthetic

    def save_outputs(self, output_dir: str | Path) -> None:
        if self.synthetic_df is None:
            raise RuntimeError("Call sample before save_outputs.")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.synthetic_df.to_csv(output / "synthetic_review.csv", index=False)
        self.save_debug(output / "debug")
        self.save_metadata(output / "metadata.json")

    def save_debug(self, debug_dir: str | Path) -> None:
        if self.real_df is None:
            raise RuntimeError("Call fit before save_debug.")
        debug = Path(debug_dir)
        debug.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({self.customer_id_col: list(self.customer_blocks), "customer_block": list(self.customer_blocks.values())}).to_csv(
            debug / "customer_blocks.csv",
            index=False,
        )
        pd.DataFrame({self.product_id_col: list(self.product_blocks), "product_block": list(self.product_blocks.values())}).to_csv(
            debug / "product_blocks.csv",
            index=False,
        )
        self.block_pair_time_counts.to_csv(debug / "block_pair_time_counts.csv", index=False)
        self.customer_activity.save_summary(debug / "customer_time_activity_summary.json")
        self.product_activity.save_summary(debug / "product_time_activity_summary.json")
        self.affinity_model.save_summary(debug / "lowrank_time_gated_affinity_summary.json")
        write_json(self.slot_summary, debug / "slot_summary.json")
        write_json(self.customer_assignment_summary, debug / "customer_initial_assignment_summary.json")
        write_json(self.customer_repair_summary, debug / "customer_degree_repair_summary.json")
        write_json(self.product_assignment_summary, debug / "product_initial_assignment_summary.json")
        write_json(self.product_repair_summary, debug / "product_degree_repair_summary.json")
        write_json(self.overlap_repair_summary, debug / "overlap_repair_summary.json")

    def save_metadata(self, path: str | Path) -> None:
        write_json(self.metadata(), path)

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, compute_c2st: bool = False) -> Dict[str, Any]:
        return evaluate_fast_event_spine(
            real_df,
            synthetic_df,
            customer_col=self.customer_id_col,
            product_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            compute_c2st=compute_c2st,
            metadata=self.metadata(),
        )

    def metadata(self) -> Dict[str, Any]:
        if self.real_df is None:
            raise RuntimeError("Call fit before metadata.")
        affinity_summary = self.affinity_model.summary() if self.affinity_model is not None else {}
        metadata = {
            "method": METHOD_NAME,
            "alias": METHOD_ALIAS,
            "class_name": "UltraFastLowRankTemporalEventGenerator",
            "preserves_old_generators": True,
            "generates_joint_events": True,
            "event_tuple": [self.customer_id_col, self.product_id_col, self.timestamp_col],
            "slot_based_assignment": True,
            "vectorized_assignment": True,
            "degree_repair": bool(self.enable_degree_repair),
            "uses_time_conditioned_customer_assignment": True,
            "uses_time_conditioned_product_assignment": True,
            "uses_time_dependent_pairing_score": self.pairing_mode != "random",
            "uses_dense_F_u_i_t": False,
            "dynamic_affinity_type": "time_gated_low_rank",
            "dynamic_affinity_formula": "(z_u * g_t)^T z_i",
            "per_event_candidate_pool_scoring": False,
            "cell_level_quota_rejection_sampling": False,
            "batch_cell_pairing": True,
            "block_pair_time_mode": self.block_pair_time_mode,
            "time_granularity": self.time_granularity,
            "time_gate_granularity": self.time_gate_granularity,
            "rank": int(self.rank),
            "alpha_customer_time": float(self.customer_activity.alpha_resolved),
            "alpha_product_time": float(self.product_activity.alpha_resolved),
            "alpha_time_gate": float(affinity_summary.get("alpha_time_gate", 0.0)),
            "pairing_mode": self.pairing_mode,
            "max_exact_affinity_cell_size": int(self.max_exact_affinity_cell_size),
            "enable_fast_overlap_repair": bool(self.enable_fast_overlap_repair),
            "repair_max_passes": int(self.repair_max_passes),
            "num_customers": int(len(self.customer_degrees)),
            "num_products": int(len(self.product_degrees)),
            "num_events": int(len(self.real_df)),
            "num_time_buckets": int(self.real_df["_time_bucket"].nunique()),
            "num_time_gate_buckets": int(affinity_summary.get("num_time_gate_buckets", 0)),
            "num_customer_blocks": int(len(set(self.customer_blocks.values()))),
            "num_product_blocks": int(len(set(self.product_blocks.values()))),
            "block_warnings": list(self.block_warnings),
            "lowrank_affinity": affinity_summary,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        metadata.update(self.slot_summary)
        metadata.update(self.runtime_metadata)
        return metadata

    def _load_or_fallback_blocks(self, frame: pd.DataFrame) -> tuple[Dict[Any, int], Dict[Any, int], List[str]]:
        warnings = []
        customer_blocks = load_entity_blocks(self.structure_debug_dir, "customer_blocks.csv", self.customer_id_col, "customer_block")
        product_blocks = load_entity_blocks(self.structure_debug_dir, "product_blocks.csv", self.product_id_col, "product_block")
        if not customer_blocks:
            warnings.append("customer blocks missing; fell back to one customer block")
            customer_blocks = {entity: 0 for entity in frame[self.customer_id_col].unique()}
        if not product_blocks:
            warnings.append("product blocks missing; fell back to one product block")
            product_blocks = {entity: 0 for entity in frame[self.product_id_col].unique()}
        return customer_blocks, product_blocks, warnings

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
        if self.block_pair_time_mode == "exact":
            return exact.sort_values(["time_bucket", "customer_block", "product_block"]).reset_index(drop=True)
        rng = np.random.default_rng(self.seed)
        global_pairs = exact.groupby(["customer_block", "product_block"])["count"].sum()
        pair_index = list(global_pairs.index)
        global_probs = normalize_probs(global_pairs.to_numpy(dtype=float))
        rows = []
        for time_bucket, group in frame.groupby("_time_bucket"):
            n = len(group)
            if self.block_pair_time_mode == "sampled":
                day_pairs = (
                    group.groupby(["_customer_block", "_product_block"])
                    .size()
                    .reindex(pair_index, fill_value=0)
                    .to_numpy(dtype=float)
                )
                probs = normalize_probs(day_pairs + 5.0 * global_probs)
            else:
                probs = global_probs
            draws = rng.choice(len(pair_index), size=n, replace=True, p=probs)
            for pair_pos, count in Counter(draws).items():
                cblock, pblock = pair_index[int(pair_pos)]
                rows.append({"customer_block": int(cblock), "product_block": int(pblock), "time_bucket": time_bucket, "count": int(count)})
        return pd.DataFrame(rows).sort_values(["time_bucket", "customer_block", "product_block"]).reset_index(drop=True)

    def _build_slots(self) -> None:
        start = time.time()
        counts = self.block_pair_time_counts["count"].to_numpy(dtype=int)
        self.slot_customer_block = np.repeat(self.block_pair_time_counts["customer_block"].to_numpy(dtype=int), counts)
        self.slot_product_block = np.repeat(self.block_pair_time_counts["product_block"].to_numpy(dtype=int), counts)
        self.slot_time_bucket = np.repeat(self.block_pair_time_counts["time_bucket"].to_numpy(dtype=object), counts)
        self.slot_time_gate_bucket = canonical_time_bucket(pd.Series(self.slot_time_bucket), self.time_gate_granularity).to_numpy(dtype=object)
        self.slot_customer_id = np.empty(len(self.slot_time_bucket), dtype=object)
        self.slot_product_id = np.empty(len(self.slot_time_bucket), dtype=object)
        self.slot_summary = {
            "num_slots": int(len(self.slot_time_bucket)),
            "num_block_pair_time_cells": int(len(self.block_pair_time_counts)),
            "average_cell_size": float(len(self.slot_time_bucket) / max(len(self.block_pair_time_counts), 1)),
            "max_cell_size": int(counts.max()) if len(counts) else 0,
            "num_time_buckets": int(pd.Series(self.slot_time_bucket).nunique()) if len(self.slot_time_bucket) else 0,
            "slot_build_seconds": float(time.time() - start),
        }
        print(
            f"[slots] built {self.slot_summary['num_slots']:,} slots from "
            f"{self.slot_summary['num_block_pair_time_cells']:,} block-pair-time cells "
            f"in {self.slot_summary['slot_build_seconds']:.2f}s",
            flush=True,
        )

    def _pair_slots(self, rng: np.random.Generator) -> None:
        pair_counts: Counter = Counter()
        repair = {"badness_before": 0, "badness_after": 0, "num_swaps": 0}
        for indices in grouped_slot_indices(
            self.slot_customer_block,
            self.slot_product_block,
            self.slot_time_bucket,
        ):
            customers = self.slot_customer_id[indices]
            products = self.slot_product_id[indices]
            time_bucket = self.slot_time_bucket[indices[0]]
            reordered = reorder_products_for_cell(
                customers,
                products,
                time_bucket,
                self.affinity_model,
                rng,
                pairing_mode=self.pairing_mode,
                max_exact_affinity_cell_size=self.max_exact_affinity_cell_size,
            )
            if self.enable_fast_overlap_repair:
                reordered, repair_summary = repair_cell_pairs_by_swaps(
                    customers,
                    reordered,
                    time_bucket,
                    self.real_event_set,
                    pair_counts,
                    rng,
                    max_attempts=self.fast_repair_attempts,
                )
                for key in repair:
                    repair[key] += int(repair_summary.get(key, 0))
            self.slot_product_id[indices] = reordered
            update_pair_counts(pair_counts, customers, reordered)
        self.overlap_repair_summary = repair


def assign_entities_to_slots_vectorized(
    slot_blocks: np.ndarray,
    slot_times: np.ndarray,
    target_degrees: Dict[Any, int],
    activity_model: FastTemporalActivityModel,
    rng: np.random.Generator,
    label: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    start = time.time()
    assigned = np.empty(len(slot_blocks), dtype=object)
    zero_fallbacks = 0
    group_count = 0
    for indices in grouped_block_time_indices(slot_blocks, slot_times):
        group_count += 1
        block = int(slot_blocks[indices[0]])
        time_bucket = slot_times[indices[0]]
        entity_ids, probs = activity_model.probabilities_for_block_time(block, time_bucket)
        if len(entity_ids) == 0:
            raise RuntimeError(f"No candidate {label}s found for block {block}")
        degrees = np.asarray([target_degrees.get(entity, 0) for entity in entity_ids], dtype=float)
        weights = degrees * np.clip(probs, 1e-12, None)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 1e-12:
            weights = degrees.copy()
            zero_fallbacks += 1
        if float(weights.sum()) <= 1e-12:
            weights = np.ones(len(entity_ids), dtype=float)
            zero_fallbacks += 1
        assigned[indices] = rng.choice(entity_ids, size=len(indices), replace=True, p=weights / weights.sum())
    return assigned, {
        "num_block_time_groups": int(group_count),
        "total_assigned": int(len(assigned)),
        "groups_with_zero_activity_fallback": int(zero_fallbacks),
        "assignment_seconds": float(time.time() - start),
    }


def grouped_block_time_indices(slot_blocks: np.ndarray, slot_times: np.ndarray) -> List[np.ndarray]:
    keys = np.asarray([f"{int(block)}\x1f{time}" for block, time in zip(slot_blocks, slot_times)], dtype=object)
    return grouped_indices_from_keys(keys)


def grouped_slot_indices(slot_customer_blocks: np.ndarray, slot_product_blocks: np.ndarray, slot_times: np.ndarray) -> List[np.ndarray]:
    keys = np.asarray(
        [f"{int(cblock)}\x1f{int(pblock)}\x1f{time}" for cblock, pblock, time in zip(slot_customer_blocks, slot_product_blocks, slot_times)],
        dtype=object,
    )
    return grouped_indices_from_keys(keys)


def grouped_indices_from_keys(keys: np.ndarray) -> List[np.ndarray]:
    if len(keys) == 0:
        return []
    order = np.argsort(keys)
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    return [order[start:end] for start, end in zip(starts, ends)]


def no_repair_summary(slot_entity_ids: np.ndarray, target_degrees: Dict[Any, int]) -> Dict[str, Any]:
    current = Counter(slot_entity_ids.tolist())
    errors = [abs(int(current.get(entity, 0)) - int(target)) for entity, target in target_degrees.items()]
    l1 = int(sum(errors))
    max_abs = int(max(errors) if errors else 0)
    return {
        "l1_error_before": l1,
        "l1_error_after": l1,
        "max_abs_error_before": max_abs,
        "max_abs_error_after": max_abs,
        "num_replacements": 0,
        "num_unresolved_entities": int(sum(error != 0 for error in errors)),
        "repair_seconds": 0.0,
    }


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
