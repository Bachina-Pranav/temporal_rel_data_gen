"""Time-biased exact block-stub matching event-spine generator."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .fast_event_spine_metrics import evaluate_fast_event_spine
from .fast_lowrank_temporal_event import load_entity_blocks
from .fast_temporal_activity import FastTemporalActivityModel, canonical_time_bucket
from .lowrank_time_gated_affinity import LowRankTimeGatedAffinity
from .stub_dynamic_pairing import reorder_products_within_cells_by_dynamic_affinity
from .stub_time_assignment import assign_stubs_to_slots_by_time


METHOD_NAME = "time_biased_block_stub_matching"
METHOD_ALIAS = "temporal_stub_matching_event"


class TimeBiasedBlockStubMatchingGenerator:
    """TimeBiasedBlockStubMatchingGenerator generates temporal relational
    event spines by exactly matching entity stubs to block-pair-time slots.
    For each customer/product block, it creates exact degree stubs using
    np.repeat, samples a desired event time for each stub from the entity's
    smoothed temporal activity distribution, and then sort-matches desired
    times to the exact slots required by the block-pair-time tensor M[a,b,t].
    This preserves exact customer degrees, exact product degrees, and exact
    block-pair-time counts without rejection sampling or repair loops.
    Within each (customer_block, product_block, time) cell, products are
    reordered using the low-rank dynamic affinity
    F_{u,i,t} = (z_u * g_t)^T z_i, retaining individual-level
    customer-product-time compatibility.
    """

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        structure_debug_dir: Optional[str | Path] = None,
        time_granularity: str = "day",
        time_gate_granularity: str = "month",
        rank: int = 32,
        alpha_customer_time: Any = "auto",
        alpha_product_time: Any = "auto",
        alpha_time_gate: Any = "auto",
        block_time_smoothing: float = 5.0,
        pairing_mode: str = "dynamic_projection",
        max_exact_affinity_cell_size: int = 128,
        large_cell_pairing: str = "projection_sort",
        desired_time_jitter: float = 1e-3,
        enable_fast_overlap_repair: bool = False,
        overlap_resample_prob: float = 0.0,
        seed: int = 42,
    ):
        if time_granularity != "day":
            raise ValueError("TimeBiasedBlockStubMatchingGenerator currently supports day event buckets only")
        if pairing_mode not in {"random", "static_projection", "dynamic_projection", "dynamic_exact_small"}:
            raise ValueError("unsupported pairing_mode")
        if large_cell_pairing not in {"projection_sort", "exact_greedy"}:
            raise ValueError("large_cell_pairing must be projection_sort or exact_greedy")
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.structure_debug_dir = Path(structure_debug_dir) if structure_debug_dir else None
        self.time_granularity = time_granularity
        self.time_gate_granularity = time_gate_granularity
        self.rank = int(rank)
        self.alpha_customer_time = alpha_customer_time
        self.alpha_product_time = alpha_product_time
        self.alpha_time_gate = alpha_time_gate
        self.block_time_smoothing = float(block_time_smoothing)
        self.pairing_mode = pairing_mode
        self.max_exact_affinity_cell_size = int(max_exact_affinity_cell_size)
        self.large_cell_pairing = large_cell_pairing
        self.desired_time_jitter = float(desired_time_jitter)
        self.enable_fast_overlap_repair = bool(enable_fast_overlap_repair)
        self.overlap_resample_prob = float(overlap_resample_prob)
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
        self.day_to_code: Dict[str, int] = {}
        self.code_to_day: List[str] = []
        self.gate_to_code: Dict[str, int] = {}
        self.code_to_gate: List[str] = []
        self.slot_id = np.asarray([], dtype=int)
        self.slot_customer_block = np.asarray([], dtype=int)
        self.slot_product_block = np.asarray([], dtype=int)
        self.slot_time_code = np.asarray([], dtype=int)
        self.slot_time_bucket = np.asarray([], dtype=object)
        self.slot_time_gate_code = np.asarray([], dtype=int)
        self.slot_customer_idx = np.asarray([], dtype=np.int64)
        self.slot_product_idx = np.asarray([], dtype=np.int64)
        self.slot_customer_id = np.asarray([], dtype=object)
        self.slot_product_id = np.asarray([], dtype=object)
        self.slot_summary: Dict[str, Any] = {}
        self.customer_assignment_summary: Dict[str, Any] = {}
        self.product_assignment_summary: Dict[str, Any] = {}
        self.dynamic_pairing_summary: Dict[str, Any] = {}
        self.runtime_metadata: Dict[str, Any] = {}

    def fit(self, real_df: pd.DataFrame) -> "TimeBiasedBlockStubMatchingGenerator":
        fit_start = time.time()
        print("[fit] loading data", flush=True)
        required = [self.customer_id_col, self.product_id_col, self.timestamp_col]
        missing = [col for col in required if col not in real_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        frame = real_df[required].copy()
        frame["_time_bucket"] = canonical_time_bucket(frame[self.timestamp_col], self.time_granularity)
        frame[self.timestamp_col] = frame["_time_bucket"]
        self.code_to_day = sorted(frame["_time_bucket"].unique().tolist())
        self.day_to_code = {day: idx for idx, day in enumerate(self.code_to_day)}
        frame["_time_code"] = frame["_time_bucket"].map(self.day_to_code).astype(int)
        frame["_time_gate_bucket"] = canonical_time_bucket(frame["_time_bucket"], self.time_gate_granularity)
        self.code_to_gate = sorted(frame["_time_gate_bucket"].unique().tolist())
        self.gate_to_code = {gate: idx for idx, gate in enumerate(self.code_to_gate)}
        frame["_time_gate_code"] = frame["_time_gate_bucket"].map(self.gate_to_code).astype(int)
        self.customer_blocks, self.product_blocks, self.block_warnings = self._load_or_fallback_blocks(frame)
        frame["_customer_block"] = frame[self.customer_id_col].map(self.customer_blocks).fillna(0).astype(int)
        frame["_product_block"] = frame[self.product_id_col].map(self.product_blocks).fillna(0).astype(int)
        self.real_df = frame
        self.customer_degrees = frame[self.customer_id_col].value_counts().astype(int).to_dict()
        self.product_degrees = frame[self.product_id_col].value_counts().astype(int).to_dict()
        self.real_event_set = set(
            zip(
                frame[self.customer_id_col].to_numpy(dtype=object),
                frame[self.product_id_col].to_numpy(dtype=object),
                frame["_time_bucket"].to_numpy(dtype=object),
            )
        )
        self.block_pair_time_counts = self._build_block_pair_time_counts(frame)
        print("[fit] fitting temporal activity models", flush=True)
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
        sample_start = time.time()
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        self._build_slots()
        print("[customer-stubs] assigning exact customer stubs by time-biased sort", flush=True)
        self.slot_customer_idx, self.customer_assignment_summary = assign_stubs_to_slots_by_time(
            self.customer_activity.entity_ids,
            self.customer_degrees,
            self.customer_blocks,
            self.slot_customer_block,
            self.slot_time_code,
            self.customer_activity,
            rng,
            jitter=self.desired_time_jitter,
            log_label="customer-stubs",
            return_entity_indices=True,
        )
        print(f"[customer-stubs] done in {self.customer_assignment_summary['assignment_seconds']:.2f}s", flush=True)
        print("[product-stubs] assigning exact product stubs by time-biased sort", flush=True)
        self.slot_product_idx, self.product_assignment_summary = assign_stubs_to_slots_by_time(
            self.product_activity.entity_ids,
            self.product_degrees,
            self.product_blocks,
            self.slot_product_block,
            self.slot_time_code,
            self.product_activity,
            rng,
            jitter=self.desired_time_jitter,
            log_label="product-stubs",
            return_entity_indices=True,
        )
        print(f"[product-stubs] done in {self.product_assignment_summary['assignment_seconds']:.2f}s", flush=True)
        self.slot_customer_id = self.customer_activity.entity_ids[self.slot_customer_idx]
        self.slot_product_id = self.product_activity.entity_ids[self.slot_product_idx]
        print(f"[pairing] {self.pairing_mode} over {len(self.block_pair_time_counts):,} cells", flush=True)
        self.slot_product_id, self.dynamic_pairing_summary = reorder_products_within_cells_by_dynamic_affinity(
            self.slot_customer_id,
            self.slot_product_id,
            self.slot_customer_block,
            self.slot_product_block,
            self.slot_time_code,
            self.slot_time_gate_code,
            self.affinity_model,
            self.pairing_mode,
            self.max_exact_affinity_cell_size,
            rng,
            large_cell_pairing=self.large_cell_pairing,
            code_to_time_gate=self.code_to_gate,
            code_to_time_bucket=self.code_to_day,
            enable_fast_overlap_repair=self.enable_fast_overlap_repair,
            real_event_set=self.real_event_set,
        )
        print(f"[pairing] done in {self.dynamic_pairing_summary['dynamic_pairing_seconds']:.2f}s", flush=True)
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
                "customer_stub_assignment_seconds": float(self.customer_assignment_summary["assignment_seconds"]),
                "product_stub_assignment_seconds": float(self.product_assignment_summary["assignment_seconds"]),
                "customer_stub_construction_seconds": float(self.customer_assignment_summary["stub_construction_seconds"]),
                "customer_desired_time_sampling_seconds": float(self.customer_assignment_summary["desired_time_sampling_seconds"]),
                "customer_sorting_seconds": float(self.customer_assignment_summary["sorting_seconds"]),
                "customer_slot_assignment_seconds": float(self.customer_assignment_summary["slot_assignment_seconds"]),
                "product_stub_construction_seconds": float(self.product_assignment_summary["stub_construction_seconds"]),
                "product_desired_time_sampling_seconds": float(self.product_assignment_summary["desired_time_sampling_seconds"]),
                "product_sorting_seconds": float(self.product_assignment_summary["sorting_seconds"]),
                "product_slot_assignment_seconds": float(self.product_assignment_summary["slot_assignment_seconds"]),
                "dynamic_pairing_seconds": float(self.dynamic_pairing_summary["dynamic_pairing_seconds"]),
            }
        )
        verify_start = time.time()
        self._verify_exact_constraints(synthetic)
        self.runtime_metadata["verification_seconds"] = float(time.time() - verify_start)
        sample_seconds = float(time.time() - sample_start)
        self.runtime_metadata.update(
            {
                "sample_seconds": sample_seconds,
                "total_seconds": sample_seconds + float(self.runtime_metadata.get("fit_seconds", 0.0)),
                "events_per_second": float(len(synthetic) / max(sample_seconds, 1e-9)),
            }
        )
        print("[verify] all exact constraints passed", flush=True)
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
        write_json(self.customer_assignment_summary, debug / "customer_stub_assignment_summary.json")
        write_json(self.product_assignment_summary, debug / "product_stub_assignment_summary.json")
        write_json(self.dynamic_pairing_summary, debug / "dynamic_pairing_summary.json")
        write_json(self.array_dtype_summary(), debug / "array_dtype_summary.json")

    def save_metadata(self, path: str | Path) -> None:
        write_json(self.metadata(), path)

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, compute_c2st: bool = False) -> Dict[str, Any]:
        metrics = evaluate_fast_event_spine(
            real_df,
            synthetic_df,
            customer_col=self.customer_id_col,
            product_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            compute_c2st=compute_c2st,
            metadata=self.metadata(),
        )
        real_bpt = self._bpt_counts(real_df)
        syn_bpt = self._bpt_counts(synthetic_df)
        cells = sorted(set(real_bpt.index).union(set(syn_bpt.index)))
        bpt_l1 = float(sum(abs(real_bpt.get(cell, 0) - syn_bpt.get(cell, 0)) for cell in cells) / max(len(real_df), 1))
        metrics["block_pair_time_count_l1"] = bpt_l1
        metrics["block_pair_time_exact_match"] = bool(bpt_l1 == 0.0)
        metrics.update(self.dynamic_affinity_diagnostics(real_df, synthetic_df))
        return metrics

    def metadata(self) -> Dict[str, Any]:
        if self.real_df is None:
            raise RuntimeError("Call fit before metadata.")
        affinity_summary = self.affinity_model.summary() if self.affinity_model is not None else {}
        metadata = {
            "method": METHOD_NAME,
            "alias": METHOD_ALIAS,
            "class_name": "TimeBiasedBlockStubMatchingGenerator",
            "generates_joint_events": True,
            "event_tuple": [self.customer_id_col, self.product_id_col, self.timestamp_col],
            "preserves_total_events_exactly": True,
            "preserves_customer_degrees_exactly": True,
            "preserves_product_degrees_exactly": True,
            "preserves_block_pair_time_counts_exactly": True,
            "preserves_daily_counts_exactly": True,
            "preserves_blocks_exactly": True,
            "uses_time_biased_stub_assignment": True,
            "uses_customer_time_activity": True,
            "uses_product_time_activity": True,
            "uses_time_dependent_pairing_score": self.pairing_mode != "random",
            "uses_dense_F_u_i_t": False,
            "dynamic_affinity_type": "time_gated_low_rank",
            "dynamic_affinity_formula": "(z_u * g_t)^T z_i",
            "no_degree_repair": True,
            "no_quota_rejection_sampling": True,
            "no_per_event_candidate_pool_scoring": True,
            "no_cell_level_repair_loops": True,
            "pairing_mode": self.pairing_mode,
            "large_cell_pairing": self.large_cell_pairing,
            "max_exact_affinity_cell_size": int(self.max_exact_affinity_cell_size),
            "desired_time_jitter": float(self.desired_time_jitter),
            "enable_fast_overlap_repair": bool(self.enable_fast_overlap_repair),
            "overlap_resample_prob": float(self.overlap_resample_prob),
            "time_granularity": self.time_granularity,
            "time_gate_granularity": self.time_gate_granularity,
            "rank": int(self.rank),
            "alpha_customer_time": float(self.customer_activity.alpha_resolved),
            "alpha_product_time": float(self.product_activity.alpha_resolved),
            "alpha_time_gate": float(affinity_summary.get("alpha_time_gate", 0.0)),
            "num_customers": int(len(self.customer_degrees)),
            "num_products": int(len(self.product_degrees)),
            "num_events": int(len(self.real_df)),
            "num_time_buckets": int(len(self.code_to_day)),
            "num_customer_blocks": int(len(set(self.customer_blocks.values()))),
            "num_product_blocks": int(len(set(self.product_blocks.values()))),
            "num_block_pair_time_cells": int(len(self.block_pair_time_counts)),
            "block_warnings": list(self.block_warnings),
            "lowrank_affinity": affinity_summary,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        metadata.update(self.slot_summary)
        metadata.update(self.runtime_metadata)
        return metadata

    def dynamic_affinity_diagnostics(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        real = real_df[[self.customer_id_col, self.product_id_col, self.timestamp_col]].copy()
        syn = synthetic_df[[self.customer_id_col, self.product_id_col, self.timestamp_col]].copy()
        real[self.timestamp_col] = canonical_time_bucket(real[self.timestamp_col], self.time_granularity)
        syn[self.timestamp_col] = canonical_time_bucket(syn[self.timestamp_col], self.time_granularity)
        real_scores = self._score_pairs_by_time(real)
        syn_scores = self._score_pairs_by_time(syn)
        return {
            "mean_dynamic_affinity_real": float(np.mean(real_scores)) if len(real_scores) else 0.0,
            "mean_dynamic_affinity_synthetic": float(np.mean(syn_scores)) if len(syn_scores) else 0.0,
            "dynamic_affinity_distribution_ks": ks_stat(real_scores, syn_scores),
        }

    def _score_pairs_by_time(self, frame: pd.DataFrame) -> np.ndarray:
        if self.affinity_model is None or len(frame) == 0:
            return np.asarray([], dtype=float)
        scores = []
        for time_bucket, group in frame.groupby(self.timestamp_col, sort=False):
            scores.append(
                self.affinity_model.score_pairs(
                    group[self.customer_id_col].to_numpy(dtype=object),
                    group[self.product_id_col].to_numpy(dtype=object),
                    time_bucket,
                )
            )
        return np.concatenate(scores) if scores else np.asarray([], dtype=float)

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
        return (
            frame.groupby(["_customer_block", "_product_block", "_time_bucket", "_time_code", "_time_gate_code"])
            .size()
            .reset_index(name="count")
            .rename(
                columns={
                    "_customer_block": "customer_block",
                    "_product_block": "product_block",
                    "_time_bucket": "time_bucket",
                    "_time_code": "time_code",
                    "_time_gate_code": "time_gate_code",
                }
            )
            .sort_values(["time_code", "customer_block", "product_block"])
            .reset_index(drop=True)
        )

    def _build_slots(self) -> None:
        start = time.time()
        counts = self.block_pair_time_counts["count"].to_numpy(dtype=int)
        self.slot_id = np.arange(int(counts.sum()), dtype=int)
        self.slot_customer_block = np.repeat(self.block_pair_time_counts["customer_block"].to_numpy(dtype=int), counts)
        self.slot_product_block = np.repeat(self.block_pair_time_counts["product_block"].to_numpy(dtype=int), counts)
        self.slot_time_code = np.repeat(self.block_pair_time_counts["time_code"].to_numpy(dtype=int), counts)
        self.slot_time_gate_code = np.repeat(self.block_pair_time_counts["time_gate_code"].to_numpy(dtype=int), counts)
        self.slot_time_bucket = np.asarray([self.code_to_day[int(code)] for code in self.slot_time_code], dtype=object)
        self.slot_summary = {
            "num_slots": int(len(self.slot_id)),
            "num_block_pair_time_cells": int(len(self.block_pair_time_counts)),
            "average_cell_size": float(len(self.slot_id) / max(len(self.block_pair_time_counts), 1)),
            "max_cell_size": int(counts.max()) if len(counts) else 0,
            "num_time_buckets": int(len(self.code_to_day)),
            "num_time_gate_buckets": int(len(self.code_to_gate)),
            "slot_build_seconds": float(time.time() - start),
        }
        print(
            f"[slots] built {self.slot_summary['num_slots']:,} slots from "
            f"{self.slot_summary['num_block_pair_time_cells']:,} block-pair-time cells "
            f"in {self.slot_summary['slot_build_seconds']:.2f}s",
            flush=True,
        )

    def _verify_exact_constraints(self, synthetic: pd.DataFrame) -> None:
        real = self.real_df
        if len(synthetic) != len(real):
            raise RuntimeError(f"Row count mismatch: synthetic={len(synthetic)} real={len(real)}")
        customer_target = self._target_degree_array(self.customer_activity, self.customer_degrees)
        product_target = self._target_degree_array(self.product_activity, self.product_degrees)
        customer_counts = np.bincount(self.slot_customer_idx, minlength=len(customer_target))
        if not np.array_equal(customer_counts, customer_target):
            raise RuntimeError("Customer degree sequence mismatch")
        product_lookup = self.product_activity.entity_to_index
        product_idx_after_pairing = np.fromiter(
            (product_lookup[product] for product in self.slot_product_id),
            dtype=np.int64,
            count=len(self.slot_product_id),
        )
        product_counts = np.bincount(product_idx_after_pairing, minlength=len(product_target))
        if not np.array_equal(product_counts, product_target):
            raise RuntimeError("Product degree sequence mismatch")
        expected_time_counts = np.bincount(real["_time_code"].to_numpy(dtype=np.int64), minlength=len(self.code_to_day))
        synthetic_time_counts = np.bincount(self.slot_time_code.astype(np.int64), minlength=len(self.code_to_day))
        if not np.array_equal(expected_time_counts, synthetic_time_counts):
            raise RuntimeError("Daily count sequence mismatch")
        customer_blocks_after = self.customer_activity.get_fast_sampling_state()["entity_block"][self.slot_customer_idx]
        if not np.array_equal(customer_blocks_after.astype(np.int64), self.slot_customer_block.astype(np.int64)):
            raise RuntimeError("Customer block assignment mismatch")
        product_blocks_after = self.product_activity.get_fast_sampling_state()["entity_block"][product_idx_after_pairing]
        if not np.array_equal(product_blocks_after.astype(np.int64), self.slot_product_block.astype(np.int64)):
            raise RuntimeError("Product block assignment mismatch")
        expected_codes = self._cell_codes(
            self.block_pair_time_counts["customer_block"].to_numpy(dtype=np.int64),
            self.block_pair_time_counts["product_block"].to_numpy(dtype=np.int64),
            self.block_pair_time_counts["time_code"].to_numpy(dtype=np.int64),
        )
        generated_codes = self._cell_codes(
            self.slot_customer_block.astype(np.int64),
            self.slot_product_block.astype(np.int64),
            self.slot_time_code.astype(np.int64),
        )
        expected_counts = np.zeros(max(int(expected_codes.max()) if len(expected_codes) else 0, int(generated_codes.max()) if len(generated_codes) else 0) + 1, dtype=np.int64)
        np.add.at(expected_counts, expected_codes, self.block_pair_time_counts["count"].to_numpy(dtype=np.int64))
        generated_counts = np.bincount(generated_codes, minlength=len(expected_counts))
        if len(generated_counts) > len(expected_counts):
            expected_counts = np.pad(expected_counts, (0, len(generated_counts) - len(expected_counts)))
        if not np.array_equal(expected_counts, generated_counts):
            raise RuntimeError("Block-pair-time counts mismatch")

    def _target_degree_array(self, activity_model: FastTemporalActivityModel, degrees: Dict[Any, int]) -> np.ndarray:
        return np.asarray([int(degrees.get(entity, 0)) for entity in activity_model.entity_ids], dtype=np.int64)

    def _cell_codes(self, customer_block: np.ndarray, product_block: np.ndarray, time_code: np.ndarray) -> np.ndarray:
        max_product = max(
            int(self.block_pair_time_counts["product_block"].max()) if len(self.block_pair_time_counts) else 0,
            int(self.slot_product_block.max()) if len(self.slot_product_block) else 0,
        ) + 1
        max_time = max(
            int(self.block_pair_time_counts["time_code"].max()) if len(self.block_pair_time_counts) else 0,
            int(self.slot_time_code.max()) if len(self.slot_time_code) else 0,
        ) + 1
        return (customer_block.astype(np.int64) * max_product + product_block.astype(np.int64)) * max_time + time_code.astype(np.int64)

    def array_dtype_summary(self) -> Dict[str, Any]:
        arrays = {
            "slot_id": self.slot_id,
            "slot_customer_block": self.slot_customer_block,
            "slot_product_block": self.slot_product_block,
            "slot_time_code": self.slot_time_code,
            "slot_time_gate_code": self.slot_time_gate_code,
            "slot_customer_idx": self.slot_customer_idx,
            "slot_product_idx": self.slot_product_idx,
            "slot_customer_id": self.slot_customer_id,
            "slot_product_id": self.slot_product_id,
        }
        summary = {}
        for name, array in arrays.items():
            arr = np.asarray(array)
            summary[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
        if self.customer_activity is not None:
            customer_state = self.customer_activity.get_fast_sampling_state()
            summary["customer_empirical_time_values"] = {
                "dtype": str(customer_state["empirical_time_values"].dtype),
                "shape": list(customer_state["empirical_time_values"].shape),
            }
        if self.product_activity is not None:
            product_state = self.product_activity.get_fast_sampling_state()
            summary["product_empirical_time_values"] = {
                "dtype": str(product_state["empirical_time_values"].dtype),
                "shape": list(product_state["empirical_time_values"].shape),
            }
        return summary

    def _bpt_counts(self, frame: pd.DataFrame) -> pd.Series:
        tmp = frame.copy()
        time_col = "_time_bucket" if "_time_bucket" in tmp.columns else self.timestamp_col
        tmp["_customer_block_eval"] = tmp[self.customer_id_col].map(self.customer_blocks).fillna(-1).astype(int)
        tmp["_product_block_eval"] = tmp[self.product_id_col].map(self.product_blocks).fillna(-1).astype(int)
        tmp["_time_eval"] = canonical_time_bucket(tmp[time_col], self.time_granularity)
        return tmp.groupby(["_customer_block_eval", "_product_block_eval", "_time_eval"]).size().sort_index()


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
