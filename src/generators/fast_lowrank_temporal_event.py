"""Scalable fast low-rank temporal event-spine generator."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .fast_event_pairing import (
    pair_stubs_by_dynamic_affinity,
    repair_bad_pairs_within_cell,
    sample_entities_with_quotas,
    update_pair_counts,
)
from .fast_event_spine_metrics import evaluate_fast_event_spine
from .fast_temporal_activity import FastTemporalActivityModel, canonical_time_bucket, normalize_probs
from .lowrank_time_gated_affinity import LowRankTimeGatedAffinity


METHOD_NAME = "fast_lowrank_temporal_event"
METHOD_ALIAS = "flte_event_spine"


class FastLowRankTemporalEventGenerator:
    """Generate (customer_id, product_id, review_time) with batched low-rank pairing."""

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        structure_debug_dir: Optional[str | Path] = None,
        time_granularity: str = "day",
        time_gate_granularity: str = "month",
        preserve_daily_counts: bool = True,
        preserve_degrees: bool = True,
        block_pair_time_mode: str = "exact",
        rank: int = 32,
        alpha_customer_time: Any = "auto",
        alpha_product_time: Any = "auto",
        alpha_time_gate: Any = "auto",
        block_time_smoothing: float = 5.0,
        max_exact_affinity_cell_size: int = 512,
        large_cell_pairing: str = "projection_sort",
        nearest_neighbor_topk: int = 10,
        enable_fast_repair: bool = True,
        fast_repair_attempts: int = 10,
        allow_degree_slack: bool = False,
        seed: int = 42,
    ):
        if time_granularity != "day":
            raise ValueError("FastLowRankTemporalEventGenerator currently supports day event buckets only")
        if block_pair_time_mode not in {"exact", "sampled", "none"}:
            raise ValueError("block_pair_time_mode must be one of: exact, sampled, none")
        if large_cell_pairing not in {"projection_sort", "nearest_neighbor"}:
            raise ValueError("large_cell_pairing must be 'projection_sort' or 'nearest_neighbor'")
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.structure_debug_dir = Path(structure_debug_dir) if structure_debug_dir else None
        self.time_granularity = time_granularity
        self.time_gate_granularity = time_gate_granularity
        self.preserve_daily_counts = bool(preserve_daily_counts)
        self.preserve_degrees = bool(preserve_degrees)
        self.block_pair_time_mode = block_pair_time_mode
        self.rank = int(rank)
        self.alpha_customer_time = alpha_customer_time
        self.alpha_product_time = alpha_product_time
        self.alpha_time_gate = alpha_time_gate
        self.block_time_smoothing = float(block_time_smoothing)
        self.max_exact_affinity_cell_size = int(max_exact_affinity_cell_size)
        self.large_cell_pairing = large_cell_pairing
        self.nearest_neighbor_topk = int(nearest_neighbor_topk)
        self.enable_fast_repair = bool(enable_fast_repair)
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
        self.sampling_failures: List[Dict[str, Any]] = []
        self.runtime_metadata: Dict[str, Any] = {}
        self.sample_diagnostics: Dict[str, Any] = {}

    def fit(self, real_df: pd.DataFrame) -> "FastLowRankTemporalEventGenerator":
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
        rng = np.random.default_rng(self.seed if seed is None else int(seed))
        sample_start = time.time()
        rem_customer = Counter(self.customer_degrees)
        rem_product = Counter(self.product_degrees)
        pair_counts: Counter = Counter()
        self.sampling_failures = []
        events: List[tuple[Any, Any, str]] = []
        cells = self._ordered_cells(rng)
        total_cells = len(cells)
        total_events = int(cells["count"].sum()) if total_cells else 0
        max_cell_size = int(cells["count"].max()) if total_cells else 0
        large_cells = int((cells["count"] > self.max_exact_affinity_cell_size).sum()) if total_cells else 0
        print(f"[sample] processing {total_cells:,} cells, {total_events:,} events", flush=True)
        progress_every = max(1, total_cells // 10)
        for cell_index, row in cells.iterrows():
            customer_block = int(row["customer_block"])
            product_block = int(row["product_block"])
            time_bucket = row["time_bucket"]
            n = int(row["count"])
            sampled_customers, sampled_products = self._sample_cell_stubs(
                customer_block,
                product_block,
                time_bucket,
                n,
                rem_customer,
                rem_product,
                rng,
            )
            paired_products = pair_stubs_by_dynamic_affinity(
                sampled_customers,
                sampled_products,
                time_bucket,
                self.affinity_model,
                rng,
                max_exact_cell_size=self.max_exact_affinity_cell_size,
                large_cell_pairing=self.large_cell_pairing,
                nearest_neighbor_topk=self.nearest_neighbor_topk,
            )
            if self.enable_fast_repair:
                paired_products = repair_bad_pairs_within_cell(
                    sampled_customers,
                    paired_products,
                    time_bucket,
                    self.real_event_set,
                    pair_counts,
                    rng,
                    max_attempts=self.fast_repair_attempts,
                )
            for customer, product in zip(sampled_customers, paired_products):
                events.append((customer, product, time_bucket))
            self._decrement_quotas(rem_customer, sampled_customers)
            self._decrement_quotas(rem_product, paired_products)
            update_pair_counts(pair_counts, sampled_customers, paired_products)
            if (cell_index + 1) % progress_every == 0 or cell_index + 1 == total_cells:
                elapsed = time.time() - sample_start
                eps = len(events) / max(elapsed, 1e-9)
                pct = 100.0 * (cell_index + 1) / max(total_cells, 1)
                print(f"[sample] processed {pct:.0f}%, events/sec {eps:.1f}", flush=True)
        self._validate_residuals(rem_customer, rem_product, events, total_events)
        synthetic = pd.DataFrame(events, columns=[self.customer_id_col, self.product_id_col, self.timestamp_col])
        synthetic = synthetic.sort_values([self.timestamp_col, self.customer_id_col, self.product_id_col]).reset_index(drop=True)
        self.synthetic_df = synthetic
        self._finish_runtime(sample_start, len(synthetic), total_cells, max_cell_size, large_cells)
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
        with (debug / "block_pair_time_summary.json").open("w") as handle:
            json.dump(self.block_pair_time_summary(), handle, indent=2)
            handle.write("\n")
        self.customer_activity.save_summary(debug / "customer_time_activity_summary.json")
        self.product_activity.save_summary(debug / "product_time_activity_summary.json")
        self.affinity_model.save_summary(debug / "lowrank_time_gated_affinity_summary.json")
        with (debug / "sampling_failures.json").open("w") as handle:
            json.dump(self.sampling_failures, handle, indent=2)
            handle.write("\n")

    def save_metadata(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.metadata(), handle, indent=2)
            handle.write("\n")

    def evaluate(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, compute_c2st: bool = False) -> Dict[str, Any]:
        return evaluate_fast_event_spine(
            real_df,
            synthetic_df,
            structure_debug_dir=None,
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
            "class_name": "FastLowRankTemporalEventGenerator",
            "generates_joint_events": True,
            "event_tuple": [self.customer_id_col, self.product_id_col, self.timestamp_col],
            "preserves_old_joint_temporal_2k_sbm_event_code": True,
            "uses_time_conditioned_customer_sampling": True,
            "uses_time_conditioned_product_sampling": True,
            "uses_time_dependent_pairing_score": True,
            "uses_dense_F_u_i_t": False,
            "dynamic_affinity_type": "time_gated_low_rank",
            "dynamic_affinity_formula": "(z_u * g_t)^T z_i",
            "per_event_candidate_pool_scoring": False,
            "batch_cell_pairing": True,
            "time_granularity": self.time_granularity,
            "time_gate_granularity": self.time_gate_granularity,
            "rank": int(self.rank),
            "alpha_customer_time": float(self.customer_activity.alpha_resolved),
            "alpha_product_time": float(self.product_activity.alpha_resolved),
            "alpha_time_gate": float(affinity_summary.get("alpha_time_gate", 0.0)),
            "block_pair_time_mode": self.block_pair_time_mode,
            "max_exact_affinity_cell_size": int(self.max_exact_affinity_cell_size),
            "large_cell_pairing": self.large_cell_pairing,
            "nearest_neighbor_topk": int(self.nearest_neighbor_topk),
            "preserve_daily_counts": bool(self.preserve_daily_counts),
            "preserve_degrees": bool(self.preserve_degrees),
            "enable_fast_repair": bool(self.enable_fast_repair),
            "fast_repair_attempts": int(self.fast_repair_attempts),
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
        metadata.update(self.runtime_metadata)
        metadata.update(self.sample_diagnostics)
        return metadata

    def block_pair_time_summary(self) -> Dict[str, Any]:
        counts = self.block_pair_time_counts["count"].to_numpy(dtype=float)
        return {
            "block_pair_time_mode": self.block_pair_time_mode,
            "num_customer_blocks": int(len(set(self.customer_blocks.values()))),
            "num_product_blocks": int(len(set(self.product_blocks.values()))),
            "num_time_buckets": int(self.block_pair_time_counts["time_bucket"].nunique()) if len(self.block_pair_time_counts) else 0,
            "num_nonzero_cells": int(len(self.block_pair_time_counts)),
            "total_count": int(counts.sum()) if len(counts) else 0,
            "max_cell_count": int(counts.max()) if len(counts) else 0,
            "mean_nonzero_cell_count": float(counts.mean()) if len(counts) else 0.0,
        }

    def _load_or_fallback_blocks(self, frame: pd.DataFrame) -> tuple[Dict[Any, int], Dict[Any, int], List[str]]:
        warnings = []
        customer_blocks = load_entity_blocks(
            self.structure_debug_dir,
            "customer_blocks.csv",
            self.customer_id_col,
            "customer_block",
        )
        product_blocks = load_entity_blocks(
            self.structure_debug_dir,
            "product_blocks.csv",
            self.product_id_col,
            "product_block",
        )
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
                rows.append(
                    {
                        "customer_block": int(cblock),
                        "product_block": int(pblock),
                        "time_bucket": time_bucket,
                        "count": int(count),
                    }
                )
        return pd.DataFrame(rows).sort_values(["time_bucket", "customer_block", "product_block"]).reset_index(drop=True)

    def _ordered_cells(self, rng: np.random.Generator) -> pd.DataFrame:
        cells = self.block_pair_time_counts.copy()
        cells["_shuffle"] = rng.random(len(cells)) if len(cells) else []
        return cells.sort_values(["time_bucket", "count", "_shuffle"], ascending=[True, False, True]).drop(columns=["_shuffle"]).reset_index(drop=True)

    def _sample_cell_stubs(
        self,
        customer_block: int,
        product_block: int,
        time_bucket: str,
        n: int,
        rem_customer: Counter,
        rem_product: Counter,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        customer_ids, customer_probs = self.customer_activity.probabilities_for_block_time(customer_block, time_bucket)
        product_ids, product_probs = self.product_activity.probabilities_for_block_time(product_block, time_bucket)
        sampled_customers, customer_ok, customer_diag = sample_entities_with_quotas(
            customer_ids,
            rem_customer,
            customer_probs,
            n,
            rng,
        )
        sampled_products, product_ok, product_diag = sample_entities_with_quotas(
            product_ids,
            rem_product,
            product_probs,
            n,
            rng,
        )
        if customer_ok and product_ok:
            return sampled_customers, sampled_products
        return self._fallback_cell_sampling(
            customer_block,
            product_block,
            time_bucket,
            n,
            rem_customer,
            rem_product,
            rng,
            customer_diag,
            product_diag,
        )

    def _fallback_cell_sampling(
        self,
        customer_block: int,
        product_block: int,
        time_bucket: str,
        n: int,
        rem_customer: Counter,
        rem_product: Counter,
        rng: np.random.Generator,
        customer_diag: Dict[str, Any],
        product_diag: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        failure = {
            "type": "cell_quota_fallback",
            "customer_block": int(customer_block),
            "product_block": int(product_block),
            "time_bucket": time_bucket,
            "count": int(n),
            "customer_diagnostics": customer_diag,
            "product_diagnostics": product_diag,
        }
        customer_ids, _ = self.customer_activity.degree_weights_for_block(customer_block)
        product_ids, _ = self.product_activity.degree_weights_for_block(product_block)
        sampled_customers, customer_ok, customer_fallback = sample_entities_with_quotas(
            customer_ids,
            rem_customer,
            np.ones(len(customer_ids), dtype=float),
            n,
            rng,
        )
        sampled_products, product_ok, product_fallback = sample_entities_with_quotas(
            product_ids,
            rem_product,
            np.ones(len(product_ids), dtype=float),
            n,
            rng,
        )
        failure["fallback_customer_diagnostics"] = customer_fallback
        failure["fallback_product_diagnostics"] = product_fallback
        self.sampling_failures.append(failure)
        if customer_ok and product_ok:
            return sampled_customers, sampled_products
        if self.block_pair_time_mode != "exact":
            all_customer_ids = np.asarray(list(rem_customer.keys()), dtype=object)
            all_product_ids = np.asarray(list(rem_product.keys()), dtype=object)
            sampled_customers, customer_ok, broad_customer = sample_entities_with_quotas(
                all_customer_ids,
                rem_customer,
                np.ones(len(all_customer_ids), dtype=float),
                n,
                rng,
            )
            sampled_products, product_ok, broad_product = sample_entities_with_quotas(
                all_product_ids,
                rem_product,
                np.ones(len(all_product_ids), dtype=float),
                n,
                rng,
            )
            failure["broad_customer_diagnostics"] = broad_customer
            failure["broad_product_diagnostics"] = broad_product
            if customer_ok and product_ok:
                return sampled_customers, sampled_products
        if not self.allow_degree_slack:
            raise RuntimeError(
                "Cannot fill strict block-pair-time cell "
                f"({customer_block}, {product_block}, {time_bucket}) with n={n}. "
                "Try --block-pair-time-mode sampled or --allow-degree-slack."
            )
        return sampled_customers, sampled_products

    def _decrement_quotas(self, remaining: Counter, sampled: np.ndarray) -> None:
        for entity, count in Counter(sampled).items():
            remaining[entity] -= int(count)

    def _validate_residuals(
        self,
        rem_customer: Counter,
        rem_product: Counter,
        events: List[tuple[Any, Any, str]],
        expected_events: int,
    ) -> None:
        residual_customers = {k: int(v) for k, v in rem_customer.items() if v != 0}
        residual_products = {k: int(v) for k, v in rem_product.items() if v != 0}
        if len(events) != expected_events:
            message = {
                "type": "row_count_mismatch",
                "expected": int(expected_events),
                "actual": int(len(events)),
            }
            self.sampling_failures.append(message)
            if not self.allow_degree_slack:
                raise RuntimeError(f"Generated row count mismatch: {message}")
        if self.preserve_degrees and (residual_customers or residual_products):
            message = {
                "type": "residual_degree_quota",
                "residual_customers": residual_customers,
                "residual_products": residual_products,
            }
            self.sampling_failures.append(message)
            if not self.allow_degree_slack:
                raise RuntimeError(f"Degree quota repair failed: {message}")

    def _finish_runtime(self, sample_start: float, num_events: int, num_cells: int, max_cell_size: int, large_cells: int) -> None:
        sample_seconds = float(time.time() - sample_start)
        fit_seconds = float(self.runtime_metadata.get("fit_seconds", 0.0))
        self.runtime_metadata.update(
            {
                "sample_seconds": sample_seconds,
                "total_seconds": fit_seconds + sample_seconds,
                "events_per_second": float(num_events / max(sample_seconds, 1e-9)),
                "num_cells_processed": int(num_cells),
                "average_cell_size": float(num_events / max(num_cells, 1)),
                "max_cell_size": int(max_cell_size),
                "percent_large_cells_projection_sort": float(large_cells / max(num_cells, 1)),
            }
        )
        print(
            f"[done] total_seconds={self.runtime_metadata['total_seconds']:.2f}, "
            f"events_per_second={self.runtime_metadata['events_per_second']:.1f}",
            flush=True,
        )


def load_entity_blocks(
    root: Optional[str | Path],
    filename: str,
    preferred_entity_col: str,
    preferred_block_col: str,
) -> Dict[Any, int]:
    if not root:
        return {}
    path = Path(root) / filename
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    entity_col = first_present(frame, [preferred_entity_col, "entity_id", "id", "customer_id", "product_id"])
    block_col = first_present(frame, [preferred_block_col, "block", "customer_block", "product_block"])
    if entity_col is None or block_col is None:
        return {}
    return {row[entity_col]: int(row[block_col]) for _, row in frame[[entity_col, block_col]].dropna().iterrows()}


def first_present(frame: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in frame.columns:
            return col
    return None
