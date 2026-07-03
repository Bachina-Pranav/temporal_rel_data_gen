#!/usr/bin/env python3
"""Benchmark the fast time-biased stub assignment kernel."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.stub_time_assignment import assign_stubs_to_slots_by_time  # noqa: E402


class SyntheticActivityModel:
    def __init__(
        self,
        entity_ids: np.ndarray,
        entity_blocks: np.ndarray,
        degrees: np.ndarray,
        num_time_codes: int,
        num_blocks: int,
        rng: np.random.Generator,
    ):
        self.entity_ids = np.asarray(entity_ids, dtype=object)
        self.entity_to_index = {entity: idx for idx, entity in enumerate(self.entity_ids)}
        self.time_buckets = [str(idx) for idx in range(num_time_codes)]
        empirical_offsets = np.zeros(len(entity_ids) + 1, dtype=np.int64)
        empirical_offsets[1:] = np.cumsum(degrees, dtype=np.int64)
        empirical_time_values = rng.integers(0, num_time_codes, size=int(degrees.sum()), dtype=np.int32)
        block_time_values = {}
        block_time_cdfs = {}
        uniform_values = np.arange(num_time_codes, dtype=np.int32)
        uniform_cdf = np.linspace(1.0 / num_time_codes, 1.0, num_time_codes)
        for block in range(num_blocks):
            block_time_values[block] = uniform_values
            block_time_cdfs[block] = uniform_cdf
        self._state = {
            "entity_ids": self.entity_ids,
            "entity_to_index": self.entity_to_index,
            "entity_mix_weight": np.full(len(entity_ids), 0.5, dtype=float),
            "entity_block": entity_blocks.astype(np.int64),
            "empirical_offsets": empirical_offsets,
            "empirical_time_values": empirical_time_values,
            "block_time_values": block_time_values,
            "block_time_cdfs": block_time_cdfs,
            "global_time_values": uniform_values,
            "global_time_cdf": uniform_cdf,
        }

    def get_fast_sampling_state(self):
        return self._state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark time-biased block stub assignment.")
    parser.add_argument("--num-entities", type=int, default=50_000)
    parser.add_argument("--num-stubs", type=int, default=100_000)
    parser.add_argument("--num-blocks", type=int, default=5)
    parser.add_argument("--num-time-codes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    entity_ids = np.arange(args.num_entities, dtype=np.int64)
    entity_blocks = rng.integers(0, args.num_blocks, size=args.num_entities, dtype=np.int64)
    degrees = rng.multinomial(args.num_stubs, np.ones(args.num_entities, dtype=float) / args.num_entities).astype(np.int64)
    slot_blocks = np.repeat(np.arange(args.num_blocks, dtype=np.int64), np.bincount(entity_blocks, weights=degrees, minlength=args.num_blocks).astype(np.int64))
    rng.shuffle(slot_blocks)
    slot_time_codes = rng.integers(0, args.num_time_codes, size=args.num_stubs, dtype=np.int32)
    activity = SyntheticActivityModel(entity_ids, entity_blocks, degrees, args.num_time_codes, args.num_blocks, rng)
    degree_map = {int(entity): int(degree) for entity, degree in zip(entity_ids, degrees) if int(degree) > 0}
    block_map = {int(entity): int(block) for entity, block in zip(entity_ids, entity_blocks)}

    start = time.time()
    _, summary = assign_stubs_to_slots_by_time(
        entity_ids,
        degree_map,
        block_map,
        slot_blocks,
        slot_time_codes,
        activity,
        rng,
        log_label="benchmark-stubs",
        return_entity_indices=True,
    )
    total = float(time.time() - start)
    print(f"stub construction seconds: {summary['stub_construction_seconds']:.4f}")
    print(f"desired-time sampling seconds: {summary['desired_time_sampling_seconds']:.4f}")
    print(f"sorting seconds: {summary['sorting_seconds']:.4f}")
    print(f"total seconds: {total:.4f}")


if __name__ == "__main__":
    main()
