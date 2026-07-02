"""Conditioning features for temporal summary Text V1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ConditionFeatureResult:
    features: pd.DataFrame
    metadata: Dict[str, Any]


class ConditionFeatureNormalizer:
    """Column-preserving mean/std normalizer for Text V1 conditioning features."""

    def __init__(self, columns: Optional[List[str]] = None, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        self.columns = list(columns or [])
        self.mean = np.asarray(mean if mean is not None else [], dtype=np.float32)
        self.std = np.asarray(std if std is not None else [], dtype=np.float32)

    def fit(self, frame: pd.DataFrame) -> "ConditionFeatureNormalizer":
        self.columns = [str(col) for col in frame.columns]
        values = frame[self.columns].to_numpy(dtype=np.float32)
        self.mean = np.nanmean(values, axis=0).astype(np.float32)
        self.std = np.nanstd(values, axis=0).astype(np.float32)
        self.mean = np.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std > 1e-6), self.std, 1.0).astype(np.float32)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        aligned = frame.reindex(columns=self.columns, fill_value=0.0)
        values = aligned.to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionFeatureNormalizer":
        return cls(columns=list(data["columns"]), mean=list(data["mean"]), std=list(data["std"]))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ConditionFeatureNormalizer":
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))


def build_text_condition_features(
    df: pd.DataFrame,
    structure_debug_dir: Optional[str | Path] = None,
    mode: str = "train",
    generated_history: Optional[pd.DataFrame] = None,
    sampled_effects_dir: Optional[str | Path] = None,
    customer_id_col: str = "customer_id",
    product_id_col: str = "product_id",
    timestamp_col: str = "review_time",
    rating_col: str = "rating",
    verified_col: str = "verified",
) -> ConditionFeatureResult:
    """Build numeric conditioning features without using text content."""

    del generated_history
    frame = df.copy().reset_index(drop=True)
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if timestamps.isna().all():
        timestamps = pd.Series(pd.Timestamp("1970-01-01"), index=frame.index)
    timestamps = timestamps.fillna(timestamps.dropna().min())
    first_ts = timestamps.min()
    last_ts = timestamps.max()
    total_seconds = max(float((last_ts - first_ts).total_seconds()), 1.0)

    rating = pd.to_numeric(frame.get(rating_col, 0), errors="coerce").fillna(0.0).astype(float)
    verified = frame.get(verified_col, False).map(_to_binary) if verified_col in frame.columns else pd.Series(0.0, index=frame.index)
    verified = verified.fillna(0.0).astype(float)

    customer_blocks, product_blocks, block_metadata = load_block_maps(
        structure_debug_dir, customer_id_col, product_id_col
    )
    customer_block = frame[customer_id_col].map(customer_blocks).fillna(-1).astype(float)
    product_block = frame[product_id_col].map(product_blocks).fillna(-1).astype(float)
    block_pair = customer_block.astype(int).astype(str) + ":" + product_block.astype(int).astype(str)
    block_pair_code = pd.factorize(block_pair, sort=True)[0].astype(float)

    features = pd.DataFrame(index=frame.index)
    features["time_year"] = timestamps.dt.year.astype(float)
    features["time_month"] = timestamps.dt.month.astype(float)
    features["time_year_month"] = (timestamps.dt.year * 12 + timestamps.dt.month).astype(float)
    features["time_normalized"] = ((timestamps - first_ts).dt.total_seconds() / total_seconds).astype(float)
    features["days_since_first_review"] = ((timestamps - first_ts).dt.total_seconds() / 86400.0).astype(float)
    features["month_sin"] = np.sin(2.0 * np.pi * timestamps.dt.month.astype(float) / 12.0)
    features["month_cos"] = np.cos(2.0 * np.pi * timestamps.dt.month.astype(float) / 12.0)
    features["rating"] = rating
    for value in range(1, 6):
        features[f"rating_is_{value}"] = (rating.round().astype(int) == value).astype(float)
    features["verified"] = verified
    features["customer_block"] = customer_block
    features["product_block"] = product_block
    features["block_pair_code"] = block_pair_code

    features = pd.concat(
        [
            features,
            structural_features(frame, timestamps, customer_id_col, product_id_col),
            causal_history_features(
                frame,
                timestamps,
                rating,
                verified,
                customer_id_col,
                product_id_col,
                block_pair,
            ),
            optional_sampled_effect_features(
                frame,
                sampled_effects_dir,
                customer_id_col,
                product_id_col,
            ),
        ],
        axis=1,
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    metadata = {
        "mode": mode,
        "uses_text_content": False,
        "uses_previous_text": False,
        "structure_debug_dir": str(structure_debug_dir) if structure_debug_dir else None,
        "sampled_effects_dir": str(sampled_effects_dir) if sampled_effects_dir else None,
        "num_features": int(features.shape[1]),
        **block_metadata,
    }
    return ConditionFeatureResult(features=features, metadata=metadata)


def structural_features(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    customer_id_col: str,
    product_id_col: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for entity_col, prefix in [(customer_id_col, "customer"), (product_id_col, "product")]:
        degree = frame.groupby(entity_col)[entity_col].transform("size").astype(float)
        first = timestamps.groupby(frame[entity_col]).transform("min")
        last = timestamps.groupby(frame[entity_col]).transform("max")
        out[f"{prefix}_degree"] = degree
        out[f"log1p_{prefix}_degree"] = np.log1p(degree)
        out[f"{prefix}_lifecycle_start"] = ((first - timestamps.min()).dt.total_seconds() / 86400.0).astype(float)
        out[f"{prefix}_lifecycle_end"] = ((last - timestamps.min()).dt.total_seconds() / 86400.0).astype(float)
    return out


def causal_history_features(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    rating: pd.Series,
    verified: pd.Series,
    customer_id_col: str,
    product_id_col: str,
    block_pair: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    sorted_idx = list(np.argsort(timestamps.to_numpy(dtype="datetime64[ns]")))
    state: Dict[str, Dict[Any, Dict[str, float]]] = {
        "customer": {},
        "product": {},
        "block_pair": {},
    }
    global_state = {"count": 0.0, "rating_sum": 0.0, "verified_sum": 0.0}
    rows: Dict[int, Dict[str, float]] = {}

    def stats(bucket: Dict[Any, Dict[str, float]], key: Any, prefix: str) -> Dict[str, float]:
        item = bucket.get(key, {"count": 0.0, "rating_sum": 0.0, "verified_sum": 0.0})
        count = item["count"]
        return {
            f"{prefix}_past_review_count": count,
            f"{prefix}_past_avg_rating": item["rating_sum"] / count if count > 0 else 0.0,
            f"{prefix}_past_verified_rate": item["verified_sum"] / count if count > 0 else 0.0,
        }

    def update(bucket: Dict[Any, Dict[str, float]], key: Any, r: float, v: float) -> None:
        item = bucket.setdefault(key, {"count": 0.0, "rating_sum": 0.0, "verified_sum": 0.0})
        item["count"] += 1.0
        item["rating_sum"] += r
        item["verified_sum"] += v

    for idx in sorted_idx:
        customer = frame.at[idx, customer_id_col]
        product = frame.at[idx, product_id_col]
        pair = block_pair.iloc[idx]
        row = {}
        row.update(stats(state["customer"], customer, "customer"))
        row.update(stats(state["product"], product, "product"))
        row.update(stats(state["block_pair"], pair, "block_pair"))
        g_count = global_state["count"]
        row["global_past_avg_rating"] = global_state["rating_sum"] / g_count if g_count > 0 else 0.0
        row["global_past_verified_rate"] = global_state["verified_sum"] / g_count if g_count > 0 else 0.0
        rows[int(idx)] = row
        r = float(rating.iloc[idx])
        v = float(verified.iloc[idx])
        update(state["customer"], customer, r, v)
        update(state["product"], product, r, v)
        update(state["block_pair"], pair, r, v)
        global_state["count"] += 1.0
        global_state["rating_sum"] += r
        global_state["verified_sum"] += v
    return pd.DataFrame.from_dict(rows, orient="index").reindex(frame.index).fillna(0.0)


def optional_sampled_effect_features(
    frame: pd.DataFrame,
    sampled_effects_dir: Optional[str | Path],
    customer_id_col: str,
    product_id_col: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    if sampled_effects_dir is None:
        out["sampled_customer_rating_effect_norm"] = 0.0
        out["sampled_product_rating_effect_norm"] = 0.0
        out["sampled_customer_verified_effect"] = 0.0
        out["sampled_product_verified_effect"] = 0.0
        return out
    root = Path(sampled_effects_dir)
    customer_path = root / "sampled_customer_effects_v3.csv"
    product_path = root / "sampled_product_effects_v3.csv"
    out = merge_effect_file(out, frame, customer_path, customer_id_col, "customer")
    out = merge_effect_file(out, frame, product_path, product_id_col, "product")
    return out.fillna(0.0)


def merge_effect_file(
    out: pd.DataFrame,
    frame: pd.DataFrame,
    path: Path,
    id_col: str,
    prefix: str,
) -> pd.DataFrame:
    if not path.exists():
        out[f"sampled_{prefix}_rating_effect_norm"] = 0.0
        out[f"sampled_{prefix}_verified_effect"] = 0.0
        return out
    effects = pd.read_csv(path)
    if id_col not in effects.columns:
        out[f"sampled_{prefix}_rating_effect_norm"] = 0.0
        out[f"sampled_{prefix}_verified_effect"] = 0.0
        return out
    rating_cols = sorted(col for col in effects.columns if col.startswith("sampled_rating_effect_"))
    verified_col = "sampled_verified_effect" if "sampled_verified_effect" in effects.columns else None
    keep = [id_col] + rating_cols + ([verified_col] if verified_col else [])
    merged = frame[[id_col]].merge(effects[keep], on=id_col, how="left")
    rating_values = merged[rating_cols].fillna(0.0).to_numpy(dtype=float) if rating_cols else np.zeros((len(frame), 1))
    out[f"sampled_{prefix}_rating_effect_norm"] = np.linalg.norm(rating_values, axis=1)
    if verified_col:
        out[f"sampled_{prefix}_verified_effect"] = pd.to_numeric(merged[verified_col], errors="coerce").fillna(0.0)
    else:
        out[f"sampled_{prefix}_verified_effect"] = 0.0
    for col in rating_cols:
        out[f"sampled_{prefix}_{col.replace('sampled_', '')}"] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return out


def load_block_maps(
    structure_debug_dir: Optional[str | Path],
    customer_id_col: str,
    product_id_col: str,
) -> tuple[Dict[Any, int], Dict[Any, int], Dict[str, Any]]:
    if not structure_debug_dir:
        return {}, {}, {"uses_block_features": False}
    root = Path(structure_debug_dir)
    customer = load_one_block_map(root / "customer_blocks.csv", customer_id_col, "customer_block")
    product = load_one_block_map(root / "product_blocks.csv", product_id_col, "product_block")
    return customer, product, {
        "uses_block_features": bool(customer or product),
        "num_customer_blocks_loaded": int(len(set(customer.values()))) if customer else 0,
        "num_product_blocks_loaded": int(len(set(product.values()))) if product else 0,
    }


def load_one_block_map(path: Path, id_col: str, block_col: str) -> Dict[Any, int]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if id_col not in frame.columns or block_col not in frame.columns:
        return {}
    return dict(zip(frame[id_col], pd.to_numeric(frame[block_col], errors="coerce").fillna(-1).astype(int)))


def _to_binary(value: Any) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"true", "1", "yes", "y", "t"} else 0.0
    return 1.0 if bool(value) else 0.0
