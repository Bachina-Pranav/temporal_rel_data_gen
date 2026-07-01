"""Temporal non-text attribute diffusion conditioned on review event spines."""

from __future__ import annotations

import json
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .nontext_diffusion_model import TemporalFeatureDiffusionModel
from .nontext_diffusion_schedules import gaussian_sigma, mask_probability
from .temporal_attribute_sampler import chronological_groups
from .temporal_causal_features import (
    CAUSAL_CONTINUOUS_FEATURES,
    CAUSAL_DISCRETE_FEATURES,
    TemporalCausalFeatureBuilder,
    load_block_maps,
    normalize_verified,
    save_json,
)


GENERATOR_NAME = "temporal_nontext_attr_diffusion"


@dataclass
class TemporalNonTextTrainingResult:
    output_dir: Path
    best_checkpoint: Path
    latest_checkpoint: Path
    history: List[Dict[str, float]]


class TemporalNonTextAttributeDiffusion:
    """Generate rating/verified/numerical attributes for a fixed temporal spine."""

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.cat_cols = list(cat_cols or ["rating", "verified"])
        self.num_cols = list(num_cols or [])
        self.seed = int(seed)

        self.category_values: Dict[str, List[Any]] = {}
        self.category_lookup: Dict[str, Dict[Any, int]] = {}
        self.feature_builder: Optional[TemporalCausalFeatureBuilder] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.discrete_feature_maps: Dict[str, Dict[Any, int]] = {}
        self.numerical_metadata: Dict[str, Dict[str, Any]] = {}
        self.model: Optional[TemporalFeatureDiffusionModel] = None
        self.model_config: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}

    @classmethod
    def train_from_csv(
        cls,
        reviews_path: str | Path,
        output_dir: str | Path,
        structure_debug_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> TemporalNonTextTrainingResult:
        reviews = pd.read_csv(reviews_path)
        generator = cls(
            customer_id_col=kwargs.get("customer_id_col", "customer_id"),
            product_id_col=kwargs.get("product_id_col", "product_id"),
            timestamp_col=kwargs.get("timestamp_col", "review_time"),
            cat_cols=kwargs.get("cat_cols"),
            num_cols=kwargs.get("num_cols"),
            seed=kwargs.get("seed", 42),
        )
        return generator.train(
            reviews,
            output_dir=output_dir,
            structure_debug_dir=structure_debug_dir,
            epochs=kwargs.get("epochs", 5),
            batch_size=kwargs.get("batch_size", 256),
            learning_rate=kwargs.get("learning_rate", 1e-3),
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 4),
            dropout=kwargs.get("dropout", 0.1),
            validation_fraction=kwargs.get("validation_fraction", 0.1),
            random_split=kwargs.get("random_split", False),
            lambda_cat=kwargs.get("lambda_cat", 1.0),
            lambda_num=kwargs.get("lambda_num", 1.0),
            mask_schedule=kwargs.get("mask_schedule", "cosine"),
            device=kwargs.get("device", "cpu"),
        )

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "cpu"
    ) -> "TemporalNonTextAttributeDiffusion":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        generator = cls(
            customer_id_col=config["customer_id_col"],
            product_id_col=config["product_id_col"],
            timestamp_col=config["timestamp_col"],
            cat_cols=config["cat_cols"],
            num_cols=config["num_cols"],
            seed=config["seed"],
        )
        generator.config = config
        generator.category_values = checkpoint["category_values"]
        generator.category_lookup = {
            col: {canonical_category_key(value): idx for idx, value in enumerate(values)}
            for col, values in generator.category_values.items()
        }
        generator.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        generator.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        generator.discrete_feature_maps = checkpoint["discrete_feature_maps"]
        generator.numerical_metadata = checkpoint["numerical_metadata"]
        generator.model_config = checkpoint["model_config"]
        generator.model = TemporalFeatureDiffusionModel(**generator.model_config).to(device)
        generator.model.load_state_dict(checkpoint["model_state_dict"])
        generator.model.eval()
        generator._checkpoint = checkpoint
        return generator

    @classmethod
    def sample_from_checkpoint(
        cls,
        synthetic_spine_path: str | Path,
        checkpoint_path: str | Path,
        output_path: str | Path,
        structure_debug_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        generator = cls.load_checkpoint(
            checkpoint_path, device=kwargs.get("device", "cpu")
        )
        spine = pd.read_csv(synthetic_spine_path)
        output = generator.sample(
            spine,
            structure_debug_dir=structure_debug_dir,
            seed=kwargs.get("seed", generator.seed),
            num_steps=kwargs.get("num_steps", 50),
            cat_sampling_strategy=kwargs.get("cat_sampling_strategy", "sample"),
            temperature=kwargs.get("temperature", 1.0),
            sampling_time_group=kwargs.get("sampling_time_group", "date"),
            sampling_window_days=kwargs.get("sampling_window_days", 1.0),
            device=kwargs.get("device", "cpu"),
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        return output

    def train(
        self,
        reviews: pd.DataFrame,
        output_dir: str | Path,
        structure_debug_dir: str | Path | None = None,
        epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        validation_fraction: float = 0.1,
        random_split: bool = False,
        lambda_cat: float = 1.0,
        lambda_num: float = 1.0,
        mask_schedule: str = "cosine",
        device: str = "cpu",
    ) -> TemporalNonTextTrainingResult:
        self._set_seeds(self.seed)
        output_dir = Path(output_dir)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        reviews = self._preprocess_training_reviews(reviews)
        self._fit_categories(reviews)
        self.num_cols = [col for col in self.num_cols if col in reviews.columns]
        missing_num = [col for col in self.config.get("requested_num_cols", []) if col not in reviews.columns]
        for col in missing_num:
            warnings.warn(f"Numerical column {col!r} is absent; skipping.")
        self.config["num_cols"] = list(self.num_cols)

        customer_blocks, product_blocks = load_block_maps(
            structure_debug_dir, self.customer_id_col, self.product_id_col
        )
        self.feature_builder = TemporalCausalFeatureBuilder(
            customer_id_col=self.customer_id_col,
            product_id_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            rating_col=self.cat_cols[0] if self.cat_cols else "rating",
            verified_col="verified" if "verified" in self.cat_cols else self.cat_cols[-1],
            customer_blocks=customer_blocks,
            product_blocks=product_blocks,
        )
        features = self.feature_builder.transform_training(reviews)
        self._fit_feature_metadata(features)
        encoded = self._encode_dataset(reviews, features)
        train_idx, val_idx = self._split_indices(
            reviews, validation_fraction, random_split=random_split
        )

        self.model_config = {
            "cat_cols": self.cat_cols,
            "cat_vocab_sizes": {
                col: len(self.category_values[col]) for col in self.cat_cols
            },
            "num_numerical": len(self.num_cols),
            "continuous_feature_dim": len(CAUSAL_CONTINUOUS_FEATURES),
            "discrete_feature_vocab_sizes": {
                col: len(mapping) + 1 for col, mapping in self.discrete_feature_maps.items()
            },
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
        }
        self.model = TemporalFeatureDiffusionModel(**self.model_config).to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        history: List[Dict[str, float]] = []
        best_val = float("inf")
        best_path = checkpoint_dir / "best.pt"
        latest_path = checkpoint_dir / "latest.pt"
        for epoch in range(1, int(epochs) + 1):
            train_losses = self._run_epoch(
                encoded,
                train_idx,
                optimizer=optimizer,
                batch_size=batch_size,
                device=device,
                train=True,
                lambda_cat=lambda_cat,
                lambda_num=lambda_num,
                mask_schedule=mask_schedule,
            )
            val_losses = self._run_epoch(
                encoded,
                val_idx if len(val_idx) else train_idx,
                optimizer=None,
                batch_size=batch_size,
                device=device,
                train=False,
                lambda_cat=lambda_cat,
                lambda_num=lambda_num,
                mask_schedule=mask_schedule,
            )
            row = {
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_losses.items()},
                **{f"val_{key}": value for key, value in val_losses.items()},
            }
            history.append(row)
            self._save_checkpoint(latest_path, history)
            if val_losses["total"] <= best_val:
                best_val = val_losses["total"]
                self._save_checkpoint(best_path, history)
            print(
                f"epoch {epoch:03d} train={train_losses['total']:.4f} "
                f"val={val_losses['total']:.4f}"
            )

        self._write_metadata(output_dir)
        return TemporalNonTextTrainingResult(output_dir, best_path, latest_path, history)

    def sample(
        self,
        synthetic_spine: pd.DataFrame,
        structure_debug_dir: str | Path | None = None,
        seed: Optional[int] = None,
        num_steps: int = 50,
        cat_sampling_strategy: str = "sample",
        temperature: float = 1.0,
        sampling_time_group: str = "date",
        sampling_window_days: float = 1.0,
        device: str = "cpu",
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if cat_sampling_strategy not in {"sample", "argmax"}:
            raise ValueError("cat_sampling_strategy must be sample or argmax.")
        self._set_seeds(self.seed if seed is None else int(seed))
        self.model.to(device)
        self.model.eval()

        spine = synthetic_spine.copy()
        spine[self.timestamp_col] = pd.to_datetime(spine[self.timestamp_col], errors="coerce")
        customer_blocks, product_blocks = load_block_maps(
            structure_debug_dir, self.customer_id_col, self.product_id_col
        )
        metadata = self._checkpoint.get("feature_builder_metadata", {})
        builder = TemporalCausalFeatureBuilder(
            customer_id_col=self.customer_id_col,
            product_id_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            rating_col=self.cat_cols[0] if self.cat_cols else "rating",
            verified_col="verified" if "verified" in self.cat_cols else self.cat_cols[-1],
            customer_blocks=customer_blocks,
            product_blocks=product_blocks,
            date_only=metadata.get("date_only"),
            marginal_rating=metadata.get("marginal_rating", 0.0),
            marginal_verified=metadata.get("marginal_verified", 0.0),
        )
        if metadata.get("min_time"):
            builder.min_time = pd.Timestamp(metadata["min_time"])
        if metadata.get("max_time"):
            builder.max_time = pd.Timestamp(metadata["max_time"])
        builder.prepare_sampling(spine)

        generated_groups = []
        with torch.no_grad():
            for _, group in chronological_groups(
                spine, self.timestamp_col, mode=sampling_time_group, window_days=sampling_window_days
            ):
                features = builder.transform_current_group(group)
                encoded_features = self._encode_features_for_sampling(features)
                batch_size = len(group)
                cat_tokens = {
                    col: torch.full(
                        (batch_size,),
                        len(self.category_values[col]),
                        dtype=torch.long,
                        device=device,
                    )
                    for col in self.cat_cols
                }
                numerical = (
                    torch.randn(batch_size, len(self.num_cols), device=device)
                    if self.num_cols
                    else None
                )
                continuous = torch.tensor(
                    encoded_features["continuous"], dtype=torch.float32, device=device
                )
                discrete = {
                    col: torch.tensor(values, dtype=torch.long, device=device)
                    for col, values in encoded_features["discrete"].items()
                }
                steps = np.linspace(1.0, 0.0, max(int(num_steps), 1))
                for step in steps:
                    t = torch.full((batch_size,), float(step), device=device)
                    out = self.model(
                        cat_tokens,
                        continuous,
                        discrete,
                        t,
                        numerical_noisy=numerical,
                    )
                    for col in self.cat_cols:
                        logits = out[f"{col}_logits"] / max(float(temperature), 1e-6)
                        if cat_sampling_strategy == "argmax":
                            cat_tokens[col] = torch.argmax(logits, dim=1)
                        else:
                            probs = torch.softmax(logits, dim=1)
                            cat_tokens[col] = torch.multinomial(probs, num_samples=1).squeeze(1)
                    if numerical is not None and "num_pred" in out:
                        sigma = gaussian_sigma(t).view(-1, 1)
                        numerical = numerical - out["num_pred"] * sigma / max(len(steps), 1)

                generated = group.copy()
                for col in self.cat_cols:
                    indices = cat_tokens[col].detach().cpu().numpy().astype(int)
                    generated[col] = [self.category_values[col][idx] for idx in indices]
                if self.num_cols and numerical is not None:
                    decoded_num = self._decode_numerical(
                        numerical.detach().cpu().numpy()
                    )
                    for i, col in enumerate(self.num_cols):
                        generated[col] = decoded_num[:, i]
                generated_groups.append(generated)
                builder.update_history(generated)

        output = pd.concat(generated_groups).sort_index()
        columns = [self.customer_id_col, self.product_id_col, self.timestamp_col] + self.cat_cols + self.num_cols
        return output[columns].reset_index(drop=True)

    def _preprocess_training_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        reviews = reviews.copy()
        required = [self.customer_id_col, self.product_id_col, self.timestamp_col] + self.cat_cols
        missing = [col for col in required if col not in reviews.columns]
        if missing:
            raise ValueError(f"Training reviews are missing required columns: {missing}")
        reviews[self.timestamp_col] = pd.to_datetime(reviews[self.timestamp_col], errors="coerce")
        reviews = reviews.dropna(subset=required).sort_values(
            self.timestamp_col, kind="mergesort"
        ).reset_index(drop=True)
        self.config = {
            "generator": GENERATOR_NAME,
            "customer_id_col": self.customer_id_col,
            "product_id_col": self.product_id_col,
            "timestamp_col": self.timestamp_col,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "requested_num_cols": list(self.num_cols),
            "seed": self.seed,
        }
        return reviews

    def _fit_categories(self, reviews: pd.DataFrame) -> None:
        self.category_values = {}
        self.category_lookup = {}
        for col in self.cat_cols:
            values = sorted(reviews[col].dropna().unique().tolist(), key=category_sort_key)
            self.category_values[col] = values
            self.category_lookup[col] = {
                canonical_category_key(value): idx for idx, value in enumerate(values)
            }

    def _fit_feature_metadata(self, features: pd.DataFrame) -> None:
        continuous = features[CAUSAL_CONTINUOUS_FEATURES].to_numpy(dtype=np.float32)
        self.feature_mean = continuous.mean(axis=0)
        self.feature_std = continuous.std(axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        self.discrete_feature_maps = {}
        for col in CAUSAL_DISCRETE_FEATURES:
            values = sorted(features[col].dropna().unique().tolist())
            self.discrete_feature_maps[col] = {value: i + 1 for i, value in enumerate(values)}

    def _encode_dataset(self, reviews: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        cat_targets = {
            col: np.asarray(
                [self.category_lookup[col][canonical_category_key(value)] for value in reviews[col]],
                dtype=np.int64,
            )
            for col in self.cat_cols
        }
        numerical = self._fit_transform_numerical(reviews)
        encoded_features = self._encode_features_for_sampling(features)
        return {
            "cat_targets": cat_targets,
            "numerical": numerical,
            **encoded_features,
        }

    def _fit_transform_numerical(self, reviews: pd.DataFrame) -> np.ndarray:
        if not self.num_cols:
            return np.zeros((len(reviews), 0), dtype=np.float32)
        arrays = []
        self.numerical_metadata = {}
        for col in self.num_cols:
            values = pd.to_numeric(reviews[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            use_log1p = bool(np.nanmin(values) >= 0 and any(token in col.lower() for token in ("vote", "count", "helpful")))
            transformed = np.log1p(values) if use_log1p else values
            mean = float(np.mean(transformed))
            std = float(np.std(transformed))
            if std < 1e-6:
                std = 1.0
            self.numerical_metadata[col] = {
                "mean": mean,
                "std": std,
                "log1p": use_log1p,
            }
            arrays.append(((transformed - mean) / std).astype(np.float32))
        return np.stack(arrays, axis=1).astype(np.float32)

    def _encode_features_for_sampling(self, features: pd.DataFrame) -> Dict[str, Any]:
        continuous = features[CAUSAL_CONTINUOUS_FEATURES].to_numpy(dtype=np.float32)
        continuous = (continuous - self.feature_mean) / self.feature_std
        discrete = {
            col: np.asarray(
                [self.discrete_feature_maps[col].get(value, 0) for value in features[col]],
                dtype=np.int64,
            )
            for col in CAUSAL_DISCRETE_FEATURES
        }
        return {"continuous": continuous.astype(np.float32), "discrete": discrete}

    def _split_indices(
        self, reviews: pd.DataFrame, validation_fraction: float, random_split: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.arange(len(reviews), dtype=int)
        if random_split:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        val_size = int(round(len(indices) * float(validation_fraction)))
        val_size = max(1, val_size) if len(indices) > 4 else 0
        return indices[:-val_size] if val_size else indices, indices[-val_size:] if val_size else np.asarray([], dtype=int)

    def _run_epoch(
        self,
        encoded: Dict[str, Any],
        indices: np.ndarray,
        optimizer: Optional[torch.optim.Optimizer],
        batch_size: int,
        device: str,
        train: bool,
        lambda_cat: float,
        lambda_num: float,
        mask_schedule: str,
    ) -> Dict[str, float]:
        assert self.model is not None
        self.model.train(train)
        indices = np.asarray(indices, dtype=int)
        if train:
            np.random.default_rng().shuffle(indices)
        totals = Counter()
        count = 0
        for start in range(0, len(indices), int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            if len(batch_idx) == 0:
                continue
            tensors = self._batch_tensors(encoded, batch_idx, device)
            t = torch.rand(len(batch_idx), device=device)
            p_mask = mask_probability(t, mask_schedule)
            cat_tokens = {}
            cat_masks = {}
            cat_loss = torch.tensor(0.0, device=device)
            for col in self.cat_cols:
                target = tensors["cat_targets"][col]
                mask = torch.rand(len(batch_idx), device=device) < p_mask
                if not bool(mask.any()):
                    mask[0] = True
                token = target.clone()
                token[mask] = len(self.category_values[col])
                cat_tokens[col] = token
                cat_masks[col] = mask
            numerical_noisy = None
            noise = None
            if self.num_cols:
                sigma = gaussian_sigma(t).view(-1, 1)
                noise = torch.randn_like(tensors["numerical"])
                numerical_noisy = tensors["numerical"] + sigma * noise
            out = self.model(
                cat_tokens,
                tensors["continuous"],
                tensors["discrete"],
                t,
                numerical_noisy=numerical_noisy,
            )
            for col in self.cat_cols:
                cat_loss = cat_loss + F.cross_entropy(
                    out[f"{col}_logits"][cat_masks[col]],
                    tensors["cat_targets"][col][cat_masks[col]],
                )
            cat_loss = cat_loss / max(len(self.cat_cols), 1)
            num_loss = torch.tensor(0.0, device=device)
            if self.num_cols and noise is not None and "num_pred" in out:
                num_loss = F.mse_loss(out["num_pred"], noise)
            loss = lambda_cat * cat_loss + lambda_num * num_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
            totals["total"] += float(loss.detach().cpu()) * len(batch_idx)
            totals["cat"] += float(cat_loss.detach().cpu()) * len(batch_idx)
            totals["num"] += float(num_loss.detach().cpu()) * len(batch_idx)
            count += len(batch_idx)
        return {key: float(value / max(count, 1)) for key, value in totals.items()}

    def _batch_tensors(
        self, encoded: Dict[str, Any], indices: np.ndarray, device: str
    ) -> Dict[str, Any]:
        return {
            "continuous": torch.tensor(
                encoded["continuous"][indices], dtype=torch.float32, device=device
            ),
            "discrete": {
                col: torch.tensor(values[indices], dtype=torch.long, device=device)
                for col, values in encoded["discrete"].items()
            },
            "cat_targets": {
                col: torch.tensor(values[indices], dtype=torch.long, device=device)
                for col, values in encoded["cat_targets"].items()
            },
            "numerical": torch.tensor(
                encoded["numerical"][indices], dtype=torch.float32, device=device
            ),
        }

    def _decode_numerical(self, values: np.ndarray) -> np.ndarray:
        decoded = []
        for i, col in enumerate(self.num_cols):
            metadata = self.numerical_metadata[col]
            transformed = values[:, i] * metadata["std"] + metadata["mean"]
            if metadata.get("log1p"):
                transformed = np.expm1(transformed)
            decoded.append(transformed)
        return np.stack(decoded, axis=1) if decoded else np.zeros((len(values), 0))

    def _save_checkpoint(self, path: Path, history: List[Dict[str, float]]) -> None:
        checkpoint = {
            "config": self.config,
            "category_values": self.category_values,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "discrete_feature_maps": self.discrete_feature_maps,
            "numerical_metadata": self.numerical_metadata,
            "model_config": self.model_config,
            "model_state_dict": self.model.state_dict(),
            "feature_builder_metadata": self.feature_builder.to_metadata()
            if self.feature_builder is not None
            else {},
            "history": history,
        }
        torch.save(checkpoint, path)

    def _write_metadata(self, output_dir: Path) -> None:
        save_json(output_dir / "config.json", to_jsonable(self.config))
        save_json(
            output_dir / "category_mappings.json",
            to_jsonable({"category_values": self.category_values}),
        )
        save_json(
            output_dir / "numerical_transform_metadata.json",
            to_jsonable(self.numerical_metadata),
        )
        save_json(
            output_dir / "feature_normalization.json",
            to_jsonable(
                {
                    "continuous_feature_names": CAUSAL_CONTINUOUS_FEATURES,
                    "discrete_feature_names": CAUSAL_DISCRETE_FEATURES,
                    "mean": self.feature_mean.tolist(),
                    "std": self.feature_std.tolist(),
                    "discrete_feature_maps": self.discrete_feature_maps,
                }
            ),
        )

    def _set_seeds(self, seed: int) -> None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))


def category_sort_key(value: Any) -> Tuple[str, str]:
    try:
        return ("0", f"{float(value):020.8f}")
    except Exception:
        return ("1", str(value))


def canonical_category_key(value: Any) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value
