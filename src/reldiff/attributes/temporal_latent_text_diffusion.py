"""Temporal latent-text attribute diffusion for review event spines."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .temporal_attr_model import TemporalRelDiffAttrModel
from .temporal_neighbor_sampler import TemporalReviewNeighborSampler
from .text_latents import (
    NearestNeighborTextDecoder,
    TextLatentEncoder,
    TextLatentNormalizer,
)


GENERATOR_NAME = "temporal_latent_text_attr_diffusion"
DEFAULT_MAX_TEXT_LENGTHS = {"summary": 64, "review_text": 256}


@dataclass
class TemporalAttrTrainingResult:
    output_dir: Path
    best_checkpoint: Path
    latest_checkpoint: Path
    history: List[Dict[str, float]]


class TemporalLatentTextAttributeDiffusion:
    """Train and sample attributes conditioned on a temporal review spine."""

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        cat_cols: Optional[List[str]] = None,
        text_cols: Optional[List[str]] = None,
        temporal_window_days: float = 365.0,
        max_customer_history: int = 32,
        max_product_history: int = 32,
        temporal_mode: str = "causal_window",
        seed: int = 42,
    ):
        self.customer_id_col = customer_id_col
        self.product_id_col = product_id_col
        self.timestamp_col = timestamp_col
        self.cat_cols = list(cat_cols or ["rating", "verified"])
        self.text_cols = list(text_cols or ["summary", "review_text"])
        self.temporal_window_days = float(temporal_window_days)
        self.max_customer_history = int(max_customer_history)
        self.max_product_history = int(max_product_history)
        self.temporal_mode = temporal_mode
        self.seed = int(seed)

        self.model: Optional[TemporalRelDiffAttrModel] = None
        self.reviews: Optional[pd.DataFrame] = None
        self.category_values: Dict[str, List[Any]] = {}
        self.category_lookup: Dict[str, Dict[Any, int]] = {}
        self.cat_targets: Optional[np.ndarray] = None
        self.text_latents: Dict[str, np.ndarray] = {}
        self.text_latents_norm: Dict[str, np.ndarray] = {}
        self.text_normalizer: Optional[TextLatentNormalizer] = None
        self.time_features: Optional[np.ndarray] = None
        self.context_features: Optional[np.ndarray] = None
        self.text_encoder_metadata: Dict[str, Any] = {}
        self.train_min_time: Optional[pd.Timestamp] = None
        self.train_max_time: Optional[pd.Timestamp] = None

    @classmethod
    def train_from_csv(
        cls,
        reviews_path: str | Path,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> TemporalAttrTrainingResult:
        reviews = pd.read_csv(reviews_path)
        generator = cls(**_constructor_kwargs(kwargs))
        return generator.train(reviews, output_dir=output_dir, **_training_kwargs(kwargs))

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "cpu"
    ) -> "TemporalLatentTextAttributeDiffusion":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        generator = cls(
            customer_id_col=config["customer_id_col"],
            product_id_col=config["product_id_col"],
            timestamp_col=config["timestamp_col"],
            cat_cols=config["cat_cols"],
            text_cols=config["text_cols"],
            temporal_window_days=config["temporal_window_days"],
            max_customer_history=config["max_customer_history"],
            max_product_history=config["max_product_history"],
            temporal_mode=config["temporal_mode"],
            seed=config["seed"],
        )
        generator.category_values = checkpoint["category_values"]
        generator.category_lookup = {
            column: {value: index for index, value in enumerate(values)}
            for column, values in generator.category_values.items()
        }
        generator.text_normalizer = TextLatentNormalizer.from_checkpoint(
            checkpoint["text_normalizer"]
        )
        generator.text_latents_norm = {
            column: np.asarray(values, dtype=np.float32)
            for column, values in checkpoint["train_text_latents_norm"].items()
        }
        generator.text_encoder_metadata = checkpoint.get("text_encoder_metadata", {})
        generator.train_min_time = pd.Timestamp(checkpoint["train_min_time"])
        generator.train_max_time = pd.Timestamp(checkpoint["train_max_time"])
        generator.model = TemporalRelDiffAttrModel(**checkpoint["model_config"]).to(device)
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
        **kwargs: Any,
    ) -> pd.DataFrame:
        generator = cls.load_checkpoint(
            checkpoint_path, device=kwargs.get("device", "cpu")
        )
        spine = pd.read_csv(synthetic_spine_path)
        output = generator.sample(
            spine,
            num_steps=kwargs.get("num_steps", 50),
            batch_size=kwargs.get("batch_size", 512),
            seed=kwargs.get("seed", generator.seed),
            device=kwargs.get("device", "cpu"),
            decoder_mode=kwargs.get("decoder_mode", "nearest_neighbor"),
            categorical_temperature=kwargs.get("categorical_temperature", 1.0),
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        return output

    def train(
        self,
        reviews: pd.DataFrame,
        output_dir: str | Path,
        epochs: int = 5,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        hidden_dim: int = 128,
        text_encoder_backend: str = "auto",
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_latent_dim: int = 384,
        lambda_cat: float = 1.0,
        lambda_text: float = 1.0,
        lambda_summary: float = 0.5,
        lambda_review_text: float = 1.0,
        validation_fraction: float = 0.1,
        force_recompute_text_latents: bool = False,
        device: str = "cpu",
    ) -> TemporalAttrTrainingResult:
        self._set_seeds(self.seed)
        output_dir = Path(output_dir)
        cache_dir = output_dir / "cache"
        checkpoint_dir = output_dir / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.reviews = self._preprocess_reviews(reviews)
        self.train_min_time = self.reviews[self.timestamp_col].min()
        self.train_max_time = self.reviews[self.timestamp_col].max()
        self._fit_categories()
        self._fit_text_latents(
            cache_dir=cache_dir,
            backend=text_encoder_backend,
            model_name=text_model_name,
            latent_dim=text_latent_dim,
            force_recompute=force_recompute_text_latents,
            device=device,
        )
        self._fit_conditioning_features()

        train_indices, val_indices = self._temporal_split(validation_fraction)
        model_config = {
            "cat_cols": self.cat_cols,
            "num_classes": [len(self.category_values[col]) for col in self.cat_cols],
            "text_dims": {
                column: int(values.shape[1])
                for column, values in self.text_latents_norm.items()
            },
            "time_feature_dim": int(self.time_features.shape[1]),
            "context_feature_dim": int(self.context_features.shape[1]),
            "hidden_dim": int(hidden_dim),
        }
        self.model = TemporalRelDiffAttrModel(**model_config).to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        history: List[Dict[str, float]] = []
        best_val = float("inf")
        best_path = checkpoint_dir / "best.pt"
        latest_path = checkpoint_dir / "latest.pt"

        for epoch in range(1, int(epochs) + 1):
            train_loss = self._run_epoch(
                train_indices,
                optimizer=optimizer,
                batch_size=batch_size,
                device=device,
                train=True,
                lambda_cat=lambda_cat,
                lambda_text=lambda_text,
                lambda_summary=lambda_summary,
                lambda_review_text=lambda_review_text,
            )
            val_loss = self._run_epoch(
                val_indices if len(val_indices) else train_indices,
                optimizer=None,
                batch_size=batch_size,
                device=device,
                train=False,
                lambda_cat=lambda_cat,
                lambda_text=lambda_text,
                lambda_summary=lambda_summary,
                lambda_review_text=lambda_review_text,
            )
            row = {"epoch": float(epoch), "train_total": train_loss, "val_total": val_loss}
            history.append(row)
            self._save_checkpoint(
                latest_path,
                model_config=model_config,
                history=history,
                train_indices=train_indices,
                val_indices=val_indices,
            )
            if val_loss <= best_val:
                best_val = val_loss
                self._save_checkpoint(
                    best_path,
                    model_config=model_config,
                    history=history,
                    train_indices=train_indices,
                    val_indices=val_indices,
                )
            print(
                f"Epoch {epoch:03d}: train_total={train_loss:.4f} "
                f"val_total={val_loss:.4f}"
            )

        with (output_dir / "training_history.json").open("w") as handle:
            json.dump(history, handle, indent=2)
            handle.write("\n")
        return TemporalAttrTrainingResult(
            output_dir=output_dir,
            best_checkpoint=best_path,
            latest_checkpoint=latest_path,
            history=history,
        )

    def sample(
        self,
        synthetic_spine: pd.DataFrame,
        num_steps: int = 50,
        batch_size: int = 512,
        seed: Optional[int] = None,
        device: str = "cpu",
        decoder_mode: str = "nearest_neighbor",
        categorical_temperature: float = 1.0,
    ) -> pd.DataFrame:
        if decoder_mode != "nearest_neighbor":
            raise NotImplementedError(
                "Only nearest_neighbor text decoding is implemented in v1."
            )
        if self.model is None:
            raise RuntimeError("No trained model is loaded.")
        checkpoint = getattr(self, "_checkpoint", None)
        if checkpoint is None:
            raise RuntimeError("Sampling requires a model loaded from a checkpoint.")

        self._set_seeds(self.seed if seed is None else int(seed))
        spine = self._preprocess_spine(synthetic_spine)
        time_features = self._build_time_features(
            spine[self.timestamp_col], self.train_min_time, self.train_max_time
        )
        sampler = TemporalReviewNeighborSampler(
            spine,
            customer_id_col=self.customer_id_col,
            product_id_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            temporal_window_days=self.temporal_window_days,
            max_customer_history=self.max_customer_history,
            max_product_history=self.max_product_history,
            temporal_mode=self.temporal_mode,
        )
        context_features = sampler.context_features(np.arange(len(spine)))

        num_rows = len(spine)
        cat_current = np.column_stack(
            [
                np.full(num_rows, len(self.category_values[column]), dtype=np.int64)
                for column in self.cat_cols
            ]
        )
        text_current = {
            column: np.random.default_rng(self.seed if seed is None else seed)
            .normal(0.0, 1.0, size=(num_rows, latents.shape[1]))
            .astype(np.float32)
            for column, latents in self.text_latents_norm.items()
        }

        self.model.to(device)
        self.model.eval()
        num_steps = max(int(num_steps), 1)
        for step in tqdm(
            range(num_steps, 0, -1),
            desc="Sampling temporal attributes",
            unit="step",
        ):
            diffusion_value = step / num_steps
            for batch_indices in _batched_indices(num_rows, batch_size):
                outputs = self._model_forward_numpy(
                    batch_indices,
                    cat_current,
                    text_current,
                    time_features,
                    context_features,
                    diffusion_value,
                    device=device,
                )
                for column_index, column in enumerate(self.cat_cols):
                    logits = outputs["cat_logits"][column] / max(
                        categorical_temperature, 1e-6
                    )
                    probs = torch.softmax(logits, dim=1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
                    cat_current[batch_indices, column_index] = sampled.cpu().numpy()
                for column in self.text_cols:
                    eps = outputs["text_eps"][column].cpu().numpy()
                    sigma = 0.05 + diffusion_value
                    text_current[column][batch_indices] = (
                        text_current[column][batch_indices]
                        - eps.astype(np.float32) * sigma / num_steps
                    )

        generated_categories = self._decode_categories(cat_current)
        decoder = NearestNeighborTextDecoder(
            train_latents=checkpoint["train_text_latents_norm"],
            train_texts=checkpoint["train_texts"],
            train_categories=checkpoint["train_categories"],
            metric="cosine",
        )
        generated_text = {
            column: decoder.decode(
                column,
                text_current[column],
                generated_categories=generated_categories,
            )
            for column in self.text_cols
        }

        output = spine[[self.customer_id_col, self.product_id_col, self.timestamp_col]].copy()
        for column in self.cat_cols:
            output[column] = generated_categories[column]
        for column in self.text_cols:
            output[column] = generated_text[column]
            output[column] = output[column].fillna("").astype(str)
        return output[
            [self.customer_id_col, self.product_id_col, self.timestamp_col]
            + self.cat_cols
            + self.text_cols
        ]

    def _run_epoch(
        self,
        indices: np.ndarray,
        optimizer: Optional[torch.optim.Optimizer],
        batch_size: int,
        device: str,
        train: bool,
        lambda_cat: float,
        lambda_text: float,
        lambda_summary: float,
        lambda_review_text: float,
    ) -> float:
        assert self.model is not None
        dataset = TensorDataset(torch.from_numpy(indices.astype(np.int64)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
        self.model.train(train)
        losses = []
        for (batch_indices_tensor,) in loader:
            batch_indices = batch_indices_tensor.numpy()
            loss = self._batch_loss(
                batch_indices,
                device=device,
                lambda_cat=lambda_cat,
                lambda_text=lambda_text,
                lambda_summary=lambda_summary,
                lambda_review_text=lambda_review_text,
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else 0.0

    def _batch_loss(
        self,
        batch_indices: np.ndarray,
        device: str,
        lambda_cat: float,
        lambda_text: float,
        lambda_summary: float,
        lambda_review_text: float,
    ) -> torch.Tensor:
        assert self.model is not None
        cat_target = torch.from_numpy(self.cat_targets[batch_indices]).long().to(device)
        time_features = torch.from_numpy(self.time_features[batch_indices]).float().to(device)
        context_features = torch.from_numpy(self.context_features[batch_indices]).float().to(device)
        diffusion_step = torch.rand(len(batch_indices), device=device)

        cat_noisy = cat_target.clone()
        mask_prob = (0.15 + 0.7 * diffusion_step).view(-1, 1)
        mask = torch.rand_like(cat_noisy.float()) < mask_prob
        for column_index, column in enumerate(self.cat_cols):
            cat_noisy[mask[:, column_index], column_index] = len(
                self.category_values[column]
            )

        text_noisy = {}
        text_noise = {}
        for column in self.text_cols:
            target = torch.from_numpy(self.text_latents_norm[column][batch_indices]).float().to(device)
            eps = torch.randn_like(target)
            sigma = (0.05 + diffusion_step).view(-1, 1)
            text_noisy[column] = target + sigma * eps
            text_noise[column] = eps

        outputs = self.model(
            cat_noisy=cat_noisy,
            text_noisy=text_noisy,
            time_features=time_features,
            context_features=context_features,
            diffusion_step=diffusion_step,
        )

        cat_losses = []
        for column_index, column in enumerate(self.cat_cols):
            logits = outputs["cat_logits"][column]
            losses = F.cross_entropy(
                logits, cat_target[:, column_index], reduction="none"
            )
            column_mask = mask[:, column_index]
            cat_losses.append(losses[column_mask].mean() if column_mask.any() else losses.mean())
        cat_loss = torch.stack(cat_losses).mean() if cat_losses else torch.tensor(0.0, device=device)

        text_losses = []
        for column in self.text_cols:
            weight = lambda_review_text
            if column.lower() == "summary":
                weight = lambda_summary
            loss = F.mse_loss(outputs["text_eps"][column], text_noise[column])
            text_losses.append(weight * loss)
        text_loss = torch.stack(text_losses).mean() if text_losses else torch.tensor(0.0, device=device)
        return lambda_cat * cat_loss + lambda_text * text_loss

    def _model_forward_numpy(
        self,
        batch_indices: np.ndarray,
        cat_current: np.ndarray,
        text_current: Dict[str, np.ndarray],
        time_features: np.ndarray,
        context_features: np.ndarray,
        diffusion_value: float,
        device: str,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            return self.model(
                cat_noisy=torch.from_numpy(cat_current[batch_indices]).long().to(device),
                text_noisy={
                    column: torch.from_numpy(values[batch_indices]).float().to(device)
                    for column, values in text_current.items()
                },
                time_features=torch.from_numpy(time_features[batch_indices]).float().to(device),
                context_features=torch.from_numpy(context_features[batch_indices]).float().to(device),
                diffusion_step=torch.full(
                    (len(batch_indices),), float(diffusion_value), device=device
                ),
            )

    def _preprocess_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
        required = [self.customer_id_col, self.product_id_col, self.timestamp_col]
        required.extend(self.cat_cols)
        required.extend(self.text_cols)
        missing = [column for column in required if column not in reviews.columns]
        if missing:
            raise ValueError(f"Reviews table is missing required columns: {missing}")

        reviews = reviews.copy()
        reviews[self.timestamp_col] = pd.to_datetime(
            reviews[self.timestamp_col], errors="coerce"
        )
        reviews = reviews.dropna(
            subset=[self.customer_id_col, self.product_id_col, self.timestamp_col]
        )
        reviews = reviews.sort_values(self.timestamp_col, kind="mergesort").reset_index(
            drop=True
        )
        if reviews.empty:
            raise ValueError("No valid reviews remain after preprocessing.")
        for column in self.text_cols:
            reviews[column] = reviews[column].fillna("").astype(str)
        for column in self.cat_cols:
            reviews[column] = reviews[column].where(~reviews[column].isna(), "__MISSING__")
        return reviews

    def _preprocess_spine(self, spine: pd.DataFrame) -> pd.DataFrame:
        required = [self.customer_id_col, self.product_id_col, self.timestamp_col]
        missing = [column for column in required if column not in spine.columns]
        if missing:
            raise ValueError(f"Synthetic spine is missing required columns: {missing}")
        spine = spine[required].copy()
        spine[self.timestamp_col] = pd.to_datetime(spine[self.timestamp_col], errors="coerce")
        spine = spine.dropna(subset=required).sort_values(
            self.timestamp_col, kind="mergesort"
        )
        return spine.reset_index(drop=True)

    def _fit_categories(self) -> None:
        self.category_values = {}
        self.category_lookup = {}
        encoded = []
        for column in self.cat_cols:
            values = sorted(pd.unique(self.reviews[column]), key=lambda value: str(value))
            values = [_to_python_scalar(value) for value in values]
            self.category_values[column] = values
            self.category_lookup[column] = {value: index for index, value in enumerate(values)}
            encoded.append(
                np.asarray(
                    [self.category_lookup[column][_to_python_scalar(value)] for value in self.reviews[column]],
                    dtype=np.int64,
                )
            )
        self.cat_targets = np.stack(encoded, axis=1) if encoded else np.zeros((len(self.reviews), 0), dtype=np.int64)

    def _fit_text_latents(
        self,
        cache_dir: Path,
        backend: str,
        model_name: str,
        latent_dim: int,
        force_recompute: bool,
        device: str,
    ) -> None:
        max_lengths = {
            column: DEFAULT_MAX_TEXT_LENGTHS.get(column, 256)
            for column in self.text_cols
        }
        encoder = TextLatentEncoder(
            model_name=model_name,
            backend=backend,
            latent_dim=latent_dim,
            device=device,
            local_files_only=True,
        )
        self.text_encoder_metadata = encoder.metadata()
        self.text_latents = encoder.fit_transform_columns(
            self.reviews,
            self.text_cols,
            cache_dir=cache_dir,
            max_lengths=max_lengths,
            force_recompute=force_recompute,
        )
        self.text_normalizer = TextLatentNormalizer()
        self.text_latents_norm = self.text_normalizer.fit_transform(self.text_latents)

    def _fit_conditioning_features(self) -> None:
        self.time_features = self._build_time_features(
            self.reviews[self.timestamp_col], self.train_min_time, self.train_max_time
        )
        sampler = TemporalReviewNeighborSampler(
            self.reviews,
            customer_id_col=self.customer_id_col,
            product_id_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            temporal_window_days=self.temporal_window_days,
            max_customer_history=self.max_customer_history,
            max_product_history=self.max_product_history,
            temporal_mode=self.temporal_mode,
        )
        rating_values = None
        verified_values = None
        if "rating" in self.reviews.columns:
            rating_values = _normalized_numeric(self.reviews["rating"])
        if "verified" in self.reviews.columns:
            verified_values = _normalized_bool_or_numeric(self.reviews["verified"])
        sampler.assert_no_future(np.arange(len(self.reviews)))
        self.context_features = sampler.context_features(
            np.arange(len(self.reviews)),
            rating_values=rating_values,
            verified_values=verified_values,
        )

    def _temporal_split(self, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
        n_rows = len(self.reviews)
        indices = np.arange(n_rows, dtype=np.int64)
        if n_rows < 5:
            return indices, indices
        val_size = max(1, int(round(n_rows * validation_fraction)))
        train_end = max(1, n_rows - val_size)
        return indices[:train_end], indices[train_end:]

    @staticmethod
    def _build_time_features(
        timestamps: pd.Series,
        min_time: Optional[pd.Timestamp],
        max_time: Optional[pd.Timestamp],
    ) -> np.ndarray:
        timestamps = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
        min_time = pd.Timestamp(min_time if min_time is not None else timestamps.min())
        max_time = pd.Timestamp(max_time if max_time is not None else timestamps.max())
        span = max((max_time - min_time).total_seconds(), 1.0)
        normalized = ((timestamps - min_time).dt.total_seconds() / span).clip(0.0, 1.0)
        day = timestamps.dt.dayofweek.astype(float)
        month = timestamps.dt.month.astype(float)
        years = timestamps.dt.year.astype(float)
        year_span = max(float(years.max() - years.min()), 1.0)
        year_norm = (years - years.min()) / year_span
        features = np.column_stack(
            [
                normalized.to_numpy(dtype=np.float32),
                np.sin(2 * math.pi * day / 7.0),
                np.cos(2 * math.pi * day / 7.0),
                np.sin(2 * math.pi * (month - 1) / 12.0),
                np.cos(2 * math.pi * (month - 1) / 12.0),
                year_norm.to_numpy(dtype=np.float32),
            ]
        )
        return features.astype(np.float32)

    def _decode_categories(self, cat_current: np.ndarray) -> Dict[str, List[Any]]:
        decoded = {}
        for column_index, column in enumerate(self.cat_cols):
            values = self.category_values[column]
            indices = np.clip(cat_current[:, column_index], 0, len(values) - 1)
            decoded[column] = [values[int(index)] for index in indices]
        return decoded

    def _save_checkpoint(
        self,
        path: Path,
        model_config: Dict[str, Any],
        history: List[Dict[str, float]],
        train_indices: np.ndarray,
        val_indices: np.ndarray,
    ) -> None:
        assert self.model is not None
        assert self.text_normalizer is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "generator": GENERATOR_NAME,
            "model_state_dict": self.model.state_dict(),
            "model_config": model_config,
            "config": self._config_dict(),
            "category_values": self.category_values,
            "text_normalizer": self.text_normalizer.to_checkpoint(),
            "train_text_latents_norm": self.text_latents_norm,
            "train_texts": {
                column: self.reviews[column].fillna("").astype(str).tolist()
                for column in self.text_cols
            },
            "train_categories": {
                column: self.reviews[column].tolist() for column in self.cat_cols
            },
            "text_encoder_metadata": self.text_encoder_metadata,
            "train_min_time": str(self.train_min_time),
            "train_max_time": str(self.train_max_time),
            "history": history,
            "train_indices": train_indices,
            "val_indices": val_indices,
        }
        torch.save(checkpoint, path)

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "customer_id_col": self.customer_id_col,
            "product_id_col": self.product_id_col,
            "timestamp_col": self.timestamp_col,
            "cat_cols": self.cat_cols,
            "text_cols": self.text_cols,
            "temporal_window_days": self.temporal_window_days,
            "max_customer_history": self.max_customer_history,
            "max_product_history": self.max_product_history,
            "temporal_mode": self.temporal_mode,
            "seed": self.seed,
        }

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def _batched_indices(num_rows: int, batch_size: int) -> List[np.ndarray]:
    return [
        np.arange(start, min(start + batch_size, num_rows), dtype=np.int64)
        for start in range(0, num_rows, max(int(batch_size), 1))
    ]


def _constructor_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    keys = {
        "customer_id_col",
        "product_id_col",
        "timestamp_col",
        "cat_cols",
        "text_cols",
        "temporal_window_days",
        "max_customer_history",
        "max_product_history",
        "temporal_mode",
        "seed",
    }
    return {key: value for key, value in kwargs.items() if key in keys}


def _training_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key not in _constructor_kwargs(kwargs)}


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalized_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(values).all():
        return np.zeros(len(series), dtype=np.float32)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    span = max(max_value - min_value, 1e-6)
    return np.nan_to_num((values - min_value) / span).astype(np.float32)


def _normalized_bool_or_numeric(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().all():
        values = numeric.to_numpy(dtype=np.float32)
        max_value = max(float(np.nanmax(values)), 1.0)
        return np.nan_to_num(values / max_value).astype(np.float32)
    lowered = series.fillna("").astype(str).str.lower()
    return lowered.isin({"true", "1", "yes", "y", "verified"}).astype(np.float32).to_numpy()
