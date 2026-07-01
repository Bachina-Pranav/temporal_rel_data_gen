"""Temporal non-text attribute diffusion with generative entity latent effects."""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .entity_effect_priors import load_customer_product_priors
from .entity_latent_effects import (
    compute_entity_structural_features,
    estimate_entity_latent_effects,
    save_entity_effect_estimate,
)
from .nontext_diffusion_model_v2 import TemporalFeatureDiffusionModelV2
from .nontext_diffusion_schedules import gaussian_sigma, mask_probability
from .temporal_attribute_sampler import chronological_groups
from .temporal_causal_features import (
    CAUSAL_CONTINUOUS_FEATURES,
    TemporalCausalFeatureBuilder,
    load_block_maps,
    save_json,
)
from .temporal_nontext_diffusion import (
    TemporalNonTextAttributeDiffusion,
    TemporalNonTextTrainingResult,
    canonical_category_key,
    to_jsonable,
)


GENERATOR_NAME_V2 = "temporal_nontext_attr_diffusion_v2"
GENERATOR_ALIAS_V2 = "temporal_nontext_attr_diffusion_entity_latents"

ENTITY_LATENT_FEATURES = [
    "customer_rating_effect",
    "customer_verified_effect",
    "product_rating_effect",
    "product_verified_effect",
]


class TemporalNonTextAttributeDiffusionV2(TemporalNonTextAttributeDiffusion):
    """V2 attribute generator conditioned on sampled entity latent effects."""

    def __init__(
        self,
        customer_id_col: str = "customer_id",
        product_id_col: str = "product_id",
        timestamp_col: str = "review_time",
        cat_cols: Optional[List[str]] = None,
        num_cols: Optional[List[str]] = None,
        seed: int = 42,
    ):
        super().__init__(
            customer_id_col=customer_id_col,
            product_id_col=product_id_col,
            timestamp_col=timestamp_col,
            cat_cols=cat_cols,
            num_cols=num_cols,
            seed=seed,
        )
        self.effect_noise_std = 0.05
        self.effect_dropout = 0.1

    def continuous_feature_names(self) -> List[str]:
        return list(CAUSAL_CONTINUOUS_FEATURES) + list(ENTITY_LATENT_FEATURES)

    def make_model(self) -> TemporalFeatureDiffusionModelV2:
        return TemporalFeatureDiffusionModelV2(**self.model_config)

    @classmethod
    def train_from_csv(
        cls,
        reviews_path: str | Path,
        output_dir: str | Path,
        structure_debug_dir: str | Path | None = None,
        entity_prior_dir: str | Path | None = None,
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
            entity_prior_dir=entity_prior_dir,
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
            effect_noise_std=kwargs.get("effect_noise_std", 0.05),
            effect_dropout=kwargs.get("effect_dropout", 0.1),
            device=kwargs.get("device", "cpu"),
        )

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "cpu"
    ) -> "TemporalNonTextAttributeDiffusionV2":
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
        generator.effect_noise_std = float(config.get("effect_noise_std", 0.05))
        generator.effect_dropout = float(config.get("effect_dropout", 0.1))
        generator.model = generator.make_model().to(device)
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
        entity_prior_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        generator = cls.load_checkpoint(
            checkpoint_path, device=kwargs.get("device", "cpu")
        )
        spine = pd.read_csv(synthetic_spine_path)
        output_path = Path(output_path)
        sampled_dir = kwargs.get("sampled_effects_output_dir", output_path.parent)
        output = generator.sample(
            spine,
            structure_debug_dir=structure_debug_dir,
            entity_prior_dir=entity_prior_dir,
            sampled_effects_output_dir=sampled_dir,
            seed=kwargs.get("seed", generator.seed),
            num_steps=kwargs.get("num_steps", 50),
            cat_sampling_strategy=kwargs.get("cat_sampling_strategy", "sample"),
            temperature=kwargs.get("temperature", 1.0),
            sampling_time_group=kwargs.get("sampling_time_group", "date"),
            sampling_window_days=kwargs.get("sampling_window_days", 1.0),
            debug_use_posterior_effects=kwargs.get("debug_use_posterior_effects", False),
            device=kwargs.get("device", "cpu"),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        metadata = generator.synthetic_metadata(
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            entity_prior_dir=entity_prior_dir,
            seed=kwargs.get("seed", generator.seed),
        )
        with output_path.with_name(output_path.stem + "_metadata.json").open("w") as handle:
            json.dump(to_jsonable(metadata), handle, indent=2)
            handle.write("\n")
        return output

    def train(
        self,
        reviews: pd.DataFrame,
        output_dir: str | Path,
        structure_debug_dir: str | Path | None = None,
        entity_prior_dir: str | Path | None = None,
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
        effect_noise_std: float = 0.05,
        effect_dropout: float = 0.1,
        device: str = "cpu",
    ) -> TemporalNonTextTrainingResult:
        self._set_seeds(self.seed)
        self.effect_noise_std = float(effect_noise_std)
        self.effect_dropout = float(effect_dropout)
        output_dir = Path(output_dir)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        reviews = self._preprocess_training_reviews(reviews)
        self.config["generator"] = GENERATOR_NAME_V2
        self.config["generator_alias"] = GENERATOR_ALIAS_V2
        self.config["uses_entity_latents"] = True
        self.config["entity_prior_dir"] = str(entity_prior_dir) if entity_prior_dir else None
        self.config["effect_noise_std"] = self.effect_noise_std
        self.config["effect_dropout"] = self.effect_dropout
        self.config["uses_real_entity_effect_lookup"] = False
        self.config["samples_entity_effects_from_prior"] = True

        self._fit_categories(reviews)
        self.num_cols = [col for col in self.num_cols if col in reviews.columns]
        missing_num = [
            col for col in self.config.get("requested_num_cols", []) if col not in reviews.columns
        ]
        for col in missing_num:
            import warnings

            warnings.warn(f"Numerical column {col!r} is absent; skipping.")
        self.config["num_cols"] = list(self.num_cols)

        rating_col = self.cat_cols[0] if self.cat_cols else "rating"
        verified_col = "verified" if "verified" in self.cat_cols else self.cat_cols[-1]
        estimate = estimate_entity_latent_effects(
            reviews,
            structure_debug_dir=structure_debug_dir,
            customer_id_col=self.customer_id_col,
            product_id_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            rating_col=rating_col,
            verified_col=verified_col,
        )
        save_entity_effect_estimate(estimate, output_dir)
        self.config["global_effect_stats"] = estimate.global_stats

        customer_blocks, product_blocks = load_block_maps(
            structure_debug_dir, self.customer_id_col, self.product_id_col
        )
        self.feature_builder = TemporalCausalFeatureBuilder(
            customer_id_col=self.customer_id_col,
            product_id_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            rating_col=rating_col,
            verified_col=verified_col,
            customer_blocks=customer_blocks,
            product_blocks=product_blocks,
        )
        features = self.feature_builder.transform_training(reviews)
        latent_features = training_entity_latent_rows(
            reviews,
            estimate.customer_effects,
            estimate.product_effects,
            self.customer_id_col,
            self.product_id_col,
        )
        features = pd.concat([features, latent_features], axis=1)
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
            "continuous_feature_dim": len(self.continuous_feature_names()),
            "discrete_feature_vocab_sizes": {
                col: len(mapping) + 1 for col, mapping in self.discrete_feature_maps.items()
            },
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
        }
        self.model = self.make_model().to(device)
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
        save_json(
            output_dir / "entity_latent_metadata.json",
            {
                "method": GENERATOR_NAME_V2,
                "uses_entity_latents": True,
                "effect_feature_names": ENTITY_LATENT_FEATURES,
                "uses_real_entity_effect_lookup": False,
                "samples_entity_effects_from_prior": True,
                "debug_use_posterior_effects_default": False,
            },
        )
        return TemporalNonTextTrainingResult(output_dir, best_path, latest_path, history)

    def sample(
        self,
        synthetic_spine: pd.DataFrame,
        structure_debug_dir: str | Path | None = None,
        entity_prior_dir: str | Path | None = None,
        sampled_effects_output_dir: str | Path | None = None,
        seed: Optional[int] = None,
        num_steps: int = 50,
        cat_sampling_strategy: str = "sample",
        temperature: float = 1.0,
        sampling_time_group: str = "date",
        sampling_window_days: float = 1.0,
        debug_use_posterior_effects: bool = False,
        device: str = "cpu",
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if cat_sampling_strategy not in {"sample", "argmax"}:
            raise ValueError("cat_sampling_strategy must be sample or argmax.")
        if entity_prior_dir is None:
            entity_prior_dir = self.config.get("entity_prior_dir")
        if entity_prior_dir is None:
            raise ValueError("V2 sampling requires --entity-prior-dir.")

        seed = self.seed if seed is None else int(seed)
        self._set_seeds(seed)
        rng = np.random.default_rng(seed)
        self.model.to(device)
        self.model.eval()

        spine = synthetic_spine.copy()
        spine[self.timestamp_col] = pd.to_datetime(spine[self.timestamp_col], errors="coerce")
        customer_blocks, product_blocks = load_block_maps(
            structure_debug_dir, self.customer_id_col, self.product_id_col
        )
        sampled_customer_effects, sampled_product_effects = self._sample_or_load_entity_effects(
            spine,
            entity_prior_dir=entity_prior_dir,
            customer_blocks=customer_blocks,
            product_blocks=product_blocks,
            rng=rng,
            debug_use_posterior_effects=debug_use_posterior_effects,
        )
        if sampled_effects_output_dir is not None:
            sample_dir = Path(sampled_effects_output_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)
            sampled_customer_effects.to_csv(
                sample_dir / "sampled_customer_effects.csv", index=False
            )
            sampled_product_effects.to_csv(
                sample_dir / "sampled_product_effects.csv", index=False
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
                spine,
                self.timestamp_col,
                mode=sampling_time_group,
                window_days=sampling_window_days,
            ):
                features = builder.transform_current_group(group)
                latent_features = sampling_entity_latent_rows(
                    group,
                    sampled_customer_effects,
                    sampled_product_effects,
                    self.customer_id_col,
                    self.product_id_col,
                )
                features = pd.concat([features, latent_features], axis=1)
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
        self._last_sampling_metadata = {
            "uses_real_entity_effect_lookup": bool(debug_use_posterior_effects),
            "diagnostic_upper_bound_only": bool(debug_use_posterior_effects),
            "entity_latent_source": "posterior_effects_debug"
            if debug_use_posterior_effects
            else "sampled_prior",
            "samples_entity_effects_from_prior": not bool(debug_use_posterior_effects),
        }
        return output[columns].reset_index(drop=True)

    def _sample_or_load_entity_effects(
        self,
        spine: pd.DataFrame,
        entity_prior_dir: str | Path,
        customer_blocks: Dict[Any, int],
        product_blocks: Dict[Any, int],
        rng: np.random.Generator,
        debug_use_posterior_effects: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if debug_use_posterior_effects:
            return load_posterior_effects_for_debug(
                entity_prior_dir,
                self.customer_id_col,
                self.product_id_col,
            )
        customer_prior, product_prior = load_customer_product_priors(entity_prior_dir)
        customer_struct = compute_entity_structural_features(
            spine,
            entity_col=self.customer_id_col,
            timestamp_col=self.timestamp_col,
            block_map=customer_blocks,
            block_col="customer_block",
            id_output_col=self.customer_id_col,
        )
        product_struct = compute_entity_structural_features(
            spine,
            entity_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            block_map=product_blocks,
            block_col="product_block",
            id_output_col=self.product_id_col,
        )
        customer_sample = customer_prior.sample(customer_struct, rng=rng).effects
        product_sample = product_prior.sample(product_struct, rng=rng).effects
        return customer_sample, product_sample

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
            if train:
                tensors["continuous"] = self._corrupt_effect_features(tensors["continuous"])
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

    def _corrupt_effect_features(self, continuous: torch.Tensor) -> torch.Tensor:
        if not ENTITY_LATENT_FEATURES:
            return continuous
        effect_start = len(CAUSAL_CONTINUOUS_FEATURES)
        output = continuous.clone()
        effects = output[:, effect_start : effect_start + len(ENTITY_LATENT_FEATURES)]
        if self.effect_noise_std > 0:
            effects = effects + torch.randn_like(effects) * float(self.effect_noise_std)
        if self.effect_dropout > 0:
            keep = (
                torch.rand((effects.shape[0], 1), device=effects.device)
                >= float(self.effect_dropout)
            ).float()
            effects = effects * keep
        output[:, effect_start : effect_start + len(ENTITY_LATENT_FEATURES)] = effects
        return output

    def synthetic_metadata(
        self,
        output_path: str | Path,
        checkpoint_path: str | Path,
        entity_prior_dir: str | Path | None,
        seed: int,
    ) -> Dict[str, Any]:
        sampling = getattr(self, "_last_sampling_metadata", {})
        return {
            "method": GENERATOR_NAME_V2,
            "uses_entity_latents": True,
            "entity_latent_source": sampling.get("entity_latent_source", "sampled_prior"),
            "uses_real_entity_effect_lookup": sampling.get("uses_real_entity_effect_lookup", False),
            "samples_entity_effects_from_prior": sampling.get("samples_entity_effects_from_prior", True),
            "diagnostic_upper_bound_only": sampling.get("diagnostic_upper_bound_only", False),
            "structure_source": "ct_2k_sbm_temporal_kde_stubs",
            "checkpoint": str(checkpoint_path),
            "entity_prior_dir": str(entity_prior_dir) if entity_prior_dir else None,
            "output": str(output_path),
            "seed": int(seed),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


def training_entity_latent_rows(
    reviews: pd.DataFrame,
    customer_effects: pd.DataFrame,
    product_effects: pd.DataFrame,
    customer_id_col: str,
    product_id_col: str,
) -> pd.DataFrame:
    customer = customer_effects[
        [customer_id_col, "rating_effect", "verified_effect"]
    ].rename(
        columns={
            "rating_effect": "customer_rating_effect",
            "verified_effect": "customer_verified_effect",
        }
    )
    product = product_effects[
        [product_id_col, "rating_effect", "verified_effect"]
    ].rename(
        columns={
            "rating_effect": "product_rating_effect",
            "verified_effect": "product_verified_effect",
        }
    )
    merged = (
        reviews[[customer_id_col, product_id_col]]
        .merge(customer, on=customer_id_col, how="left")
        .merge(product, on=product_id_col, how="left")
    )
    return merged[ENTITY_LATENT_FEATURES].fillna(0.0).reset_index(drop=True)


def sampling_entity_latent_rows(
    group: pd.DataFrame,
    sampled_customer_effects: pd.DataFrame,
    sampled_product_effects: pd.DataFrame,
    customer_id_col: str,
    product_id_col: str,
) -> pd.DataFrame:
    customer = sampled_customer_effects[
        [customer_id_col, "sampled_rating_effect", "sampled_verified_effect"]
    ].rename(
        columns={
            "sampled_rating_effect": "customer_rating_effect",
            "sampled_verified_effect": "customer_verified_effect",
        }
    )
    product = sampled_product_effects[
        [product_id_col, "sampled_rating_effect", "sampled_verified_effect"]
    ].rename(
        columns={
            "sampled_rating_effect": "product_rating_effect",
            "sampled_verified_effect": "product_verified_effect",
        }
    )
    merged = (
        group[[customer_id_col, product_id_col]]
        .merge(customer, on=customer_id_col, how="left")
        .merge(product, on=product_id_col, how="left")
    )
    return merged[ENTITY_LATENT_FEATURES].fillna(0.0).set_index(group.index)


def load_posterior_effects_for_debug(
    entity_prior_dir: str | Path,
    customer_id_col: str,
    product_id_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(entity_prior_dir)
    nested = root / "entity_effects"
    if nested.exists():
        root = nested
    customer = pd.read_csv(root / "customer_effects.csv")
    product = pd.read_csv(root / "product_effects.csv")
    customer = customer.rename(
        columns={
            "customer_block": "block",
            "rating_effect": "sampled_rating_effect",
            "verified_effect": "sampled_verified_effect",
        }
    )
    product = product.rename(
        columns={
            "product_block": "block",
            "rating_effect": "sampled_rating_effect",
            "verified_effect": "sampled_verified_effect",
        }
    )
    for frame, id_col in ((customer, customer_id_col), (product, product_id_col)):
        if "degree_bin" not in frame.columns:
            frame["degree_bin"] = "posterior"
        frame["prior_cell_used"] = "posterior_effect_upper_bound"
        for col in ["degree", "block", "sampled_rating_effect", "sampled_verified_effect"]:
            if col not in frame.columns:
                frame[col] = 0
        if id_col not in frame.columns:
            raise ValueError(f"Posterior effect debug file missing {id_col!r}.")
    return (
        customer[
            [
                customer_id_col,
                "block",
                "degree",
                "degree_bin",
                "sampled_rating_effect",
                "sampled_verified_effect",
                "prior_cell_used",
            ]
        ],
        product[
            [
                product_id_col,
                "block",
                "degree",
                "degree_bin",
                "sampled_rating_effect",
                "sampled_verified_effect",
                "prior_cell_used",
            ]
        ],
    )
