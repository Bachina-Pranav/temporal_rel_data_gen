"""V3 non-text attribute generator with temporal priors and residual diffusion."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .block_attribute_priors import BlockAttributePrior
from .entity_effect_priors_v3 import (
    ConditionalGaussianEffectPriorV3,
    estimate_entity_effects_v3,
    fit_entity_priors_v3,
)
from .entity_latent_effects import compute_entity_structural_features
from .residual_diffusion_model_v3 import ResidualTemporalFeatureDiffusionModelV3
from .temporal_attribute_sampler import chronological_groups
from .temporal_calibration import calibrate_logits_torch
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
from .temporal_nontext_diffusion_v2 import load_posterior_effects_for_debug
from .temporal_priors import TemporalAttributePrior, temporal_bucket
from .nontext_diffusion_schedules import gaussian_sigma, mask_probability


GENERATOR_NAME_V3 = "temporal_nontext_attr_diffusion_v3"
GENERATOR_ALIAS_V3 = "residual_temporal_entity_attr_diffusion"


class TemporalNonTextAttributeDiffusionV3(TemporalNonTextAttributeDiffusion):
    """Residual temporal/entity non-text attribute generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_feature_names: List[str] = []
        self.temporal_prior: Optional[TemporalAttributePrior] = None
        self.block_prior: Optional[BlockAttributePrior] = None
        self.customer_prior_v3: Optional[ConditionalGaussianEffectPriorV3] = None
        self.product_prior_v3: Optional[ConditionalGaussianEffectPriorV3] = None
        self.effect_noise_std = 0.05
        self.effect_dropout = 0.1
        self.lambda_block = 1.0
        self.lambda_product_effect = 1.0
        self.lambda_customer_effect = 0.7
        self.lambda_block_verified = 1.0
        self.lambda_product_verified_effect = 1.0
        self.lambda_customer_verified_effect = 0.7
        self.lambda_residual_l2 = 1e-4

    def continuous_feature_names(self) -> List[str]:
        return list(CAUSAL_CONTINUOUS_FEATURES) + list(self.entity_feature_names)

    def make_model(self) -> ResidualTemporalFeatureDiffusionModelV3:
        return ResidualTemporalFeatureDiffusionModelV3(**self.model_config)

    @classmethod
    def train_from_csv(cls, reviews_path: str | Path, output_dir: str | Path, structure_debug_dir=None, **kwargs):
        reviews = pd.read_csv(reviews_path)
        generator = cls(
            customer_id_col=kwargs.get("customer_id_col", "customer_id"),
            product_id_col=kwargs.get("product_id_col", "product_id"),
            timestamp_col=kwargs.get("timestamp_col", "review_time"),
            cat_cols=kwargs.get("cat_cols"),
            num_cols=kwargs.get("num_cols"),
            seed=kwargs.get("seed", 42),
        )
        return generator.train(reviews, output_dir=output_dir, structure_debug_dir=structure_debug_dir, **kwargs)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str | Path, device: str = "cpu") -> "TemporalNonTextAttributeDiffusionV3":
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
        generator.entity_feature_names = config.get("entity_feature_names", [])
        generator.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        generator.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        generator.discrete_feature_maps = checkpoint["discrete_feature_maps"]
        generator.numerical_metadata = checkpoint["numerical_metadata"]
        generator.model_config = checkpoint["model_config"]
        generator.temporal_prior = TemporalAttributePrior.from_dict(checkpoint["temporal_prior"])
        generator.block_prior = BlockAttributePrior.from_dict(checkpoint["block_prior"])
        generator.customer_prior_v3 = ConditionalGaussianEffectPriorV3.from_dict(checkpoint["customer_prior_v3"])
        generator.product_prior_v3 = ConditionalGaussianEffectPriorV3.from_dict(checkpoint["product_prior_v3"])
        for name in [
            "effect_noise_std", "effect_dropout", "lambda_block", "lambda_product_effect",
            "lambda_customer_effect", "lambda_block_verified", "lambda_product_verified_effect",
            "lambda_customer_verified_effect", "lambda_residual_l2",
        ]:
            setattr(generator, name, float(config.get(name, getattr(generator, name))))
        generator.model = generator.make_model().to(device)
        generator.model.load_state_dict(checkpoint["model_state_dict"])
        generator.model.eval()
        generator._checkpoint = checkpoint
        return generator

    @classmethod
    def sample_from_checkpoint(cls, synthetic_spine_path, checkpoint_path, output_path, structure_debug_dir=None, **kwargs):
        generator = cls.load_checkpoint(checkpoint_path, device=kwargs.get("device", "cpu"))
        spine = pd.read_csv(synthetic_spine_path)
        output_path = Path(output_path)
        output = generator.sample(
            spine,
            structure_debug_dir=structure_debug_dir,
            sampled_effects_output_dir=kwargs.get("sampled_effects_output_dir", output_path.parent),
            seed=kwargs.get("seed", generator.seed),
            num_steps=kwargs.get("num_steps", 50),
            cat_sampling_strategy=kwargs.get("cat_sampling_strategy", "sample"),
            temperature=kwargs.get("temperature", 1.0),
            sampling_time_group=kwargs.get("sampling_time_group", "date"),
            sampling_window_days=kwargs.get("sampling_window_days", 1.0),
            use_temporal_calibration=kwargs.get("use_temporal_calibration", False),
            temporal_calibration_strength=kwargs.get("temporal_calibration_strength", 0.75),
            debug_use_posterior_effects=kwargs.get("debug_use_posterior_effects", False),
            device=kwargs.get("device", "cpu"),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        with output_path.with_name(output_path.stem + "_metadata.json").open("w") as handle:
            json.dump(to_jsonable(generator.synthetic_metadata(output_path, checkpoint_path, kwargs.get("seed", generator.seed))), handle, indent=2)
            handle.write("\n")
        return output

    def train(self, reviews: pd.DataFrame, output_dir: str | Path, structure_debug_dir=None, **kwargs) -> TemporalNonTextTrainingResult:
        self._set_seeds(self.seed)
        output_dir = Path(output_dir)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        reviews = self._preprocess_training_reviews(reviews)
        self.config["generator"] = GENERATOR_NAME_V3
        self.config["generator_alias"] = GENERATOR_ALIAS_V3
        self.config["output_dir"] = str(output_dir)
        for key, default in {
            "effect_noise_std": 0.05,
            "effect_dropout": 0.1,
            "lambda_block": 1.0,
            "lambda_product_effect": 1.0,
            "lambda_customer_effect": 0.7,
            "lambda_block_verified": 1.0,
            "lambda_product_verified_effect": 1.0,
            "lambda_customer_verified_effect": 0.7,
            "lambda_residual_l2": 1e-4,
        }.items():
            setattr(self, key, float(kwargs.get(key, default)))
            self.config[key] = getattr(self, key)
        self.config.update(
            {
                "uses_real_entity_effect_lookup": False,
                "samples_entity_effects_from_prior": True,
                "entity_effect_source": "sampled_prior",
                "diagnostic_upper_bound": False,
            }
        )
        self._fit_categories(reviews)
        self.num_cols = [col for col in self.num_cols if col in reviews.columns]
        self.config["num_cols"] = list(self.num_cols)
        rating_col = self.cat_cols[0]
        verified_col = "verified" if "verified" in self.cat_cols else self.cat_cols[-1]
        self.entity_feature_names = entity_feature_names(len(self.category_values[rating_col]))
        self.config["entity_feature_names"] = self.entity_feature_names

        temporal_level = kwargs.get("temporal_prior_level", "month")
        self.temporal_prior = TemporalAttributePrior(self.category_values[rating_col], temporal_prior_level=temporal_level).fit(reviews, self.timestamp_col, rating_col, verified_col)
        self.block_prior = BlockAttributePrior(self.category_values[rating_col]).fit(reviews, structure_debug_dir, self.customer_id_col, self.product_id_col, rating_col, verified_col)
        customer_effects, product_effects, effect_stats = estimate_entity_effects_v3(
            reviews,
            self.category_values[rating_col],
            structure_debug_dir=structure_debug_dir,
            customer_id_col=self.customer_id_col,
            product_id_col=self.product_id_col,
            timestamp_col=self.timestamp_col,
            rating_col=rating_col,
            verified_col=verified_col,
        )
        effects_dir = output_dir / "entity_effects_v3"
        effects_dir.mkdir(parents=True, exist_ok=True)
        customer_effects.to_csv(effects_dir / "customer_effects.csv", index=False)
        product_effects.to_csv(effects_dir / "product_effects.csv", index=False)
        self.customer_prior_v3, self.product_prior_v3 = fit_entity_priors_v3(
            customer_effects,
            product_effects,
            self.customer_id_col,
            self.product_id_col,
            num_degree_bins=int(kwargs.get("num_degree_bins", 4)),
            min_entities_per_cell=int(kwargs.get("min_entities_per_cell", 20)),
            product_effect_scale=float(kwargs.get("product_effect_scale", 1.0)),
            customer_effect_scale=float(kwargs.get("customer_effect_scale", 1.15)),
        )
        save_json(output_dir / "entity_effect_stats_v3.json", effect_stats)

        customer_blocks, product_blocks = load_block_maps(structure_debug_dir, self.customer_id_col, self.product_id_col)
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
        latent_features = posterior_effect_rows_v3(reviews, customer_effects, product_effects, self.customer_id_col, self.product_id_col, self.entity_feature_names)
        features = pd.concat([features, latent_features], axis=1)
        base_rating, base_verified = self.compute_base_logits(reviews, latent_features, customer_blocks, product_blocks)
        self._fit_feature_metadata(features)
        encoded = self._encode_dataset(reviews, features)
        encoded["base_rating_logits"] = base_rating
        encoded["base_verified_logits"] = base_verified
        train_idx, val_idx = self._split_indices(reviews, kwargs.get("validation_fraction", 0.1), kwargs.get("random_split", False))
        self.model_config = {
            "cat_cols": self.cat_cols,
            "cat_vocab_sizes": {col: len(self.category_values[col]) for col in self.cat_cols},
            "num_numerical": len(self.num_cols),
            "continuous_feature_dim": len(self.continuous_feature_names()),
            "discrete_feature_vocab_sizes": {col: len(mapping) + 1 for col, mapping in self.discrete_feature_maps.items()},
            "rating_num_classes": len(self.category_values[rating_col]),
            "hidden_dim": int(kwargs.get("hidden_dim", 256)),
            "num_layers": int(kwargs.get("num_layers", 4)),
            "dropout": float(kwargs.get("dropout", 0.1)),
        }
        self.model = self.make_model().to(kwargs.get("device", "cpu"))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=kwargs.get("learning_rate", 1e-3), weight_decay=1e-4)
        history = []
        best_val = float("inf")
        best_path = checkpoint_dir / "best.pt"
        latest_path = checkpoint_dir / "latest.pt"
        for epoch in range(1, int(kwargs.get("epochs", 5)) + 1):
            train_losses = self._run_epoch_v3(encoded, train_idx, optimizer, kwargs.get("batch_size", 256), kwargs.get("device", "cpu"), True, kwargs.get("mask_schedule", "cosine"))
            val_losses = self._run_epoch_v3(encoded, val_idx if len(val_idx) else train_idx, None, kwargs.get("batch_size", 256), kwargs.get("device", "cpu"), False, kwargs.get("mask_schedule", "cosine"))
            row = {"epoch": float(epoch), **{f"train_{k}": v for k, v in train_losses.items()}, **{f"val_{k}": v for k, v in val_losses.items()}}
            history.append(row)
            self._save_checkpoint(latest_path, history)
            if val_losses["total"] <= best_val:
                best_val = val_losses["total"]
                self._save_checkpoint(best_path, history)
            print(f"epoch {epoch:03d} train={train_losses['total']:.4f} val={val_losses['total']:.4f}")
        self._write_metadata(output_dir)
        self.temporal_prior.save(output_dir / "temporal_priors.json")
        self.block_prior.save(output_dir / "block_attribute_priors.json")
        self.customer_prior_v3.save(output_dir / "customer_entity_effect_prior_v3.json")
        self.product_prior_v3.save(output_dir / "product_entity_effect_prior_v3.json")
        return TemporalNonTextTrainingResult(output_dir, best_path, latest_path, history)

    def sample(self, synthetic_spine: pd.DataFrame, structure_debug_dir=None, sampled_effects_output_dir=None, seed=None, num_steps=50, cat_sampling_strategy="sample", temperature=1.0, sampling_time_group="date", sampling_window_days=1.0, use_temporal_calibration=False, temporal_calibration_strength=0.75, debug_use_posterior_effects=False, device="cpu") -> pd.DataFrame:
        seed = self.seed if seed is None else int(seed)
        self._set_seeds(seed)
        rng = np.random.default_rng(seed)
        self.model.to(device)
        self.model.eval()
        spine = synthetic_spine.copy()
        spine[self.timestamp_col] = pd.to_datetime(spine[self.timestamp_col], errors="coerce")
        customer_blocks, product_blocks = load_block_maps(structure_debug_dir, self.customer_id_col, self.product_id_col)
        customer_effects, product_effects = self.sample_effects_for_spine(spine, customer_blocks, product_blocks, rng, debug_use_posterior_effects)
        if sampled_effects_output_dir is not None:
            sample_dir = Path(sampled_effects_output_dir)
            sample_dir.mkdir(parents=True, exist_ok=True)
            customer_effects.to_csv(sample_dir / "sampled_customer_effects_v3.csv", index=False)
            product_effects.to_csv(sample_dir / "sampled_product_effects_v3.csv", index=False)
        metadata = self._checkpoint.get("feature_builder_metadata", {})
        builder = TemporalCausalFeatureBuilder(
            self.customer_id_col, self.product_id_col, self.timestamp_col,
            self.cat_cols[0], "verified" if "verified" in self.cat_cols else self.cat_cols[-1],
            customer_blocks=customer_blocks, product_blocks=product_blocks,
            date_only=metadata.get("date_only"),
            marginal_rating=metadata.get("marginal_rating", 0.0),
            marginal_verified=metadata.get("marginal_verified", 0.0),
        )
        builder.prepare_sampling(spine)
        generated_groups = []
        calibration_norms = []
        with torch.no_grad():
            for bucket, group in chronological_groups(spine, self.timestamp_col, mode=sampling_time_group, window_days=sampling_window_days):
                features = builder.transform_current_group(group)
                latent_features = sampled_effect_rows_v3(group, customer_effects, product_effects, self.customer_id_col, self.product_id_col, self.entity_feature_names)
                features = pd.concat([features, latent_features], axis=1)
                base_rating, base_verified = self.compute_base_logits(group, latent_features, customer_blocks, product_blocks)
                encoded_features = self._encode_features_for_sampling(features)
                batch_size = len(group)
                cat_tokens = {col: torch.full((batch_size,), len(self.category_values[col]), dtype=torch.long, device=device) for col in self.cat_cols}
                numerical = torch.randn(batch_size, len(self.num_cols), device=device) if self.num_cols else None
                continuous = torch.tensor(encoded_features["continuous"], dtype=torch.float32, device=device)
                discrete = {col: torch.tensor(values, dtype=torch.long, device=device) for col, values in encoded_features["discrete"].items()}
                base_rating_t = torch.tensor(base_rating, dtype=torch.float32, device=device)
                base_verified_t = torch.tensor(base_verified, dtype=torch.float32, device=device)
                steps = np.linspace(1.0, 0.0, max(int(num_steps), 1))
                out = None
                for step in steps:
                    t = torch.full((batch_size,), float(step), device=device)
                    out = self.model(cat_tokens, continuous, discrete, t, base_rating_t, base_verified_t, numerical_noisy=numerical)
                    rating_logits = out["rating_logits"] / max(float(temperature), 1e-6)
                    verified_logits = out["verified_logits"] / max(float(temperature), 1e-6)
                    if use_temporal_calibration:
                        b = temporal_bucket(group[self.timestamp_col], self.temporal_prior.temporal_prior_level).iloc[0]
                        rating_logits, verified_logits, norm = calibrate_logits_torch(rating_logits, verified_logits, self.temporal_prior.target_rating_distribution(b), self.temporal_prior.target_verified_rate(b), temporal_calibration_strength)
                        calibration_norms.append(norm)
                    cat_tokens[self.cat_cols[0]] = sample_logits(rating_logits, cat_sampling_strategy)
                    verified_col = "verified" if "verified" in self.cat_cols else self.cat_cols[-1]
                    cat_tokens[verified_col] = sample_logits(verified_logits, cat_sampling_strategy)
                    if numerical is not None and "num_pred" in out:
                        numerical = numerical - out["num_pred"] * gaussian_sigma(t).view(-1, 1) / max(len(steps), 1)
                generated = group.copy()
                for col in self.cat_cols:
                    indices = cat_tokens[col].detach().cpu().numpy().astype(int)
                    generated[col] = [self.category_values[col][idx] for idx in indices]
                generated_groups.append(generated)
                builder.update_history(generated)
        self._last_sampling_metadata = {
            "uses_temporal_calibration": bool(use_temporal_calibration),
            "temporal_calibration_average_correction_norm": float(np.mean(calibration_norms)) if calibration_norms else 0.0,
            "uses_real_entity_effect_lookup": bool(debug_use_posterior_effects),
            "diagnostic_upper_bound": bool(debug_use_posterior_effects),
            "samples_entity_effects_from_prior": not bool(debug_use_posterior_effects),
            "entity_effect_source": "posterior_effects_debug" if debug_use_posterior_effects else "sampled_prior",
        }
        output = pd.concat(generated_groups).sort_index()
        columns = [self.customer_id_col, self.product_id_col, self.timestamp_col] + self.cat_cols + self.num_cols
        return output[columns].reset_index(drop=True)

    def compute_base_logits(self, rows: pd.DataFrame, latent_features: pd.DataFrame, customer_blocks, product_blocks) -> Tuple[np.ndarray, np.ndarray]:
        rating_temporal = self.temporal_prior.rating_logits_for_timestamps(rows[self.timestamp_col])
        verified_temporal_scalar = self.temporal_prior.verified_logits_for_timestamps(rows[self.timestamp_col])
        rating_block, verified_block = self.block_prior.residuals_for_rows(rows, customer_blocks, product_blocks, self.customer_id_col, self.product_id_col)
        k = rating_temporal.shape[1]
        product_rating = latent_features[[f"product_rating_effect_{i}" for i in range(k)]].to_numpy(dtype=np.float32)
        customer_rating = latent_features[[f"customer_rating_effect_{i}" for i in range(k)]].to_numpy(dtype=np.float32)
        base_rating = rating_temporal + self.lambda_block * rating_block + self.lambda_product_effect * product_rating + self.lambda_customer_effect * customer_rating
        verified_scalar = verified_temporal_scalar + self.lambda_block_verified * verified_block + self.lambda_product_verified_effect * latent_features["product_verified_effect"].to_numpy(dtype=np.float32) + self.lambda_customer_verified_effect * latent_features["customer_verified_effect"].to_numpy(dtype=np.float32)
        base_verified = np.stack([-0.5 * verified_scalar, 0.5 * verified_scalar], axis=1).astype(np.float32)
        return base_rating.astype(np.float32), base_verified

    def _batch_tensors(self, encoded: Dict[str, Any], indices: np.ndarray, device: str) -> Dict[str, Any]:
        tensors = super()._batch_tensors(encoded, indices, device)
        tensors["base_rating_logits"] = torch.tensor(encoded["base_rating_logits"][indices], dtype=torch.float32, device=device)
        tensors["base_verified_logits"] = torch.tensor(encoded["base_verified_logits"][indices], dtype=torch.float32, device=device)
        return tensors

    def _run_epoch_v3(self, encoded, indices, optimizer, batch_size, device, train, mask_schedule):
        self.model.train(train)
        indices = np.asarray(indices, dtype=int)
        if train:
            np.random.default_rng().shuffle(indices)
        totals = Counter()
        count = 0
        rating_col = self.cat_cols[0]
        verified_col = "verified" if "verified" in self.cat_cols else self.cat_cols[-1]
        for start in range(0, len(indices), int(batch_size)):
            batch_idx = indices[start:start + int(batch_size)]
            if len(batch_idx) == 0:
                continue
            tensors = self._batch_tensors(encoded, batch_idx, device)
            if train:
                tensors["continuous"] = corrupt_entity_features(tensors["continuous"], len(CAUSAL_CONTINUOUS_FEATURES), len(self.entity_feature_names), self.effect_noise_std, self.effect_dropout)
            t = torch.rand(len(batch_idx), device=device)
            p_mask = mask_probability(t, mask_schedule)
            cat_tokens = {}
            masks = {}
            for col in self.cat_cols:
                target = tensors["cat_targets"][col]
                mask = torch.rand(len(batch_idx), device=device) < p_mask
                if not bool(mask.any()):
                    mask[0] = True
                token = target.clone()
                token[mask] = len(self.category_values[col])
                cat_tokens[col] = token
                masks[col] = mask
            numerical_noisy = None
            noise = None
            if self.num_cols:
                sigma = gaussian_sigma(t).view(-1, 1)
                noise = torch.randn_like(tensors["numerical"])
                numerical_noisy = tensors["numerical"] + sigma * noise
            out = self.model(cat_tokens, tensors["continuous"], tensors["discrete"], t, tensors["base_rating_logits"], tensors["base_verified_logits"], numerical_noisy=numerical_noisy)
            cat_loss = F.cross_entropy(out["rating_logits"][masks[rating_col]], tensors["cat_targets"][rating_col][masks[rating_col]])
            cat_loss = cat_loss + F.cross_entropy(out["verified_logits"][masks[verified_col]], tensors["cat_targets"][verified_col][masks[verified_col]])
            cat_loss = cat_loss / 2.0
            num_loss = torch.tensor(0.0, device=device)
            if self.num_cols and noise is not None and "num_pred" in out:
                num_loss = F.mse_loss(out["num_pred"], noise)
            residual_l2 = out["rating_residual_logits"].pow(2).mean() + out["verified_residual_logits"].pow(2).mean()
            loss = cat_loss + num_loss + self.lambda_residual_l2 * residual_l2
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
            totals["total"] += float(loss.detach().cpu()) * len(batch_idx)
            totals["cat"] += float(cat_loss.detach().cpu()) * len(batch_idx)
            totals["num"] += float(num_loss.detach().cpu()) * len(batch_idx)
            totals["residual_l2"] += float(residual_l2.detach().cpu()) * len(batch_idx)
            count += len(batch_idx)
        return {key: float(value / max(count, 1)) for key, value in totals.items()}

    def sample_effects_for_spine(self, spine, customer_blocks, product_blocks, rng, debug_use_posterior_effects):
        if debug_use_posterior_effects:
            return load_posterior_effects_v3_for_debug(
                self.config.get("output_dir"),
                self.customer_id_col,
                self.product_id_col,
            )
        customer_struct = compute_entity_structural_features(spine, self.customer_id_col, self.timestamp_col, customer_blocks, "customer_block", self.customer_id_col)
        product_struct = compute_entity_structural_features(spine, self.product_id_col, self.timestamp_col, product_blocks, "product_block", self.product_id_col)
        return self.customer_prior_v3.sample(customer_struct, rng), self.product_prior_v3.sample(product_struct, rng)

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
            "feature_builder_metadata": self.feature_builder.to_metadata() if self.feature_builder is not None else {},
            "temporal_prior": self.temporal_prior.to_dict(),
            "block_prior": self.block_prior.to_dict(),
            "customer_prior_v3": self.customer_prior_v3.to_dict(),
            "product_prior_v3": self.product_prior_v3.to_dict(),
            "history": history,
        }
        torch.save(checkpoint, path)

    def synthetic_metadata(self, output_path, checkpoint_path, seed):
        sampling = getattr(self, "_last_sampling_metadata", {})
        return {
            "method": GENERATOR_NAME_V3,
            "uses_real_entity_effect_lookup": sampling.get("uses_real_entity_effect_lookup", False),
            "samples_entity_effects_from_prior": sampling.get("samples_entity_effects_from_prior", True),
            "entity_effect_source": sampling.get("entity_effect_source", "sampled_prior"),
            "diagnostic_upper_bound": sampling.get("diagnostic_upper_bound", False),
            "uses_temporal_calibration": sampling.get("uses_temporal_calibration", False),
            "temporal_calibration_average_correction_norm": sampling.get("temporal_calibration_average_correction_norm", 0.0),
            "structure_source": "ct_2k_sbm_temporal_kde_stubs",
            "checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "seed": int(seed),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


def entity_feature_names(k: int) -> List[str]:
    return [f"customer_rating_effect_{i}" for i in range(k)] + ["customer_verified_effect"] + [f"product_rating_effect_{i}" for i in range(k)] + ["product_verified_effect"]


def posterior_effect_rows_v3(reviews, customer_effects, product_effects, customer_id_col, product_id_col, names):
    customer_cols = [col for col in customer_effects.columns if col.startswith("rating_effect_")] + ["verified_effect"]
    product_cols = [col for col in product_effects.columns if col.startswith("rating_effect_")] + ["verified_effect"]
    customer = customer_effects[[customer_id_col] + customer_cols].rename(columns={col: f"customer_{col}" for col in customer_cols})
    product = product_effects[[product_id_col] + product_cols].rename(columns={col: f"product_{col}" for col in product_cols})
    merged = reviews[[customer_id_col, product_id_col]].merge(customer, on=customer_id_col, how="left").merge(product, on=product_id_col, how="left")
    return merged[names].fillna(0.0).reset_index(drop=True)


def sampled_effect_rows_v3(group, customer_effects, product_effects, customer_id_col, product_id_col, names):
    customer_cols = [col for col in customer_effects.columns if col.startswith("sampled_rating_effect_")] + ["sampled_verified_effect"]
    product_cols = [col for col in product_effects.columns if col.startswith("sampled_rating_effect_")] + ["sampled_verified_effect"]
    customer = customer_effects[[customer_id_col] + customer_cols].rename(columns={col: col.replace("sampled_", "customer_") for col in customer_cols})
    product = product_effects[[product_id_col] + product_cols].rename(columns={col: col.replace("sampled_", "product_") for col in product_cols})
    merged = group[[customer_id_col, product_id_col]].merge(customer, on=customer_id_col, how="left").merge(product, on=product_id_col, how="left")
    return merged[names].fillna(0.0).set_index(group.index)


def corrupt_entity_features(continuous, start, width, noise_std, dropout):
    output = continuous.clone()
    if width <= 0:
        return output
    effects = output[:, start:start + width]
    if noise_std > 0:
        effects = effects + torch.randn_like(effects) * float(noise_std)
    if dropout > 0:
        keep = (torch.rand((effects.shape[0], 1), device=effects.device) >= float(dropout)).float()
        effects = effects * keep
    output[:, start:start + width] = effects
    return output


def sample_logits(logits, strategy):
    if strategy == "argmax":
        return torch.argmax(logits, dim=1)
    return torch.multinomial(torch.softmax(logits, dim=1), num_samples=1).squeeze(1)


def load_posterior_effects_v3_for_debug(output_dir, customer_id_col, product_id_col):
    if output_dir is None:
        raise ValueError("V3 checkpoint missing output_dir; cannot load diagnostic posterior effects.")
    root = Path(output_dir) / "entity_effects_v3"
    customer = pd.read_csv(root / "customer_effects.csv")
    product = pd.read_csv(root / "product_effects.csv")
    customer = convert_posterior_effects_to_sampled(customer, customer_id_col)
    product = convert_posterior_effects_to_sampled(product, product_id_col)
    return customer, product


def convert_posterior_effects_to_sampled(frame, id_col):
    out = frame.copy()
    for col in list(out.columns):
        if col.startswith("rating_effect_"):
            out = out.rename(columns={col: f"sampled_{col}"})
    if "verified_effect" in out.columns:
        out = out.rename(columns={"verified_effect": "sampled_verified_effect"})
    if "block" not in out.columns:
        block_cols = [col for col in out.columns if col.endswith("_block")]
        out["block"] = out[block_cols[0]] if block_cols else -1
    if "degree_bin" not in out.columns:
        out["degree_bin"] = "posterior"
    out["prior_cell_used"] = "posterior_effect_upper_bound"
    return out
