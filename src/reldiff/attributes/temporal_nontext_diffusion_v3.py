"""V3 non-text attribute generator with temporal priors and residual diffusion."""

from __future__ import annotations

import json
import warnings
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
from .temporal_calibration import (
    calibrate_logits_torch,
    calibration_group_stats_torch,
    js_divergence_probs,
)
from .temporal_buckets import infer_bucket_format
from .temporal_causal_features import (
    CAUSAL_CONTINUOUS_FEATURES,
    TemporalCausalFeatureBuilder,
    load_block_maps,
    normalize_verified,
    save_json,
)
from .temporal_nontext_diffusion import (
    TemporalNonTextAttributeDiffusion,
    TemporalNonTextTrainingResult,
    canonical_category_key,
    to_jsonable,
)
from .temporal_nontext_diffusion_v2 import load_posterior_effects_for_debug
from .temporal_priors import (
    TemporalAttributePrior,
    check_temporal_bucket_consistency,
    temporal_bucket,
)
from .nontext_diffusion_schedules import gaussian_sigma, mask_probability


GENERATOR_NAME_V3 = "temporal_nontext_attr_diffusion_v3"
GENERATOR_ALIAS_V3 = "residual_temporal_entity_attr_diffusion"

DECOMPOSITION_KEYS = [
    "average_norm_base_rating_logits",
    "average_norm_residual_rating_logits",
    "average_norm_final_rating_logits_pre_calibration",
    "average_norm_final_rating_logits_post_calibration",
    "residual_to_base_norm_ratio",
    "average_abs_base_verified_logit",
    "average_abs_residual_verified_logit",
    "verified_residual_to_base_abs_ratio",
    "sampled_product_rating_effect_variance",
    "sampled_customer_rating_effect_variance",
    "sampled_product_verified_effect_variance",
    "sampled_customer_verified_effect_variance",
    "average_temporal_rating_prior_entropy",
    "average_model_rating_entropy_pre_calibration",
    "average_model_rating_entropy_post_calibration",
    "temporal_calibration_average_correction_norm",
    "temporal_calibration_max_correction_norm",
    "temporal_calibration_num_groups_calibrated",
    "average_precal_rating_target_js",
    "average_postcal_rating_target_js",
    "average_precal_verified_target_abs_error",
    "average_postcal_verified_target_abs_error",
    "sampled_product_effect_variance",
    "sampled_customer_effect_variance",
]


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
            diagnostics_dir=kwargs.get("diagnostics_dir"),
            diagnostic_row_sample_size=kwargs.get("diagnostic_row_sample_size", 5000),
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

        temporal_level = kwargs.get("temporal_prior_level", "year_month")
        self.config["temporal_prior_level"] = temporal_level
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

    def sample(self, synthetic_spine: pd.DataFrame, structure_debug_dir=None, sampled_effects_output_dir=None, seed=None, num_steps=50, cat_sampling_strategy="sample", temperature=1.0, sampling_time_group="date", sampling_window_days=1.0, use_temporal_calibration=False, temporal_calibration_strength=0.75, debug_use_posterior_effects=False, device="cpu", diagnostics_dir=None, diagnostic_row_sample_size=5000) -> pd.DataFrame:
        seed = self.seed if seed is None else int(seed)
        self._set_seeds(seed)
        rng = np.random.default_rng(seed)
        self.model.to(device)
        self.model.eval()
        spine = synthetic_spine.copy()
        spine[self.timestamp_col] = pd.to_datetime(spine[self.timestamp_col], errors="coerce")
        temporal_prior_level = self.temporal_prior.temporal_prior_level
        prior_bucket_keys = set(self.temporal_prior.per_bucket_rating_distribution)
        sampling_bucket_keys = set(temporal_bucket(spine[self.timestamp_col], temporal_prior_level).dropna().astype(str))
        sampling_bucket_format = infer_bucket_format(sampling_bucket_keys)
        missing_sampling_buckets = sorted(sampling_bucket_keys - prior_bucket_keys)
        checkpoint_uses_legacy_buckets = self.temporal_prior.uses_legacy_temporal_buckets()
        if checkpoint_uses_legacy_buckets:
            warnings.warn(
                "Checkpoint uses legacy month-number temporal priors. "
                "Retrain V3 with --temporal-prior-level year_month.",
                RuntimeWarning,
            )
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
        calibration_rows = []
        component_rows = []
        row_diagnostics = []
        norm_records = []
        rating_values_numeric = np.asarray(
            [float(value) for value in self.category_values[self.cat_cols[0]]],
            dtype=np.float32,
        )
        with torch.no_grad():
            for bucket, group in chronological_groups(spine, self.timestamp_col, mode=sampling_time_group, window_days=sampling_window_days):
                features = builder.transform_current_group(group)
                latent_features = sampled_effect_rows_v3(group, customer_effects, product_effects, self.customer_id_col, self.product_id_col, self.entity_feature_names)
                features = pd.concat([features, latent_features], axis=1)
                base_components = self.compute_base_logit_components(
                    group, latent_features, customer_blocks, product_blocks
                )
                base_rating = base_components["base_rating"]
                base_verified = base_components["base_verified"]
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
                final_precal_rating = None
                final_postcal_rating = None
                final_precal_verified = None
                final_postcal_verified = None
                final_residual_rating = None
                final_residual_verified = None
                for step_index, step in enumerate(steps):
                    t = torch.full((batch_size,), float(step), device=device)
                    out = self.model(cat_tokens, continuous, discrete, t, base_rating_t, base_verified_t, numerical_noisy=numerical)
                    rating_logits = out["rating_logits"] / max(float(temperature), 1e-6)
                    verified_logits = out["verified_logits"] / max(float(temperature), 1e-6)
                    precal_rating_logits = rating_logits
                    precal_verified_logits = verified_logits
                    if use_temporal_calibration:
                        b = temporal_bucket(group[self.timestamp_col], self.temporal_prior.temporal_prior_level).iloc[0]
                        rating_logits, verified_logits, norm = calibrate_logits_torch(rating_logits, verified_logits, self.temporal_prior.target_rating_distribution(b), self.temporal_prior.target_verified_rate(b), temporal_calibration_strength)
                        calibration_norms.append(norm)
                        if step_index == len(steps) - 1:
                            calibration_rows.append(
                                calibration_group_stats_torch(
                                    b,
                                    precal_rating_logits,
                                    rating_logits,
                                    precal_verified_logits,
                                    verified_logits,
                                    self.temporal_prior.target_rating_distribution(b),
                                    self.temporal_prior.target_verified_rate(b),
                                    temporal_calibration_strength,
                                    self.category_values[self.cat_cols[0]],
                                )
                            )
                    if step_index == len(steps) - 1:
                        final_precal_rating = precal_rating_logits.detach().cpu().numpy()
                        final_postcal_rating = rating_logits.detach().cpu().numpy()
                        final_precal_verified = precal_verified_logits.detach().cpu().numpy()
                        final_postcal_verified = verified_logits.detach().cpu().numpy()
                        final_residual_rating = out["rating_residual_logits"].detach().cpu().numpy()
                        final_residual_verified = out["verified_residual_logits"].detach().cpu().numpy()
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
                if final_precal_rating is not None:
                    group_component = component_curve_row(
                        group,
                        generated,
                        base_components,
                        final_precal_rating,
                        final_postcal_rating,
                        final_precal_verified,
                        final_postcal_verified,
                        rating_values_numeric,
                        self.timestamp_col,
                        self.cat_cols[0],
                        "verified" if "verified" in self.cat_cols else self.cat_cols[-1],
                        self.temporal_prior.temporal_prior_level,
                    )
                    component_rows.append(group_component)
                    norm_records.append(
                        norm_record(
                            base_rating,
                            final_residual_rating,
                            final_precal_rating,
                            final_postcal_rating,
                            base_verified,
                            final_residual_verified,
                            final_precal_verified,
                            final_postcal_verified,
                        )
                    )
                    row_diagnostics.extend(
                        row_level_logit_records(
                            group,
                            generated,
                            latent_features,
                            customer_blocks,
                            product_blocks,
                            base_rating,
                            final_residual_rating,
                            final_precal_rating,
                            final_postcal_rating,
                            base_verified,
                            final_residual_verified,
                            final_precal_verified,
                            final_postcal_verified,
                            self.customer_id_col,
                            self.product_id_col,
                            self.timestamp_col,
                            self.cat_cols[0],
                            "verified" if "verified" in self.cat_cols else self.cat_cols[-1],
                            self.category_values[self.cat_cols[0]],
                            self.temporal_prior.temporal_prior_level,
                        )
                    )
                builder.update_history(generated)
        decomposition = build_decomposition_summary(
            norm_records,
            calibration_rows,
            product_effects,
            customer_effects,
            self.category_values[self.cat_cols[0]],
            self.temporal_prior,
            spine[self.timestamp_col],
        )
        self._last_sampling_metadata = {
            "uses_temporal_calibration": bool(use_temporal_calibration),
            "temporal_calibration_strength": float(temporal_calibration_strength),
            "temporal_calibration_level": self.temporal_prior.temporal_prior_level,
            "temporal_prior_level_used_for_sampling": temporal_prior_level,
            "temporal_bucket_format_used_for_sampling": sampling_bucket_format,
            "temporal_prior_num_buckets": int(len(prior_bucket_keys)),
            "sampling_num_buckets": int(len(sampling_bucket_keys)),
            "num_sampling_buckets_missing_in_prior": int(len(missing_sampling_buckets)),
            "sampling_buckets_missing_in_prior_examples": missing_sampling_buckets[:20],
            "checkpoint_uses_legacy_temporal_buckets": bool(checkpoint_uses_legacy_buckets),
            "temporal_calibration_average_correction_norm": float(np.mean(calibration_norms)) if calibration_norms else 0.0,
            "temporal_calibration_num_groups_calibrated": int(len(calibration_rows)),
            "calibration_applied_before_sampling": bool(use_temporal_calibration),
            "uses_real_entity_effect_lookup": bool(debug_use_posterior_effects),
            "diagnostic_upper_bound": bool(debug_use_posterior_effects),
            "samples_entity_effects_from_prior": not bool(debug_use_posterior_effects),
            "entity_effect_source": "posterior_effects_debug" if debug_use_posterior_effects else "sampled_prior",
            **decomposition,
        }
        if use_temporal_calibration and not calibration_rows:
            raise RuntimeError(
                "Temporal calibration was requested but no calibration groups were calibrated."
            )
        output = pd.concat(generated_groups).sort_index()
        if diagnostics_dir is not None:
            write_v3_sampling_diagnostics(
                diagnostics_dir,
                decomposition,
                calibration_rows,
                component_rows,
                row_diagnostics,
                int(diagnostic_row_sample_size),
                self.temporal_prior,
                spine[self.timestamp_col],
                sampling_metadata=self._last_sampling_metadata,
            )
        columns = [self.customer_id_col, self.product_id_col, self.timestamp_col] + self.cat_cols + self.num_cols
        return output[columns].reset_index(drop=True)

    def compute_base_logits(self, rows: pd.DataFrame, latent_features: pd.DataFrame, customer_blocks, product_blocks) -> Tuple[np.ndarray, np.ndarray]:
        components = self.compute_base_logit_components(
            rows, latent_features, customer_blocks, product_blocks
        )
        return components["base_rating"], components["base_verified"]

    def compute_base_logit_components(self, rows: pd.DataFrame, latent_features: pd.DataFrame, customer_blocks, product_blocks) -> Dict[str, np.ndarray]:
        rating_temporal = self.temporal_prior.rating_logits_for_timestamps(rows[self.timestamp_col])
        verified_temporal_scalar = self.temporal_prior.verified_logits_for_timestamps(rows[self.timestamp_col])
        rating_block, verified_block = self.block_prior.residuals_for_rows(rows, customer_blocks, product_blocks, self.customer_id_col, self.product_id_col)
        k = rating_temporal.shape[1]
        product_rating = latent_features[[f"product_rating_effect_{i}" for i in range(k)]].to_numpy(dtype=np.float32)
        customer_rating = latent_features[[f"customer_rating_effect_{i}" for i in range(k)]].to_numpy(dtype=np.float32)
        product_verified = latent_features["product_verified_effect"].to_numpy(dtype=np.float32)
        customer_verified = latent_features["customer_verified_effect"].to_numpy(dtype=np.float32)
        temporal_block_rating = rating_temporal + self.lambda_block * rating_block
        temporal_block_product_rating = temporal_block_rating + self.lambda_product_effect * product_rating
        base_rating = temporal_block_product_rating + self.lambda_customer_effect * customer_rating
        temporal_block_verified = verified_temporal_scalar + self.lambda_block_verified * verified_block
        temporal_block_product_verified = temporal_block_verified + self.lambda_product_verified_effect * product_verified
        verified_scalar = temporal_block_product_verified + self.lambda_customer_verified_effect * customer_verified
        return {
            "temporal_rating": rating_temporal.astype(np.float32),
            "temporal_block_rating": temporal_block_rating.astype(np.float32),
            "temporal_block_product_rating": temporal_block_product_rating.astype(np.float32),
            "base_rating": base_rating.astype(np.float32),
            "temporal_verified": verified_logits_from_scalar(verified_temporal_scalar),
            "temporal_block_verified": verified_logits_from_scalar(temporal_block_verified),
            "temporal_block_product_verified": verified_logits_from_scalar(temporal_block_product_verified),
            "base_verified": verified_logits_from_scalar(verified_scalar),
        }

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
        metadata = {
            "method": GENERATOR_NAME_V3,
            "uses_real_entity_effect_lookup": sampling.get("uses_real_entity_effect_lookup", False),
            "samples_entity_effects_from_prior": sampling.get("samples_entity_effects_from_prior", True),
            "entity_effect_source": sampling.get("entity_effect_source", "sampled_prior"),
            "diagnostic_upper_bound": sampling.get("diagnostic_upper_bound", False),
            "uses_temporal_calibration": sampling.get("uses_temporal_calibration", False),
            "temporal_calibration_average_correction_norm": sampling.get("temporal_calibration_average_correction_norm", 0.0),
            "temporal_calibration_strength": sampling.get("temporal_calibration_strength"),
            "temporal_calibration_level": sampling.get("temporal_calibration_level"),
            "temporal_prior_level_used_for_sampling": sampling.get("temporal_prior_level_used_for_sampling"),
            "temporal_bucket_format_used_for_sampling": sampling.get("temporal_bucket_format_used_for_sampling"),
            "temporal_prior_num_buckets": sampling.get("temporal_prior_num_buckets"),
            "sampling_num_buckets": sampling.get("sampling_num_buckets"),
            "num_sampling_buckets_missing_in_prior": sampling.get("num_sampling_buckets_missing_in_prior"),
            "checkpoint_uses_legacy_temporal_buckets": sampling.get("checkpoint_uses_legacy_temporal_buckets", False),
            "temporal_calibration_num_groups_calibrated": sampling.get("temporal_calibration_num_groups_calibrated", 0),
            "calibration_applied_before_sampling": sampling.get("calibration_applied_before_sampling", False),
            "structure_source": "ct_2k_sbm_temporal_kde_stubs",
            "checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "seed": int(seed),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        for key in DECOMPOSITION_KEYS:
            metadata[key] = sampling.get(key)
        return metadata


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


def verified_logits_from_scalar(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.stack([-0.5 * values, 0.5 * values], axis=1).astype(np.float32)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def entropy_np(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)


def expected_rating_mean(logits: np.ndarray, rating_values: np.ndarray) -> float:
    return float(np.mean(softmax_np(logits) @ rating_values))


def expected_verified_rate(logits: np.ndarray) -> float:
    return float(np.mean(softmax_np(logits)[:, 1]))


def logit_norm(values: np.ndarray) -> float:
    if values is None or len(values) == 0:
        return 0.0
    return float(np.linalg.norm(values, axis=1).mean())


def verified_scalar(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits, dtype=float)[:, 1] - np.asarray(logits, dtype=float)[:, 0]


def norm_record(
    base_rating,
    residual_rating,
    final_pre_rating,
    final_post_rating,
    base_verified,
    residual_verified,
    final_pre_verified,
    final_post_verified,
) -> Dict[str, float]:
    return {
        "base_rating_norm": logit_norm(base_rating),
        "residual_rating_norm": logit_norm(residual_rating),
        "final_pre_rating_norm": logit_norm(final_pre_rating),
        "final_post_rating_norm": logit_norm(final_post_rating),
        "base_verified_abs": float(np.mean(np.abs(verified_scalar(base_verified)))),
        "residual_verified_abs": float(np.mean(np.abs(verified_scalar(residual_verified)))),
        "final_pre_verified_abs": float(np.mean(np.abs(verified_scalar(final_pre_verified)))),
        "final_post_verified_abs": float(np.mean(np.abs(verified_scalar(final_post_verified)))),
        "pre_rating_entropy": float(np.mean(entropy_np(softmax_np(final_pre_rating)))),
        "post_rating_entropy": float(np.mean(entropy_np(softmax_np(final_post_rating)))),
    }


def build_decomposition_summary(
    norm_records,
    calibration_rows,
    product_effects: pd.DataFrame,
    customer_effects: pd.DataFrame,
    rating_values: List[Any],
    temporal_prior: TemporalAttributePrior,
    timestamps: pd.Series,
) -> Dict[str, Any]:
    records = pd.DataFrame(norm_records)
    summary: Dict[str, Any] = {}
    if records.empty:
        for key in DECOMPOSITION_KEYS:
            summary[key] = None
        return summary
    summary["average_norm_base_rating_logits"] = float(records["base_rating_norm"].mean())
    summary["average_norm_residual_rating_logits"] = float(records["residual_rating_norm"].mean())
    summary["average_norm_final_rating_logits_pre_calibration"] = float(records["final_pre_rating_norm"].mean())
    summary["average_norm_final_rating_logits_post_calibration"] = float(records["final_post_rating_norm"].mean())
    summary["residual_to_base_norm_ratio"] = float(
        summary["average_norm_residual_rating_logits"]
        / max(summary["average_norm_base_rating_logits"], 1e-12)
    )
    summary["average_abs_base_verified_logit"] = float(records["base_verified_abs"].mean())
    summary["average_abs_residual_verified_logit"] = float(records["residual_verified_abs"].mean())
    summary["verified_residual_to_base_abs_ratio"] = float(
        summary["average_abs_residual_verified_logit"]
        / max(summary["average_abs_base_verified_logit"], 1e-12)
    )
    product_rating_cols = [col for col in product_effects.columns if col.startswith("sampled_rating_effect_")]
    customer_rating_cols = [col for col in customer_effects.columns if col.startswith("sampled_rating_effect_")]
    summary["sampled_product_rating_effect_variance"] = float(product_effects[product_rating_cols].to_numpy(dtype=float).var()) if product_rating_cols else None
    summary["sampled_customer_rating_effect_variance"] = float(customer_effects[customer_rating_cols].to_numpy(dtype=float).var()) if customer_rating_cols else None
    summary["sampled_product_verified_effect_variance"] = float(pd.to_numeric(product_effects.get("sampled_verified_effect", pd.Series(dtype=float)), errors="coerce").var()) if "sampled_verified_effect" in product_effects.columns else None
    summary["sampled_customer_verified_effect_variance"] = float(pd.to_numeric(customer_effects.get("sampled_verified_effect", pd.Series(dtype=float)), errors="coerce").var()) if "sampled_verified_effect" in customer_effects.columns else None
    summary["sampled_product_effect_variance"] = summary["sampled_product_rating_effect_variance"]
    summary["sampled_customer_effect_variance"] = summary["sampled_customer_rating_effect_variance"]
    temporal_probs = np.exp(temporal_prior.rating_logits_for_timestamps(timestamps))
    temporal_probs = temporal_probs / temporal_probs.sum(axis=1, keepdims=True)
    summary["average_temporal_rating_prior_entropy"] = float(np.mean(entropy_np(temporal_probs)))
    summary["average_model_rating_entropy_pre_calibration"] = float(records["pre_rating_entropy"].mean())
    summary["average_model_rating_entropy_post_calibration"] = float(records["post_rating_entropy"].mean())
    if calibration_rows:
        calibration = pd.DataFrame(calibration_rows)
        summary["temporal_calibration_average_correction_norm"] = float(calibration["rating_correction_norm"].mean())
        summary["temporal_calibration_max_correction_norm"] = float(calibration["rating_correction_norm"].max())
        summary["temporal_calibration_num_groups_calibrated"] = int(len(calibration))
        summary["average_precal_rating_target_js"] = float(calibration["precal_rating_target_js"].mean())
        summary["average_postcal_rating_target_js"] = float(calibration["postcal_rating_target_js"].mean())
        summary["average_precal_verified_target_abs_error"] = float(calibration["precal_verified_target_abs_error"].mean())
        summary["average_postcal_verified_target_abs_error"] = float(calibration["postcal_verified_target_abs_error"].mean())
    else:
        summary["temporal_calibration_average_correction_norm"] = 0.0
        summary["temporal_calibration_max_correction_norm"] = 0.0
        summary["temporal_calibration_num_groups_calibrated"] = 0
        summary["average_precal_rating_target_js"] = None
        summary["average_postcal_rating_target_js"] = None
        summary["average_precal_verified_target_abs_error"] = None
        summary["average_postcal_verified_target_abs_error"] = None
    return summary


def component_curve_row(
    group: pd.DataFrame,
    generated: pd.DataFrame,
    base_components: Dict[str, np.ndarray],
    final_pre_rating: np.ndarray,
    final_post_rating: np.ndarray,
    final_pre_verified: np.ndarray,
    final_post_verified: np.ndarray,
    rating_values: np.ndarray,
    timestamp_col: str,
    rating_col: str,
    verified_col: str,
    temporal_level: str,
) -> Dict[str, Any]:
    month = temporal_bucket(group[timestamp_col], temporal_level).iloc[0]
    return {
        "month": str(month),
        "num_rows": int(len(group)),
        "temporal_only_avg_rating": expected_rating_mean(base_components["temporal_rating"], rating_values),
        "temporal_block_avg_rating": expected_rating_mean(base_components["temporal_block_rating"], rating_values),
        "temporal_block_product_avg_rating": expected_rating_mean(base_components["temporal_block_product_rating"], rating_values),
        "full_base_avg_rating": expected_rating_mean(base_components["base_rating"], rating_values),
        "final_precal_avg_rating": expected_rating_mean(final_pre_rating, rating_values),
        "final_postcal_avg_rating": expected_rating_mean(final_post_rating, rating_values),
        "synthetic_sampled_avg_rating": float(pd.to_numeric(generated[rating_col], errors="coerce").mean()),
        "real_avg_rating": None,
        "temporal_only_verified_rate": expected_verified_rate(base_components["temporal_verified"]),
        "temporal_block_verified_rate": expected_verified_rate(base_components["temporal_block_verified"]),
        "temporal_block_product_verified_rate": expected_verified_rate(base_components["temporal_block_product_verified"]),
        "full_base_verified_rate": expected_verified_rate(base_components["base_verified"]),
        "final_precal_verified_rate": expected_verified_rate(final_pre_verified),
        "final_postcal_verified_rate": expected_verified_rate(final_post_verified),
        "synthetic_sampled_verified_rate": float(normalize_verified(generated[verified_col]).mean()),
        "real_verified_rate": None,
    }


def row_level_logit_records(
    group,
    generated,
    latent_features,
    customer_blocks,
    product_blocks,
    base_rating,
    residual_rating,
    final_pre_rating,
    final_post_rating,
    base_verified,
    residual_verified,
    final_pre_verified,
    final_post_verified,
    customer_id_col,
    product_id_col,
    timestamp_col,
    rating_col,
    verified_col,
    rating_values,
    temporal_level,
) -> List[Dict[str, Any]]:
    records = []
    for row_pos, (idx, row) in enumerate(group.iterrows()):
        customer_id = row[customer_id_col]
        product_id = row[product_id_col]
        customer_block = int(customer_blocks.get(customer_id, -1))
        product_block = int(product_blocks.get(product_id, -1))
        record = {
            "row_id": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "customer_id": customer_id,
            "product_id": product_id,
            "review_time": row[timestamp_col],
            "month": temporal_bucket(pd.Series([row[timestamp_col]]), temporal_level).iloc[0],
            "customer_block": customer_block,
            "product_block": product_block,
            "block_pair": f"{customer_block}:{product_block}",
            "sampled_product_rating_effect_norm": float(
                np.linalg.norm(
                    latent_features.filter(regex=r"^product_rating_effect_").iloc[row_pos].to_numpy(dtype=float)
                )
            ),
            "sampled_customer_rating_effect_norm": float(
                np.linalg.norm(
                    latent_features.filter(regex=r"^customer_rating_effect_").iloc[row_pos].to_numpy(dtype=float)
                )
            ),
            "sampled_product_verified_effect": float(latent_features["product_verified_effect"].iloc[row_pos]),
            "sampled_customer_verified_effect": float(latent_features["customer_verified_effect"].iloc[row_pos]),
            "base_verified_logit": float(verified_scalar(base_verified[[row_pos]])[0]),
            "residual_verified_logit": float(verified_scalar(residual_verified[[row_pos]])[0]),
            "final_precal_verified_logit": float(verified_scalar(final_pre_verified[[row_pos]])[0]),
            "final_postcal_verified_logit": float(verified_scalar(final_post_verified[[row_pos]])[0]),
            "generated_rating": generated.loc[idx, rating_col],
            "generated_verified": generated.loc[idx, verified_col],
        }
        for value_pos, value in enumerate(rating_values):
            suffix = str(value)
            record[f"base_rating_logit_{suffix}"] = float(base_rating[row_pos, value_pos])
            record[f"residual_rating_logit_{suffix}"] = float(residual_rating[row_pos, value_pos])
            record[f"final_precal_rating_logit_{suffix}"] = float(final_pre_rating[row_pos, value_pos])
            record[f"final_postcal_rating_logit_{suffix}"] = float(final_post_rating[row_pos, value_pos])
        records.append(record)
    return records


def write_v3_sampling_diagnostics(
    diagnostics_dir,
    decomposition,
    calibration_rows,
    component_rows,
    row_diagnostics,
    diagnostic_row_sample_size,
    temporal_prior: TemporalAttributePrior,
    synthetic_timestamps: pd.Series,
    sampling_metadata=None,
) -> None:
    diagnostics_dir = Path(diagnostics_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    save_json(diagnostics_dir / "decomposition_diagnostics.json", to_jsonable(decomposition))
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(diagnostics_dir / "temporal_calibration_by_group.csv", index=False)
    component_df = aggregate_component_rows(pd.DataFrame(component_rows))
    component_df.to_csv(diagnostics_dir / "component_curve_by_month.csv", index=False)
    row_df = pd.DataFrame(row_diagnostics)
    if len(row_df) > diagnostic_row_sample_size:
        row_df = row_df.sample(n=diagnostic_row_sample_size, random_state=0)
    row_df.to_csv(diagnostics_dir / "v3_row_level_logit_components_sample.csv", index=False)
    consistency = check_temporal_bucket_consistency(
        temporal_prior, synthetic_timestamps, synthetic_timestamps
    )
    if not consistency["is_consistent"]:
        warnings.warn(
            "Temporal bucket inconsistency detected; inspect temporal_bucket_consistency.json.",
            RuntimeWarning,
        )
    save_json(diagnostics_dir / "temporal_bucket_consistency.json", to_jsonable(consistency))
    sampling_metadata = sampling_metadata or {}
    save_json(
        diagnostics_dir / "temporal_missing_bucket_diagnostics.json",
        to_jsonable(
            {
                "temporal_prior_level_used_for_sampling": sampling_metadata.get("temporal_prior_level_used_for_sampling", temporal_prior.temporal_prior_level),
                "temporal_bucket_format_used_for_sampling": sampling_metadata.get("temporal_bucket_format_used_for_sampling", temporal_prior.bucket_format),
                "temporal_prior_num_buckets": sampling_metadata.get("temporal_prior_num_buckets", len(temporal_prior.per_bucket_rating_distribution)),
                "sampling_num_buckets": sampling_metadata.get("sampling_num_buckets"),
                "num_sampling_buckets_missing_in_prior": sampling_metadata.get("num_sampling_buckets_missing_in_prior"),
                "sampling_buckets_missing_in_prior_examples": sampling_metadata.get("sampling_buckets_missing_in_prior_examples", []),
                "checkpoint_uses_legacy_temporal_buckets": sampling_metadata.get("checkpoint_uses_legacy_temporal_buckets", temporal_prior.uses_legacy_temporal_buckets()),
            }
        ),
    )
    prior_curve, prior_summary = temporal_prior_diagnostics_from_prior(
        temporal_prior, synthetic_timestamps
    )
    prior_curve.to_csv(diagnostics_dir / "temporal_rating_prior_monthly_avg_curve.csv", index=False)
    save_json(diagnostics_dir / "temporal_prior_diagnostics.json", to_jsonable(prior_summary))


def aggregate_component_rows(component_df: pd.DataFrame) -> pd.DataFrame:
    if component_df.empty or "month" not in component_df.columns:
        return component_df
    numeric_cols = [col for col in component_df.columns if col not in {"month"}]
    rows = []
    for month, group in component_df.groupby("month", sort=True):
        weights = group["num_rows"].to_numpy(dtype=float)
        weights = weights / max(weights.sum(), 1.0)
        row = {"month": month, "num_rows": int(group["num_rows"].sum())}
        for col in numeric_cols:
            if col == "num_rows":
                continue
            values = pd.to_numeric(group[col], errors="coerce")
            row[col] = float(np.nansum(values.to_numpy(dtype=float) * weights)) if values.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def temporal_prior_diagnostics_from_prior(
    temporal_prior: TemporalAttributePrior, timestamps: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rating_values = np.asarray([float(value) for value in temporal_prior.rating_values], dtype=float)
    rows = []
    for bucket, dist in temporal_prior.per_bucket_rating_distribution.items():
        probs = np.asarray(dist, dtype=float)
        rows.append(
            {
                "month": bucket,
                "prior_avg_rating": float(probs @ rating_values),
                "prior_verified_rate": float(temporal_prior.per_bucket_verified_rate.get(bucket, temporal_prior.verified_global_rate)),
                "prior_rating_entropy": float(entropy_np(probs.reshape(1, -1))[0]),
                "review_count": int(temporal_prior.bucket_counts.get(bucket, 0)),
            }
        )
    curve = pd.DataFrame(rows).sort_values("month") if rows else pd.DataFrame()
    summary = {
        "prior_monthly_avg_rating_std": float(curve["prior_avg_rating"].std()) if not curve.empty else None,
        "real_monthly_avg_rating_std": None,
        "prior_to_real_monthly_avg_rating_corr": None,
        "prior_monthly_verified_std": float(curve["prior_verified_rate"].std()) if not curve.empty else None,
        "real_monthly_verified_std": None,
        "prior_to_real_monthly_verified_corr": None,
        "temporal_prior_level": temporal_prior.temporal_prior_level,
        "prior_curve_month_format": temporal_prior.bucket_format,
        "temporal_prior_num_buckets": temporal_prior.num_buckets,
        "synthetic_bucket_examples": sorted(set(temporal_bucket(timestamps, temporal_prior.temporal_prior_level)))[:5],
    }
    return curve, summary


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
