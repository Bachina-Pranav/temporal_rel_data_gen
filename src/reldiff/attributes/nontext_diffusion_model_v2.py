"""V2 non-text diffusion model with generative entity latent effects."""

from __future__ import annotations

from .nontext_diffusion_model import TemporalFeatureDiffusionModel


class TemporalFeatureDiffusionModelV2(TemporalFeatureDiffusionModel):
    """Feature-conditioned diffusion model with appended entity latent effects."""

    pass
