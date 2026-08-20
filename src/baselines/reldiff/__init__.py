"""Thin, schema-driven adapter around the bundled upstream RelDiff code."""

from .schema import RelDiffDatasetConfig, load_dataset_config

__all__ = ["RelDiffDatasetConfig", "load_dataset_config"]

