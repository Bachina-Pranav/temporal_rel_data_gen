"""Reusable preprocessing for source-entity-induced interaction benchmarks."""

from .registry import get_adapter, list_datasets
from .download import download_dataset

__all__ = ["download_dataset", "get_adapter", "list_datasets"]
