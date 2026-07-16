"""Shared download entry point for interaction dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DownloadResult
from .registry import get_adapter


def download_dataset(
    dataset: str,
    *,
    raw_root: str | Path = "data/raw",
    force: bool = False,
    verify_only: bool = False,
    archive: str | Path | None = None,
    **kwargs: Any,
) -> DownloadResult:
    adapter = get_adapter(dataset)
    return adapter.download(raw_root, force=force, verify_only=verify_only, archive=archive, **kwargs)
