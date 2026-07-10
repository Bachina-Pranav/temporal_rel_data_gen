"""Conditional TABDLM-style attribute generator."""

from pathlib import Path

try:
    from tempdir_bootstrap import configure_tempdir

    configure_tempdir(Path.cwd())
except Exception:
    pass

from .schema import ConditionalTABDLMConfig, ConditionalTABDLMSchema, load_config
from .model import ConditionalTABDLM

__all__ = [
    "ConditionalTABDLM",
    "ConditionalTABDLMConfig",
    "ConditionalTABDLMSchema",
    "load_config",
]
