"""Configure a writable temp directory before importing Torch."""

from __future__ import annotations

import os
from pathlib import Path


def configure_tempdir(anchor: str | Path | None = None) -> Path | None:
    """Set TMPDIR/TEMP/TMP to a repo-local fallback if needed.

    Some server images have `/tmp` missing or unwritable. PyTorch imports dill,
    and dill calls `tempfile.gettempdir()` during import, so this must run before
    importing torch or package modules that import torch.
    """

    for key in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(key)
        if value and _usable(Path(value)):
            _set_all(value)
            return Path(value)
    bases: list[Path] = []
    if anchor is not None:
        bases.append(Path(anchor).resolve())
    bases.append(Path.cwd().resolve())
    candidates: list[Path] = []
    for base in bases:
        candidates.extend([base / "outputs" / "_tmp", base / ".tmp"])
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if _usable(candidate):
                _set_all(str(candidate))
                return candidate
        except OSError:
            continue
    return None


def _set_all(value: str) -> None:
    os.environ["TMPDIR"] = value
    os.environ["TEMP"] = value
    os.environ["TMP"] = value


def _usable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".reldiff_tempdir_test"
        test.write_text("ok", encoding="utf-8")
        try:
            test.unlink()
        except FileNotFoundError:
            pass
        return True
    except OSError:
        return False
