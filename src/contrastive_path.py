"""Resolve vendored contrastive code on ``sys.path``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONTRASTIVE = _BENCHMARK_ROOT / "contrastive"


def contrastive_root() -> Path:
    env = os.environ.get("CONTRASTIVE_CODE_ROOT")
    if env:
        return Path(env)
    return _DEFAULT_CONTRASTIVE


def ensure_contrastive_on_path() -> Path:
    root = contrastive_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Contrastive code not found at {root}. "
            "Expected vendored copy in contrastive/ or set CONTRASTIVE_CODE_ROOT."
        )
    root_s = str(root.resolve())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
