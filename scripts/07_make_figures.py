#!/usr/bin/env python3
"""Build paper figures from committed result CSVs (standalone)."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_SCRIPT = ROOT / "scripts" / "make_figure_pack.py"


def main() -> None:
    if not FIG_SCRIPT.exists():
        raise FileNotFoundError(f"Missing {FIG_SCRIPT}")
    runpy.run_path(str(FIG_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
