#!/usr/bin/env python3
"""Build cross-negative ROC-AUC heatmap table from ERE outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import cross_negative_pivot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cross-csv",
        default=str(ROOT / "results" / "ere_v2" / "cross_negative_metrics.csv"),
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "results" / "ere_v2" / "cross_negative_heatmap.csv"),
    )
    args = p.parse_args()
    pivot = cross_negative_pivot(args.cross_csv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.out)
    print(pivot.round(3).to_string())
    print("Saved", args.out)


if __name__ == "__main__":
    main()
