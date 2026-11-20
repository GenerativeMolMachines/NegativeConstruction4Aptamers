#!/usr/bin/env python3
"""Audit the negative pairs written by the main ERE runner."""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_aptabench_indexed_pairs, normalize_sequence
from ere_lgbm import make_folds

KEY = ["sequence", "canonical_smiles"]
METHODS = {"random_sequence", "composition_shuffle", "random_pairing",
           "structure_filtered", "hard_negative", "learned_generated"}
CHECKS = ["heldout_sequences", "heldout_molecules", "test_pairs", "known_positives",
          "outside_train_positive_pool"]


def pairs(frame):
    return set(frame[KEY].itertuples(index=False, name=None))


def audit(pool_root, data):
    full, _ = load_aptabench_indexed_pairs(data)
    known = pairs(full.loc[full.label.eq(1)])
    rows = []
    for split_type in ("random", "aptamer", "molecule"):
        for fold, split in enumerate(make_folds(full, split_type, 5, 42)):
            train = pd.concat([split.train_pos, split.train_neg_exp])
            test = pd.concat([split.test_pos, split.test_neg_exp])
            # Shared entities are allowed unless an entity is exclusive to test.
            held_sequences = set(test.sequence) - set(train.sequence)
            held_molecules = set(test.canonical_smiles) - set(train.canonical_smiles)
            test_pairs = pairs(test)
            train_positive_sequences = set(split.train_pos.sequence)
            for path in sorted((pool_root / split_type / f"fold{fold}" / "train").glob("*.csv")):
                if path.stem not in METHODS:
                    continue
                neg = pd.read_csv(path)
                neg.sequence = neg.sequence.map(normalize_sequence)
                keys = pairs(neg)
                rows.append(dict(
                    split_type=split_type, fold=fold, method=path.stem, n_rows=len(neg),
                    heldout_sequences=len(set(neg.sequence) & held_sequences),
                    heldout_molecules=len(set(neg.canonical_smiles) & held_molecules),
                    test_pairs=len(keys & test_pairs), known_positives=len(keys & known),
                    outside_train_positive_pool=(
                        len(set(neg.sequence) - train_positive_sequences)
                        if path.stem in {"random_pairing", "hard_negative"} else 0
                    ),
                ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_root", type=Path)
    parser.add_argument("--data", default=ROOT / "data/aptabench/embeddings/aptabench_embeddings.parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.pool_root, args.data)
    if result.empty:
        raise SystemExit("No pools found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.groupby("method")[CHECKS].max().to_string())
    if not args.allow_partial and (
        len(result) != 90 or result.n_rows.eq(0).any() or result[CHECKS].ne(0).any().any()
    ):
        raise SystemExit("Audit failed or incomplete")


if __name__ == "__main__":
    main()
