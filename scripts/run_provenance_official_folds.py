#!/usr/bin/env python3
"""Source holdout inside the official AptaBench folds, with a hard leak fence.

The global leave-one-source-out split is not this experiment. Here the test
pairs are the official held-out fold, restricted to one database source.
Training rows are the official training fold without that source and without
any aptamer or molecule that appears in that test slice. Synthetic negatives
are built only from that filtered train pool and are dropped if they recreate
a held-out entity, a test pair, or a known binder. A cell with any remaining
overlap is not scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import known_pairs, load_official_indexed_pairs, normalize_sequence  # noqa: E402
from ere_lgbm import _eval_model, _fit_model, _labeled_eval, make_folds  # noqa: E402
from negative_dataset_controls import build_controlled_negative_dataset  # noqa: E402
from provenance_splits import official_fold_source_holdouts  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "provenance_official_folds_v1"
METHODS = ("experimental", "random_sequence", "random_pairing")
SPLIT_TYPES = ("stratified", "aptamer", "molecule")


def _seq_set(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    return set(frame["sequence"].map(normalize_sequence))


def _mol_set(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    return set(frame["canonical_smiles"].astype(str).str.strip())


def _pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    if frame.empty:
        return set()
    seq = frame["sequence"].map(normalize_sequence)
    mol = frame["canonical_smiles"].astype(str).str.strip()
    return set(zip(seq, mol))


def _leak_counts(
    frame: pd.DataFrame,
    *,
    test_frame: pd.DataFrame,
    known_pos: set[tuple[str, str]],
    train_pos: pd.DataFrame,
    method: str,
) -> dict[str, int]:
    seqs = _seq_set(frame)
    mols = _mol_set(frame)
    pairs = _pair_set(frame)
    counts = {
        "n_seq_in_test": len(seqs & _seq_set(test_frame)),
        "n_mol_in_test": len(mols & _mol_set(test_frame)),
        "n_exact_test_pairs": len(pairs & _pair_set(test_frame)),
        "n_known_positive_pairs": 0 if method == "experimental" else len(pairs & known_pos),
        "n_seq_outside_pos_pool": 0,
    }
    if method == "random_pairing":
        counts["n_seq_outside_pos_pool"] = len(seqs - _seq_set(train_pos))
    return counts


def _filter_clean(
    frame: pd.DataFrame,
    *,
    test_frame: pd.DataFrame,
    known_pos: set[tuple[str, str]],
    train_pos: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    ban_seq = _seq_set(test_frame)
    ban_mol = _mol_set(test_frame)
    ban_pairs = _pair_set(test_frame)
    pos_pool = _seq_set(train_pos)
    seq = frame["sequence"].map(normalize_sequence)
    mol = frame["canonical_smiles"].astype(str).str.strip()
    keep = []
    for s, m in zip(seq, mol):
        if s in ban_seq or m in ban_mol or (s, m) in ban_pairs:
            keep.append(False)
            continue
        if method != "experimental" and (s, m) in known_pos:
            keep.append(False)
            continue
        if method == "random_pairing" and s not in pos_pool:
            keep.append(False)
            continue
        keep.append(True)
    return frame.loc[keep].reset_index(drop=True)


def _load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    prev = pd.read_csv(path)
    return {
        (str(r.split_type), int(r.fold), str(r.held_out_source), str(r.train_negative))
        for r in prev.itertuples(index=False)
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--splits", default=",".join(SPLIT_TYPES))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df, _ = load_official_indexed_pairs()
    known_pos = known_pairs(df, {1})
    known_pos = {(normalize_sequence(s), m.strip()) for s, m in known_pos}
    split_types = [s.strip() for s in args.splits.split(",") if s.strip()]
    metric_path = args.out / "fold_source_metrics.csv"
    done = _load_done(metric_path)
    rows: list[dict] = []
    if metric_path.exists():
        rows = pd.read_csv(metric_path).to_dict("records")

    audit_rows = []
    for split_type in split_types:
        folds = make_folds(df, split_type, 5, args.seed)
        cells = official_fold_source_holdouts(folds, split_type=split_type)
        for cell in cells:
            audit_rows.append({k: v for k, v in cell.items() if not isinstance(v, pd.DataFrame)})
        if args.audit_only:
            continue
        for cell in cells:
            source = cell["held_out_source"]
            fold = int(cell["fold"])
            if cell["skipped"]:
                print(
                    f"SKIP {split_type} fold{fold} {source}: {cell['skip_reason']} "
                    f"test {cell['n_test_pos']}/{cell['n_test_neg']}",
                    flush=True,
                )
                continue
            if (
                cell["n_test_molecules_in_train"]
                or cell["n_test_aptamers_in_train"]
                or cell["n_test_pairs_in_train"]
                or cell["n_train_rows_same_source"]
            ):
                raise RuntimeError(f"leak fence failed for {split_type} fold{fold} {source}")
            test_frame = pd.concat([cell["test_pos"], cell["test_neg_exp"]], ignore_index=True)
            test = _labeled_eval(cell["test_pos"], cell["test_neg_exp"])
            rng = np.random.default_rng(args.seed + fold)
            for method in METHODS:
                key = (split_type, fold, source, method)
                if key in done:
                    print(f"RESUME skip {split_type} fold{fold} {source} {method}", flush=True)
                    continue
                neg = build_controlled_negative_dataset(
                    method,
                    positives=cell["train_pos"],
                    reference_negatives=cell["train_neg_exp"],
                    full_df=cell["train_pos"],
                    rng=rng,
                    target_n=len(cell["train_neg_exp"]),
                    generated_path="",
                )
                clean = _filter_clean(
                    neg.frame,
                    test_frame=test_frame,
                    known_pos=known_pos,
                    train_pos=cell["train_pos"],
                    method=method,
                )
                leaks = _leak_counts(
                    clean,
                    test_frame=test_frame,
                    known_pos=known_pos,
                    train_pos=cell["train_pos"],
                    method=method,
                )
                base = {
                    "split_type": split_type,
                    "fold": fold,
                    "held_out_source": source,
                    "train_negative": method,
                    "n_train_pos": cell["n_train_pos"],
                    "n_train_neg": int(len(clean)),
                    "n_test_pos": cell["n_test_pos"],
                    "n_test_neg": cell["n_test_neg"],
                    "n_test_molecules_in_train": 0,
                    "n_test_aptamers_in_train": 0,
                    **leaks,
                }
                if clean.empty or any(leaks.values()):
                    row = {**base, "skipped": True, "skip_reason": "empty_or_leak_after_filter"}
                    rows.append(row)
                    done.add(key)
                    pd.DataFrame(rows).to_csv(metric_path, index=False)
                    print(f"SKIP {split_type} fold{fold} {source} {method}: filter {leaks}", flush=True)
                    continue
                train = _labeled_eval(cell["train_pos"], clean)
                model, threshold, scaler = _fit_model(train, df, split_type, args.seed + fold)
                metrics = _eval_model(model, test, df, split_type, threshold, scaler)
                row = {
                    **base,
                    "skipped": False,
                    "skip_reason": "",
                    "test_roc_auc": metrics["roc_auc"],
                    "test_pr_auc": metrics["pr_auc"],
                    "test_mcc": metrics["mcc"],
                }
                rows.append(row)
                done.add(key)
                pd.DataFrame(rows).to_csv(metric_path, index=False)
                print(
                    f"{split_type} fold{fold} {source} {method}: "
                    f"AUC={row['test_roc_auc']:.4f} "
                    f"train {row['n_train_pos']}/{row['n_train_neg']} "
                    f"test {row['n_test_pos']}/{row['n_test_neg']}",
                    flush=True,
                )

    pd.DataFrame(audit_rows).to_csv(args.out / "fold_source_audit.csv", index=False)
    meta = {
        "protocol": "official_fold_source_holdout_v1",
        "splits": split_types,
        "methods": list(METHODS),
        "entity_overlap_allowed": False,
        "same_source_in_train_allowed": False,
        "uses_global_leave_one_source_out": False,
        "audit_only": bool(args.audit_only),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out / 'fold_source_audit.csv'}", flush=True)
    if not args.audit_only:
        print(f"Wrote {metric_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
