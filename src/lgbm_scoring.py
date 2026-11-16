"""Bootstrap LightGBM pair scorer for hard-negative mining.

Trains the same ERE feature stack (k-mer / OneHot + CLAMP / ChemBERTa /
Morgan) on train positives + experimental train negatives, then returns
P(binder) for candidate (aptamer, molecule) pairs. Higher score = harder
negative candidate.

Molecule features are cached at fit-time so hard-negative mining stays fast.
"""

from __future__ import annotations

from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aptabench_features import (
    build_aptamer_features,
    build_molecule_features,
    build_pair_features,
    feature_spec_for_split,
)
from data import normalize_sequence

HARD_SCORE_METHODS = frozenset({"hard_negative", "learned_real_hard_negative"})


def _labeled_pairs(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    cols = ["sequence", "canonical_smiles"] + (["type"] if "type" in pos.columns else [])
    p = pos.loc[:, cols].copy()
    p["sequence"] = p["sequence"].map(normalize_sequence)
    p["label"] = 1
    n = neg.loc[:, cols].copy()
    n["sequence"] = n["sequence"].map(normalize_sequence)
    n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


class BootstrapLgbmScorer:
    """Fold-local LGBM used to rank hard-negative candidates."""

    def __init__(self) -> None:
        self.model: lgb.LGBMClassifier | None = None
        self.scaler: StandardScaler | None = None
        self.reference_df: pd.DataFrame | None = None
        self.split_type: str | None = None
        self._mol_cache: dict[str, np.ndarray] = {}
        self._apt_cache: dict[str, np.ndarray] = {}
        self._apt_encoder: str | None = None
        self._mol_encoder: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.reference_df is not None and self.split_type is not None

    def fit(
        self,
        train_pos: pd.DataFrame,
        train_neg: pd.DataFrame,
        reference_df: pd.DataFrame,
        split_type: str,
        seed: int,
    ) -> BootstrapLgbmScorer:
        if train_pos.empty or train_neg.empty:
            raise ValueError("Bootstrap LGBM scorer needs non-empty train positives and negatives")
        frame = _labeled_pairs(train_pos, train_neg)
        x, _ = build_pair_features(frame, reference_df, split_type)
        y = frame["label"].to_numpy(dtype=int)
        if len(set(y)) < 2:
            raise ValueError("Bootstrap LGBM scorer needs both classes in train fold")
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(x, y)
        self.reference_df = reference_df
        self.split_type = split_type
        spec = feature_spec_for_split(split_type)
        self._apt_encoder = spec.aptamer_encoder
        self._mol_encoder = spec.molecule_encoder
        self._mol_cache = {}
        self._apt_cache = {}
        self._warm_molecule_cache(reference_df)
        return self

    def _warm_molecule_cache(self, reference_df: pd.DataFrame) -> None:
        assert self._mol_encoder is not None
        smiles = reference_df["canonical_smiles"].astype(str).str.strip().unique().tolist()
        mol_x = build_molecule_features(
            smiles,
            reference_df,
            self._mol_encoder,
            morgan_bits=2048,
            morgan_radius=2,
        )
        for smi, vec in zip(smiles, mol_x):
            self._mol_cache[smi] = np.asarray(vec, dtype=np.float32)

    def _aptamer_matrix(self, sequences: list[str]) -> np.ndarray:
        assert self._apt_encoder is not None
        seqs = [normalize_sequence(s) for s in sequences]
        missing = [s for s in seqs if s not in self._apt_cache]
        if missing:
            uniq = list(dict.fromkeys(missing))
            mats = build_aptamer_features(uniq, self._apt_encoder, max_len=200)
            for seq, vec in zip(uniq, mats):
                self._apt_cache[seq] = np.asarray(vec, dtype=np.float32)
        return np.stack([self._apt_cache[s] for s in seqs]).astype(np.float32)

    def _molecule_matrix(self, smiles: list[str]) -> np.ndarray:
        assert self.reference_df is not None
        assert self._mol_encoder is not None
        keys = [str(s).strip() for s in smiles]
        missing = [s for s in keys if s not in self._mol_cache]
        if missing:
            uniq = list(dict.fromkeys(missing))
            mats = build_molecule_features(
                uniq,
                self.reference_df,
                self._mol_encoder,
                morgan_bits=2048,
                morgan_radius=2,
            )
            for smi, vec in zip(uniq, mats):
                self._mol_cache[smi] = np.asarray(vec, dtype=np.float32)
        return np.stack([self._mol_cache[s] for s in keys]).astype(np.float32)

    def score(self, sequence: str, smiles: str) -> float:
        scores = self.score_many([sequence], smiles)
        return float(scores[0]) if len(scores) else float("nan")

    def score_many(self, sequences: list[str], smiles: str) -> np.ndarray:
        if not self.ready or not sequences:
            return np.array([], dtype=np.float32)
        smi = str(smiles).strip()
        frame = pd.DataFrame(
            {
                "sequence": [normalize_sequence(s) for s in sequences],
                "canonical_smiles": [smi] * len(sequences),
            }
        )
        return self.score_table(frame)

    def score_table(self, pairs: pd.DataFrame) -> np.ndarray:
        if not self.ready or pairs.empty:
            return np.array([], dtype=np.float32)
        assert self.model is not None
        seqs = pairs["sequence"].astype(str).tolist()
        smiles = pairs["canonical_smiles"].astype(str).tolist()
        apt_x = self._aptamer_matrix(seqs)
        mol_x = self._molecule_matrix(smiles)
        x = np.concatenate([apt_x, mol_x], axis=1).astype(np.float32)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return self.model.predict_proba(x)[:, 1].astype(np.float32)

    def as_score_fn(self) -> Callable[[str, str], float]:
        return self.score


def build_bootstrap_lgbm_scorer(
    train_pos: pd.DataFrame,
    train_neg: pd.DataFrame,
    reference_df: pd.DataFrame,
    split_type: str,
    seed: int,
) -> BootstrapLgbmScorer:
    return BootstrapLgbmScorer().fit(train_pos, train_neg, reference_df, split_type, seed)
