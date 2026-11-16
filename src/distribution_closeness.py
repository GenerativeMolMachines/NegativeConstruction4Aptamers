"""Distribution closeness metrics for synthetic negatives.

The benchmark treats experimental AptaBench negatives as the reference
distribution. For every synthetic negative set, this module quantifies how
close it is to that reference along sequence, ligand, structure, and optional
interaction-score axes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from aptabench_features import kmer_matrix, morgan_matrix
from data import gc_content, normalize_sequence
from structure import cached_features

try:
    from scipy.stats import wasserstein_distance
except Exception:  # pragma: no cover
    wasserstein_distance = None


def distribution_closeness(
    candidate: pd.DataFrame,
    experimental: pd.DataFrame,
    *,
    method: str,
    score_fn: Callable[[str, str], float] | None = None,
    positives: pd.DataFrame | None = None,
    sample_n: int = 1000,
    seed: int = 42,
) -> dict:
    """Return numeric distances from candidate negatives to experimental negatives."""
    cand = _sample(candidate, sample_n, seed)
    exp = _sample(experimental, sample_n, seed + 1)
    cseq = cand["sequence"].astype(str).map(normalize_sequence)
    eseq = exp["sequence"].astype(str).map(normalize_sequence)

    ckmer = kmer_matrix(cseq.tolist(), k=4)
    ekmer = kmer_matrix(eseq.tolist(), k=4)
    c_basic = _basic_sequence_features(cand)
    e_basic = _basic_sequence_features(exp)
    c_struct = _structure_matrix(cseq.tolist())
    e_struct = _structure_matrix(eseq.tolist())

    out = {
        "method": method,
        "n_candidate": int(len(candidate)),
        "n_experimental": int(len(experimental)),
        "n_compared_candidate": int(len(cand)),
        "n_compared_experimental": int(len(exp)),
        "length_wasserstein": _wasserstein(cseq.map(len).to_numpy(), eseq.map(len).to_numpy()),
        "gc_wasserstein": _wasserstein(cseq.map(gc_content).to_numpy(), eseq.map(gc_content).to_numpy()),
        "dna_fraction_abs_delta": abs(_dna_fraction(cand) - _dna_fraction(exp)),
        "kmer_centroid_l2": _centroid_l2(ckmer, ekmer),
        "kmer_mmd_rbf": _mmd_rbf(ckmer, ekmer),
        "sequence_c2st_auc": _two_sample_auc(np.hstack([c_basic, ckmer]), np.hstack([e_basic, ekmer]), seed),
        "mfe_per_nt_wasserstein": _wasserstein(c_struct[:, 0], e_struct[:, 0]),
        "paired_frac_wasserstein": _wasserstein(c_struct[:, 1], e_struct[:, 1]),
        "mean_nearest_exp_kmer_jaccard": _mean_nearest_jaccard(cseq.tolist(), eseq.tolist()),
        "ligand_jaccard": _ligand_jaccard(cand, exp),
        "candidate_ligand_in_experimental_fraction": _ligand_in_reference_fraction(cand, exp),
    }

    if positives is not None and not positives.empty:
        pseq = positives["sequence"].astype(str).map(normalize_sequence)
        p_sample = pseq.sample(n=min(len(pseq), sample_n), random_state=seed + 2).tolist() if len(pseq) else []
        c_pos = _nearest_jaccard_array(cseq.tolist(), p_sample)
        e_pos = _nearest_jaccard_array(eseq.tolist(), p_sample)
        out.update(
            {
                "mean_nearest_pos_kmer_jaccard": float(np.mean(c_pos)) if len(c_pos) else None,
                "pos_sim_wasserstein": _wasserstein(c_pos, e_pos),
            }
        )
    else:
        out.update({"mean_nearest_pos_kmer_jaccard": None, "pos_sim_wasserstein": None})

    mol_metrics = _molecule_metrics(cand, exp)
    out.update(mol_metrics)

    if score_fn is not None:
        cscore = _pair_scores(cand, score_fn)
        escore = _pair_scores(exp, score_fn)
        out.update(
            {
                "interaction_score_mean_delta": _mean_delta(cscore, escore),
                "interaction_score_wasserstein": _wasserstein(cscore, escore),
            }
        )
    else:
        out.update({"interaction_score_mean_delta": None, "interaction_score_wasserstein": None})

    return out


def _sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def _basic_sequence_features(frame: pd.DataFrame) -> np.ndarray:
    seq = frame["sequence"].astype(str).map(normalize_sequence)
    length = seq.map(len).to_numpy(dtype=np.float32)
    gc = seq.map(gc_content).to_numpy(dtype=np.float32)
    dna = (frame.get("type", pd.Series(["DNA"] * len(frame))).astype(str).str.upper() == "DNA").to_numpy(dtype=np.float32)
    if len(length):
        length = length / max(float(np.percentile(length, 95)), 1.0)
    return np.vstack([length, gc, dna]).T.astype(np.float32)


def _structure_matrix(sequences: list[str]) -> np.ndarray:
    rows = []
    for seq in sequences:
        feats = cached_features(seq)
        rows.append(
            [
                float(feats.get("mfe_per_nt", np.nan)),
                float(feats.get("paired_frac", np.nan)),
            ]
        )
    arr = np.asarray(rows, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return None
    if wasserstein_distance is not None:
        return float(wasserstein_distance(a, b))
    return float(abs(np.mean(a) - np.mean(b)))


def _centroid_l2(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def _mmd_rbf(a: np.ndarray, b: np.ndarray, *, max_n: int = 500) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    a = a[:max_n].astype(np.float32)
    b = b[:max_n].astype(np.float32)
    pooled = np.vstack([a, b])
    gamma = 1.0 / max(float(np.median(_sq_dists(pooled[: min(len(pooled), 300)], pooled[: min(len(pooled), 300)]))), 1e-6)
    kxx = np.exp(-gamma * _sq_dists(a, a)).mean()
    kyy = np.exp(-gamma * _sq_dists(b, b)).mean()
    kxy = np.exp(-gamma * _sq_dists(a, b)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)


def _two_sample_auc(cand_x: np.ndarray, exp_x: np.ndarray, seed: int) -> float | None:
    if len(cand_x) < 20 or len(exp_x) < 20:
        return None
    x = np.vstack([cand_x, exp_x]).astype(np.float32)
    y = np.r_[np.zeros(len(cand_x), dtype=int), np.ones(len(exp_x), dtype=int)]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=500, solver="liblinear", random_state=seed)
    clf.fit(x_train, y_train)
    score = clf.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, score)
    return float(max(auc, 1.0 - auc))


def _dna_fraction(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    typ = frame.get("type", pd.Series(["DNA"] * len(frame))).astype(str).str.upper()
    return float((typ == "DNA").mean())


def _ligand_jaccard(candidate: pd.DataFrame, experimental: pd.DataFrame) -> float:
    c = set(candidate["canonical_smiles"].astype(str))
    e = set(experimental["canonical_smiles"].astype(str))
    return len(c & e) / max(len(c | e), 1)


def _ligand_in_reference_fraction(candidate: pd.DataFrame, experimental: pd.DataFrame) -> float:
    ref = set(experimental["canonical_smiles"].astype(str))
    return float(candidate["canonical_smiles"].astype(str).isin(ref).mean()) if len(candidate) else 0.0


def _molecule_metrics(candidate: pd.DataFrame, experimental: pd.DataFrame) -> dict:
    smiles_c = candidate["canonical_smiles"].astype(str).tolist()
    smiles_e = experimental["canonical_smiles"].astype(str).tolist()
    try:
        cfp = morgan_matrix(smiles_c, n_bits=256, radius=2)
        efp = morgan_matrix(smiles_e, n_bits=256, radius=2)
    except Exception:
        return {"morgan_centroid_l2": None, "molecule_c2st_auc": None}
    return {
        "morgan_centroid_l2": _centroid_l2(cfp, efp),
        "molecule_c2st_auc": _two_sample_auc(cfp, efp, seed=123),
    }


def _pair_scores(frame: pd.DataFrame, score_fn: Callable[[str, str], float]) -> np.ndarray:
    batch_scorer = getattr(score_fn, "__self__", None)
    if batch_scorer is not None and hasattr(batch_scorer, "score_table"):
        return np.asarray(batch_scorer.score_table(frame), dtype=np.float32)
    vals = [
        score_fn(str(seq), str(smi))
        for seq, smi in zip(frame["sequence"], frame["canonical_smiles"])
    ]
    return np.asarray(vals, dtype=np.float32)


def _mean_delta(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return None
    return float(np.mean(a) - np.mean(b))


def _kmer_set(seq: str, k: int = 4) -> set[str]:
    seq = normalize_sequence(seq)
    if len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _nearest_jaccard_array(queries: list[str], pool: list[str], k: int = 4) -> np.ndarray:
    if not queries or not pool:
        return np.asarray([], dtype=np.float32)
    pool_sets = [_kmer_set(s, k) for s in pool]
    vals = []
    for q in queries:
        qk = _kmer_set(q, k)
        if not qk:
            vals.append(0.0)
            continue
        best = 0.0
        for pk in pool_sets:
            union = len(qk | pk)
            if union:
                best = max(best, len(qk & pk) / union)
        vals.append(best)
    return np.asarray(vals, dtype=np.float32)


def _mean_nearest_jaccard(queries: list[str], pool: list[str]) -> float | None:
    arr = _nearest_jaccard_array(queries, pool)
    if len(arr) == 0:
        return None
    return float(np.mean(arr))
