"""
Secondary-structure utilities for aptamer generation.

Uses ViennaRNA (RNAfold MFE) when available. Thresholds were calibrated on
AptaBench positives (18-80 nt): typical aptamers fold with moderate pairing
and expose a binding loop.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

try:
    import RNA

    HAS_VIENNARNA = True
except ImportError:  # pragma: no cover - optional dependency
    RNA = None
    HAS_VIENNARNA = False


DEFAULT_STRUCTURE_FILTER = {
    # Calibrated vs AptaBench positives (p10-p90).
    "min_mfe_per_nt": -0.12,
    "min_paired_frac": 0.30,
    "max_paired_frac": 0.82,
    "min_unpaired_frac": 0.18,
    "max_unpaired_frac": 0.72,
    "min_loop_len": 3,
    "max_loop_len": 18,
    "min_loop_count": 1,
}


def normalize_to_rna(sequence: str) -> str:
    return str(sequence).strip().upper().replace("T", "U")


def _loop_runs(dot_bracket: str) -> list[int]:
    runs: list[int] = []
    current = 0
    for ch in dot_bracket:
        if ch == ".":
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _stem_pair_count(dot_bracket: str) -> int:
    return dot_bracket.count("(")


@lru_cache(maxsize=4096)
def _fold_rna_cached(rna_sequence: str) -> Tuple[str, float]:
    if not HAS_VIENNARNA:
        raise RuntimeError("ViennaRNA is not installed. Install with: pip install ViennaRNA")
    fc = RNA.fold_compound(rna_sequence)
    structure, mfe = fc.mfe()
    return structure, float(mfe)


def compute_structure_features(sequence: str, molecule_type: str = "dna") -> Dict[str, Any]:
    """Predict MFE structure and return summary features for one aptamer."""
    rna = normalize_to_rna(sequence)
    n = len(rna)
    if n == 0:
        return {
            "sequence_length": 0,
            "mfe": 0.0,
            "mfe_per_nt": 0.0,
            "paired_frac": 0.0,
            "unpaired_frac": 0.0,
            "stem_pairs": 0,
            "loop_runs": [],
            "max_loop_len": 0,
            "valid_loop_count": 0,
            "dot_bracket": "",
            "molecule_type": molecule_type,
            "backend": "none",
        }

    if HAS_VIENNARNA:
        dot_bracket, mfe = _fold_rna_cached(rna)
        backend = "viennarna"
    else:
        dot_bracket, mfe = _fallback_fold(rna)
        backend = "fallback"

    paired_nt = dot_bracket.count("(") + dot_bracket.count(")")
    paired_frac = paired_nt / n
    unpaired_frac = 1.0 - paired_frac
    loop_runs = _loop_runs(dot_bracket)
    valid_loops = [
        run for run in loop_runs
        if DEFAULT_STRUCTURE_FILTER["min_loop_len"] <= run <= DEFAULT_STRUCTURE_FILTER["max_loop_len"]
    ]

    return {
        "sequence_length": n,
        "mfe": float(mfe),
        "mfe_per_nt": float(mfe / n),
        "paired_frac": float(paired_frac),
        "unpaired_frac": float(unpaired_frac),
        "stem_pairs": int(_stem_pair_count(dot_bracket)),
        "loop_runs": loop_runs,
        "max_loop_len": int(max(loop_runs) if loop_runs else 0),
        "valid_loop_count": int(len(valid_loops)),
        "dot_bracket": dot_bracket,
        "molecule_type": molecule_type,
        "backend": backend,
    }


def _fallback_fold(rna: str) -> Tuple[str, float]:
    """
    Very small Nussinov-style fallback when ViennaRNA is unavailable.
    Not as accurate as RNAfold, but keeps the pipeline usable offline.
    """
    n = len(rna)
    pair_scores = {
        ("A", "U"): 2,
        ("U", "A"): 2,
        ("G", "C"): 3,
        ("C", "G"): 3,
        ("G", "U"): 1,
        ("U", "G"): 1,
    }
    dp = [[0] * n for _ in range(n)]
    trace = [[None] * n for _ in range(n)]

    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            best = dp[i + 1][j]
            choice = "skip_left"
            right = dp[i][j - 1]
            if right > best:
                best = right
                choice = "skip_right"
            split_best = 0
            split_k = None
            for k in range(i, j):
                value = dp[i][k] + dp[k + 1][j]
                if value > split_best:
                    split_best = value
                    split_k = k
            if split_best > best:
                best = split_best
                choice = ("split", split_k)
            if j - i >= 3:
                score = pair_scores.get((rna[i], rna[j]), 0) + (dp[i + 1][j - 1] if j - i > 1 else 0)
                if score > best:
                    best = score
                    choice = "pair"
            dp[i][j] = best
            trace[i][j] = choice

    dot = ["."] * n
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        choice = trace[i][j]
        if choice == "pair":
            dot[i] = "("
            dot[j] = ")"
            if j - i > 1:
                stack.append((i + 1, j - 1))
        elif choice == "skip_left":
            stack.append((i + 1, j))
        elif choice == "skip_right":
            stack.append((i, j - 1))
        elif isinstance(choice, tuple) and choice[0] == "split":
            k = choice[1]
            stack.append((k + 1, j))
            stack.append((i, k))

    mfe = -float(dp[0][n - 1]) if n else 0.0
    return "".join(dot), mfe


def evaluate_secondary_structure(
    sequence: str,
    molecule_type: str = "dna",
    *,
    min_mfe_per_nt: float = DEFAULT_STRUCTURE_FILTER["min_mfe_per_nt"],
    min_paired_frac: float = DEFAULT_STRUCTURE_FILTER["min_paired_frac"],
    max_paired_frac: float = DEFAULT_STRUCTURE_FILTER["max_paired_frac"],
    min_unpaired_frac: float = DEFAULT_STRUCTURE_FILTER["min_unpaired_frac"],
    max_unpaired_frac: float = DEFAULT_STRUCTURE_FILTER["max_unpaired_frac"],
    min_loop_len: int = DEFAULT_STRUCTURE_FILTER["min_loop_len"],
    max_loop_len: int = DEFAULT_STRUCTURE_FILTER["max_loop_len"],
    min_loop_count: int = DEFAULT_STRUCTURE_FILTER["min_loop_count"],
    min_stem_pairs: int = 2,
) -> Tuple[bool, float, bool, Dict[str, Any]]:
    """
    Evaluate whether a candidate has a realistic aptamer-like secondary structure.

    Returns:
        passes: bool
        penalty: float (higher = worse; 0 if passes)
        reject: bool (hard reject)
        details: feature dict + rejection reasons
    """
    features = compute_structure_features(sequence, molecule_type=molecule_type)
    details = dict(features)
    reasons: list[str] = []
    penalty = 0.0

    if features["mfe_per_nt"] > min_mfe_per_nt:
        reasons.append("weak_folding")
        penalty += 0.20 * (features["mfe_per_nt"] - min_mfe_per_nt) / max(abs(min_mfe_per_nt), 1e-6)

    if features["paired_frac"] < min_paired_frac:
        reasons.append("too_unpaired")
        penalty += 0.15 * (min_paired_frac - features["paired_frac"])

    if features["paired_frac"] > max_paired_frac:
        reasons.append("too_paired")
        penalty += 0.10 * (features["paired_frac"] - max_paired_frac)

    if features["unpaired_frac"] < min_unpaired_frac:
        reasons.append("no_binding_loop")
        penalty += 0.15 * (min_unpaired_frac - features["unpaired_frac"])

    if features["unpaired_frac"] > max_unpaired_frac:
        reasons.append("unstructured")
        penalty += 0.20 * (features["unpaired_frac"] - max_unpaired_frac)

    if features["valid_loop_count"] < min_loop_count:
        reasons.append("missing_hairpin_loop")
        penalty += 0.15

    if features["stem_pairs"] < min_stem_pairs:
        reasons.append("insufficient_stem")
        penalty += 0.10

    if features["max_loop_len"] > max_loop_len and features["valid_loop_count"] == 0:
        reasons.append("loop_length_out_of_range")
        penalty += 0.10

    details["reject_reasons"] = reasons
    reject = bool(reasons)
    passes = not reject
    return passes, float(penalty), reject, details


def passes_secondary_structure_filter(
    sequence: str,
    molecule_type: str = "dna",
    **kwargs: Any,
) -> Tuple[bool, Dict[str, Any]]:
    passes, _, _, details = evaluate_secondary_structure(sequence, molecule_type=molecule_type, **kwargs)
    return passes, details


def structure_details_to_json(details: Dict[str, Any]) -> str:
    payload = {
        key: details.get(key)
        for key in (
            "mfe_per_nt",
            "paired_frac",
            "unpaired_frac",
            "stem_pairs",
            "valid_loop_count",
            "max_loop_len",
            "reject_reasons",
            "backend",
        )
    }
    return json.dumps(payload, ensure_ascii=False)
