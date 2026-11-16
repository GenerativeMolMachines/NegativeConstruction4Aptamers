"""Secondary-structure helpers for PPAI-style and AptaBLE filters."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from contrastive_path import ensure_contrastive_on_path

try:
    ensure_contrastive_on_path()
    from aptamer_structure import (  # type: ignore
        HAS_VIENNARNA,
        compute_structure_features,
        evaluate_secondary_structure,
        passes_secondary_structure_filter,
    )
except Exception:  # pragma: no cover - optional
    HAS_VIENNARNA = False
    compute_structure_features = None  # type: ignore
    evaluate_secondary_structure = None  # type: ignore
    passes_secondary_structure_filter = None  # type: ignore


def _simple_features(seq: str) -> dict[str, Any]:
    s = str(seq).strip().upper().replace("T", "U")
    n = max(len(s), 1)
    gc = (s.count("G") + s.count("C")) / n
    # crude stem proxy: complementary GC/AU runs in first vs last quarter
    left, right = s[: n // 4], s[-(n // 4) :][::-1]
    complement = {"A": "U", "U": "A", "G": "C", "C": "G"}
    pairs = sum(complement.get(a) == b for a, b in zip(left, right))
    paired_frac = min(1.0, 2 * pairs / n)
    return {
        "sequence_length": n,
        "mfe": -0.5 * pairs,
        "mfe_per_nt": -0.5 * pairs / n,
        "paired_frac": paired_frac,
        "gc_content": gc,
        "backend": "simple",
    }


def structure_features(seq: str) -> dict[str, Any]:
    if compute_structure_features is not None:
        try:
            return compute_structure_features(seq)
        except Exception:
            pass
    return _simple_features(seq)


def is_aptamer_like(seq: str) -> tuple[bool, dict[str, Any]]:
    if passes_secondary_structure_filter is not None:
        try:
            ok, details = passes_secondary_structure_filter(seq)
            if isinstance(details, dict):
                return bool(ok), details
            return bool(ok), structure_features(seq)
        except Exception:
            pass
    feats = structure_features(seq)
    gc = gc_of(seq)
    ok = (
        0.25 <= gc <= 0.75
        and feats.get("paired_frac", 0) >= 0.20
        and feats.get("paired_frac", 1) <= 0.85
        and feats.get("mfe_per_nt", 0) <= -0.02
    )
    feats["reject_reasons"] = [] if ok else ["simple_structure_gate"]
    return ok, feats


def structure_diff_frac(a: str, b: str) -> float:
    """Fraction of positions whose paired/unpaired state differs (AptaBLE-style)."""
    fa = structure_features(a)
    fb = structure_features(b)
    da = str(fa.get("dot_bracket") or "")
    db = str(fb.get("dot_bracket") or "")
    n = min(len(da), len(db))
    if n == 0:
        return 1.0
    return sum(int((da[i] in "()") != (db[i] in "()")) for i in range(n)) / n


def gc_of(seq: str) -> float:
    s = str(seq).upper().replace("U", "T")
    if not s:
        return 0.0
    return (s.count("G") + s.count("C")) / len(s)


@lru_cache(maxsize=8192)
def cached_features(seq: str) -> dict[str, Any]:
    return structure_features(seq)
