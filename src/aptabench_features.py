"""Feature builders matching the AptaBench Table 13 baseline choices.

The intended mapping is:
- random (Stratified): Kmer4 aptamer + CLAMP molecule features
- molecule-disjoint: OneHot aptamer + MorganFP molecule features
- aptamer-disjoint: Kmer4 aptamer + ChemBERTa molecule features
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from data import embedding_columns, normalize_sequence

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None


SPLIT_FEATURE_SPECS: dict[str, tuple[str, str]] = {
    "random": ("kmer4", "clamp"),
    "molecule": ("onehot", "morgan"),
    "aptamer": ("kmer4", "chemberta"),
}


@dataclass(frozen=True)
class FeatureSpec:
    aptamer_encoder: str
    molecule_encoder: str


def feature_spec_for_split(split_type: str) -> FeatureSpec:
    key = "random" if split_type == "stratified" else split_type
    if key not in SPLIT_FEATURE_SPECS:
        raise ValueError(f"Unknown split type for AptaBench baseline: {split_type}")
    apt, mol = SPLIT_FEATURE_SPECS[key]
    return FeatureSpec(aptamer_encoder=apt, molecule_encoder=mol)


def build_pair_features(
    frame: pd.DataFrame,
    reference_df: pd.DataFrame,
    split_type: str,
    *,
    max_len: int = 200,
    morgan_bits: int = 2048,
    morgan_radius: int = 2,
) -> tuple[np.ndarray, FeatureSpec]:
    """Return X = aptamer features || molecule features for the split's best baseline."""
    spec = feature_spec_for_split(split_type)
    seqs = frame["sequence"].astype(str).tolist()
    smiles = frame["canonical_smiles"].astype(str).tolist()
    apt_x = build_aptamer_features(seqs, spec.aptamer_encoder, max_len=max_len)
    mol_x = build_molecule_features(smiles, reference_df, spec.molecule_encoder, morgan_bits, morgan_radius)
    return np.concatenate([apt_x, mol_x], axis=1).astype(np.float32), spec


def build_aptamer_features(sequences: Iterable[str], encoder: str, *, max_len: int = 200) -> np.ndarray:
    seqs = [normalize_sequence(s).replace("U", "T") for s in sequences]
    if encoder == "kmer4":
        return kmer_matrix(seqs, k=4)
    if encoder == "onehot":
        return onehot_matrix(seqs, max_len=max_len)
    raise ValueError(f"Unsupported aptamer encoder: {encoder}")


def kmer_matrix(sequences: list[str], k: int = 4) -> np.ndarray:
    alphabet = ("A", "C", "G", "T")
    kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    index = {kmer: i for i, kmer in enumerate(kmers)}
    out = np.zeros((len(sequences), len(kmers)), dtype=np.float32)
    for row_i, seq in enumerate(sequences):
        total = max(len(seq) - k + 1, 0)
        if total == 0:
            continue
        for i in range(total):
            j = index.get(seq[i : i + k])
            if j is not None:
                out[row_i, j] += 1.0
        out[row_i] /= max(out[row_i].sum(), 1.0)
    return out


def onehot_matrix(sequences: list[str], *, max_len: int = 200) -> np.ndarray:
    alphabet = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = np.zeros((len(sequences), max_len, 4), dtype=np.float32)
    for row_i, seq in enumerate(sequences):
        for pos_i, base in enumerate(seq[:max_len]):
            j = alphabet.get(base)
            if j is not None:
                out[row_i, pos_i, j] = 1.0
    return out.reshape(len(sequences), max_len * 4)


def build_molecule_features(
    smiles: list[str],
    reference_df: pd.DataFrame,
    encoder: str,
    morgan_bits: int,
    morgan_radius: int,
) -> np.ndarray:
    if encoder == "morgan":
        return morgan_matrix(smiles, n_bits=morgan_bits, radius=morgan_radius)
    if encoder == "chemberta":
        return lookup_embedding_matrix(smiles, reference_df, "canonical_smiles", ("smi_emb_", "chemberta_"))
    if encoder == "clamp":
        return lookup_embedding_matrix(smiles, reference_df, "canonical_smiles", ("clamp_", "clamp_emb_"))
    raise ValueError(f"Unsupported molecule encoder: {encoder}")


def lookup_embedding_matrix(
    keys: list[str],
    reference_df: pd.DataFrame,
    key_col: str,
    prefixes: tuple[str, ...],
) -> np.ndarray:
    cols: list[str] = []
    for prefix in prefixes:
        cols = embedding_columns(reference_df, prefix)
        if cols:
            break
    if not cols:
        raise ValueError(f"Missing feature columns for prefixes: {prefixes}")

    lookup = (
        reference_df.drop_duplicates(key_col)
        .set_index(key_col)[cols]
        .astype(np.float32)
        .to_dict(orient="index")
    )
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise ValueError(f"Missing molecule embeddings for {len(missing)} SMILES; first={missing[0]}")
    return np.stack([np.fromiter(lookup[k].values(), dtype=np.float32) for k in keys]).astype(np.float32)


def morgan_matrix(smiles: list[str], *, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    if Chem is None or AllChem is None or DataStructs is None:
        raise RuntimeError("RDKit is required for MorganFP features")
    rows = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, rows[i])
    return rows
