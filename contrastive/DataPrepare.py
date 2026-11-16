import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

DEFAULT_NEGATIVE_RATIO = 16


def _negative_mix_counts(negative_ratio):
    """25% same-molecule, 50% semi-hard global, 25% random cross-molecule."""
    n_same = max(0, int(round(0.25 * negative_ratio)))
    n_semi = max(0, int(round(0.50 * negative_ratio)))
    n_random = max(0, negative_ratio - n_same - n_semi)
    if negative_ratio > 0 and n_same + n_semi + n_random == 0:
        n_random = negative_ratio
    return n_same, n_semi, n_random


def _smiles_column(df):
    for col in df.columns:
        if col.lower() in ('canonical_smiles', 'smiles'):
            return col
    raise ValueError("Need canonical_smiles or smiles column for molecule split")


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        norm = np.linalg.norm(values)
        return values / np.clip(norm, 1e-8, None)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def _sequence_column(df):
    for col in df.columns:
        if col.lower() in ('sequence', 'aptamer'):
            return col
    raise ValueError("Need sequence or aptamer column for pair cleaning/splitting")


def normalize_aptamer_sequence(values):
    """Canonical sequence key used for deduplication and split grouping."""
    return (
        values.astype(str)
        .str.strip()
        .str.upper()
        .str.replace('U', 'T', regex=False)
    )


def clean_pair_dataframe(df):
    """Drop duplicate and contradictory (aptamer, molecule) supervision."""
    clean = df.copy()
    seq_col = _sequence_column(clean)
    smiles_col = _smiles_column(clean)
    clean['label'] = pd.to_numeric(clean['label'], errors='raise').astype(int)
    if not set(clean['label'].unique()).issubset({0, 1}):
        raise ValueError("Contrastive labels must be 0/1")

    clean['_seq_norm'] = normalize_aptamer_sequence(clean[seq_col])
    clean['_mol_norm'] = clean[smiles_col].astype(str).str.strip()
    raw_rows = len(clean)
    exact_duplicates = int(
        clean.duplicated(['_seq_norm', '_mol_norm', 'label']).sum()
    )
    clean = clean.drop_duplicates(
        ['_seq_norm', '_mol_norm', 'label'], keep='first'
    ).reset_index(drop=True)

    label_counts = clean.groupby(
        ['_seq_norm', '_mol_norm']
    )['label'].transform('nunique')
    conflicting_rows = int((label_counts > 1).sum())
    conflicting_pairs = int(
        clean.loc[label_counts > 1, ['_seq_norm', '_mol_norm']]
        .drop_duplicates()
        .shape[0]
    )
    clean = clean.loc[label_counts == 1].reset_index(drop=True)
    stats = {
        'raw_rows': int(raw_rows),
        'exact_duplicates_removed': exact_duplicates,
        'conflicting_pairs_removed': conflicting_pairs,
        'conflicting_rows_removed': conflicting_rows,
        'clean_rows': int(len(clean)),
        'clean_pos': int((clean['label'] == 1).sum()),
        'clean_neg': int((clean['label'] == 0).sum()),
    }
    return clean.drop(columns=['_seq_norm', '_mol_norm']), stats


def _group_disjoint_split(df, group_keys, seed=42, train_frac=0.7, val_frac=0.15):
    """Split all groups while preserving positive-bearing groups per fold."""
    keys = pd.Series(group_keys, index=df.index).astype(str)
    pos_mask = df['label'].to_numpy() == 1
    pos_keys = keys[pos_mask].unique()
    all_keys = keys.unique()
    neg_only_keys = np.setdiff1d(all_keys, pos_keys, assume_unique=False)
    rng = np.random.default_rng(seed)

    def partition(values):
        values = np.asarray(values, dtype=object)
        values = values[rng.permutation(len(values))]
        train_end = int(train_frac * len(values))
        val_end = int((train_frac + val_frac) * len(values))
        return (
            set(values[:train_end]),
            set(values[train_end:val_end]),
            set(values[val_end:]),
        )

    pos_parts = partition(pos_keys)
    neg_only_parts = partition(neg_only_keys)
    fold_keys = tuple(pos_parts[i] | neg_only_parts[i] for i in range(3))
    pos_group_keys = keys[pos_mask].to_numpy()
    neg_group_keys = keys[~pos_mask].to_numpy()
    pos_indices = tuple(
        np.where(np.isin(pos_group_keys, list(fold)))[0] for fold in fold_keys
    )
    neg_indices = tuple(
        np.where(np.isin(neg_group_keys, list(fold)))[0] for fold in fold_keys
    )
    assigned = sum(map(len, pos_indices)) + sum(map(len, neg_indices))
    if assigned != len(df):
        raise RuntimeError(f"Split dropped {len(df) - assigned} rows")
    return pos_indices, neg_indices, fold_keys, len(neg_only_keys)


def molecule_disjoint_split(df, seed=42, train_frac=0.7, val_frac=0.15):
    """
    Split positives and negatives by molecule (canonical_smiles).
    All pairs for a molecule land in the same fold — no train/val leakage.
    """
    smiles_col = _smiles_column(df)

    pos_idx, neg_idx, folds, n_neg_only = _group_disjoint_split(
        df,
        df[smiles_col].astype(str).str.strip(),
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    train_pos_idx, val_pos_idx, test_pos_idx = pos_idx
    train_neg_idx, val_neg_idx, test_neg_idx = neg_idx
    train_smiles, val_smiles, test_smiles = folds

    stats = {
        'split_mode': 'molecule',
        'train_molecules': len(train_smiles),
        'val_molecules': len(val_smiles),
        'test_molecules': len(test_smiles),
        'negative_only_molecules': int(n_neg_only),
        'train_pos': len(train_pos_idx),
        'val_pos': len(val_pos_idx),
        'test_pos': len(test_pos_idx),
        'train_neg': len(train_neg_idx),
        'val_neg': len(val_neg_idx),
        'test_neg': len(test_neg_idx),
    }
    return (
        train_pos_idx, val_pos_idx, test_pos_idx,
        train_neg_idx, val_neg_idx, test_neg_idx,
        stats,
    )


def aptamer_disjoint_split(df, seed=42, train_frac=0.7, val_frac=0.15):
    """Sequence-disjoint benchmark; molecules may occur in multiple folds."""
    seq_col = _sequence_column(df)
    pos_idx, neg_idx, folds, n_neg_only = _group_disjoint_split(
        df,
        normalize_aptamer_sequence(df[seq_col]),
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    train_pos_idx, val_pos_idx, test_pos_idx = pos_idx
    train_neg_idx, val_neg_idx, test_neg_idx = neg_idx
    stats = {
        'split_mode': 'aptamer',
        'train_aptamers': len(folds[0]),
        'val_aptamers': len(folds[1]),
        'test_aptamers': len(folds[2]),
        'negative_only_aptamers': int(n_neg_only),
        'train_pos': len(train_pos_idx),
        'val_pos': len(val_pos_idx),
        'test_pos': len(test_pos_idx),
        'train_neg': len(train_neg_idx),
        'val_neg': len(val_neg_idx),
        'test_neg': len(test_neg_idx),
    }
    return (
        train_pos_idx, val_pos_idx, test_pos_idx,
        train_neg_idx, val_neg_idx, test_neg_idx,
        stats,
    )


def _mol_group_key(smi_vec, mol_key=None):
    if mol_key is not None:
        return str(mol_key)
    return tuple(np.asarray(smi_vec, dtype=np.float32).round(6).tolist())


class FinalContrastiveDataset(Dataset):
    def __init__(
        self,
        apt_pos,
        smi_pos,
        apt_neg,
        smi_neg,
        mol_keys_pos=None,
        mol_keys_neg=None,
        negative_ratio=DEFAULT_NEGATIVE_RATIO,
        semi_hard_low_pct=0.10,
        semi_hard_high_pct=0.40,
        training=False,
        online_candidate_multiplier=4,
        eval_seed=42,
    ):
        self.smi_to_pos = defaultdict(list)
        self.smi_to_neg = defaultdict(list)
        self.key_to_smi = {}

        for i, (apt, smi) in enumerate(zip(apt_pos, smi_pos)):
            mk = _mol_group_key(smi, mol_keys_pos[i] if mol_keys_pos is not None else None)
            self.smi_to_pos[mk].append(apt)
            self.key_to_smi.setdefault(mk, np.asarray(smi, dtype=np.float32))

        for i, (apt, smi) in enumerate(zip(apt_neg, smi_neg)):
            mk = _mol_group_key(smi, mol_keys_neg[i] if mol_keys_neg is not None else None)
            self.smi_to_neg[mk].append(apt)
            self.key_to_smi.setdefault(mk, np.asarray(smi, dtype=np.float32))

        self.smis = list(self.smi_to_pos.keys())
        self.pos_items = [
            (mk, np.asarray(apt, dtype=np.float32))
            for mk, apts in self.smi_to_pos.items()
            for apt in apts
        ]
        # Always molecule-balanced for loaders: hubs must not dominate train, and
        # retrieval Acc stays comparable (1 positive / molecule). Full pair AUC
        # is measured separately via analyze_fixed_pairs on every labeled row.
        self.sample_by_molecule = True
        self.negative_ratio = negative_ratio
        self.n_same_mol, self.n_semi_hard, self.n_random = _negative_mix_counts(negative_ratio)
        self.semi_hard_low_pct = semi_hard_low_pct
        self.semi_hard_high_pct = semi_hard_high_pct
        self.training = bool(training)
        self.online_candidate_multiplier = max(1, int(online_candidate_multiplier))
        self.eval_seed = int(eval_seed)

        self.global_neg_pool = [np.asarray(a, dtype=np.float32) for a in apt_neg]
        if self.global_neg_pool:
            self.global_neg_norm = _normalize_rows(np.stack(self.global_neg_pool))
        else:
            self.global_neg_norm = np.zeros((0, apt_pos.shape[1]), dtype=np.float32)

        same_mol_with_neg = sum(1 for s in self.smis if self.smi_to_neg.get(s))
        print("Dataset created:")
        print(f"  Unique molecules: {len(self.smis)}")
        print(f"  Total positive pairs: {len(self.pos_items)}")
        print(f"  Samples / epoch: {len(self)}")
        print(f"  Total negative pairs (split): {len(self.global_neg_pool)}")
        print(f"  Molecules with same-mol negatives: {same_mol_with_neg} ({100 * same_mol_with_neg / max(len(self.smis), 1):.1f}%)")
        print(
            f"  Negative mix per sample: {self.n_same_mol} same-mol + "
            f"{self.n_semi_hard} semi-hard + {self.n_random} random "
            f"(total={negative_ratio})"
        )
        if self.training:
            print("  Indexing mode: molecule-balanced train (1 random pos / molecule)")
        else:
            print("  Indexing mode: molecule-balanced eval (1 fixed pos / molecule)")

    def __len__(self):
        return len(self.smis)

    def _choose_indices(self, pool_size, n, exclude, replace=False):
        if pool_size <= 0 or n <= 0:
            return []
        available = [i for i in range(pool_size) if i not in exclude]
        if not available:
            available = list(range(pool_size))
        if len(available) >= n and not replace:
            return np.random.choice(available, size=n, replace=False).tolist()
        return np.random.choice(available, size=n, replace=True).tolist()

    def _sample_semi_hard_from_pool(self, positive_vec, n, pool=None, pool_norm=None, exclude=None):
        if pool is None:
            pool = self.global_neg_pool
        if pool_norm is None:
            pool_norm = self.global_neg_norm
        if not pool or n <= 0:
            return []

        exclude = set(exclude or [])
        pos_norm = _normalize_rows(positive_vec)
        sims = pool_norm @ pos_norm
        ranked = np.argsort(-sims)

        pool_size = len(pool)
        lo = int(self.semi_hard_low_pct * pool_size)
        hi = max(lo + 1, int(self.semi_hard_high_pct * pool_size))
        band = [i for i in ranked[lo:hi] if i not in exclude]

        if len(band) < n:
            extra = [i for i in ranked[:lo] if i not in exclude]
            band = band + extra

        if not band:
            band = [i for i in ranked if i not in exclude]
        if not band:
            band = list(range(pool_size))

        chosen = np.random.choice(band, size=n, replace=len(band) < n).tolist()
        return [pool[i] for i in chosen]

    def _sample_random_from_pool(self, n, pool=None, exclude=None):
        if pool is None:
            pool = self.global_neg_pool
        if not pool or n <= 0:
            return []
        exclude = set(exclude or [])
        chosen = self._choose_indices(len(pool), n, exclude, replace=True)
        return [pool[i] for i in chosen]

    def _sample_same_mol_negatives(self, neg_list, positive_vec, n):
        if n <= 0 or not neg_list:
            return []

        neg_arr = np.stack([np.asarray(a, dtype=np.float32) for a in neg_list])
        neg_norm = _normalize_rows(neg_arr)
        pos_norm = _normalize_rows(positive_vec)
        sims = neg_norm @ pos_norm
        ranked = np.argsort(-sims)

        lo = max(0, int(0.05 * len(neg_list)))
        hi = max(lo + 1, int(0.50 * len(neg_list)))
        band = ranked[lo:hi].tolist()
        if len(band) < n:
            band = ranked[: max(n, hi)].tolist()

        chosen = np.random.choice(band, size=n, replace=len(band) < n).tolist()
        return [neg_list[i] for i in chosen]

    def _sample_negatives_mixed(self, mol_key, positive_apts_np):
        neg_list = self.smi_to_neg.get(mol_key, [])
        n_same = self.n_same_mol if neg_list else 0
        n_semi = self.n_semi_hard
        n_random = self.n_random + max(0, self.n_same_mol - n_same)

        negatives = []
        used_pool_idx = set()

        if n_same > 0:
            negatives.extend(self._sample_same_mol_negatives(neg_list, positive_apts_np, n_same))

        if n_semi > 0:
            semi = self._sample_semi_hard_from_pool(
                positive_apts_np, n_semi, exclude=used_pool_idx
            )
            negatives.extend(semi)

        if n_random > 0:
            negatives.extend(self._sample_random_from_pool(n_random, exclude=used_pool_idx))

        if len(negatives) < self.negative_ratio:
            extra = self._sample_semi_hard_from_pool(
                positive_apts_np,
                self.negative_ratio - len(negatives),
            )
            negatives.extend(extra)

        return negatives[: self.negative_ratio]

    def _sample_online_candidates(self, mol_key):
        """Candidate pool for learned mol–apt mining in FinalTrainer."""
        local = [
            np.asarray(a, dtype=np.float32)
            for a in self.smi_to_neg.get(mol_key, [])
        ]
        n_same_candidates = min(
            len(local),
            max(self.n_same_mol, self.n_same_mol * self.online_candidate_multiplier),
        )
        if n_same_candidates:
            local_idx = np.random.choice(
                len(local), size=n_same_candidates, replace=False
            )
            candidates = [local[i] for i in local_idx]
        else:
            candidates = []

        total_candidates = self.negative_ratio * self.online_candidate_multiplier
        n_global = total_candidates - len(candidates)
        candidates.extend(self._sample_random_from_pool(n_global))
        while len(candidates) < self.negative_ratio:
            candidates.extend(
                self._sample_random_from_pool(self.negative_ratio - len(candidates))
            )
        return candidates, n_same_candidates

    def _deterministic_eval_item(self, idx):
        """Fixed positive and fixed negative set for reproducible val/test."""
        mol_key = self.smis[idx]
        positive = np.asarray(self.smi_to_pos[mol_key][0], dtype=np.float32)
        local = [
            np.asarray(a, dtype=np.float32)
            for a in self.smi_to_neg.get(mol_key, [])
        ]
        rng = np.random.default_rng(self.eval_seed + idx)
        negatives = []
        if local:
            n_local = min(self.n_same_mol, len(local))
            chosen = rng.choice(len(local), size=n_local, replace=False)
            negatives.extend(local[i] for i in chosen)
        n_needed = self.negative_ratio - len(negatives)
        if n_needed > 0:
            if not self.global_neg_pool:
                raise ValueError("Cannot evaluate without negative pairs")
            chosen = rng.choice(
                len(self.global_neg_pool),
                size=n_needed,
                replace=len(self.global_neg_pool) < n_needed,
            )
            negatives.extend(self.global_neg_pool[i] for i in chosen)
        return mol_key, positive, negatives[:self.negative_ratio]

    def get_negatives_for_molecule(self, mol_key, max_local=50, max_global=50):
        """Negatives for generation / clustering: local + semi-hard/random global."""
        local = [np.asarray(a, dtype=np.float32) for a in self.smi_to_neg.get(mol_key, [])]
        local_arr = local[:max_local]

        n_needed = max(0, max_global - len(local_arr))
        global_arr = []
        if n_needed > 0 and self.global_neg_pool:
            pos_list = self.smi_to_pos.get(mol_key, [])
            ref = _normalize_rows(np.asarray(pos_list[0], dtype=np.float32)) if pos_list else None
            if ref is not None:
                n_semi = max(1, int(round(0.6 * n_needed)))
                n_rand = n_needed - n_semi
                global_arr.extend(self._sample_semi_hard_from_pool(ref, n_semi))
                global_arr.extend(self._sample_random_from_pool(n_rand))
            else:
                global_arr.extend(self._sample_random_from_pool(n_needed))

        combined = local_arr + global_arr
        if not combined and self.global_neg_pool:
            combined = self._sample_random_from_pool(min(max_global, len(self.global_neg_pool)))
        return combined

    def __getitem__(self, idx):
        if self.training:
            # Equal molecule weight: hubs with many aptamers do not dominate.
            smi = self.smis[idx]
            pos_list = self.smi_to_pos[smi]
            pick = int(np.random.randint(0, len(pos_list)))
            positive_apts_np = np.asarray(pos_list[pick], dtype=np.float32)
            negatives, same_mol_count = self._sample_online_candidates(smi)
        else:
            smi, positive_apts_np, negatives = self._deterministic_eval_item(idx)
            same_mol_count = min(
                self.n_same_mol, len(self.smi_to_neg.get(smi, []))
            )

        anchor_smi = torch.FloatTensor(self.key_to_smi[smi])
        positive_apts = torch.FloatTensor(positive_apts_np)
        negative_apts = torch.stack([
            torch.FloatTensor(np.asarray(n, dtype=np.float32)) for n in negatives
        ])

        if self.training and torch.rand(1).item() > 0.5:
            anchor_smi = anchor_smi + torch.randn_like(anchor_smi) * 0.01

        if self.training and torch.rand(1).item() > 0.5:
            positive_apts = positive_apts + torch.randn_like(positive_apts) * 0.01

        return {
            'anchor_smi': anchor_smi,
            'positive_apts': positive_apts,
            'negative_apts': negative_apts,
            'same_mol_count': torch.tensor(same_mol_count, dtype=torch.long),
        }
