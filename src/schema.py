"""Canonical method IDs, literature tags, and output columns."""

from __future__ import annotations

from typing import TypedDict


class MethodSpec(TypedDict):
    method: str
    method_name: str
    literature: str
    is_reference: bool
    invents_sequence: bool
    description: str


SYNTHETIC_METHODS: tuple[str, ...] = (
    "random_sequence",
    "composition_shuffle",
    "random_pairing",
    "structure_filtered",
    "hard_negative",
    "learned_generated",
    "edit_based",
    "gru_edit",
)

REFERENCE_METHOD = "experimental"

METHOD_SPECS: dict[str, MethodSpec] = {
    "random_sequence": {
        "method": "random_sequence",
        "method_name": "Random sequence",
        "literature": "standard decoy baseline (Li et al. 2014; AptaNet; many SELEX-ML pipelines)",
        "is_reference": False,
        "invents_sequence": True,
        "description": (
            "Sample a uniform random oligonucleotide of the same length and alphabet "
            "as the positive aptamer. Easy, but ignores composition, structure, and chemistry."
        ),
    },
    "composition_shuffle": {
        "method": "composition_shuffle",
        "method_name": "Composition-preserving shuffle (AptaBLE)",
        "literature": "AptaBLE (bioRxiv 2026): 4 shuffled decoys / positive, same length and nucleotide composition",
        "is_reference": False,
        "invents_sequence": True,
        "description": (
            "Permute the positive sequence. Preserves length and base composition; "
            "optionally reject shuffles whose secondary structure is too similar (AptaBLE ViennaRNA rule)."
        ),
    },
    "random_pairing": {
        "method": "random_pairing",
        "method_name": "Random aptamer-target pairing",
        "literature": (
            "interaction-prediction literature; SMRTnet (Fei et al., Nat Biotechnol 2025) "
            "randomly pairs RNA fragments with small molecules after filtering known interactions; "
            "also Li/AptaTrans/PPAI-style protein–aptamer random pairing"
        ),
        "is_reference": False,
        "invents_sequence": False,
        "description": (
            "Keep real aptamer sequences and real molecules, but rewire pairs. "
            "Drop any pair that is a known interaction (and, by default, known experimental non-binders)."
        ),
    },
    "structure_filtered": {
        "method": "structure_filtered",
        "method_name": "Structure-filtered random negatives (PPAI-style)",
        "literature": "PPAI-style aptamer negatives: random sequences filtered by secondary structure / MFE",
        "is_reference": False,
        "invents_sequence": True,
        "description": (
            "Draw random same-length sequences until they look aptamer-like by MFE, "
            "paired fraction, and hairpin-loop constraints."
        ),
    },
    "hard_negative": {
        "method": "hard_negative",
        "method_name": "Hard-negative sampling",
        "literature": "Robinson et al., Contrastive Learning with Hard Negative Samples, ICLR 2021",
        "is_reference": False,
        "invents_sequence": False,
        "description": (
            "From a pool of plausible non-interacting pairs, keep those the current model "
            "scores as most binder-like (highest contrastive cosine)."
        ),
    },
    "learned_generated": {
        "method": "learned_generated",
        "method_name": "Learned / generated negatives (ours)",
        "literature": "this work: contrastive representation + target-conditioned generation",
        "is_reference": False,
        "invents_sequence": True,
        "description": (
            "Target-conditioned GRU (or equivalent) samples from the contrastive non-binding "
            "region for that molecule. Primary proposed method."
        ),
    },
    "edit_based": {
        "method": "edit_based",
        "method_name": "Edit-based negatives (real seed rewrite)",
        "literature": "this work: edit real aptamers toward non-binding + experimental profile",
        "is_reference": False,
        "invents_sequence": True,
        "description": (
            "Start from a real aptamer not known to bind the ligand; locally rewrite "
            "windows/bases to lower contrastive binding score while matching train-fold "
            "experimental negative length/GC/k-mer profile."
        ),
    },
    "gru_edit": {
        "method": "gru_edit",
        "method_name": "GRU-edit negatives (seed flanks + GRU window)",
        "literature": "this work: ConditionalGRUDecoder infills a window in a real aptamer seed",
        "is_reference": False,
        "invents_sequence": True,
        "description": (
            "Keep flanks of a real aptamer; the target-conditioned GRU rewrites a contiguous "
            "middle window using a negative latent, then rank by experimental profile + non-binding score."
        ),
    },
    "experimental": {
        "method": "experimental",
        "method_name": "Experimental negatives (AptaBench)",
        "literature": "AptaBench reported non-binders / low-affinity pairs",
        "is_reference": True,
        "invents_sequence": False,
        "description": (
            "Held-out experimental non-binders. Not a competing generator: the reference "
            "distribution that synthetic methods should approximate."
        ),
    },
}

OUTPUT_COLUMNS: tuple[str, ...] = (
    "method",
    "method_name",
    "literature",
    "is_reference",
    "label",
    "sequence",
    "canonical_smiles",
    "type",
    "source_positive_idx",
    "pair_rank",
    "length",
    "gc_content",
    "alphabet",
    "contrastive_sim",
    "mfe_per_nt",
    "paired_frac",
    "notes",
)
