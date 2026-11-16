"""Build paper figures from committed benchmark result CSVs."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

OFFICIAL6 = [
    "random_sequence",
    "composition_shuffle",
    "random_pairing",
    "structure_filtered",
    "hard_negative",
    "learned_generated",
]
LABELS = {
    "random_sequence": "Random seq",
    "composition_shuffle": "Comp. shuffle",
    "random_pairing": "Random pairing",
    "structure_filtered": "Struct-filtered",
    "hard_negative": "Hard negative",
    "learned_generated": "Learned gen.",
    "experimental": "Experimental",
}
METHOD_ORDER = OFFICIAL6 + ["experimental"]
SPLIT_ORDER = ["random", "aptamer", "molecule"]
SPLIT_TITLE = {
    "random": "Stratified",
    "aptamer": "Aptamer-disjoint",
    "molecule": "Molecule-disjoint",
}
COLORS = {
    "random_sequence": "#aec7e8",
    "composition_shuffle": "#ffbb78",
    "random_pairing": "#1f77b4",
    "structure_filtered": "#9467bd",
    "hard_negative": "#d62728",
    "learned_generated": "#2ca02c",
    "experimental": "#ff7f0e",
}


def _official_merged_crossneg_df() -> pd.DataFrame:
    scaled = ROOT / "results" / "ere_official_scaled_v1" / "cross_negative_metrics.csv"
    v1 = ROOT / "results" / "ere_official_v1" / "cross_negative_metrics.csv"
    if not scaled.exists() or not v1.exists():
        raise FileNotFoundError("Need official_scaled_v1 and official_v1 cross_negative_metrics.csv")
    dfs = []
    v1df = pd.read_csv(v1)
    if "skipped" in v1df.columns:
        v1df = v1df[~v1df["skipped"].fillna(False).astype(bool)]
    dfs.append(v1df[v1df["split_type"] == "random"])
    sdf = pd.read_csv(scaled)
    if "skipped" in sdf.columns:
        sdf = sdf[~sdf["skipped"].fillna(False).astype(bool)]
    dfs.append(sdf[sdf["split_type"].isin(["aptamer", "molecule"])])
    return pd.concat(dfs, ignore_index=True)


def _official_merged_ere_df() -> pd.DataFrame:
    scaled = ROOT / "results" / "ere_official_scaled_v1" / "ere_metrics.csv"
    v1 = ROOT / "results" / "ere_official_v1" / "ere_metrics.csv"
    if not scaled.exists() or not v1.exists():
        raise FileNotFoundError("Need official_scaled_v1 and official_v1 ere_metrics.csv")
    dfs = []
    v1df = pd.read_csv(v1)
    if "skipped" in v1df.columns:
        v1df = v1df[~v1df["skipped"].fillna(False).astype(bool)]
    dfs.append(v1df[v1df["split_type"] == "random"])
    sdf = pd.read_csv(scaled)
    if "skipped" in sdf.columns:
        sdf = sdf[~sdf["skipped"].fillna(False).astype(bool)]
    dfs.append(sdf[sdf["split_type"].isin(["aptamer", "molecule"])])
    return pd.concat(dfs, ignore_index=True)


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.family"] = "DejaVu Sans"


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)


def fig0_pipeline_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.2, 1.8, 0.9, "AptaBench\npositives", "#fff3cd"),
        (2.4, 1.2, 2.2, 0.9, "6 synthetic\nnegative methods", "#d1ecf1"),
        (5.0, 1.2, 2.0, 0.9, "LightGBM\ntrain", "#e2e3e5"),
        (7.4, 1.2, 2.2, 0.9, "Test on\nexperimental negs", "#f8d7da"),
        (10.0, 1.2, 0.8, 0.9, "AUC", "#ffffff"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                linewidth=1.2,
                edgecolor="#333333",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for x0, x1 in [(2.0, 2.4), (4.6, 5.0), (7.0, 7.4), (9.6, 10.0)]:
        ax.add_patch(FancyArrowPatch((x0, 1.65), (x1, 1.65), arrowstyle="->", mutation_scale=12, lw=1.4))
    ax.text(
        5.5,
        0.35,
        "Only train-negative construction changes → different boundaries → different experimental AUC",
        ha="center",
        fontsize=10,
        style="italic",
    )
    fig.suptitle("ERE: controlled negative-construction benchmark", y=0.98)
    _save(fig, "fig0_pipeline_schematic")


def fig_ere_bars() -> None:
    df = _official_merged_ere_df()
    src_label = "official_scaled_v1 + official_v1 (stratified)"
    auc_col = "test_roc_auc_experimental" if "test_roc_auc_experimental" in df.columns else "test_roc_auc"
    g = (
        df[df["train_negative"].isin(METHOD_ORDER)]
        .groupby(["split_type", "train_negative"], as_index=False)
        .agg(auc_mean=(auc_col, "mean"), auc_std=(auc_col, "std"))
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), sharey=True)
    for ax, split in zip(axes, SPLIT_ORDER):
        sub = g[g["split_type"] == split].set_index("train_negative").reindex(METHOD_ORDER)
        x = np.arange(len(METHOD_ORDER))
        colors = [COLORS.get(m, "#888888") for m in METHOD_ORDER]
        ax.bar(
            x,
            sub["auc_mean"],
            yerr=sub["auc_std"].fillna(0),
            color=colors,
            capsize=2,
            width=0.75,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in METHOD_ORDER], rotation=55, ha="right", fontsize=8)
        ax.set_title(SPLIT_TITLE[split])
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="0.65", ls="--", lw=0.8)
    axes[0].set_ylabel("Test ROC-AUC (experimental negatives)")
    fig.suptitle(f"ERE — source: {src_label}", y=1.02)
    _save(fig, "fig1_ere_bars")


def fig_crossneg_heatmaps() -> None:
    df = _official_merged_crossneg_df()
    src_label = "official_scaled_v1 + official_v1 (stratified)"
    train_col = "train_negative" if "train_negative" in df.columns else "train_method"
    test_col = "test_negative" if "test_negative" in df.columns else "test_method"
    value = next(c for c in ("test_roc_auc", "roc_auc", "test_auc") if c in df.columns)
    methods = [m for m in METHOD_ORDER if m in set(df[train_col]) | set(df[test_col])]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    for ax, split in zip(axes, SPLIT_ORDER):
        sub = df[df["split_type"] == split]
        pivot = sub.groupby([train_col, test_col])[value].mean().unstack()
        pivot = pivot.reindex(index=methods, columns=methods)
        sns.heatmap(
            pivot,
            ax=ax,
            vmin=0.3,
            vmax=1.0,
            cmap="YlGnBu",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar=split == "molecule",
            square=True,
            xticklabels=[LABELS.get(m, m) for m in methods],
            yticklabels=[LABELS.get(m, m) for m in methods],
        )
        ax.set_title(SPLIT_TITLE[split])
        ax.set_xlabel("Test negatives")
        ax.set_ylabel("Train negatives" if split == "random" else "")
    fig.suptitle(f"Cross-negative evaluation (ROC-AUC) — {src_label}", y=1.03)
    _save(fig, "fig2_crossneg_heatmaps")


def fig_mix_curves() -> None:
    src = ROOT / "results" / "ere_fraction_retrain_v1" / "mix_summary.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    methods = [m for m in OFFICIAL6 if m in set(df["train_negative"])]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), sharey=True)
    for ax, split in zip(axes, SPLIT_ORDER):
        sub = df[df["split_type"] == split]
        if sub.empty:
            ax.set_visible(False)
            continue
        for method in methods:
            m = sub[sub["train_negative"] == method].sort_values("exp_fraction")
            if m.empty:
                continue
            ax.errorbar(
                m["exp_fraction"] * 100,
                m["auc_mean"],
                yerr=m["auc_std"].fillna(0),
                marker="o",
                ms=4,
                lw=1.6,
                color=COLORS.get(method, "#333333"),
                label=LABELS[method],
                capsize=2,
            )
        ax.set_title(SPLIT_TITLE.get(split, split))
        ax.set_xlabel("% experimental supervision / mix fraction")
        ax.set_xlim(-2, 102)
        ax.set_ylim(0.15, 1.0)
        ax.grid(True, alpha=0.35)
    axes[0].set_ylabel("Test ROC-AUC")
    if methods:
        axes[-1].legend(fontsize=7, loc="lower right", frameon=True, ncol=1)
    fig.suptitle("Mix experiment (fraction-retrain v2, official scaled)", y=1.03)
    _save(fig, "fig3_mix_curves")


def fig_methods_overview() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    rows = [
        ("random_sequence", "Random sequence decoy", 4.2),
        ("composition_shuffle", "Composition shuffle (AptaBLE)", 3.5),
        ("random_pairing", "Real seq × random ligand", 2.8),
        ("structure_filtered", "Random + MFE filter", 2.1),
        ("hard_negative", "High-score non-interactors", 1.4),
        ("learned_generated", "Contrastive + GRU generation", 0.7),
    ]
    for method, desc, y in rows:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.3, y),
                2.4,
                0.55,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=COLORS.get(method, "#ddd"),
                edgecolor="#333",
                linewidth=0.8,
            )
        )
        ax.text(1.5, y + 0.275, LABELS[method], ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(3.0, y + 0.275, desc, ha="left", va="center", fontsize=9)
    ax.text(0.3, 4.85, "Official-6 synthetic methods", fontsize=11, fontweight="bold")
    ax.text(0.3, 0.05, "Reference: experimental AptaBench label=0", fontsize=9, style="italic")
    _save(fig, "fig_methods_overview")


def main() -> None:
    _style()
    fig0_pipeline_schematic()
    fig_methods_overview()
    fig_ere_bars()
    fig_crossneg_heatmaps()
    fig_mix_curves()
    print("Wrote figures to", OUT)
    for p in sorted(OUT.glob("*")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
