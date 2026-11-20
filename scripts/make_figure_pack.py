"""Build paper figures from committed benchmark result CSVs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
INPUT = ROOT / "analyses" / "results"
PREFIX = ""

APTABENCH6 = [
    "random_sequence",
    "composition_shuffle",
    "random_pairing",
    "structure_filtered",
    "hard_negative",
    "learned_generated",
]
LABELS = {
    "random_sequence": "Random seq",
    "composition_shuffle": "Composition shuffle",
    "random_pairing": "Random pairing",
    "structure_filtered": "Struct-filtered",
    "hard_negative": "Hard negatives",
    "learned_generated": "Profile-GRU",
    "experimental": "Experimental",
}
METHOD_ORDER = APTABENCH6 + ["experimental"]
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


def _figure_input(name: str) -> pd.DataFrame:
    frame = pd.read_csv(INPUT / f"{PREFIX}{name}")
    if "skipped" in frame:
        frame = frame[~frame["skipped"].fillna(False).astype(bool)]
    return frame


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
    df = _figure_input("ere_metrics.csv")
    src_label = "Archived AptaBench ERE" if PREFIX else "AptaBench ERE"
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
    df = _figure_input("cross_negative_metrics.csv")
    src_label = "Archived AptaBench ERE" if PREFIX else "AptaBench ERE"
    train_col = "train_negative" if "train_negative" in df.columns else "train_method"
    test_col = "test_negative" if "test_negative" in df.columns else "test_method"
    value = next(c for c in ("test_roc_auc", "roc_auc", "test_auc") if c in df.columns)
    methods = [m for m in METHOD_ORDER if m in set(df[train_col]) | set(df[test_col])]
    fig = plt.figure(figsize=(13.8, 4.5), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045])
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    color_axis = fig.add_subplot(grid[0, 3])
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
            cbar_ax=color_axis if split == "molecule" else None,
            square=True,
            xticklabels=[LABELS.get(m, m) for m in methods],
            yticklabels=[LABELS.get(m, m) for m in methods] if split == "random" else False,
        )
        ax.set_title(SPLIT_TITLE[split])
        ax.set_xlabel("Test negatives")
        ax.set_ylabel("Train negatives" if split == "random" else "")
    fig.suptitle(f"Cross-negative evaluation (ROC-AUC) — {src_label}")
    _save(fig, "fig2_crossneg_heatmaps")


def fig_mix_curves() -> None:
    df = pd.read_csv(ROOT / "analyses" / "results" / "archived_mix_summary.csv")
    methods = [m for m in APTABENCH6 if m in set(df["train_negative"])]
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
    fig.suptitle("Mix experiment (fraction-retrain v2, scaled features)", y=1.03)
    _save(fig, "fig3_mix_curves")


def fig_methods_overview() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    rows = [
        ("random_sequence", "Random sequence decoy", 4.2),
        ("composition_shuffle", "Composition shuffle", 3.5),
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
    ax.text(0.3, 4.85, "Six synthetic methods", fontsize=11, fontweight="bold")
    ax.text(0.3, 0.05, "Reference: experimental AptaBench label=0", fontsize=9, style="italic")
    _save(fig, "fig_methods_overview")


def fig_negative_volume() -> None:
    summary = pd.read_csv(ROOT / "analyses" / "results" / "archived_blend_volume_summary.csv")
    volume = summary.loc[summary.arm.eq("volume")]
    historical_ere = pd.read_csv(ROOT / "analyses" / "results" / "archived_ere_metrics.csv")
    reference = historical_ere.loc[historical_ere.train_negative.eq("experimental")]
    ref_means = reference.groupby("split_type").test_roc_auc.mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, split in zip(axes, SPLIT_ORDER):
        for method in APTABENCH6:
            rows = volume.loc[volume.split_type.eq(split) & volume.train_negative.eq(method)]
            rows = rows.sort_values("syn_to_pos_ratio")
            ax.errorbar(rows.syn_to_pos_ratio, rows.auc_mean, yerr=rows.auc_std, marker="o",
                        color=COLORS[method], label=LABELS[method], capsize=2)
        ax.set_title(SPLIT_TITLE[split])
        ax.axhline(ref_means.loc[split], ls="--", color="0.35", label="Experimental reference")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 2, 4], labels=["0.25", "0.5", "1", "2", "4"])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Experimental-test ROC-AUC")
    fig.supxlabel("Synthetic negatives / training positives", y=0.19)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.89, bottom=0.28, wspace=0.24)
    _save(fig, "negative_volume")


def fig_main_comparison() -> None:
    """Render the three-panel paper figure from released numerical inputs."""
    folds = _figure_input("ere_metrics.csv")
    stats = folds.groupby(["split_type", "train_negative"]).test_roc_auc.agg(["mean", "std"])
    molecule = folds.loc[folds.split_type.eq("molecule")].groupby("train_negative")[
        ["test_roc_auc_experimental", "test_roc_auc_synthetic_matched"]
    ].mean()
    low = pd.read_csv(ROOT / "analyses/results/amplification_summary.csv").set_index("mode")
    colors = dict.fromkeys(APTABENCH6, "#248e82")
    colors.update(experimental="#6d6586", random_pairing="#414ba8",
                  hard_negative="#414ba8", learned_generated="#de791d")
    labels = dict(LABELS, random_sequence="Random sequence", composition_shuffle="Composition shuffle",
                  structure_filtered="Structure-filtered", hard_negative="Hard negatives")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17.3, 5.55),
                                 gridspec_kw={"width_ratios": [1.08, 1, 1]})
    fig.subplots_adjust(left=0.092, right=0.988, top=0.87, bottom=0.26, wspace=0.47)
    for ax, title in zip((a, b, c), ("A   Experimental reference", "B   Synthetic-to-experimental transfer",
                                    "C   10% experimental control")):
        ax.set_title(title, loc="left", weight="bold", fontsize=12, pad=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params(length=3)
        ax.grid(False)
    order = ["experimental", "random_pairing", "learned_generated", "structure_filtered",
             "random_sequence", "composition_shuffle", "hard_negative"]
    a.axhspan(-0.43, 0.43, color=colors["experimental"], alpha=0.16, lw=0)
    a.axvline(0.5, color="0.5", ls=":", lw=1.2)
    markers = ("o", "s", "D")
    for y, method in enumerate(order):
        for split, marker, offset in zip(SPLIT_ORDER, markers, (-0.23, 0, 0.23)):
            point = stats.loc[split, method]
            a.errorbar(point["mean"], y + offset, xerr=point["std"], fmt=marker,
                       color=colors[method], lw=1.6, capsize=3, ms=5)
    a.set_yticks(range(len(order)), [labels[m] for m in order])
    a.set_ylim(6.65, -0.65)
    a.set_xlabel("Experimental-test ROC-AUC")
    a.legend(handles=[Line2D([], [], marker=m, ls="", color="0.4", label=SPLIT_TITLE[s])
                      for s, m in zip(SPLIT_ORDER, markers)], loc="upper center",
             bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8)

    order = ["hard_negative", "composition_shuffle", "random_sequence", "structure_filtered",
             "learned_generated", "random_pairing"]
    for y, method in enumerate(order):
        exp, syn = molecule.loc[method]
        color = colors[method]
        b.annotate("", xy=(syn, y), xytext=(exp, y),
                   arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8,
                               "shrinkA": 4, "shrinkB": 4})
        b.plot(exp, y, "o", mfc="white", mec=color, mew=1.8, ms=7)
        b.plot(syn, y, "s", color=color, ms=6)
        b.text((exp + syn) / 2, y - 0.17, f"{syn-exp:+.2f}", ha="center", color=color, weight="bold")
    b.set_yticks(range(len(order)), [labels[m] for m in order])
    b.set_ylim(5.65, -0.62)
    b.set_xlabel("ROC-AUC (molecule-disjoint)")
    b.legend(handles=[Line2D([], [], marker="o", mfc="white", mec="0.4", ls="", label="Experimental test"),
                      Line2D([], [], marker="s", color="0.4", ls="", label="Matched synthetic test")],
             loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8)
    for ax in (a, b):
        ax.set_xlim(0.14, 1.03)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis="x", alpha=0.23)

    modes = ["exp_10_only", "exp_10_oversample", "exp_10_plus_rp", "exp_10_plus_generated"]
    reference = stats.loc["molecule", "experimental"]
    mean, sd = reference["mean"], reference["std"]
    c.axhspan(mean - sd, mean + sd, color=colors["experimental"], alpha=0.10, lw=0)
    c.axhline(mean, color=colors["experimental"], ls="--", lw=1.5)
    c.text(0.54, mean + 0.009, "100% experimental", transform=c.get_yaxis_transform(),
           ha="center", color=colors["experimental"], fontsize=8)
    for x, (mode, color, marker) in enumerate(zip(
        modes, [colors[k] for k in ("experimental", "experimental", "random_pairing", "learned_generated")],
        ("o", "o", "D", "P"),
    )):
        row = low.loc[mode]
        c.errorbar(x, row.mean_roc_auc, yerr=row.std_roc_auc, fmt=marker,
                   color=color, lw=1.6, capsize=4, ms=7)
        c.text(x, row.mean_roc_auc + row.std_roc_auc + 0.012, f"{row.mean_roc_auc:.3f}",
               ha="center", color=color, fontsize=9)
    c.set_xticks(range(4), ["10% exp.\nonly", "10% exp.\noversampled",
                            "10% exp. +\nRandom pairing", "10% exp. +\nProfile-GRU"], fontsize=8)
    c.set_xlim(-0.48, 3.48)
    c.set_ylim(0.54, 0.98)
    c.set_ylabel("Experimental-test ROC-AUC")
    c.grid(axis="y", alpha=0.23)
    _save(fig, "fig_main_comparison")


def main() -> None:
    global INPUT, PREFIX, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ere-inputs", type=Path, help="Directory containing a complete new ERE run")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.ere_inputs:
        INPUT, PREFIX = args.ere_inputs, ""
        metrics = _figure_input("ere_metrics.csv")
        metrics = metrics.loc[metrics.train_negative.isin(METHOD_ORDER)]
        counts = metrics.groupby(["split_type", "train_negative"]).fold.nunique()
        if len(counts) != 21 or not counts.eq(5).all():
            raise SystemExit("The revised figure input requires five complete folds per method and split.")
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    fig0_pipeline_schematic()
    fig_methods_overview()
    fig_ere_bars()
    fig_main_comparison()
    fig_crossneg_heatmaps()
    fig_mix_curves()
    fig_negative_volume()
    print("Wrote figures to", OUT)
    for p in sorted(OUT.glob("*")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
