#!/usr/bin/env python3
"""Generate all figures for the clustering analysis report (PDF, Arial font).

Three experiments:
  1. Block1 balanced PCA+UMAP clustering (phylum + elements), overall + per-class
     UMAP + metrics vs 4-mer baseline.
  2. 6-block depth sweep (phylum + elements), 5 metrics vs block.
  3. Checkpoint ablation (phase1 first, then phase2), 5 metrics vs checkpoint.

Formatting requirements:
  - PDF output
  - UMAP plots: no axis info
  - Legend: consistent small markers, no overlap
  - Bar/line charts: reasonable y-axis range (not always 0-1)
  - Exp3: phase1 first, dashed separator, equal x-spacing
  - All fonts Arial
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Arial font ──
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = PROJECT_ROOT / "docs/experiment_report_clustering"
FIG_DIR = DOCS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palettes ──
PHYLUM_NAMES = ["Ascomycota", "Basidiomycota", "Blastocladiomycota",
                "Chytridiomycota", "Mucoromycota", "Zoopagomycota"]
PHYLUM_COLORS = ["#7B1FA2", "#2E7D32", "#C62828", "#1565C0", "#F9A825", "#EF6C00"]

ELEMENT_NAMES = ["cds", "intron", "intergenic", "five_prime_utr", "three_prime_utr"]
ELEMENT_COLORS = ["#7B1FA2", "#2E7D32", "#C62828", "#1565C0", "#EF6C00"]

METRIC_NAMES = ["ARI", "NMI", "Homogeneity", "Completeness", "V-measure"]
METRIC_KEYS = ["ari", "nmi", "homogeneity", "completeness", "v_measure"]
METRIC_COLORS = ["#1565C0", "#7B1FA2", "#2E7D32", "#EF6C00", "#C62828"]

GRAY = "#C9C9C9"
DPI = 300


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)


def _legend(ax, markerscale=5.0, fontsize=10, loc="best"):
    """Consistent legend with small, uniform markers, no overlap."""
    return ax.legend(
        markerscale=markerscale,
        handlelength=1.0,
        handletextpad=0.35,
        borderpad=0.5,
        fontsize=fontsize,
        loc=loc,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
    )


def _auto_ylim(values, pad=0.15):
    """Reasonable y-range around data (floor 0 for metrics, headroom on top)."""
    hi = max(values)
    return 0.0, hi * (1.0 + pad)


# ═══════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════
p_umap = np.load(PROJECT_ROOT / "analysis/umap_phylum_depth/block1/umap_pca.npy")
p_labels = np.load(PROJECT_ROOT / "analysis/umap_phylum_depth/labels/labels.npy")
p_metrics_all = json.load(open(PROJECT_ROOT / "analysis/umap_phylum_depth/all_metrics.json"))

e_umap = np.load(PROJECT_ROOT / "analysis/umap_elements/balanced/block1/umap_pca.npy")
e_labels = np.load(PROJECT_ROOT / "analysis/umap_elements/labels/balanced_labels.npy")
e_metrics_all = json.load(open(PROJECT_ROOT / "analysis/umap_elements/all_metrics.json"))

ckpt_df = pd.read_csv(PROJECT_ROOT / "analysis/checkpoint_ablation.csv")


# ═══════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════

def _umap_axis_off(ax):
    """Remove all axis info from a UMAP plot."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_overall_umap(coords, labels, names, colors, title, fname):
    fig, ax = plt.subplots(figsize=(10, 9))
    order = sorted(range(len(names)), key=lambda l: (labels == l).sum())
    for lbl in order:
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[lbl],
                   label=f"{names[lbl]}  ({mask.sum():,})",
                   s=3, alpha=0.7, rasterized=True, edgecolors="none")
    xlim = tuple(np.percentile(coords[:, 0], [1, 99]))
    ylim = tuple(np.percentile(coords[:, 1], [1, 99]))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    _umap_axis_off(ax)
    _legend(ax, markerscale=5.0, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, format="pdf", bbox_inches="tight")
    plt.close()


def plot_per_class_umap(coords, labels, names, colors, class_lbl, fname):
    fig, ax = plt.subplots(figsize=(10, 9))
    others = labels != class_lbl
    ax.scatter(coords[others, 0], coords[others, 1], c=GRAY,
               s=2, alpha=0.4, rasterized=True, edgecolors="none", label="Other")
    mask = labels == class_lbl
    ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[class_lbl],
               s=4, alpha=0.85, rasterized=True, edgecolors="none",
               label=f"{names[class_lbl]} (n={mask.sum():,})")
    xlim = tuple(np.percentile(coords[:, 0], [1, 99]))
    ylim = tuple(np.percentile(coords[:, 1], [1, 99]))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    _umap_axis_off(ax)
    _legend(ax, markerscale=5.0, fontsize=11)
    ax.set_title(f"{names[class_lbl]} (label={class_lbl})", fontsize=14,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, format="pdf", bbox_inches="tight")
    plt.close()


def plot_metrics_vs_kmer(ours_metrics, kmer_metrics, title, fname, ours_color="#1565C0"):
    x = np.arange(len(METRIC_NAMES)); width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    def _get(d, k):
        return d.get(k, d.get(f"pca_{k}", 0.0))
    ours_vals = [_get(ours_metrics, k) for k in METRIC_KEYS]
    kmer_vals = [kmer_metrics.get(f"kmer_{k}", 0.0) for k in METRIC_KEYS]
    b1 = ax.bar(x - width/2, ours_vals, width, label="Ours (Block1)", color=ours_color,
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width/2, kmer_vals, width, label="4-mer baseline", color="#9E9E9E",
                edgecolor="white", linewidth=0.5)
    for bars, color in [(b1, ours_color), (b2, "#616161")]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9,
                    fontweight="bold", color=color)
    _clean_ax(ax)
    ax.set_ylim(*_auto_ylim(ours_vals + kmer_vals))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x); ax.set_xticklabels(METRIC_NAMES, fontsize=11)
    ax.legend(fontsize=11, frameon=True, framealpha=0.92, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, format="pdf", bbox_inches="tight")
    plt.close()


def plot_depth_metrics(metrics_by_block, title, fname):
    blocks = list(range(6))
    fig, ax = plt.subplots(figsize=(9, 6))
    all_vals = []
    for mi, (mname, mkey, mcolor) in enumerate(zip(METRIC_NAMES, METRIC_KEYS, METRIC_COLORS)):
        vals = [metrics_by_block[b][f"pca_{mkey}"] for b in blocks]
        all_vals.extend(vals)
        ax.plot(blocks, vals, marker="o", linewidth=2, markersize=5,
                label=mname, color=mcolor)
    _clean_ax(ax)
    ax.set_xticks(blocks)
    ax.set_xticklabels([f"Block {i}\n(layer {(i+1)*4})" for i in blocks], fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(*_auto_ylim(all_vals))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, framealpha=0.92, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, format="pdf", bbox_inches="tight")
    plt.close()


def plot_checkpoint_metrics(df, task, title, fname):
    sub = df[df["task"] == task]
    # Phase1 first (40k/50k/60k), then phase2 (5k/10k/15k/16k)
    phase1 = sub[sub["step"] >= 40000].sort_values("step")
    phase2 = sub[sub["step"] < 40000].sort_values("step")
    ordered = pd.concat([phase1, phase2], ignore_index=True)
    n1 = len(phase1)
    x = np.arange(len(ordered))
    sep = n1 - 0.5  # dashed separator between phase1 and phase2

    fig, ax = plt.subplots(figsize=(10, 6))
    all_vals = []
    for mi, (mname, mkey, mcolor) in enumerate(zip(METRIC_NAMES, METRIC_KEYS, METRIC_COLORS)):
        vals = ordered[mkey].tolist()
        all_vals.extend(vals)
        ax.plot(x, vals, marker="o", linewidth=2, markersize=5,
                label=mname, color=mcolor)

    # Dashed separator between phases
    ax.axvline(x=sep, color="#888888", linestyle="--", linewidth=1.2, alpha=0.7)

    # Phase labels
    y_hi = max(all_vals) * 1.15
    ax.text((sep - 1) / 2, y_hi * 0.97, "Phase 1\n(MLM)", ha="center", va="top",
            fontsize=9, color="#555555")
    ax.text(sep + (len(ordered) - 1 - sep) / 2, y_hi * 0.97, "Phase 2\n(MLM + CL)",
            ha="center", va="top", fontsize=9, color="#555555")

    _clean_ax(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s):,}" for s in ordered["step"]], fontsize=9)
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(*_auto_ylim(all_vals))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, frameon=True, framealpha=0.92, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, format="pdf", bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════
# Experiment 1
# ═══════════════════════════════════════════════════
print("=== Experiment 1: Block1 clustering ===")

plot_overall_umap(p_umap, p_labels, PHYLUM_NAMES, PHYLUM_COLORS,
                  "Phylum Clustering — Block1 PCA+UMAP", "phylum_overall_umap.pdf")
for lbl in range(len(PHYLUM_NAMES)):
    plot_per_class_umap(p_umap, p_labels, PHYLUM_NAMES, PHYLUM_COLORS, lbl,
                        f"phylum_class_{lbl}_{PHYLUM_NAMES[lbl]}.pdf")
plot_metrics_vs_kmer(p_metrics_all["block1_pca"], p_metrics_all["kmer_baseline"],
                     "Phylum Clustering: Ours vs 4-mer", "phylum_metrics_vs_kmer.pdf")
print("  phylum done")

plot_overall_umap(e_umap, e_labels, ELEMENT_NAMES, ELEMENT_COLORS,
                  "Element Clustering — Block1 PCA+UMAP", "elements_overall_umap.pdf")
for lbl in range(len(ELEMENT_NAMES)):
    plot_per_class_umap(e_umap, e_labels, ELEMENT_NAMES, ELEMENT_COLORS, lbl,
                        f"elements_class_{lbl}_{ELEMENT_NAMES[lbl]}.pdf")
e_pca = {f"pca_{k}": e_metrics_all["balanced_block1"][f"pca_{k}"] for k in METRIC_KEYS}
plot_metrics_vs_kmer(e_pca, e_metrics_all["kmer_baseline"],
                     "Element Clustering: Ours vs 4-mer", "elements_metrics_vs_kmer.pdf")
print("  elements done")


# ═══════════════════════════════════════════════════
# Experiment 2
# ═══════════════════════════════════════════════════
print("=== Experiment 2: 6-block depth sweep ===")

p_blocks = {i: p_metrics_all[f"block{i}_pca"] for i in range(6)}
plot_depth_metrics(p_blocks, "Phylum: Clustering Metrics vs Depth",
                   "phylum_block_metrics.pdf")

e_blocks_pca = {i: {f"pca_{k}": e_metrics_all[f"balanced_block{i}"][f"pca_{k}"]
                    for k in METRIC_KEYS} for i in range(6)}
plot_depth_metrics(e_blocks_pca, "Element: Clustering Metrics vs Depth",
                   "elements_block_metrics.pdf")
print("  done")


# ═══════════════════════════════════════════════════
# Experiment 3
# ═══════════════════════════════════════════════════
print("=== Experiment 3: checkpoint ablation ===")

plot_checkpoint_metrics(ckpt_df, "phylum", "Phylum: Clustering Metrics vs Checkpoint",
                        "phylum_checkpoint_metrics.pdf")
plot_checkpoint_metrics(ckpt_df, "elements", "Element: Clustering Metrics vs Checkpoint",
                        "elements_checkpoint_metrics.pdf")
print("  done")

print(f"\nAll figures saved to {FIG_DIR} (PDF)")
