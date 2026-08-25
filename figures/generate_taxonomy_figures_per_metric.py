#!/usr/bin/env python3
"""Generate per-metric PDF figures for the taxonomy report.

One figure per metric (F1/MCC/AUROC/AUPRC) x experiment (five_rank/genome_isolated)
= 8 PDF figures. All English, Arial font, auto-scaled y-axis.

Layout per figure: x-axis = 5 ranks, 4 grouped bars = 4 models.
"""
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path("/home/lty/yy_projects/fungi_project/fungi_dna_model")
OUT = BASE / "docs" / "experiment_report_taxonomy"

# ── Arial font (register real Arial, not Liberation Sans alias) ──
ARIAL_REG = Path("/home/lty/.local/share/fonts/Arial.ttf")
ARIAL_BOLD = Path("/home/lty/.local/share/fonts/Arial_Bold.ttf")
if ARIAL_REG.exists():
    font_manager.fontManager.addfont(str(ARIAL_REG))
if ARIAL_BOLD.exists():
    font_manager.fontManager.addfont(str(ARIAL_BOLD))
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

RANKS = ["phylum", "subphylum", "class", "order", "family"]
MODELS = ["gena", "cnn", "mamba2", "ours"]
MODEL_DISPLAY = {"gena": "GENA-yeast", "cnn": "CNN", "mamba2": "Mamba2-1layer", "ours": "Ours"}
METRICS = ["f1", "mcc", "auroc", "auprc"]
METRIC_LABEL = {"f1": "F1 (Macro)", "mcc": "MCC", "auroc": "AUROC (Macro)", "auprc": "AUPRC (Macro)"}
COLORS = {"gena": "#ea4335", "cnn": "#34a853", "mamba2": "#f9ab00", "ours": "#1a73e8"}
EXP_TITLE = {"five_rank": "Five-fold Cross-Validation", "genome_isolated": "Genome-isolated"}


def auto_ylim(values):
    """Auto-scale y-axis to data range (not forced 0-1)."""
    vmin = min(values)
    vmax = max(values)
    span = vmax - vmin
    pad = 0.10 * span + 0.02
    ymin = max(0.0, vmin - pad)
    ymax = min(1.0, vmax + pad)
    # ensure a minimum visible range so bars aren't flattened
    if ymax - ymin < 0.15:
        center = (ymin + ymax) / 2
        ymin = max(0.0, center - 0.08)
        ymax = min(1.0, center + 0.08)
    return ymin, ymax


def plot_metric(data, metric, exp_name, out_path, has_std):
    x = np.arange(len(RANKS))
    width = 0.2
    offsets = np.linspace(-1.5, 1.5, len(MODELS)) * width

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    all_vals = []
    for j, m in enumerate(MODELS):
        vals, errs = [], []
        for rank in RANKS:
            v = data[rank][m][metric]
            if has_std:
                vals.append(v["mean"]); errs.append(v["std"])
                all_vals.append(v["mean"] + v["std"])
            else:
                vals.append(v); errs.append(0.0)
                all_vals.append(v)
        ax.bar(x + offsets[j], vals, width, label=MODEL_DISPLAY[m],
               color=COLORS[m],
               yerr=errs if has_std else None, capsize=3 if has_std else 0,
               error_kw={"elinewidth": 1, "ecolor": "dimgray"})

    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in RANKS], fontsize=12)
    ax.set_ylabel(METRIC_LABEL[metric], fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)

    ymin, ymax = auto_ylim(all_vals)
    ax.set_ylim(ymin, ymax)

    ax.set_title(f"{METRIC_LABEL[metric]} — {EXP_TITLE[exp_name]}", fontsize=14, fontweight="bold")

    # legend below the plot (outside the axes, no overlap with bars)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=11, frameon=False, ncol=4)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}  (ylim={ymin:.2f}-{ymax:.2f})")


def main():
    summary = json.load(open(OUT / "metrics_summary.json"))
    five = summary["five_rank"]
    gi = summary["genome_isolated"]

    for exp_name, data, has_std in [("five_rank", five, True),
                                     ("genome_isolated", gi, False)]:
        exp_tag = "exp1" if exp_name == "five_rank" else "exp2"
        for metric in METRICS:
            out = OUT / f"{exp_tag}_{metric}.pdf"
            plot_metric(data, metric, exp_name, out, has_std)

    print("\nDone. 8 PDF figures generated in", OUT)


if __name__ == "__main__":
    main()
