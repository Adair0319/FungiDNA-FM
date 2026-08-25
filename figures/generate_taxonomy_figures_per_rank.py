#!/usr/bin/env python3
"""Generate per-rank PDF figures for the taxonomy report.

One figure per rank (phylum/subphylum/class/order/family) x experiment
(five_rank/genome_isolated) = 10 PDF figures. All English, Arial font.

Layout per figure: x-axis = 4 models, 4 grouped bars = 4 metrics.
Metric colors follow the material dark palette used in the clustering report.
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

# ── Arial font ──
for f in ["/home/lty/.local/share/fonts/Arial.ttf",
          "/home/lty/.local/share/fonts/Arial_Bold.ttf"]:
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

RANKS = ["phylum", "subphylum", "class", "order", "family"]
MODELS = ["gena", "cnn", "mamba2", "ours"]
MODEL_DISPLAY = {"gena": "GENA-yeast", "cnn": "CNN", "mamba2": "Mamba2-1layer", "ours": "Ours"}
METRICS = ["f1", "mcc", "auroc", "auprc"]
METRIC_LABEL = {"f1": "F1 (Macro)", "mcc": "MCC", "auroc": "AUROC (Macro)", "auprc": "AUPRC (Macro)"}
# metric colors: same Google palette as the per-metric figures
METRIC_COLORS = {"f1": "#1a73e8", "mcc": "#ea4335", "auroc": "#34a853", "auprc": "#f9ab00"}
EXP_TITLE = {"five_rank": "Five-fold Cross-Validation", "genome_isolated": "Genome-isolated"}


def plot_rank(data, rank, exp_name, out_path, has_std):
    x = np.arange(len(MODELS))
    width = 0.2
    offsets = np.linspace(-1.5, 1.5, len(METRICS)) * width

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for j, mk in enumerate(METRICS):
        vals, errs = [], []
        for m in MODELS:
            v = data[rank][m][mk]
            if has_std:
                vals.append(v["mean"]); errs.append(v["std"])
            else:
                vals.append(v); errs.append(0.0)
        ax.bar(x + offsets[j], vals, width, label=METRIC_LABEL[mk],
               color=METRIC_COLORS[mk],
               yerr=errs if has_std else None, capsize=3 if has_std else 0,
               error_kw={"elinewidth": 1, "ecolor": "dimgray"})

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=12)
    ax.set_ylabel("Score", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.set_ylim(0, 1.0)

    ax.set_title(f"{rank.capitalize()} — {EXP_TITLE[exp_name]}", fontsize=14, fontweight="bold")

    # legend below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=11, frameon=False, ncol=4)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main():
    summary = json.load(open(OUT / "metrics_summary.json"))
    five = summary["five_rank"]
    gi = summary["genome_isolated"]

    for exp_name, data, has_std in [("five_rank", five, True),
                                     ("genome_isolated", gi, False)]:
        exp_tag = "exp1" if exp_name == "five_rank" else "exp2"
        for rank in RANKS:
            out = OUT / f"{exp_tag}_{rank}.pdf"
            plot_rank(data, rank, exp_name, out, has_std)

    print("\nDone. 10 PDF figures generated in", OUT)


if __name__ == "__main__":
    main()
