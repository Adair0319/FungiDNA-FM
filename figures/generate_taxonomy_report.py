#!/usr/bin/env python3
"""Generate taxonomy species-classification summary report.

Two experiments:
  1. five_rank        — 常规五折交叉验证 (5-fold CV within genome)
  2. genome_isolated  — 物种隔离 (test genomes never seen in training)

Metrics: F1 (Macro), MCC, AUROC (Macro), AUPRC (Macro).

Outputs (docs/experiment_report_taxonomy/):
  - metrics_summary.json  (consolidated numbers)
  - exp1_five_rank.png, exp2_genome_isolated.png
  - report.md
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Use CJK-capable font for Chinese labels
for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"]:
    if any(_f.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

BASE = Path("/home/lty/yy_projects/fungi_project/fungi_dna_model")
RESULTS = BASE / "results"
CKPT = BASE / "checkpoints"
DATA = BASE / "data" / "downstream"
OUT = BASE / "docs" / "experiment_report_taxonomy"
OUT.mkdir(parents=True, exist_ok=True)

RANKS = ["phylum", "subphylum", "class", "order", "family"]
RANK_CN = {"phylum": "门", "subphylum": "亚门", "class": "纲", "order": "目", "family": "科"}
MODELS = ["gena", "cnn", "mamba2", "ours"]
MODEL_DISPLAY = {"gena": "GENA-yeast", "cnn": "CNN", "mamba2": "Mamba2-1layer", "ours": "Ours"}

# ─────────────────────────────────────────────────────────
# 1. five_rank: mean±std from summary.json / fold metrics
# ─────────────────────────────────────────────────────────
def load_five_rank():
    data = {r: {} for r in RANKS}
    # corrected mamba2 AUROC/AUPRC (test-present labels)
    corrected = json.load(open(OUT / "mamba2_auroc_auprc.json"))
    for rank in RANKS:
        # ours / cnn / gena
        for m in ["ours", "cnn", "gena"]:
            p = RESULTS / "five_rank" / rank / f"{m}.json"
            d = json.load(open(p))
            data[rank][m] = {
                "f1": d["macro_f1"], "mcc": d["mcc"],
                "auroc": d["macro_auroc"], "auprc": d["macro_auprc"],
            }
        # mamba2: aggregate F1/MCC from fold metrics, AUROC/AUPRC from corrected
        folds = []
        for fold in range(1, 6):
            mp = CKPT / "baseline_mamba2_1layer" / f"task0_{rank}_five_rank" / f"fold{fold}" / "metrics.json"
            folds.append(json.load(open(mp)))
        data[rank]["mamba2"] = {
            "f1": agg(folds, "macro_f1"),
            "mcc": agg(folds, "mcc"),
            "auroc": corrected["five_rank"][rank]["auroc"],
            "auprc": corrected["five_rank"][rank]["auprc"],
        }
    return data


def agg(folds, key):
    vals = [f[key] for f in folds]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


# ─────────────────────────────────────────────────────────
# 2. genome_isolated: single values; recompute AUROC/AUPRC
# ─────────────────────────────────────────────────────────
def compute_auroc_auprc(predictions_csv, label_map_json):
    """Macro AUROC/AUPRC over labels present in the test set only."""
    df = pd.read_csv(predictions_csv)
    y_true = df["true_label"].to_numpy()
    label_map = json.load(open(label_map_json))
    idx_to_name = {v: k for k, v in label_map.items()}
    present = sorted(set(y_true.tolist()))
    aurocs, auprcs = [], []
    for lbl in present:
        name = idx_to_name[lbl]
        score = df[f"prob_{name}"].to_numpy()
        yb = (y_true == lbl).astype(int)
        if len(set(yb)) < 2:  # single class present
            continue
        aurocs.append(roc_auc_score(yb, score))
        auprcs.append(average_precision_score(yb, score))
    return float(np.mean(aurocs)), float(np.mean(auprcs))


def load_genome_isolated():
    data = {r: {} for r in RANKS}
    corrected = json.load(open(OUT / "mamba2_auroc_auprc.json"))
    for rank in RANKS:
        # mamba2: F1/MCC from metrics.json, AUROC/AUPRC from corrected
        mp = CKPT / "baseline_mamba2_1layer" / f"task0_{rank}" / "metrics.json"
        d = json.load(open(mp))
        data[rank]["mamba2"] = {
            "f1": d["macro_f1_noignore"], "mcc": d["mcc"],
            "auroc": corrected["genome_isolated"][rank]["auroc"],
            "auprc": corrected["genome_isolated"][rank]["auprc"],
        }
        # ours / cnn / gena: report.json (f1/mcc) + recompute auroc/auprc
        for m in ["ours", "cnn", "gena"]:
            rp = CKPT / "genome_isolated" / f"{m}_{rank}" / "report.json"
            r = json.load(open(rp))
            pred_csv = CKPT / "genome_isolated" / f"{m}_{rank}" / "predictions.csv"
            auroc, auprc = compute_auroc_auprc(pred_csv, DATA / f"task0_{rank}" / "label_map.json")
            data[rank][m] = {
                "f1": r["macro_f1"], "mcc": r["mcc"],
                "auroc": auroc, "auprc": auprc,
            }
    return data


# ─────────────────────────────────────────────────────────
# 3. Figures
# ─────────────────────────────────────────────────────────
METRIC_KEYS = ["f1", "mcc", "auroc", "auprc"]
METRIC_LABELS = {"f1": "F1 (Macro)", "mcc": "MCC", "auroc": "AUROC (Macro)", "auprc": "AUPRC (Macro)"}
COLORS = {"gena": "#E6A176", "cnn": "#7FB3D8", "mamba2": "#82C79B", "ours": "#C39BD3"}


def plot_experiment(data, title, out_path, has_std):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x = np.arange(len(MODELS))
    width = 0.2
    offsets = np.linspace(-1.5, 1.5, len(METRIC_KEYS)) * width

    for i, rank in enumerate(RANKS):
        ax = axes[i]
        for j, mk in enumerate(METRIC_KEYS):
            vals, errs = [], []
            for m in MODELS:
                v = data[rank][m][mk]
                if has_std:
                    vals.append(v["mean"]); errs.append(v["std"])
                else:
                    vals.append(v); errs.append(0.0)
            ax.bar(x + offsets[j], vals, width, label=METRIC_LABELS[mk],
                   yerr=errs if has_std else None, capsize=3 if has_std else 0,
                   error_kw={"elinewidth": 1, "ecolor": "dimgray"})
        ax.set_title(f"{RANK_CN[rank]} (phylum/subphylum/class/order/family 分解)" if False else RANK_CN[rank],
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", labelrotation=15)

    # legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METRIC_KEYS), fontsize=12, frameon=False)

    # hide unused 6th subplot
    axes[5].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ─────────────────────────────────────────────────────────
# 4. Markdown
# ─────────────────────────────────────────────────────────
def fmt_cv(d):
    return f"{d['mean']:.4f} ±{d['std']:.4f}"


def fmt_single(v):
    return f"{v:.4f}"


def write_markdown(five, gi, path):
    lines = []
    lines.append("# 真菌物种分类 五级分类实验总结报告\n")
    lines.append("**评估指标**: F1 (Macro) · MCC · AUROC (Macro) · AUPRC (Macro)\n")
    lines.append("**模型**: GENA-yeast · CNN · Mamba2-1layer · Ours (StripedMamba)\n")

    # Experiment 1
    lines.append("## 实验 1：常规五折交叉验证（非物种隔离）\n")
    lines.append("5-fold CV，每个 genome 的窗口打乱后分为 5 折（train/val/test）。\n")
    for rank in RANKS:
        lines.append(f"### {RANK_CN[rank]}（{rank}）\n")
        lines.append("| 模型 | F1 (Macro) | MCC | AUROC (Macro) | AUPRC (Macro) |")
        lines.append("|------|-----------|-----|---------------|---------------|")
        for m in MODELS:
            d = five[rank][m]
            lines.append(f"| {MODEL_DISPLAY[m]} | {fmt_cv(d['f1'])} | {fmt_cv(d['mcc'])} | "
                         f"{fmt_cv(d['auroc'])} | {fmt_cv(d['auprc'])} |")
        lines.append("")
    lines.append(f"![实验1：五折交叉验证](exp1_five_rank.png)\n")

    # Experiment 2
    lines.append("## 实验 2：物种隔离（Genome-isolated）\n")
    lines.append("测试集 genome 完全不在训练集中，测试泛化到未见物种的能力。\n")
    lines.append("> AUROC/AUPRC 仅对 test 集中有代表的 label 求 macro 平均（无测试样本的 label 数学上无定义）。\n")
    for rank in RANKS:
        lines.append(f"### {RANK_CN[rank]}（{rank}）\n")
        lines.append("| 模型 | F1 (Macro) | MCC | AUROC (Macro) | AUPRC (Macro) |")
        lines.append("|------|-----------|-----|---------------|---------------|")
        for m in MODELS:
            d = gi[rank][m]
            lines.append(f"| {MODEL_DISPLAY[m]} | {fmt_single(d['f1'])} | {fmt_single(d['mcc'])} | "
                         f"{fmt_single(d['auroc'])} | {fmt_single(d['auprc'])} |")
        lines.append("")
    lines.append(f"![实验2：物种隔离](exp2_genome_isolated.png)\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("Loading five_rank...")
    five = load_five_rank()
    print("Loading genome_isolated...")
    gi = load_genome_isolated()

    # consolidated
    summary = {"five_rank": five, "genome_isolated": gi}
    json.dump(summary, open(OUT / "metrics_summary.json", "w"), indent=2, ensure_ascii=False)
    print(f"  saved metrics_summary.json")

    print("Generating figures...")
    plot_experiment(five, "实验 1：常规五折交叉验证（非物种隔离）",
                    OUT / "exp1_five_rank.png", has_std=True)
    plot_experiment(gi, "实验 2：物种隔离（Genome-isolated）",
                    OUT / "exp2_genome_isolated.png", has_std=False)

    print("Generating markdown...")
    write_markdown(five, gi, OUT / "report.md")

    print("\nDone. All outputs in:", OUT)


if __name__ == "__main__":
    main()
