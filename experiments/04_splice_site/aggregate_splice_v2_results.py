"""Aggregate 5-fold CV results across splice v2 experiments."""
import json, os, sys
import numpy as np

EXPS = ["randfull", "frozen", "fullft", "onehot", "cnn"]
BASE = "/home/lty/yy_projects/fungi_project/fungi_dna_model/checkpoints/splice_v2"
METRICS = ["macro_f1", "mcc", "auroc_macro", "auprc_macro"]
CLASS_NAMES = ["Donor", "Acceptor", "Non-Site"]

all_results = {}
for exp in EXPS:
    folds = []
    for f in range(5):
        path = f"{BASE}/{exp}/fold_{f}/metrics.json"
        if os.path.exists(path):
            with open(path) as fh:
                folds.append(json.load(fh))
    if not folds:
        print(f"WARNING: {exp}: no folds found"); continue

    summary = {}
    for key in METRICS:
        vals = [m[key] for m in folds if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))]
        if vals:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    for nm in CLASS_NAMES:
        for sub in ["F1", "Recall", "Precision", "AUROC", "AUPRC"]:
            k = f"{nm}_{sub}"
            vals = [m[k] for m in folds if k in m and not (isinstance(m[k], float) and np.isnan(m[k]))]
            if vals:
                summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["n_folds"] = len(folds)
    all_results[exp] = summary

    print(f"\n=== {exp} ({len(folds)} folds) ===")
    for key in METRICS:
        if key in summary:
            s = summary[key]
            print(f"  {key:15s}: {s['mean']:.4f} ± {s['std']:.4f}")
    print("  Per-class F1:")
    for nm in CLASS_NAMES:
        k = f"{nm}_F1"
        if k in summary:
            s = summary[k]
            print(f"    {nm:10s}: {s['mean']:.4f} ± {s['std']:.4f}")

# Comparison table
print("\n" + "=" * 80)
print("CROSS-EXPERIMENT COMPARISON (Macro-F1)")
print("=" * 80)
header = f"{'Experiment':<15} {'Macro-F1':>16} {'MCC':>12} {'AUROC':>12} {'AUPRC':>12}"
print(header)
print("-" * len(header))
for exp in EXPS:
    if exp not in all_results:
        continue
    s = all_results[exp]
    f1 = s.get("macro_f1", {})
    mcc = s.get("mcc", {})
    auroc = s.get("auroc_macro", {})
    auprc = s.get("auprc_macro", {})
    print(f"{exp:<15} {f1.get('mean',0):.4f}±{f1.get('std',0):.3f}  "
          f"{mcc.get('mean',0):.4f}±{mcc.get('std',0):.3f}  "
          f"{auroc.get('mean',0):.4f}±{auroc.get('std',0):.3f}  "
          f"{auprc.get('mean',0):.4f}±{auprc.get('std',0):.3f}")

out_path = f"{BASE}/comparison.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved -> {out_path}")
