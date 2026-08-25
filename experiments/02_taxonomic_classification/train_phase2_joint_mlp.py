#!/usr/bin/env python3
"""Evaluate Phase 2 joint backbone on five-rank 5-fold CV classification.

Usage: python scripts/train_phase2_joint_mlp.py --rank phylum --gpu 0
"""
import sys, os, json, math, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, average_precision_score, matthews_corrcoef,
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Warmup
try:
    from flash_attn import flash_attn_func
    d = torch.device("cuda")
    q = torch.randn(1, 1, 8, 64, device=d, dtype=torch.bfloat16)
    flash_attn_func(q, q, q, causal=False)
except Exception: pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fungidna.model.config import FungiDNAConfig
from fungidna.model.striped_mamba import StripedMambaBackbone
from fungidna.data.tokenizer import DualTokenizer
from fungidna.data.pks_dataset import PKSDataset


def extract_features(backbone, df, tokenizer, device):
    """Extract mean-pooled 768-dim features from frozen backbone."""
    backbone.eval()
    ds = PKSDataset(df, tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=lambda b: {
                            "input_ids": torch.nn.utils.rnn.pad_sequence(
                                [x["input_ids"] for x in b], batch_first=True, padding_value=0),
                            "label": torch.tensor([x["label"] for x in b]),
                        })
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            hidden = backbone(batch["input_ids"].to(device), token_type=0)
            feats.append(hidden.mean(dim=1).cpu().float().numpy())
            labels.append(batch["label"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        targets_safe = targets.clamp(min=0)
        alpha_w = self.alpha[targets_safe]
        alpha_w[targets == self.ignore_index] = 0.0
        focal = alpha_w * (1 - pt) ** self.gamma * ce
        valid = targets != self.ignore_index
        return focal[valid].mean() if valid.any() else torch.tensor(0.0, device=logits.device)


class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_fold(X_train, y_train, X_val, y_val, X_test, y_test, test_df,
               num_classes, device, fold_dir, idx_to_name):
    """Train MLP on one fold, return test metrics + save predictions CSV."""
    # Focal Loss alpha
    class_counts = Counter(y_train.tolist())
    total = sum(class_counts.values())
    alpha = torch.tensor(
        [1.0 / math.sqrt(max(class_counts.get(i, 1), 1) / total) for i in range(num_classes)],
        device=device,
    )

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)

    model = MLP(hidden_dim=256, num_classes=num_classes, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = FocalLoss(alpha, gamma=2.0)

    ds_tr = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds_tr, batch_size=256, shuffle=True)

    best_val_f1 = -1
    best_state = None
    patience = 0

    for epoch in range(50):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v).argmax(-1)
            val_f1 = f1_score(y_v.cpu(), val_pred.cpu(), average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    # Load best & evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(-1).cpu().numpy()

    y_true = y_test
    y_pred = preds
    prec, rec, f1_arr, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes), zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Per-class metrics
    per_class = {}
    auroc_vals, auprc_vals = [], []
    for lbl in range(num_classes):
        y_bin = (y_true == lbl).astype(int)
        auroc_val = float("nan")
        auprc_val = float("nan")
        if y_bin.sum() > 0:
            try:
                fpr_, tpr_, _ = roc_curve(y_bin, probs[:, lbl])
                auroc_val = auc(fpr_, tpr_)
                if not np.isnan(auroc_val): auroc_vals.append(auroc_val)
            except: pass
            try:
                auprc_val = average_precision_score(y_bin, probs[:, lbl])
                if not np.isnan(auprc_val): auprc_vals.append(auprc_val)
            except: pass
        per_class[lbl] = {
            "precision": float(prec[lbl]), "recall": float(rec[lbl]), "f1": float(f1_arr[lbl]),
            "auroc": float(auroc_val) if not np.isnan(auroc_val) else None,
            "auprc": float(auprc_val) if not np.isnan(auprc_val) else None,
        }
    macro_auroc = float(np.mean(auroc_vals)) if auroc_vals else None
    macro_auprc = float(np.mean(auprc_vals)) if auprc_vals else None

    torch.save(best_state, os.path.join(fold_dir, "mlp_best.pt"))

    # ── Save sequence-level predictions CSV ──
    pred_rows = []
    for i in range(len(y_true)):
        row = {
            "sequence": test_df.iloc[i]["sequence"],
            "genome_id": test_df.iloc[i]["genome_id"],
            "true_label": int(y_true[i]),
            "true_name": idx_to_name.get(int(y_true[i]), "?"),
            "pred_label": int(y_pred[i]),
            "pred_name": idx_to_name.get(int(y_pred[i]), "?"),
            "correct": int(y_true[i] == y_pred[i]),
        }
        for lbl in range(num_classes):
            row[f"prob_{idx_to_name.get(lbl, lbl)}"] = float(probs[i, lbl])
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)
    print(f"    Saved {len(pred_rows)} predictions to predictions.csv")

    return {
        "macro_f1": float(macro_f1), "mcc": mcc,
        "macro_auroc": macro_auroc, "macro_auprc": macro_auprc,
        "per_class": per_class, "best_val_f1": float(best_val_f1),
    }, probs, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", required=True, choices=["phylum","subphylum","class","order","family"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backbone", default="backbone_final.pt",
                        help="backbone checkpoint filename in checkpoints/phase2_joint/")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    rank = args.rank
    data_base = PROJECT_ROOT / "data" / "downstream" / f"task0_{rank}_five_rank"
    backbone_tag = args.backbone.replace(".pt", "").replace("backbone-", "").replace("backbone_", "")
    out_base = PROJECT_ROOT / "checkpoints" / f"phase2_joint_eval_{backbone_tag}" / rank
    os.makedirs(out_base, exist_ok=True)

    # Load label_map
    label_map = json.load(open(data_base / "label_map.json"))
    num_classes = len(label_map)
    idx_to_name = {v: k for k, v in label_map.items()}

    print(f"Phase 2 Joint Eval — {rank} ({num_classes} classes) on GPU {args.gpu}")

    # Load backbone
    config = FungiDNAConfig()
    backbone = StripedMambaBackbone(config).to(device).bfloat16()
    backbone_path = PROJECT_ROOT / "checkpoints/phase2_joint" / args.backbone
    ckpt = torch.load(str(backbone_path),
                      map_location="cpu", weights_only=False)
    # backbone_final.pt is direct state dict (no 'backbone.' prefix)
    missing, unexpected = backbone.load_state_dict(ckpt, strict=False)
    print(f"Backbone: {len(missing)} missing, {len(unexpected)} unexpected")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    tokenizer = DualTokenizer(str(PROJECT_ROOT / "data/processed/bpe_fungi.model"))

    fold_metrics = []
    for fold_id in range(1, 6):
        print(f"\n--- Fold {fold_id}/5 ---")
        fold_dir = out_base / f"fold{fold_id}"
        os.makedirs(fold_dir, exist_ok=True)

        fd = data_base / f"fold{fold_id}"
        train_df = pd.read_parquet(fd / "train.parquet")
        val_df   = pd.read_parquet(fd / "val.parquet")
        test_df  = pd.read_parquet(fd / "test.parquet")

        print(f"  Extracting features: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
        X_train, y_train = extract_features(backbone, train_df, tokenizer, device)
        X_val,   y_val   = extract_features(backbone, val_df,   tokenizer, device)
        X_test,  y_test  = extract_features(backbone, test_df,  tokenizer, device)

        # Save features for future reuse
        np.save(fold_dir / "X_train.npy", X_train)
        np.save(fold_dir / "X_val.npy", X_val)
        np.save(fold_dir / "X_test.npy", X_test)
        np.save(fold_dir / "y_train.npy", y_train)
        np.save(fold_dir / "y_val.npy", y_val)
        np.save(fold_dir / "y_test.npy", y_test)

        print(f"  Training MLP...")
        metrics, probs, y_true, y_pred = train_fold(
            X_train, y_train, X_val, y_val, X_test, y_test, test_df,
            num_classes, device, fold_dir, idx_to_name)

        fold_metrics.append(metrics)
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")

        # Confusion matrix per fold (with annotations)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        names = [idx_to_name[i] for i in range(num_classes)]
        for norm, suf in [(False, "raw"), (True, "normalized")]:
            fig, ax = plt.subplots(figsize=(max(10, num_classes*0.7), max(9, num_classes*0.6)))
            pcm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) if norm else cm
            annot = np.empty_like(cm, dtype=object)
            for i in range(num_classes):
                for j in range(num_classes):
                    if cm[i, j] > 0:
                        annot[i, j] = f"{cm[i,j]:,}" if not norm else f"{pcm[i,j]:.2f}"
                    else:
                        annot[i, j] = ""
            sns.heatmap(pcm if norm else cm, annot=annot, fmt="", cmap="YlOrRd",
                        xticklabels=names, yticklabels=names, ax=ax, linewidths=0.3,
                        cbar_kws={"label": "Proportion" if norm else "Count"},
                        annot_kws={"fontsize": 7 if num_classes <= 30 else 4})
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{rank.capitalize()} Fold{fold_id} CM{' (Norm)' if norm else ''}")
            plt.xticks(rotation=45, ha="right", fontsize=7 if num_classes <= 30 else 4)
            plt.yticks(fontsize=7 if num_classes <= 30 else 4)
            plt.tight_layout()
            fig.savefig(fold_dir / f"cm_{suf}.png", dpi=150, bbox_inches="tight")
            plt.close()

    # ── Summary (all metrics on TEST set) ──
    def mean_std(values, name=None):
        valid = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not valid: return None
        return {"mean": float(np.mean(valid)), "std": float(np.std(valid))}

    summary = {
        "rank": rank, "num_classes": num_classes,
        "macro_f1":   mean_std([m["macro_f1"] for m in fold_metrics]),
        "mcc":        mean_std([m["mcc"] for m in fold_metrics]),
        "macro_auroc": mean_std([m["macro_auroc"] for m in fold_metrics]),
        "macro_auprc": mean_std([m["macro_auprc"] for m in fold_metrics]),
        "per_fold": [{
            "fold": i+1,
            "macro_f1": m["macro_f1"], "mcc": m["mcc"],
            "macro_auroc": m["macro_auroc"], "macro_auprc": m["macro_auprc"],
            "best_val_f1": m["best_val_f1"],
        } for i, m in enumerate(fold_metrics)],
    }

    # Per-class summary (aggregate across folds)
    all_classes = set()
    for m in fold_metrics:
        all_classes.update(m["per_class"].keys())
    for lbl in sorted(all_classes):
        name = idx_to_name.get(lbl, str(lbl))
        summary.setdefault("per_class", {})[name] = {}
        for metric in ["f1", "precision", "recall", "auroc", "auprc"]:
            vals = [m["per_class"].get(lbl, {}).get(metric) for m in fold_metrics]
            summary["per_class"][name][f"{metric}_mean"] = mean_std(vals)
            if mean_std(vals):
                summary["per_class"][name][f"{metric}_mean"] = mean_std(vals)["mean"]
                summary["per_class"][name][f"{metric}_std"] = mean_std(vals)["std"]

    with open(out_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"{rank}: Macro-F1 = {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
