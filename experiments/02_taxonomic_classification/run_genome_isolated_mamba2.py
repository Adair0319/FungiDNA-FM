#!/usr/bin/env python3
"""Single-layer Mamba2 baseline (one-hot DNA) on genome-isolated datasets.

Same architecture as splice-site baseline and five_rank baseline:
  OneHot(10Kbp,4) → Linear(4→256) → RMSNorm → Mamba2(d=256,s=128,conv=4,expand=2) → MeanPool → Head

Usage:
  python scripts/run_genome_isolated_mamba2.py --rank phylum --gpu 0
  python scripts/run_genome_isolated_mamba2.py --rank subphylum --gpu 0
"""
import sys, os, json, time, argparse, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support,
)
from mamba_ssm import Mamba2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=str, required=True,
                    choices=["phylum", "subphylum", "class", "order", "family"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch", type=int, default=64)
args = parser.parse_args()

DATA_DIR = PROJECT_ROOT / "data" / "downstream" / f"task0_{args.rank}"
OUT_DIR = PROJECT_ROOT / "checkpoints" / "baseline_mamba2_1layer" / f"task0_{args.rank}"

# ── Hyperparams ──
BATCH = args.batch; LR = 1e-3; EPOCHS = 30; WARMUP = 500; WD = 0.01
SEED = 42; D_MODEL = 256; EARLY_STOP_PATIENCE = 5

NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
           'a': 0, 'c': 1, 'g': 2, 't': 3}

device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES remaps
torch.manual_seed(SEED); np.random.seed(SEED)


# ═══════════════════════ Model ═══════════════════════
class UnifiedHead(nn.Module):
    def __init__(self, input_dim=D_MODEL, hidden_dim=256, num_classes=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)


class Mamba2Baseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embed = nn.Linear(4, D_MODEL)
        self.norm = nn.RMSNorm(D_MODEL)
        self.mamba = Mamba2(d_model=D_MODEL, d_state=128, d_conv=4, expand=2)
        self.head = UnifiedHead(D_MODEL, num_classes=num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.mean(dim=1)
        return self.head(x)


# ═══════════════════════ Dataset ═══════════════════════
class OneHotDNADataset(Dataset):
    def __init__(self, pq_path, seq_len=10000):
        import pandas as pd
        df = pd.read_parquet(pq_path, columns=["sequence", "label"])
        self.seqs = df["sequence"].values
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.seq_len = seq_len
        self._lut = np.zeros((256, 4), dtype=np.float32)
        for c, i in NUC_MAP.items():
            self._lut[ord(c)] = i

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        n = min(len(s), self.seq_len)
        arr = np.frombuffer(s[:n].encode('ascii'), dtype=np.uint8)
        idx = self._lut[arr].argmax(axis=1)
        mask = self._lut[arr].sum(axis=1) > 0
        x = np.zeros((self.seq_len, 4), dtype=np.float32)
        valid_positions = np.where(mask)[0]
        x[valid_positions, idx[valid_positions]] = 1.0
        return torch.from_numpy(x), torch.tensor(int(self.labels[i]))


def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch])


# ═══════════════════════ Metrics ═══════════════════════
@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    L, P = [], []
    for x, labs in loader:
        logits = model(x.to(device))
        L.append(labs.numpy())
        P.append(F.softmax(logits, dim=-1).float().cpu().numpy())
    y = np.concatenate(L); prob = np.concatenate(P); pred = prob.argmax(-1)

    r = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
    prec, rec, f1s, _ = precision_recall_fscore_support(
        y, pred, labels=range(num_classes), zero_division=0)
    r["per_class"] = {}
    for ci in range(num_classes):
        r["per_class"][str(ci)] = {
            "precision": float(prec[ci]), "recall": float(rec[ci]), "f1": float(f1s[ci]),
        }
    try:
        r["auroc_macro"] = float(roc_auc_score(
            y, prob, multi_class="ovr", average="macro"))
    except Exception:
        r["auroc_macro"] = float("nan")
    try:
        y_oh = np.eye(num_classes, dtype=int)[y]
        r["auprc_macro"] = float(average_precision_score(y_oh, prob, average="macro"))
    except Exception:
        r["auprc_macro"] = float("nan")
    r["confusion_matrix"] = confusion_matrix(y, pred, labels=range(num_classes)).tolist()
    return r


# ═══════════════════════ Main ═══════════════════════
def main():
    # Load metadata
    label_map = json.load(open(DATA_DIR / "label_map.json"))
    num_classes = len(label_map)
    idx_to_name = {v: k for k, v in label_map.items()}
    ignore_labels = set(json.load(open(DATA_DIR / "ignore_labels.json")))

    print(f"Genome-isolated Mamba2-1L: {args.rank} ({num_classes} classes) GPU {args.gpu}")
    print(f"  Ignore labels: {ignore_labels}")

    # Data
    print("Loading data...", flush=True)
    train_full = OneHotDNADataset(str(DATA_DIR / "train.parquet"))
    test_ds = OneHotDNADataset(str(DATA_DIR / "test.parquet"))

    # Split 10% of train for validation
    rng = np.random.default_rng(SEED)
    n = len(train_full)
    idx = rng.permutation(n)
    n_val = n // 10
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]

    from torch.utils.data import Subset
    train_ds = Subset(train_full, tr_idx)
    val_ds = Subset(train_full, val_idx)

    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}", flush=True)

    train_ld = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_fn,
                          num_workers=0, drop_last=True, pin_memory=True)
    val_ld = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate_fn,
                        num_workers=0, pin_memory=True)
    test_ld = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_fn,
                         num_workers=0, pin_memory=True)

    # Model
    model = Mamba2Baseline(num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per_ep = len(train_ld)
    total_steps = steps_per_ep * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total_steps,
        pct_start=WARMUP / max(total_steps, 1))
    crit = nn.CrossEntropyLoss()

    best_f1 = 0.0; best_epoch = -1; no_improve = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Training ({steps_per_ep} steps/ep, {EPOCHS} epochs, "
          f"early_stop={EARLY_STOP_PATIENCE}, batch={BATCH})...", flush=True)

    for ep in range(EPOCHS):
        model.train(); ep_loss = 0; t0 = time.time(); opt.zero_grad()
        for step, (x, labs) in enumerate(train_ld):
            x, labs = x.to(device), labs.to(device)
            loss = crit(model(x), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sched.step()
            ep_loss += loss.item()
            if step % 200 == 0:
                print(f"  ep{ep+1} step{step}: loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

        t1 = time.time(); avg_loss = ep_loss / max(step + 1, 1)
        vm = evaluate(model, val_ld, num_classes)
        improved = vm["macro_f1"] > best_f1
        es = f"es={no_improve+1}/{EARLY_STOP_PATIENCE}" if not improved else "es=0"
        print(f"Epoch {ep+1}/{EPOCHS} loss={avg_loss:.4f} [{t1-t0:.0f}s] {es}", flush=True)
        print(f"  Val: Macro-F1={vm['macro_f1']:.4f} MCC={vm['mcc']:.4f}", flush=True)

        if improved:
            best_f1 = vm["macro_f1"]; best_epoch = ep; no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": vm},
                       str(OUT_DIR / "best_model.pt"))
            print(f"  -> saved (best F1={best_f1:.4f})", flush=True)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {ep+1}", flush=True); break

    print("\n=== Final Test ===", flush=True)
    best = torch.load(str(OUT_DIR / "best_model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    tm = evaluate(model, test_ld, num_classes)

    # Filter ignored labels from metrics
    y_true_all = []; y_pred_all = []
    for x, labs in test_ld:
        logits = model(x.to(device))
        y_true_all.append(labs.numpy()); y_pred_all.append(logits.argmax(-1).cpu().numpy())
    y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all)

    # Compute un-ignored macro F1
    valid_labels = [i for i in range(num_classes) if i not in ignore_labels]
    valid_mask = np.isin(y_true, valid_labels) & np.isin(y_pred, valid_labels)
    tm["macro_f1_noignore"] = float(f1_score(y_true, y_pred, labels=valid_labels,
                                              average="macro", zero_division=0))
    tm["rank"] = args.rank; tm["n_params"] = n_params; tm["best_val_f1"] = best_f1

    print(f"Test: Macro-F1={tm['macro_f1']:.4f} "
          f"Macro-F1(no ignore)={tm['macro_f1_noignore']:.4f} "
          f"MCC={tm['mcc']:.4f}", flush=True)

    # Save predictions
    import pandas as pd
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    rows = []
    for i in range(min(len(y_true), len(test_df))):
        rows.append({
            "genome_id": test_df.iloc[i]["genome_id"],
            "true_label": int(y_true[i]), "true_name": idx_to_name.get(int(y_true[i]), "?"),
            "pred_label": int(y_pred[i]), "pred_name": idx_to_name.get(int(y_pred[i]), "?"),
            "correct": int(y_true[i] == y_pred[i]),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "predictions.csv", index=False)

    json.dump(tm, open(OUT_DIR / "metrics.json", "w"), indent=2)
    print(f"Done -> {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
