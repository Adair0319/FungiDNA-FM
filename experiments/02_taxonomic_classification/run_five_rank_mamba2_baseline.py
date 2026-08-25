#!/usr/bin/env python3
"""Single-layer Mamba2 baseline (one-hot DNA input) for five-rank 5-fold CV classification.

Architecture matches the splice-site Mamba2 baseline exactly:
  OneHot(10Kbp,4) → Linear(4→256) → RMSNorm → Mamba2(d=256, s=128, conv=4, expand=2) → MeanPool → Head

Usage:
  python scripts/run_five_rank_mamba2_baseline.py --rank phylum --fold all --gpu 0
  python scripts/run_five_rank_mamba2_baseline.py --rank all --fold all --gpu 0
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
                    choices=["phylum", "subphylum", "class", "order", "family", "all"])
parser.add_argument("--fold", type=str, default="all",
                    choices=["1", "2", "3", "4", "5", "all"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch", type=int, default=128,
                    help="Batch size (default 128, reduce to 64/32 on OOM)")
args = parser.parse_args()

DATA_BASE = PROJECT_ROOT / "data" / "downstream"
OUT_BASE = PROJECT_ROOT / "checkpoints" / "baseline_mamba2_1layer"

# ── Hyperparams (from splice baseline) ──
BATCH = args.batch
LR = 1e-3
EPOCHS = 30
WARMUP = 500
WD = 0.01
SEED = 42
D_MODEL = 256
EARLY_STOP_PATIENCE = 5

NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
           'a': 0, 'c': 1, 'g': 2, 't': 3}

device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES remaps to 0
torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════ Model ═══════════════════════
class UnifiedHead(nn.Module):
    """LayerNorm → Linear(256→256) → GELU → Dropout → Linear(256→C)"""
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
    """OneHot → Linear(4→256) → RMSNorm → Mamba2 → MeanPool → UnifiedHead"""
    def __init__(self, num_classes):
        super().__init__()
        self.embed = nn.Linear(4, D_MODEL)
        self.norm = nn.RMSNorm(D_MODEL)
        self.mamba = Mamba2(d_model=D_MODEL, d_state=128, d_conv=4, expand=2)
        self.head = UnifiedHead(D_MODEL, num_classes=num_classes)

    def forward(self, x):
        # x: (B, L, 4) one-hot
        x = self.embed(x)         # (B, L, 256)
        x = self.norm(x)
        x = self.mamba(x)         # (B, L, 256)
        x = x.mean(dim=1)         # (B, 256)
        return self.head(x)


# ═══════════════════════ Dataset ═══════════════════════
class OneHotDNADataset(Dataset):
    """Convert raw DNA strings to one-hot tensors (L, 4) using numpy vectorization."""
    def __init__(self, pq_path, seq_len=10000):
        import pandas as pd
        df = pd.read_parquet(pq_path, columns=["sequence", "label"])
        self.seqs = df["sequence"].values  # numpy array of strings
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.seq_len = seq_len
        # Lookup table for ACGT → one-hot (4-dim)
        self._lut = np.zeros((256, 4), dtype=np.float32)
        for c, i in NUC_MAP.items():
            self._lut[ord(c)] = i

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        # Convert bytes to indices using lookup
        n = min(len(s), self.seq_len)
        arr = np.frombuffer(s[:n].encode('ascii'), dtype=np.uint8)
        # Vectorized lookup: indices where ACGT, others → 0
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
    y = np.concatenate(L)
    prob = np.concatenate(P)
    pred = prob.argmax(-1)

    r = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }

    # Per-class
    prec, rec, f1s, _ = precision_recall_fscore_support(
        y, pred, labels=range(num_classes), zero_division=0)
    r["per_class"] = {}
    for ci in range(num_classes):
        r["per_class"][str(ci)] = {
            "precision": float(prec[ci]),
            "recall": float(rec[ci]),
            "f1": float(f1s[ci]),
        }

    # AUROC / AUPRC
    try:
        r["auroc_macro"] = float(roc_auc_score(
            y, prob, multi_class="ovr", average="macro"))
    except Exception:
        r["auroc_macro"] = float("nan")
    try:
        # Use one-hot for AUPRC
        y_oh = np.eye(num_classes, dtype=int)[y]
        r["auprc_macro"] = float(average_precision_score(
            y_oh, prob, average="macro"))
    except Exception:
        r["auprc_macro"] = float("nan")

    r["confusion_matrix"] = confusion_matrix(
        y, pred, labels=range(num_classes)).tolist()
    return r


# ═══════════════════════ Main ═══════════════════════
def run_rank_fold(rank, fold, num_classes, out_dir):
    """Train + eval one rank + one fold."""
    data_dir = DATA_BASE / f"task0_{rank}_five_rank" / f"fold{fold}"  # fold is 1-5
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Mamba2-1L | {rank} | Fold {fold} | GPU {args.gpu}")
    print(f"{'='*60}", flush=True)

    # Model
    model = Mamba2Baseline(num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)
    model = model.to(device)

    # Data
    print("Loading data...", flush=True)
    train_ds = OneHotDNADataset(str(data_dir / "train.parquet"))
    val_ds = OneHotDNADataset(str(data_dir / "val.parquet"))
    test_ds = OneHotDNADataset(str(data_dir / "test.parquet"))
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}", flush=True)

    train_ld = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_fn,
                          num_workers=0, drop_last=True, pin_memory=True)
    val_ld = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate_fn,
                        num_workers=0, pin_memory=True)
    test_ld = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_fn,
                         num_workers=0, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per_ep = len(train_ld)
    total_steps = steps_per_ep * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total_steps,
        pct_start=WARMUP / max(total_steps, 1))
    crit = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_epoch = -1
    no_improve = 0
    print(f"Training ({steps_per_ep} steps/ep, {EPOCHS} epochs, "
          f"early_stop={EARLY_STOP_PATIENCE}, batch={BATCH})...", flush=True)

    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0
        t0 = time.time()
        opt.zero_grad()
        for step, (x, labs) in enumerate(train_ld):
            x, labs = x.to(device), labs.to(device)
            loss = crit(model(x), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
            ep_loss += loss.item()
            if step % 200 == 0:
                print(f"  ep{ep+1} step{step}: loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

        t1 = time.time()
        avg_loss = ep_loss / max(step + 1, 1)

        vm = evaluate(model, val_ld, num_classes)
        improved = vm["macro_f1"] > best_f1
        es = f"es={no_improve+1}/{EARLY_STOP_PATIENCE}" if not improved else "es=0"
        print(f"Epoch {ep+1}/{EPOCHS} loss={avg_loss:.4f} [{t1-t0:.0f}s] {es}", flush=True)
        print(f"  Val: Macro-F1={vm['macro_f1']:.4f} MCC={vm['mcc']:.4f} "
              f"AUROC={vm.get('auroc_macro', '?'):.4f}", flush=True)

        if improved:
            best_f1 = vm["macro_f1"]
            best_epoch = ep
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": vm},
                       str(out_dir / "best_model.pt"))
            print(f"  -> saved (best F1={best_f1:.4f})", flush=True)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {ep+1}", flush=True)
                break

    # Final test
    print("\n=== Final Test ===", flush=True)
    best = torch.load(str(out_dir / "best_model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    tm = evaluate(model, test_ld, num_classes)
    tm["best_val_f1"] = best_f1
    tm["best_epoch"] = best_epoch
    tm["rank"] = rank
    tm["fold"] = fold
    tm["n_params"] = n_params
    print(f"Test: Macro-F1={tm['macro_f1']:.4f} MCC={tm['mcc']:.4f} "
          f"AUROC={tm.get('auroc_macro', '?'):.4f}", flush=True)
    with open(str(out_dir / "metrics.json"), "w") as f:
        json.dump(tm, f, indent=2)
    print(f"Done -> {out_dir}/", flush=True)
    return tm


def main():
    RANKS = ["phylum", "subphylum", "class", "order", "family"] if args.rank == "all" else [args.rank]
    folds = list(range(1, 6)) if args.fold == "all" else [int(args.fold)]

    # Load label_maps to get num_classes
    all_results = {}
    for rank in RANKS:
        label_map_path = DATA_BASE / f"task0_{rank}_five_rank" / "label_map.json"
        with open(label_map_path) as f:
            label_map = json.load(f)
        num_classes = len(label_map)
        print(f"\n{'#'*60}")
        print(f"# {rank}: {num_classes} classes")
        print(f"{'#'*60}")

        fold_results = []
        for fold in folds:
            out_dir = OUT_BASE / f"task0_{rank}_five_rank" / f"fold{fold}"
            tm = run_rank_fold(rank, fold, num_classes, out_dir)
            fold_results.append(tm)

        # Fold summary
        f1s = [r["macro_f1"] for r in fold_results]
        mccs = [r["mcc"] for r in fold_results]
        print(f"\n{rank} 5-fold summary:")
        print(f"  Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"  MCC:      {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
        all_results[rank] = {
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "mcc_mean": float(np.mean(mccs)),
            "mcc_std": float(np.std(mccs)),
            "per_fold": fold_results,
        }

    # Save summary
    summary_path = OUT_BASE / "summary.json"
    os.makedirs(OUT_BASE, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
