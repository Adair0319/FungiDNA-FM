"""
Simple baselines: single-layer Mamba2 and single-layer Transformer.
From-scratch training with one-hot encoded 401bp DNA input.
"""
import sys, os, json, time, argparse, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
)
from mamba_ssm import Mamba2

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    choices=["mamba2", "transformer"])
parser.add_argument("--fold", type=str, required=True,
                    choices=["0","1","2","3","4","all"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--out-suffix", type=str, default="")
args = parser.parse_args()

DATA_BASE = "data/downstream/task2_splice_site_v2"
OUT_BASE = f"checkpoints/splice_v2/baseline_{args.model}_1layer"
if args.out_suffix:
    OUT_BASE = f"checkpoints/splice_v2/baseline_{args.model}_1layer_{args.out_suffix}"

BATCH = 256; LR = args.lr; EPOCHS = 30; WARMUP = 500; WD = 0.01; SEED = 42
D_MODEL = 256; EARLY_STOP_PATIENCE = 5
CLASS_NAMES = ["Donor", "Acceptor", "Non-Site"]
NUC_MAP = {'A':0,'C':1,'G':2,'T':3,'N':4,'a':0,'c':1,'g':2,'t':3,'n':4}

device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)
torch.manual_seed(SEED); np.random.seed(SEED)


# ═══════════════════════ Models ═══════════════════════
class UnifiedHead(nn.Module):
    def __init__(self, input_dim=D_MODEL, hidden_dim=256, num_classes=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.norm(x); x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)


class Mamba2Baseline(nn.Module):
    """OneHot → Linear(4→256) → RMSNorm → Mamba2 → MeanPool → UnifiedHead"""
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, D_MODEL)
        self.norm = nn.RMSNorm(D_MODEL)
        self.mamba = Mamba2(d_model=D_MODEL, d_state=128, d_conv=4, expand=2)
        self.head = UnifiedHead(D_MODEL)

    def forward(self, x):
        # x: (B, 401, 4) one-hot
        x = self.embed(x)         # (B, 401, 256)
        x = self.norm(x)
        x = self.mamba(x)         # (B, 401, 256)
        x = x.mean(dim=1)         # (B, 256)
        return self.head(x)


class TransformerBaseline(nn.Module):
    """OneHot → Linear(4→256) + SinPosEmb → Pre-LN Attention + FFN → MeanPool → UnifiedHead"""
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, D_MODEL)
        self.pos_emb = SinusoidalPositionalEncoding(D_MODEL, max_len=512)
        self.attn_norm = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, num_heads=8, batch_first=True)
        self.ffn_norm = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 4),
            nn.GELU(),
            nn.Linear(D_MODEL * 4, D_MODEL),
        )
        self.head = UnifiedHead(D_MODEL)

    def forward(self, x):
        # x: (B, 401, 4)
        x = self.embed(x)                          # (B, 401, 256)
        x = x + self.pos_emb(x)                    # add position info
        # Pre-LN Attention
        x = x + self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))[0]
        # Pre-LN FFN
        x = x + self.ffn(self.ffn_norm(x))
        x = x.mean(dim=1)                          # (B, 256)
        return self.head(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


# ═══════════════════════ Dataset ═══════════════════════
class DNADataset(Dataset):
    def __init__(self, pq_path):
        import pandas as pd
        df = pd.read_parquet(pq_path, columns=["sequence", "label"])
        self.seqs = df["sequence"].tolist()
        self.labels = df["label"].to_numpy(dtype=np.int64)

    def __len__(self): return len(self.seqs)

    def __getitem__(self, i):
        x = torch.zeros(401, 4, dtype=torch.float32)
        for j, c in enumerate(self.seqs[i]):
            if c in NUC_MAP and NUC_MAP[c] < 4:
                x[j, NUC_MAP[c]] = 1.0
        return x, torch.tensor(int(self.labels[i]))


# ═══════════════════════ Metrics ═══════════════════════
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    L, P = [], []
    for x, labs in loader:
        logits = model(x.to(device))
        L.append(labs.numpy())
        P.append(F.softmax(logits, dim=-1).float().cpu().numpy())
    y = np.concatenate(L); prob = np.concatenate(P); pred = prob.argmax(-1)
    r = {"macro_f1": float(f1_score(y, pred, average="macro")),
         "mcc": float(matthews_corrcoef(y, pred)),
         "confusion_matrix": confusion_matrix(y, pred, labels=[0,1,2]).tolist()}
    try: r["auroc_macro"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except: r["auroc_macro"] = float("nan")
    try: r["auprc_macro"] = float(average_precision_score(np.eye(3,dtype=int)[y], prob, average="macro"))
    except: r["auprc_macro"] = float("nan")
    for ci, nm in enumerate(CLASS_NAMES):
        yb = (y==ci).astype(int); pb = (pred==ci).astype(int)
        r[f"{nm}_F1"] = float(f1_score(yb,pb,zero_division=0))
        r[f"{nm}_Recall"] = float(yb.sum() and (yb&pb).sum()/max(yb.sum(),1) or 0)
        try: r[f"{nm}_AUROC"] = float(roc_auc_score(yb, prob[:,ci]))
        except: r[f"{nm}_AUROC"] = float("nan")
        try: r[f"{nm}_AUPRC"] = float(average_precision_score(yb, prob[:,ci]))
        except: r[f"{nm}_AUPRC"] = float("nan")
    return r


# ═══════════════════════ Main ═══════════════════════
def main():
    folds = range(5) if args.fold == "all" else [int(args.fold)]

    for fold in folds:
        data_dir = f"{DATA_BASE}/fold_{fold}"
        out_dir = f"{OUT_BASE}/fold_{fold}"
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"=== {args.model} 1-layer | Fold {fold} | GPU {args.gpu} ===", flush=True)

        # Build fresh model per fold
        model = Mamba2Baseline() if args.model == "mamba2" else TransformerBaseline()
        n = sum(p.numel() for p in model.parameters())
        print(f"Model: {args.model}, Params: {n:,}", flush=True)
        model = model.to(device)

        # Data
        print("Loading data...", flush=True)
        train_ds = DNADataset(f"{data_dir}/train.parquet")
        val_ds   = DNADataset(f"{data_dir}/val.parquet")
        test_ds  = DNADataset(f"{data_dir}/test.parquet")
        print(f"Train: {len(train_ds):,} Val: {len(val_ds):,} Test: {len(test_ds):,}", flush=True)

        train_ld = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])),
                              num_workers=0, drop_last=True, pin_memory=True)
        val_ld   = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])),
                              num_workers=0, pin_memory=True)
        test_ld  = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])),
                              num_workers=0, pin_memory=True)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        steps_per_ep = len(train_ld)
        total_steps = steps_per_ep * EPOCHS
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total_steps,
            pct_start=WARMUP/max(total_steps,1))
        crit = nn.CrossEntropyLoss()

        best_f1 = 0.0; best_epoch = -1; no_improve = 0
        print(f"Training ({steps_per_ep} steps/epoch, {EPOCHS} epochs, early_stop={EARLY_STOP_PATIENCE})...", flush=True)

        for ep in range(EPOCHS):
            model.train(); ep_loss = 0; t0 = time.time(); opt.zero_grad()
            for step, (x, labs) in enumerate(train_ld):
                x, labs = x.to(device), labs.to(device)
                loss = crit(model(x), labs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); sched.step()
                ep_loss += loss.item()
                if step % 500 == 0:
                    print(f"  ep{ep+1} step{step}: loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)

            t1 = time.time(); avg_loss = ep_loss/max(step+1,1)
            torch.save({"model": model.state_dict(), "epoch": ep}, f"{out_dir}/checkpoint_ep{ep+1}.pt")
            vm = evaluate(model, val_ld)
            improved = vm["macro_f1"] > best_f1
            es = f"es={no_improve+1}/{EARLY_STOP_PATIENCE}" if not improved else "es=0"
            print(f"Epoch {ep+1}/{EPOCHS} loss={avg_loss:.4f} [{t1-t0:.0f}s] {es}", flush=True)
            print(f"Val: Macro-F1={vm['macro_f1']:.4f} MCC={vm['mcc']:.4f} "
                  f"AUROC={vm['auroc_macro']:.4f} AUPRC={vm['auprc_macro']:.4f}", flush=True)

            if improved:
                best_f1 = vm["macro_f1"]; best_epoch = ep; no_improve = 0
                torch.save({"model": model.state_dict(), "epoch": ep, "metrics": vm}, f"{out_dir}/best_model.pt")
                print(f"  -> saved (best F1={best_f1:.4f})", flush=True)
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {ep+1}", flush=True); break

        print("\n=== Final Test ===", flush=True)
        best = torch.load(f"{out_dir}/best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model"])
        tm = evaluate(model, test_ld)
        tm["best_val_f1"] = best_f1; tm["best_epoch"] = best_epoch
        print(f"Test: Macro-F1={tm['macro_f1']:.4f} MCC={tm['mcc']:.4f} "
              f"AUROC={tm['auroc_macro']:.4f} AUPRC={tm['auprc_macro']:.4f}", flush=True)
        with open(f"{out_dir}/metrics.json","w") as f: json.dump(tm, f, indent=2)
        print(f"Done -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
