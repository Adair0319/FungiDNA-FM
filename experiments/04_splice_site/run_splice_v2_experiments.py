"""
Unified splice site v2 experiment runner.
Supports 5 experiment types with 5-fold CV on the v2 dataset.

Usage:
  python scripts/run_splice_v2_experiments.py --exp randfull --fold 0 --gpu 0
  python scripts/run_splice_v2_experiments.py --exp onehot  --fold all --gpu 0
  python scripts/run_splice_v2_experiments.py --exp frozen  --fold 2 --gpu 1
"""
import sys, os, json, time, argparse, math
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

sys.path.insert(0, "/home/lty/yy_projects/fungi_project/fungi_dna_model")

from fungidna.model.config import FungiDNAConfig
from fungidna.model.fungi_dna import FungiDNAForSequenceClassification
from fungidna.data.tokenizer import DualTokenizer

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, required=True,
                    choices=["randfull", "frozen", "fullft", "onehot", "cnn"])
parser.add_argument("--fold", type=str, required=True,
                    choices=["0", "1", "2", "3", "4", "all"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--fewshot_pct", type=int, default=None, choices=[10,20,30,40])
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--out-suffix", type=str, default="")
args = parser.parse_args()

# ── Experiment Configs ──
EXP_CONFIGS = {
    "randfull": {"use_backbone": True, "pretrained": False, "freeze_backbone": False,
                 "lr": 1e-4, "epochs": 30, "warmup": 800, "use_amp": True},
    "frozen":   {"use_backbone": True, "pretrained": True,  "freeze_backbone": True,
                 "lr": 1e-3, "epochs": 30, "warmup": 800, "use_amp": True},
    "fullft":   {"use_backbone": True, "pretrained": True,  "freeze_backbone": False,
                 "lr": 5e-6, "epochs": 30, "warmup": 800, "use_amp": True},
    "onehot":   {"use_backbone": False, "pretrained": False, "freeze_backbone": False,
                 "lr": 1e-3, "epochs": 30, "warmup": 500, "use_amp": False},
    "cnn":      {"use_backbone": False, "pretrained": False, "freeze_backbone": False,
                 "lr": 5e-4, "epochs": 30, "warmup": 500, "use_amp": False},
}
EARLY_STOP_PATIENCE = 5

cfg = EXP_CONFIGS[args.exp]
if args.lr is not None:
    cfg["lr"] = args.lr
BATCH = 256; WD = 0.01; SEED = 42
DATA_BASE = "data/downstream/task2_splice_site_v2"
CKPT_PATH = "checkpoints/phase2_joint/backbone_final.pt"
TOK_PATH = "data/processed/bpe_fungi.model"
OUT_BASE = f"checkpoints/splice_v2/{args.exp}"
if args.fewshot_pct:
    OUT_BASE = f"checkpoints/splice_v2/fewshot/ours_fullft_{args.fewshot_pct}pct"
if args.out_suffix:
    OUT_BASE = f"checkpoints/splice_v2/{args.exp}_{args.out_suffix}"
CLASS_NAMES = ["Donor", "Acceptor", "Non-Site"]
NUC_MAP = {'A':0,'C':1,'G':2,'T':3,'N':4,'a':0,'c':1,'g':2,'t':3,'n':4}

device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)


# ═══════════════════════════════════════════════════════════
# Dataset: Backbone (pre-tokenized npz, eagerly copied to avoid mmap in workers)
# ═══════════════════════════════════════════════════════════
class BPEDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.ids = np.array(data["ids"])       # eager copy
        self.lens = np.array(data["lens"])
        self.labels = np.array(data["labels"])

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        n = int(self.lens[i])
        return torch.from_numpy(self.ids[i, :n].astype(np.int64)), torch.tensor(int(self.labels[i]))

def bpe_collate(batch):
    ids, labs = zip(*batch)
    return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0), torch.stack(labs)


# ═══════════════════════════════════════════════════════════
# Dataset: Baseline (one-hot encode on-the-fly)
# ═══════════════════════════════════════════════════════════
class OneHotDataset(Dataset):
    def __init__(self, pq_path):
        df = pd.read_parquet(pq_path, columns=["sequence", "label"])
        self.seqs = df["sequence"].tolist()
        self.labels = df["label"].to_numpy(dtype=np.int64)

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        x = torch.zeros(401, 4, dtype=torch.float32)
        for j, c in enumerate(self.seqs[i]):
            if c in NUC_MAP and NUC_MAP[c] < 4:
                x[j, NUC_MAP[c]] = 1.0
        return x, torch.tensor(self.labels[i])

def oh_collate(batch):
    x, y = zip(*batch)
    return torch.stack(x), torch.stack(y)


# ═══════════════════════════════════════════════════════════
# Unified Classification Head: Norm -> Linear(dim->256) -> GELU -> Dropout -> Linear(256->3)
# ═══════════════════════════════════════════════════════════
class UnifiedHead(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=256, dropout=0.1):
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


# ═══════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════

class OneHotMLP(nn.Module):
    """Flatten(1604) -> UnifiedHead(1604->256->3)"""
    def __init__(self):
        super().__init__()
        self.head = UnifiedHead(1604)

    def forward(self, x):
        return self.head(x.view(x.shape[0], -1))


class CNNMLP(nn.Module):
    """3-layer Conv1D -> AdaptiveMaxPool(256) -> UnifiedHead(256->256->3)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 64, 9, padding=4), nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 5, padding=2), nn.BatchNorm1d(256), nn.GELU(), nn.AdaptiveMaxPool1d(1),
        )
        self.head = UnifiedHead(256)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,401,4) -> (B,4,401)
        x = self.conv(x).squeeze(-1)  # (B,256)
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# Rich Evaluation
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, use_amp=False):
    model.eval()
    L, P = [], []
    for batch in loader:
        if cfg["use_backbone"]:
            ids, labs = batch
            ids, labs = ids.to(device), labs.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad():
                logits = model(ids)
        else:
            x, labs = batch
            x, labs = x.to(device), labs.to(device)
            logits = model(x)
        L.append(labs.cpu().numpy())
        P.append(F.softmax(logits, dim=-1).float().cpu().numpy())

    y = np.concatenate(L)
    prob = np.concatenate(P)
    pred = prob.argmax(-1)
    cm = confusion_matrix(y, pred, labels=[0,1,2])

    r = {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "mcc": float(matthews_corrcoef(y, pred)),
        "confusion_matrix": cm.tolist(),
    }

    # AUROC macro ovr
    try:
        r["auroc_macro"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except: r["auroc_macro"] = float("nan")

    # AUPRC macro
    try:
        r["auprc_macro"] = float(average_precision_score(
            np.eye(3, dtype=int)[y], prob, average="macro"))
    except: r["auprc_macro"] = float("nan")

    # Per-class metrics
    for ci, nm in enumerate(CLASS_NAMES):
        yb = (y == ci).astype(int)
        pb = (pred == ci).astype(int)
        r[f"{nm}_F1"] = float(f1_score(yb, pb, zero_division=0))
        r[f"{nm}_Recall"] = float(yb.sum() and (yb & pb).sum() / max(yb.sum(), 1) or 0)
        r[f"{nm}_Precision"] = float(pb.sum() and (yb & pb).sum() / max(pb.sum(), 1) or 0)
        try:
            r[f"{nm}_AUROC"] = float(roc_auc_score(yb, prob[:, ci]))
        except: r[f"{nm}_AUROC"] = float("nan")
        try:
            r[f"{nm}_AUPRC"] = float(average_precision_score(yb, prob[:, ci]))
        except: r[f"{nm}_AUPRC"] = float("nan")

    return r


def format_metrics(m):
    lines = [f"Macro-F1={m['macro_f1']:.4f}  MCC={m['mcc']:.4f}  "
             f"AUROC={m['auroc_macro']:.4f}  AUPRC={m['auprc_macro']:.4f}"]
    for nm in CLASS_NAMES:
        lines.append(f"  {nm}: F1={m[f'{nm}_F1']:.4f}  Recall={m[f'{nm}_Recall']:.4f}  "
                     f"Precision={m[f'{nm}_Precision']:.4f}  "
                     f"AUROC={m[f'{nm}_AUROC']:.4f}  AUPRC={m[f'{nm}_AUPRC']:.4f}")
    lines.append(f"  Confusion Matrix:\n{m['confusion_matrix']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Main Training
# ═══════════════════════════════════════════════════════════
def main():
    folds = range(5) if args.fold == "all" else [int(args.fold)]

    for fold in folds:
        fold_seed = SEED + fold
        torch.manual_seed(fold_seed); np.random.seed(fold_seed)
        data_dir = f"{DATA_BASE}/fold_{fold}"
        out_dir = f"{OUT_BASE}/fold_{fold}"
        os.makedirs(out_dir, exist_ok=True)

        exp_name = args.exp
        lr = cfg["lr"]
        epochs = cfg["epochs"]
        warmup = cfg["warmup"]
        use_amp = cfg["use_amp"]
        use_backbone = cfg["use_backbone"]

        print(f"\n=== {exp_name} | Fold {fold} | GPU {args.gpu} | LR={lr} | Epochs={epochs} ===", flush=True)

        # ── Data ──
        print("Loading data...", flush=True)
        if use_backbone:
            train_file = f"{data_dir}/train_{args.fewshot_pct}pct_tokenized.npz" if args.fewshot_pct else f"{data_dir}/train_tokenized.npz"
            train_ds = BPEDataset(train_file)
            val_ds   = BPEDataset(f"{data_dir}/val_tokenized.npz")
            test_ds  = BPEDataset(f"{data_dir}/test_tokenized.npz")
            collate_fn = bpe_collate
        else:
            train_ds = OneHotDataset(f"{data_dir}/train.parquet")
            val_ds   = OneHotDataset(f"{data_dir}/val.parquet")
            test_ds  = OneHotDataset(f"{data_dir}/test.parquet")
            collate_fn = oh_collate

        print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}", flush=True)

        train_ld = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate_fn,
                              num_workers=0, drop_last=True, pin_memory=True)
        val_ld   = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate_fn,
                              num_workers=0, pin_memory=True)
        test_ld  = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate_fn,
                              num_workers=0, pin_memory=True)

        # ── Model ──
        print("Building model...", flush=True)
        if use_backbone:
            model = FungiDNAForSequenceClassification(FungiDNAConfig(), num_classes=3)
            # Replace head with UnifiedHead + MeanPool
            model.head = UnifiedHead(FungiDNAConfig().hidden_size)
            # Override forward for mean pooling
            def forward_meanpool(input_ids):
                hidden = model.backbone(input_ids, token_type=0)
                pooled = hidden[:, 1:, :].mean(dim=1)  # mean over non-CLS tokens
                return model.head(pooled)
            model.forward = forward_meanpool

            if cfg["pretrained"]:
                print(f"Loading pretrained weights from {CKPT_PATH}...", flush=True)
                cpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
                miss, unexp = model.backbone.load_state_dict(cpt, strict=False)
                print(f"  Backbone: {len(miss)} missing, {len(unexp)} unexpected", flush=True)

            if cfg["freeze_backbone"]:
                for p in model.backbone.parameters():
                    p.requires_grad = False
            else:
                for p in model.backbone.parameters():
                    p.requires_grad = True
        else:
            if exp_name == "onehot":
                model = OneHotMLP()
            else:
                model = CNNMLP()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/max(total,1):.1f}%)", flush=True)
        model = model.to(device)

        # ── GPU warmup: trigger Triton autotuning before DataLoader ──
        if use_backbone:
            print("Warming up GPU (Triton autotuning)...", flush=True)
            with torch.no_grad():
                for slen in [50, 60, 70, 80]:
                    dummy = torch.randint(0, 4096, (BATCH, slen), device=device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad():
                        _ = model(dummy)
            print("Warmup complete.", flush=True)

        # ── Optimizer ──
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=lr, weight_decay=WD)
        steps_per_ep = len(train_ld)
        total_steps = steps_per_ep * epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps,
            pct_start=warmup/max(total_steps, 1))
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        crit = nn.CrossEntropyLoss()
        best_f1 = 0.0

        # ── Training Loop ──
        log_lines = []
        best_f1 = 0.0
        best_epoch = -1
        no_improve_count = 0
        print(f"Training ({steps_per_ep} steps/epoch, {epochs} epochs, early_stop={EARLY_STOP_PATIENCE})...", flush=True)
        for ep in range(epochs):
            model.train(); ep_loss = 0; t0 = time.time()
            for step, batch in enumerate(train_ld):
                if use_backbone:
                    ids, labs = batch; ids, labs = ids.to(device), labs.to(device)
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            loss = crit(model(ids), labs)
                    else:
                        loss = crit(model(ids), labs)
                else:
                    x, labs = batch; x, labs = x.to(device), labs.to(device)
                    loss = crit(model(x), labs)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                opt.zero_grad(); sched.step()
                ep_loss += loss.item()
                if step % 500 == 0:
                    print(f"  ep{ep+1} step{step}: loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)

            t1 = time.time()
            avg_loss = ep_loss / max(step + 1, 1)
            torch.save({"model": model.state_dict(), "epoch": ep, "loss": avg_loss},
                       f"{out_dir}/checkpoint_ep{ep+1}.pt")

            vm = evaluate(model, val_ld, use_amp)
            improved = vm["macro_f1"] > best_f1
            es_info = f"es={no_improve_count+1}/{EARLY_STOP_PATIENCE}" if not improved else "es=0"
            line = (f"Epoch {ep+1}/{epochs} loss={avg_loss:.4f} [{t1-t0:.0f}s] {es_info}\n"
                    f"Val: {format_metrics(vm)}")
            print(line, flush=True)
            log_lines.append(line)

            if improved:
                best_f1 = vm["macro_f1"]
                best_epoch = ep
                no_improve_count = 0
                torch.save({"model": model.state_dict(), "epoch": ep, "metrics": vm},
                           f"{out_dir}/best_model.pt")
                print(f"  -> saved (best F1={best_f1:.4f})", flush=True)
            else:
                no_improve_count += 1
                if no_improve_count >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {ep+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)", flush=True)
                    break

        # ── Final Test ──
        print("\n=== Final Test ===", flush=True)
        best = torch.load(f"{out_dir}/best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model"])
        tm = evaluate(model, test_ld, use_amp)
        tm["best_val_f1"] = best_f1
        tm["best_epoch"] = best_epoch
        tm["stopped_early"] = (no_improve_count >= EARLY_STOP_PATIENCE)
        print(f"Test: {format_metrics(tm)}", flush=True)
        with open(f"{out_dir}/metrics.json", "w") as f:
            json.dump(tm, f, indent=2)

        # Save training log
        with open(f"{out_dir}/train_log.txt", "w") as f:
            f.write("\n".join(log_lines))

        print(f"Done -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
