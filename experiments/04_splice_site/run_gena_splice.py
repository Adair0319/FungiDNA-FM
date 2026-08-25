"""
GENA-LM splice site prediction — correct tokenization (raw DNA, no spaces).
FullFT with unified classification head.
"""
import sys, os, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
)

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=str, required=True, choices=["0","1","2","3","4","all"])
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--mode", type=str, default="fullft", choices=["fullft", "frozen"])
parser.add_argument("--fewshot_pct", type=int, default=None, choices=[10,20,30,40])
args = parser.parse_args()

MODEL_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/baselines/models/gena-lm-bert-base-yeast"
DATA_BASE = "data/downstream/task2_splice_site_v2"
OUT_BASE = f"checkpoints/splice_v2/baseline_gena_{args.mode}"
if args.fewshot_pct:
    OUT_BASE = f"checkpoints/splice_v2/fewshot/gena_fullft_{args.fewshot_pct}pct"

if args.mode == "frozen":
    BATCH = 256; LR = 1e-3; EPOCHS = 30; WARMUP = 800
else:
    BATCH = 128; LR = 3e-5; EPOCHS = 30; WARMUP = 800
WD = 0.01; SEED = 42
EARLY_STOP_PATIENCE = 5
CLASS_NAMES = ["Donor", "Acceptor", "Non-Site"]

device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)
torch.manual_seed(SEED); np.random.seed(SEED)


# ═══════════════════════ Model ═══════════════════════
class UnifiedHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.norm(x); x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)


def build_gena_model():
    """Load gena with correct weight remapping for Pre-LN architecture."""
    sys.path.insert(0, MODEL_DIR)
    from transformers import AutoTokenizer, AutoModel

    print("Loading gena (trust_remote_code=True, custom Pre-LN BERT)...", flush=True)
    full_model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    encoder = full_model.bert  # base BERT, Pre-LN architecture

    # Verify weight loading
    ckpt = torch.load(f"{MODEL_DIR}/pytorch_model.bin", map_location="cpu", weights_only=True)
    own = encoder.state_dict()
    loaded = 0
    for ck, cv in ckpt.items():
        if not ck.startswith("bert."): continue
        mk = ck[5:]  # strip "bert." prefix
        if mk in own and own[mk].shape == cv.shape:
            loaded += 1
    total = len(own)
    print(f"  Weights loaded: {loaded}/{total} ({100*loaded//total}%)", flush=True)
    if loaded < total:
        print(f"  WARNING: {total-loaded} weights not loaded from checkpoint", flush=True)

    class GenaClassifier(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, batch):
            out = self.encoder(**batch)
            h = out.last_hidden_state
            return self.head(h.mean(dim=1))

    head = UnifiedHead(768)
    model = GenaClassifier(encoder, head)

    if args.mode == "frozen":
        for p in encoder.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total ({100*trainable//total}%)", flush=True)
    return model, tokenizer


# ═══════════════════════ Data ═══════════════════════
class GenaDataset(Dataset):
    """Pre-tokenize full 401bp raw DNA sequences (no spaces, no truncation)."""
    def __init__(self, pq_path, tokenizer):
        import pandas as pd
        df = pd.read_parquet(pq_path, columns=["sequence", "label"])
        self.labels = df["label"].to_numpy(dtype=np.int64)
        seqs = df["sequence"].tolist()
        print(f"  Tokenizing {len(seqs):,} sequences (raw DNA, full 401bp)...", flush=True)
        # Batch tokenize
        batch_size = 50000
        all_ids = []
        all_mask = []
        global_max = 0
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i+batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
            global_max = max(global_max, enc["input_ids"].shape[1])
            all_ids.append(enc["input_ids"])
            all_mask.append(enc["attention_mask"])
            if (i + batch_size) % 200000 == 0:
                print(f"    {min(i+batch_size, len(seqs)):,}/{len(seqs):,}", flush=True)
        # Pad all chunks to global_max
        padded_ids = []
        padded_mask = []
        for ids, mask in zip(all_ids, all_mask):
            cur_len = ids.shape[1]
            if cur_len < global_max:
                ids = F.pad(ids, (0, global_max - cur_len), value=0)
                mask = F.pad(mask, (0, global_max - cur_len), value=0)
            padded_ids.append(ids)
            padded_mask.append(mask)
        self.input_ids = torch.cat(padded_ids, dim=0)
        self.attention_mask = torch.cat(padded_mask, dim=0)
        self.max_len = global_max
        print(f"  Done: {self.input_ids.shape[0]:,} seqs × {self.max_len} max tokens, 0 UNK expected", flush=True)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return (
            {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]},
            torch.tensor(int(self.labels[i]))
        )


def collate(batch):
    ids = torch.nn.utils.rnn.pad_sequence([b[0]["input_ids"] for b in batch], batch_first=True, padding_value=0)
    mask = torch.nn.utils.rnn.pad_sequence([b[0]["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.stack([b[1] for b in batch])
    return {"input_ids": ids, "attention_mask": mask}, labels


# ═══════════════════════ Metrics ═══════════════════════
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    L, P = [], []
    for batch, labs in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)
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
        r[f"{nm}_Precision"] = float(pb.sum() and (yb&pb).sum()/max(pb.sum(),1) or 0)
        try: r[f"{nm}_AUROC"] = float(roc_auc_score(yb, prob[:,ci]))
        except: r[f"{nm}_AUROC"] = float("nan")
        try: r[f"{nm}_AUPRC"] = float(average_precision_score(yb, prob[:,ci]))
        except: r[f"{nm}_AUPRC"] = float("nan")
    return r


# ═══════════════════════ Main ═══════════════════════
def main():
    folds = range(5) if args.fold == "all" else [int(args.fold)]

    for fold in folds:
        print(f"\n{'='*60}", flush=True)
        print(f"=== gena | Fold {fold} | GPU {args.gpu} ===", flush=True)

        # Fresh model per fold
        print("Building model...", flush=True)
        model, tokenizer = build_gena_model()
        model = model.to(device)

        data_dir = f"{DATA_BASE}/fold_{fold}"
        out_dir = f"{OUT_BASE}/fold_{fold}"
        os.makedirs(out_dir, exist_ok=True)

        print("Loading data...", flush=True)
        train_file = f"{data_dir}/train_{args.fewshot_pct}pct.parquet" if args.fewshot_pct else f"{data_dir}/train.parquet"
        train_ds = GenaDataset(train_file, tokenizer)
        val_ds   = GenaDataset(f"{data_dir}/val.parquet", tokenizer)
        test_ds  = GenaDataset(f"{data_dir}/test.parquet", tokenizer)
        print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}", flush=True)

        train_ld = DataLoader(train_ds, BATCH, shuffle=True, collate_fn=collate,
                              num_workers=0, drop_last=True, pin_memory=True)
        val_ld   = DataLoader(val_ds, BATCH, shuffle=False, collate_fn=collate,
                              num_workers=0, pin_memory=True)
        test_ld  = DataLoader(test_ds, BATCH, shuffle=False, collate_fn=collate,
                              num_workers=0, pin_memory=True)

        # GPU warmup
        print("Warming up GPU...", flush=True)
        with torch.no_grad():
            db, _ = next(iter(train_ld))
            db = {k: v[:min(BATCH,len(v))].to(device) for k,v in db.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(db)
        print("Warmup complete.", flush=True)

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
        steps_per_ep = len(train_ld)
        total_steps = steps_per_ep * EPOCHS
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total_steps,
            pct_start=WARMUP/max(total_steps,1))
        scaler = torch.amp.GradScaler("cuda")
        crit = nn.CrossEntropyLoss()

        best_f1 = 0.0; best_epoch = -1; no_improve = 0
        print(f"Training ({steps_per_ep} steps/epoch, {EPOCHS} epochs, early_stop={EARLY_STOP_PATIENCE}, "
              f"batch={BATCH}, max_tokens={train_ds.max_len})...", flush=True)

        for ep in range(EPOCHS):
            model.train(); ep_loss = 0; t0 = time.time(); opt.zero_grad()
            for step, (batch, labs) in enumerate(train_ld):
                batch = {k: v.to(device) for k, v in batch.items()}; labs = labs.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = crit(model(batch), labs)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
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
            for nm in CLASS_NAMES:
                print(f"  {nm}: F1={vm[f'{nm}_F1']:.4f} Recall={vm[f'{nm}_Recall']:.4f}", flush=True)

            if improved:
                best_f1 = vm["macro_f1"]; best_epoch = ep; no_improve = 0
                torch.save({"model": model.state_dict(), "epoch": ep, "metrics": vm},
                           f"{out_dir}/best_model.pt")
                print(f"  -> saved (best F1={best_f1:.4f})", flush=True)
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {ep+1}", flush=True); break

        # Final Test
        print("\n=== Final Test ===", flush=True)
        best = torch.load(f"{out_dir}/best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model"])
        tm = evaluate(model, test_ld)
        tm["best_val_f1"] = best_f1; tm["best_epoch"] = best_epoch
        print(f"Test: Macro-F1={tm['macro_f1']:.4f} MCC={tm['mcc']:.4f} "
              f"AUROC={tm['auroc_macro']:.4f} AUPRC={tm['auprc_macro']:.4f}", flush=True)
        for nm in CLASS_NAMES:
            print(f"  {nm}: F1={tm[f'{nm}_F1']:.4f} Recall={tm[f'{nm}_Recall']:.4f}", flush=True)
        with open(f"{out_dir}/metrics.json","w") as f: json.dump(tm, f, indent=2)
        print(f"Done -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
