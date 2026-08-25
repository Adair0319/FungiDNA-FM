#!/usr/bin/env python3
"""Unified baseline evaluation for the taxonomic classification task.

Baselines reported in the paper: GENA-LM (frozen), CNN and Transformer
(trained end-to-end). Select them with --model gena / cnn / transformer.

NOTE: this script also contains an unreachable DNABERT-2 code path left over
from exploratory work. DNABERT-2 is NOT a baseline in the paper and has been
removed from the --model choices; the dead branch is retained only so the
shared feature-extraction code reads as it did when the reported numbers were
produced.

Usage: python evaluate_baselines.py --model cnn  --rank phylum    --gpu 0
       python evaluate_baselines.py --model gena --rank subphylum --gpu 1
"""
import sys, os, json, math, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, average_precision_score, matthews_corrcoef,
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════
# MLP + FocalLoss (same as backbone_final pipeline)
# ═══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        ts = targets.clamp(min=0)
        aw = self.alpha[ts]; aw[targets == self.ignore_index] = 0.0
        fl = aw * (1 - pt) ** self.gamma * ce
        v = targets != self.ignore_index
        return fl[v].mean() if v.any() else torch.tensor(0.0, device=logits.device)


class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════
# BPE Dataset (for CNN/Transformer)
# ═══════════════════════════════════════════════════════════════

class BPEDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=2000):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ids = self.tokenizer.encode_bpe(row["sequence"], add_cls=True)
        ids = ids[:self.max_len]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": int(row["label"]),
            "genome_id": row["genome_id"],
            "sequence": row["sequence"],
        }


def bpe_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    gids = [b["genome_id"] for b in batch]
    seqs = [b["sequence"] for b in batch]
    return {"input_ids": input_ids, "label": labels, "genome_id": gids, "sequence": seqs}


# ═══════════════════════════════════════════════════════════════
# Transformer Backbone Feature Extraction (gena-lm / DNABERT-2)
# Batched sliding window: 10Kbp → 20×512bp → mean pool, batch_size=64
# Pre-compute ONCE before fold loop (same sequences across folds)
# ═══════════════════════════════════════════════════════════════

def extract_transformer_features_batched(model, tokenizer, df, device, model_type,
                                          window_size=512, stride=512, batch_size=64):
    """Batch-extract 768-dim features via sliding window + mean pool.

    Process all windows in batches for GPU efficiency.
    """
    model.eval()
    all_features = []
    total = len(df)

    # Phase 1: collect all windows across all sequences
    all_windows = []
    window_counts = []  # number of windows per sequence
    labels = []

    for _, row in df.iterrows():
        seq = row["sequence"]
        windows = [seq[i:i+window_size] for i in range(0, len(seq)-window_size+1, stride)]
        if not windows:
            windows = [seq[:window_size]]
        all_windows.extend(windows)
        window_counts.append(len(windows))
        labels.append(int(row["label"]))

    # Phase 2: batch tokenize + forward
    print(f"    Total windows: {len(all_windows):,}, batch_size={batch_size}")
    all_embeds = []

    for i in range(0, len(all_windows), batch_size):
        batch_windows = all_windows[i:i+batch_size]
        # DNABERT-2 tokenizer handles k-mers; gena-lm uses standard BPE
        tok = tokenizer(batch_windows, return_tensors="pt", padding=True,
                       truncation=True, max_length=window_size)
        input_ids = tok["input_ids"].to(device)
        if "token_type_ids" in tok:
            token_type_ids = tok["token_type_ids"].to(device)
        else:
            token_type_ids = None

        with torch.no_grad():
            kwargs = {"input_ids": input_ids, "output_hidden_states": True}
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            outputs = model(**kwargs)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden = outputs.hidden_states[-1]
            elif isinstance(outputs, tuple):
                hidden = outputs[0]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs
            pooled = hidden.mean(dim=1).cpu().float().numpy()  # (batch, 768)
            all_embeds.append(pooled)

        if (i // batch_size + 1) % 500 == 0:
            print(f"    Processed {i+len(batch_windows):,}/{len(all_windows):,} windows")

    all_embeds = np.concatenate(all_embeds, axis=0)  # (total_windows, 768)

    # Phase 3: aggregate windows per sequence
    idx = 0
    for n_windows, label in zip(window_counts, labels):
        feat = all_embeds[idx:idx+n_windows].mean(axis=0)  # mean pool over windows
        all_features.append(feat)
        idx += n_windows

    print(f"    Extracted {len(all_features):,} sequence features ({all_features[0].shape[0]}-dim)")
    return np.array(all_features), np.array(labels)


# ═══════════════════════════════════════════════════════════════
# CNN/Transformer End-to-End Training
# ═══════════════════════════════════════════════════════════════

def train_end_to_end(model, train_loader, val_loader, test_df, tokenizer,
                     num_classes, device, fold_dir, idx_to_name, max_epochs=100, patience=15):
    """End-to-end training for CNN/Transformer."""
    # Focal Loss alpha
    all_labels = []
    for batch in train_loader:
        if isinstance(batch, (tuple, list)):
            all_labels.extend(batch[1].tolist())
        else:
            all_labels.extend(batch["label"].tolist())
    cc = Counter(all_labels)
    total = sum(cc.values())
    alpha = torch.tensor(
        [1.0 / math.sqrt(max(cc.get(i, 1), 1) / total) for i in range(num_classes)],
        device=device,
    )
    criterion = FocalLoss(alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_f1 = -1
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            xb, yb = (batch[0], batch[1]) if isinstance(batch, (tuple, list)) else (batch["input_ids"], batch["label"])
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = (batch[0], batch[1]) if isinstance(batch, (tuple, list)) else (batch["input_ids"], batch["label"])
                logits = model(xb.to(device))
                val_preds.append(logits.argmax(-1).cpu())
                val_labels.append(yb)

        val_f1 = f1_score(
            torch.cat(val_labels).numpy(), torch.cat(val_preds).numpy(),
            average="macro", zero_division=0,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f} val_f1={val_f1:.4f}")

        if patience_counter >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    # Create test loader
    test_loader = DataLoader(
        BPEDataset(test_df, tokenizer), batch_size=32, shuffle=False,
        num_workers=0, collate_fn=bpe_collate_fn,
    )

    test_preds, test_labels_all, test_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input_ids"].to(device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            test_preds.append(logits.argmax(-1).cpu().numpy())
            test_labels_all.append(batch["label"].numpy())
            test_probs.append(probs)

    y_pred = np.concatenate(test_preds)
    y_true = np.concatenate(test_labels_all)
    probs = np.concatenate(test_probs)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    torch.save(best_state, os.path.join(fold_dir, "model_best.pt"))

    # Predictions CSV
    pred_rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        r = {"sequence": row["sequence"], "genome_id": row["genome_id"],
             "true_label": int(y_true[i]), "true_name": idx_to_name.get(int(y_true[i]), "?"),
             "pred_label": int(y_pred[i]), "pred_name": idx_to_name.get(int(y_pred[i]), "?"),
             "correct": int(y_true[i] == y_pred[i])}
        for lbl in range(num_classes):
            r[f"prob_{idx_to_name.get(lbl, lbl)}"] = float(probs[i, lbl])
        pred_rows.append(r)
    pd.DataFrame(pred_rows).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

    # Per-class metrics
    prec, rec, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0)
    per_class, auroc_vals, auprc_vals = {}, [], []
    for lbl in range(num_classes):
        y_bin = (y_true == lbl).astype(int)
        auroc_val, auprc_val = float("nan"), float("nan")
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
        per_class[lbl] = {"precision": float(prec[lbl]), "recall": float(rec[lbl]),
                          "f1": float(f1_arr[lbl]), "auroc": auroc_val, "auprc": auprc_val}

    return {
        "macro_f1": float(macro_f1), "mcc": mcc,
        "macro_auroc": float(np.mean(auroc_vals)) if auroc_vals else None,
        "macro_auprc": float(np.mean(auprc_vals)) if auprc_vals else None,
        "per_class": per_class, "best_val_f1": float(best_val_f1),
    }, y_true, y_pred


# ═══════════════════════════════════════════════════════════════
# MLP Training (shared by all frozen-backbone methods)
# ═══════════════════════════════════════════════════════════════

def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, test_df,
              num_classes, device, fold_dir, idx_to_name):
    """Same as backbone_final pipeline."""
    cc = Counter(y_train.tolist())
    total = sum(cc.values())
    alpha = torch.tensor(
        [1.0 / math.sqrt(max(cc.get(i, 1), 1) / total) for i in range(num_classes)], device=device)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)

    model = MLP(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = FocalLoss(alpha, gamma=2.0)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)

    best_val_f1, best_state, patience = -1, None, 0
    for epoch in range(50):
        model.train()
        for xb, yb in loader: optimizer.zero_grad(); criterion(model(xb), yb).backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            vf1 = f1_score(y_v.cpu(), model(X_v).argmax(-1).cpu(), average="macro", zero_division=0)
        if vf1 > best_val_f1:
            best_val_f1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 10: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits = model(X_te); probs = F.softmax(logits, dim=-1).cpu().numpy(); preds = logits.argmax(-1).cpu().numpy()

    y_true, y_pred = y_test, preds
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    torch.save(best_state, os.path.join(fold_dir, "mlp_best.pt"))

    pred_rows = []
    for i in range(len(y_true)):
        r = {"sequence": test_df.iloc[i]["sequence"], "genome_id": test_df.iloc[i]["genome_id"],
             "true_label": int(y_true[i]), "true_name": idx_to_name.get(int(y_true[i]), "?"),
             "pred_label": int(y_pred[i]), "pred_name": idx_to_name.get(int(y_pred[i]), "?"),
             "correct": int(y_true[i] == y_pred[i])}
        for lbl in range(num_classes):
            r[f"prob_{idx_to_name.get(lbl, lbl)}"] = float(probs[i, lbl])
        pred_rows.append(r)
    pd.DataFrame(pred_rows).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

    prec, rec, f1_arr, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes), zero_division=0)
    per_class, auroc_vals, auprc_vals = {}, [], []
    for lbl in range(num_classes):
        y_bin = (y_true == lbl).astype(int)
        auroc_val, auprc_val = float("nan"), float("nan")
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
        per_class[lbl] = {"precision": float(prec[lbl]), "recall": float(rec[lbl]),
                          "f1": float(f1_arr[lbl]), "auroc": auroc_val, "auprc": auprc_val}

    return {"macro_f1": float(macro_f1), "mcc": mcc,
            "macro_auroc": float(np.mean(auroc_vals)) if auroc_vals else None,
            "macro_auprc": float(np.mean(auprc_vals)) if auprc_vals else None,
            "per_class": per_class, "best_val_f1": float(best_val_f1)}, probs, y_true, y_pred


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gena", "cnn", "transformer"])
    parser.add_argument("--rank", required=True, choices=["phylum","subphylum","class","order","family"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    rank = args.rank
    model_type = args.model
    data_base = PROJECT_ROOT / "data" / "downstream" / f"task0_{rank}_five_rank"
    out_base = PROJECT_ROOT / "checkpoints" / f"baseline_{model_type}" / rank
    os.makedirs(out_base, exist_ok=True)

    label_map = json.load(open(data_base / "label_map.json"))
    num_classes = len(label_map)
    idx_to_name = {v: k for k, v in label_map.items()}

    print(f"Baseline: {model_type} — {rank} ({num_classes} classes) on GPU {args.gpu}")

    # ══ Load model ══
    tokenizer = None
    if model_type == "gena":
        from transformers import AutoTokenizer, BertModel, BertConfig
        model_path = str(PROJECT_ROOT / "baselines/models/gena-lm-bert-base-yeast")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path)
        backbone = BertModel(config)
        ckpt = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        remapped = {}
        for k, v in ckpt.items():
            if k.startswith('cls.'):
                continue
            new_k = k
            # Strip 'bert.' prefix (standard BertModel expects no prefix)
            if new_k.startswith('bert.'):
                new_k = new_k[5:]
            # Remap Pre-LN keys to Post-LN names
            if 'pre_attention_ln' in new_k:
                new_k = new_k.replace('pre_attention_ln', 'attention.output.LayerNorm')
            elif 'post_attention_ln' in new_k:
                new_k = new_k.replace('post_attention_ln', 'output.LayerNorm')
            remapped[new_k] = v
        missing, unexpected = backbone.load_state_dict(remapped, strict=False)
        backbone = backbone.to(device)
        backbone.eval()
        for p in backbone.parameters(): p.requires_grad = False
        print(f"Loaded gena-lm (768-dim) — {len(missing)} missing, {len(unexpected)} unexpected")

    elif model_type == "dnabert2":
        from transformers import AutoTokenizer, AutoModel
        model_path = str(PROJECT_ROOT / "baselines/models/DNABERT-2-117M")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        backbone.eval()
        for p in backbone.parameters(): p.requires_grad = False
        print(f"Loaded DNABERT-2 (768-dim)")

    elif model_type in ("cnn", "transformer"):
        from fungidna.data.tokenizer import DualTokenizer
        from fungidna.model.baseline_models import create_baseline_model
        tokenizer = DualTokenizer(str(PROJECT_ROOT / "data/processed/bpe_fungi.model"))
        backbone = None  # Will create fresh each fold
        print(f"Loaded {model_type} (end-to-end)")
    else:
        raise ValueError(f"Unknown model: {model_type}")

    # ══ Pre-computation: extract/cache once for all 5 folds ══
    # Build a unified view: collect all (sequence, genome_id, label) pairs
    # with fold assignments from all 5 folds
    print("\n=== Pre-computation: caching all sequences across 5 folds ===")

    # Phase 1: Read all fold data into a combined cache
    fold_data = {fid: {} for fid in range(1, 6)}  # fid -> {"train": df, "val": df, "test": df}
    all_rows = {}  # (genome_id, sequence) -> label (same across folds)

    for fold_id in range(1, 6):
        fd = data_base / f"fold{fold_id}"
        for split in ["train", "val", "test"]:
            df = pd.read_parquet(fd / f"{split}.parquet")
            fold_data[fold_id][split] = df
            for _, row in df.iterrows():
                key = (row["genome_id"], row["sequence"])
                all_rows[key] = int(row["label"])

    print(f"  Total unique sequences: {len(all_rows):,}")

    # Phase 2: Pre-compute features/tokens for all unique sequences
    if model_type in ("gena", "dnabert2"):
        # Build a DataFrame from all_rows for batch extraction
        all_seqs = [k[1] for k in all_rows.keys()]
        all_gids = [k[0] for k in all_rows.keys()]
        all_lbls = list(all_rows.values())
        all_df = pd.DataFrame({"sequence": all_seqs, "genome_id": all_gids, "label": all_lbls})
        print("  Extracting transformer features once for all sequences...")
        X_all, y_all = extract_transformer_features_batched(
            backbone, tokenizer, all_df, device, model_type)
        feature_cache = {}
        for i, (gid, seq) in enumerate(zip(all_gids, all_seqs)):
            feature_cache[(gid, seq)] = X_all[i]
        del X_all, all_df, all_seqs  # free memory
        print(f"  Cached {len(feature_cache):,} features ({feature_cache[list(feature_cache.keys())[0]].shape[0]}-dim)")

    elif model_type in ("cnn", "transformer"):
        # Pre-tokenize all unique sequences
        print("  Pre-tokenizing all sequences once...")
        from torch.nn.utils.rnn import pad_sequence
        token_cache = {}
        all_seqs_sorted = sorted(all_rows.keys(), key=lambda k: k[0])  # deterministic
        for gid, seq in all_seqs_sorted:
            ids = tokenizer.encode_bpe(seq, add_cls=True)[:2000]
            token_cache[(gid, seq)] = torch.tensor(ids, dtype=torch.long)
        print(f"  Cached {len(token_cache):,} tokenized sequences")

    # ══ 5-Fold Loop (fast: just look up from cache) ══
    fold_metrics = []
    for fold_id in range(1, 6):
        print(f"\n--- Fold {fold_id}/5 ---")
        fold_dir = out_base / f"fold{fold_id}"
        os.makedirs(fold_dir, exist_ok=True)

        train_df = fold_data[fold_id]["train"]
        val_df = fold_data[fold_id]["val"]
        test_df = fold_data[fold_id]["test"]
        print(f"  train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

        if model_type in ("gena", "dnabert2"):
            # Look up features from cache
            def lookup_cache(df):
                feats = np.array([feature_cache[(r["genome_id"], r["sequence"])]
                                  for _, r in df.iterrows()])
                labels = np.array([int(r["label"]) for _, r in df.iterrows()])
                return feats, labels

            X_train, y_train = lookup_cache(train_df)
            X_val, y_val = lookup_cache(val_df)
            X_test, y_test = lookup_cache(test_df)
            np.save(fold_dir / "X_train.npy", X_train); np.save(fold_dir / "y_train.npy", y_train)
            np.save(fold_dir / "X_val.npy", X_val); np.save(fold_dir / "y_val.npy", y_val)
            np.save(fold_dir / "X_test.npy", X_test); np.save(fold_dir / "y_test.npy", y_test)

            print(f"  Training MLP...")
            metrics, probs, y_true, y_pred = train_mlp(
                X_train, y_train, X_val, y_val, X_test, y_test, test_df,
                num_classes, device, fold_dir, idx_to_name)

        elif model_type in ("cnn", "transformer"):
            # Look up tokens from cache, pad into tensors
            def lookup_cache_token(df):
                toks = [token_cache[(r["genome_id"], r["sequence"])]
                        for _, r in df.iterrows()]
                X = pad_sequence(toks, batch_first=True, padding_value=0)
                y = torch.tensor([int(r["label"]) for _, r in df.iterrows()], dtype=torch.long)
                return X, y

            from fungidna.model.baseline_models import create_baseline_model
            X_tr, y_tr = lookup_cache_token(train_df)
            X_va, y_va = lookup_cache_token(val_df)

            model = create_baseline_model(model_type, num_classes).to(device)
            train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=32, shuffle=False, num_workers=0)

            print(f"  Training {model_type} end-to-end ({sum(p.numel() for p in model.parameters()):,} params)...")
            metrics, y_true, y_pred = train_end_to_end(
                model, train_loader, val_loader, test_df, tokenizer,
                num_classes, device, fold_dir, idx_to_name)

        fold_metrics.append(metrics)
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        names = [idx_to_name[i] for i in range(num_classes)]
        for norm, suf in [("raw", False), ("normalized", True)]:
            fig, ax = plt.subplots(figsize=(max(10, num_classes*0.7), max(9, num_classes*0.6)))
            pcm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) if suf else cm
            annot = np.empty_like(cm, dtype=object)
            for i in range(num_classes):
                for j in range(num_classes):
                    annot[i,j] = f"{cm[i,j]:,}" if not suf and cm[i,j]>0 else (f"{pcm[i,j]:.2f}" if suf and cm[i,j]>0 else "")
            sns.heatmap(pcm if suf else cm, annot=annot, fmt="", cmap="YlOrRd",
                        xticklabels=names, yticklabels=names, ax=ax, linewidths=0.3,
                        annot_kws={"fontsize": 7 if num_classes <= 30 else 4})
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            plt.xticks(rotation=45, ha="right", fontsize=7 if num_classes <= 30 else 4)
            plt.yticks(fontsize=7 if num_classes <= 30 else 4); plt.tight_layout()
            fig.savefig(fold_dir / f"cm_{norm}.png", dpi=150, bbox_inches="tight"); plt.close()

    # ══ Summary ══
    def mean_std(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None

    summary = {
        "model": model_type, "rank": rank, "num_classes": num_classes,
        "macro_f1": mean_std([m["macro_f1"] for m in fold_metrics]),
        "mcc": mean_std([m["mcc"] for m in fold_metrics]),
        "macro_auroc": mean_std([m["macro_auroc"] for m in fold_metrics]),
        "macro_auprc": mean_std([m["macro_auprc"] for m in fold_metrics]),
        "per_fold": [{"fold": i+1, "macro_f1": m["macro_f1"], "mcc": m["mcc"],
                       "best_val_f1": m["best_val_f1"]} for i, m in enumerate(fold_metrics)],
    }
    for lbl in sorted(label_map.values()):
        name = idx_to_name[lbl]
        for metric in ["f1", "precision", "recall", "auroc", "auprc"]:
            vals = [m["per_class"].get(lbl, {}).get(metric) for m in fold_metrics]
            ms = mean_std(vals)
            summary.setdefault("per_class", {}).setdefault(name, {})[f"{metric}_mean"] = ms["mean"] if ms else None
            summary.setdefault("per_class", {}).setdefault(name, {})[f"{metric}_std"] = ms["std"] if ms else None

    json.dump(summary, open(out_base / "summary.json", "w"), indent=2)
    print(f"\n{'='*60}")
    print(f"{model_type}/{rank}: Macro-F1 = {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
