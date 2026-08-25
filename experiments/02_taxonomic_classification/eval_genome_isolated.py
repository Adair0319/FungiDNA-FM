#!/usr/bin/env python3
"""Evaluate models on the genome-isolated datasets (all five taxonomic ranks).

Produces the genome-isolated half of Figure 3 / Supplementary Table S2.

Baselines reported in the paper: GENA-LM (frozen), CNN (end-to-end), and
FungiDNA itself ("ours"). Exploratory DNABERT-2 / NT-v2 code paths remain in
the file but are NOT paper baselines and have been removed from --model.

Usage: python eval_genome_isolated.py --model ours --dataset phylum --gpu 0
"""
import sys, os, json, math, argparse, numpy as np, pandas as pd
from pathlib import Path; from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, matthews_corrcoef, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(PROJECT_ROOT))

# ═══ MLP + FocalLoss ═══
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, ignore_index=-100):
        super().__init__(); self.register_buffer("alpha", alpha); self.gamma = gamma; self.ignore_index = ignore_index
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce); ts = targets.clamp(min=0); aw = self.alpha[ts]; aw[targets == self.ignore_index] = 0.0
        fl = aw * (1 - pt) ** self.gamma * ce; v = targets != self.ignore_index
        return fl[v].mean() if v.any() else torch.tensor(0.0, device=logits.device)

class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
    def forward(self, x): return self.net(x)

# ═══ Feature extraction ═══
def extract_features(backbone, tokenizer, df, device, model_type, batch_size=32):
    """Extract features from backbone. For gena/dnabert2: sliding window + mean pool."""
    backbone.eval(); feats, labels = [], []
    if model_type in ("gena", "dnabert2", "ntv2"):
        # Sliding window approach
        win_size = 2048 if model_type == "ntv2" else 512
        for _, row in df.iterrows():
            seq = row["sequence"]; lbl = int(row["label"])
            windows = [seq[i:i+win_size] for i in range(0, len(seq)-win_size+1, win_size)]
            if not windows: windows = [seq[:win_size]]
            embeds = []
            for w in windows:
                tok = tokenizer(w, return_tensors="pt", truncation=True, max_length=win_size)
                ids = tok["input_ids"].to(device)
                with torch.no_grad():
                    out = backbone(input_ids=ids, output_hidden_states=True)
                    h = out.hidden_states[-1] if hasattr(out,'hidden_states') and out.hidden_states else (out[0] if isinstance(out,tuple) else out.last_hidden_state)
                    embeds.append(h.mean(dim=1).cpu().float().numpy())
            feats.append(np.mean(embeds, axis=0).squeeze(0))
            labels.append(lbl)
            if len(feats) % 10000 == 0: print(f"    {len(feats):,}/{len(df):,}")
    else:
        # Full sequence (Ours, CNN)
        from fungidna.data.pks_dataset import PKSDataset
        ds = PKSDataset(df, tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=lambda b: {"input_ids": torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in b], batch_first=True, padding_value=0), "label": torch.tensor([x["label"] for x in b])})
        with torch.no_grad():
            for batch in loader:
                if model_type == "ours":
                    h = backbone(batch["input_ids"].to(device), token_type=0)
                else:  # cnn
                    h = backbone(batch["input_ids"].to(device))
                feats.append(h.mean(dim=1).cpu().float().numpy())
                labels.append(batch["label"].numpy())
        feats = np.concatenate(feats); labels = np.concatenate(labels)
    return np.array(feats), np.array(labels)

# ═══ Main ═══
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ours","cnn","gena"])
    parser.add_argument("--dataset", required=True, choices=["phylum","subphylum","class","order","family"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    rank = args.dataset; model_type = args.model
    data_dir = PROJECT_ROOT / "data/downstream" / f"task0_{rank}"
    out_dir = PROJECT_ROOT / "checkpoints/genome_isolated" / f"{model_type}_{rank}"
    os.makedirs(out_dir, exist_ok=True)

    label_map = json.load(open(data_dir / "label_map.json"))
    num_classes = len(label_map); idx_to_name = {v:k for k,v in label_map.items()}
    print(f"Genome-isolated: {model_type} on {rank} ({num_classes} classes) GPU {args.gpu}")

    # ══ Load model ══
    if model_type == "ours":
        from fungidna.model.config import FungiDNAConfig; from fungidna.model.striped_mamba import StripedMambaBackbone
        from fungidna.data.tokenizer import DualTokenizer
        config = FungiDNAConfig(); backbone = StripedMambaBackbone(config).to(device).bfloat16()
        ckpt = torch.load(str(PROJECT_ROOT / "checkpoints/phase2_joint/backbone_final.pt"), map_location="cpu", weights_only=False)
        backbone.load_state_dict(ckpt, strict=False); backbone.eval()
        for p in backbone.parameters(): p.requires_grad = False
        tokenizer = DualTokenizer(str(PROJECT_ROOT / "data/processed/bpe_fungi.model"))
        input_dim = 768
    elif model_type == "cnn":
        from fungidna.model.baseline_models import CNNBaseline
        from fungidna.data.tokenizer import DualTokenizer
        tokenizer = DualTokenizer(str(PROJECT_ROOT / "data/processed/bpe_fungi.model"))
        backbone = CNNBaseline(num_classes=num_classes).to(device)  # end-to-end, no freeze
        input_dim = 256  # CNN output dim (from fc1 before classifier)
        # For CNN, we need to extract from intermediate layer. Simpler: just use features from conv output
    elif model_type == "gena":
        from transformers import AutoTokenizer, BertModel, BertConfig
        model_path = str(PROJECT_ROOT / "baselines/models/gena-lm-bert-base-yeast")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path); backbone = BertModel(config)
        ckpt = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        sd = {}
        for k,v in ckpt.items():
            if k.startswith('cls.'): continue
            nk = k[5:] if k.startswith('bert.') else k
            if 'pre_attention_ln' in nk: nk = nk.replace('pre_attention_ln', 'attention.output.LayerNorm')
            elif 'post_attention_ln' in nk: nk = nk.replace('post_attention_ln', 'output.LayerNorm')
            sd[nk] = v
        backbone.load_state_dict(sd, strict=False); backbone = backbone.to(device).eval()
        for p in backbone.parameters(): p.requires_grad = False
        input_dim = 768
    elif model_type == "dnabert2":
        from transformers import AutoTokenizer, AutoModel
        model_path = str(PROJECT_ROOT / "baselines/models/DNABERT-2-117M")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        for p in backbone.parameters(): p.requires_grad = False
        input_dim = 768
    elif model_type == "ntv2":
        from transformers import AutoModelForMaskedLM, AutoTokenizer as AT
        model_path = str(PROJECT_ROOT / "baselines/models/nt-v2-500m-multi-species")
        tokenizer = AT.from_pretrained(model_path)
        m = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        backbone = m.esm.to(device).eval()
        for p in backbone.parameters(): p.requires_grad = False
        input_dim = backbone.config.hidden_size
    print(f"  Loaded, input_dim={input_dim}")

    # ══ Load data ══
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    rng = np.random.default_rng(42)
    train_idx = rng.permutation(len(train_df))
    n_val = len(train_df) // 10
    val_idx = train_idx[:n_val]; tr_idx = train_idx[n_val:]
    train_df_tr = train_df.iloc[tr_idx]; train_df_val = train_df.iloc[val_idx]
    print(f"  train={len(train_df_tr):,} val={len(train_df_val):,} test={len(test_df):,}")

    # ══ For CNN: end-to-end training (not frozen+MLP) ══
    if model_type == "cnn":
        from fungidna.data.pks_dataset import PKSDataset
        def cnn_collate(batch):
            from torch.nn.utils.rnn import pad_sequence
            return {"input_ids": pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=0), "label": torch.tensor([x["label"] for x in batch])}
        tr_loader = DataLoader(PKSDataset(train_df_tr, tokenizer), batch_size=32, shuffle=True, num_workers=0, collate_fn=cnn_collate)
        val_loader = DataLoader(PKSDataset(train_df_val, tokenizer), batch_size=32, shuffle=False, num_workers=0, collate_fn=cnn_collate)
        test_loader = DataLoader(PKSDataset(test_df, tokenizer), batch_size=32, shuffle=False, num_workers=0, collate_fn=cnn_collate)

        cc = Counter(train_df_tr["label"].tolist()); total = sum(cc.values())
        alpha = torch.tensor([1.0/math.sqrt(max(cc.get(i,1),1)/total) for i in range(num_classes)], device=device)
        crit = FocalLoss(alpha); opt = torch.optim.AdamW(backbone.parameters(), lr=1e-3)

        best_f1, best_state, patience = -1, None, 0
        for epoch in range(100):
            backbone.train()
            for batch in tr_loader: opt.zero_grad(); crit(backbone(batch["input_ids"].to(device)), batch["label"].to(device)).backward(); opt.step()
            backbone.eval(); vp, vl = [], []
            with torch.no_grad():
                for batch in val_loader: vp.append(backbone(batch["input_ids"].to(device)).argmax(-1).cpu()); vl.append(batch["label"])
            vf1 = f1_score(torch.cat(vl).numpy(), torch.cat(vp).numpy(), average="macro", zero_division=0)
            if vf1 > best_f1: best_f1 = vf1; best_state = {k:v.cpu().clone() for k,v in backbone.state_dict().items()}; patience = 0
            else: patience += 1
            if patience >= 15: break

        backbone.load_state_dict(best_state); backbone.eval()
        tp, tl, tprobs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = backbone(batch["input_ids"].to(device)); probs = F.softmax(logits, dim=-1).cpu().numpy()
                tp.append(logits.argmax(-1).cpu().numpy()); tl.append(batch["label"].numpy()); tprobs.append(probs)
        y_pred = np.concatenate(tp); y_true = np.concatenate(tl); probs = np.concatenate(tprobs)
    else:
        # Frozen backbone: extract features → MLP
        print("  Extracting features...")
        X_train, y_train = extract_features(backbone, tokenizer, train_df_tr, device, model_type)
        X_val, y_val = extract_features(backbone, tokenizer, train_df_val, device, model_type)
        X_test, y_test = extract_features(backbone, tokenizer, test_df, device, model_type)
        np.save(out_dir/"X_train.npy", X_train); np.save(out_dir/"X_test.npy", X_test)

        cc = Counter(y_train.tolist()); total = sum(cc.values())
        alpha = torch.tensor([1.0/math.sqrt(max(cc.get(i,1),1)/total) for i in range(num_classes)], device=device)
        X_tr = torch.tensor(X_train, dtype=torch.float32).to(device); y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device); y_v = torch.tensor(y_val, dtype=torch.long).to(device)
        X_te = torch.tensor(X_test, dtype=torch.float32).to(device); y_te = torch.tensor(y_test, dtype=torch.long).to(device)

        model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        crit = FocalLoss(alpha, gamma=2.0)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)

        best_f1, best_state, patience = -1, None, 0
        for epoch in range(50):
            model.train()
            for xb, yb in loader: opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
            model.eval()
            with torch.no_grad(): vf1 = f1_score(y_v.cpu(), model(X_v).argmax(-1).cpu(), average="macro", zero_division=0)
            if vf1 > best_f1: best_f1 = vf1; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; patience = 0
            else: patience += 1
            if patience >= 10: break

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad(): logits = model(X_te); probs = F.softmax(logits, dim=-1).cpu().numpy(); y_pred = logits.argmax(-1).cpu().numpy(); y_true = y_test
        torch.save(best_state, out_dir/"mlp_best.pt")

    # ══ Evaluate ══
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    prec, rec, f1_arr, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes), zero_division=0)
    report = {"model": model_type, "dataset": rank, "macro_f1": float(macro_f1), "mcc": mcc}
    print(f"  Macro-F1: {macro_f1:.4f}, MCC: {mcc:.4f}")

    # Save predictions
    rows = []
    for i in range(min(len(y_true), len(test_df))):
        rows.append({"sequence": test_df.iloc[i]["sequence"], "genome_id": test_df.iloc[i]["genome_id"],
                     "true_label": int(y_true[i]), "true_name": idx_to_name.get(int(y_true[i]),"?"),
                     "pred_label": int(y_pred[i]), "pred_name": idx_to_name.get(int(y_pred[i]),"?"),
                     "correct": int(y_true[i]==y_pred[i])})
        for lbl in range(num_classes): rows[-1][f"prob_{idx_to_name.get(lbl,lbl)}"] = float(probs[i,lbl])
    pd.DataFrame(rows).to_csv(out_dir/"predictions.csv", index=False)

    # Per-class
    for lbl in range(num_classes):
        report.setdefault("per_class",{})[idx_to_name[lbl]] = {"precision": float(prec[lbl]), "recall": float(rec[lbl]), "f1": float(f1_arr[lbl])}

    # CM
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes)); names = [idx_to_name[i] for i in range(num_classes)]
    for norm,suf in [("raw",False),("normalized",True)]:
        fig,ax = plt.subplots(figsize=(max(9,num_classes*0.7),max(8,num_classes*0.55)))
        pcm = cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-8) if suf else cm
        annot = np.empty_like(cm, dtype=object)
        for i in range(num_classes):
            for j in range(num_classes): annot[i,j] = f"{cm[i,j]:,}" if not suf and cm[i,j]>0 else (f"{pcm[i,j]:.2f}" if suf and cm[i,j]>0 else "")
        sns.heatmap(pcm if suf else cm, annot=annot, fmt="", cmap="YlOrRd", xticklabels=names, yticklabels=names, ax=ax, linewidths=0.3, annot_kws={"fontsize":8})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.tight_layout()
        fig.savefig(out_dir/f"cm_{norm}.png", dpi=150, bbox_inches="tight"); plt.close()

    json.dump(report, open(out_dir/"report.json","w"), indent=2)
    print(f"  Saved to {out_dir}")

if __name__ == "__main__": main()
