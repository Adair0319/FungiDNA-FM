#!/usr/bin/env python3
"""Recompute robust AUROC/AUPRC for Mamba2-1layer baselines.

The original evaluate() used roc_auc_score(ovr, macro) which returns NaN when
a class has no/single test sample. This recomputes macro AUROC/AUPRC only over
labels present in the test set (>=2 samples).

For genome_isolated: recompute from predictions.csv.
For five_rank: re-run inference on the test set from best_model.pt.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from mamba_ssm import Mamba2
from pathlib import Path

BASE = Path("/home/lty/yy_projects/fungi_project/fungi_dna_model")
CKPT = BASE / "checkpoints" / "baseline_mamba2_1layer"
DATA = BASE / "data" / "downstream"

D_MODEL = 256
NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}


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


class OneHotDNADataset(Dataset):
    def __init__(self, pq_path, seq_len=10000):
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
        valid = np.where(mask)[0]
        x[valid, idx[valid]] = 1.0
        return torch.from_numpy(x), torch.tensor(int(self.labels[i]))


def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch])


def robust_auroc_auprc(y_true, prob):
    present = sorted(set(y_true.tolist()))
    aurocs, auprcs = [], []
    for lbl in present:
        yb = (y_true == lbl).astype(int)
        if len(set(yb.tolist())) < 2:
            continue
        score = prob[:, lbl]
        aurocs.append(roc_auc_score(yb, score))
        auprcs.append(average_precision_score(yb, score))
    return float(np.mean(aurocs)), float(np.mean(auprcs))


def recompute_from_predictions(pred_csv, label_map_json):
    df = pd.read_csv(pred_csv)
    y_true = df["true_label"].to_numpy()
    label_map = json.load(open(label_map_json))
    idx_to_name = {v: k for k, v in label_map.items()}
    prob = np.zeros((len(df), len(label_map)), dtype=np.float32)
    for lbl in range(len(label_map)):
        name = idx_to_name[lbl]
        prob[:, lbl] = df[f"prob_{name}"].to_numpy() if f"prob_{name}" in df.columns else 0.0
    return robust_auroc_auprc(y_true, prob)


def run_inference(model, pq_path, device, batch=128):
    ds = OneHotDNADataset(pq_path)
    loader = DataLoader(ds, batch, shuffle=False, collate_fn=collate_fn, num_workers=0)
    model.eval()
    all_y, all_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_y.append(y.numpy())
            all_prob.append(F.softmax(logits, dim=-1).cpu().numpy())
    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    return robust_auroc_auprc(y, prob)


def main():
    device = torch.device("cuda:0")
    results = {"genome_isolated": {}, "five_rank": {}}

    # ── genome_isolated: re-run inference on test set ──
    for rank in ["phylum", "subphylum", "class", "order", "family"]:
        label_map = json.load(open(DATA / f"task0_{rank}" / "label_map.json"))
        num_classes = len(label_map)
        ckpt_path = CKPT / f"task0_{rank}" / "best_model.pt"
        test_pq = DATA / f"task0_{rank}" / "test.parquet"
        model = Mamba2Baseline(num_classes).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        auroc, auprc = run_inference(model, str(test_pq), device)
        results["genome_isolated"][rank] = {"auroc": auroc, "auprc": auprc}
        print(f"genome_isolated/{rank}: auroc={auroc:.4f} auprc={auprc:.4f}", flush=True)

    # ── five_rank: re-run inference ──
    for rank in ["phylum", "subphylum", "class", "order", "family"]:
        label_map = json.load(open(DATA / f"task0_{rank}_five_rank" / "label_map.json"))
        num_classes = len(label_map)
        aurocs, auprcs = [], []
        for fold in range(1, 6):
            ckpt_path = CKPT / f"task0_{rank}_five_rank" / f"fold{fold}" / "best_model.pt"
            test_pq = DATA / f"task0_{rank}_five_rank" / f"fold{fold}" / "test.parquet"
            model = Mamba2Baseline(num_classes).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            auroc, auprc = run_inference(model, str(test_pq), device)
            aurocs.append(auroc); auprcs.append(auprc)
            print(f"five_rank/{rank}/fold{fold}: auroc={auroc:.4f} auprc={auprc:.4f}")
        results["five_rank"][rank] = {
            "auroc": {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))},
            "auprc": {"mean": float(np.mean(auprcs)), "std": float(np.std(auprcs))},
        }
        print(f"five_rank/{rank}: auroc={np.mean(aurocs):.4f}±{np.std(aurocs):.4f} "
              f"auprc={np.mean(auprcs):.4f}±{np.std(auprcs):.4f}")

    out = BASE / "docs" / "experiment_report_taxonomy" / "mamba2_auroc_auprc.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
