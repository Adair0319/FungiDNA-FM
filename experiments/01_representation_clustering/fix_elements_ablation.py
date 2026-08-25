#!/usr/bin/env python3
"""Fix elements ablation: replicate exact original sampling, re-run 5k/10k/15k."""
import json, time, numpy as np, pandas as pd
from pathlib import Path
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch; torch.cuda.set_device(0)

def _warmup():
    try:
        from flash_attn import flash_attn_func
        d=torch.device("cuda:0"); q=torch.randn(1,64,12,64,device=d,dtype=torch.bfloat16)
        flash_attn_func(q,q,q,causal=False)
    except Exception: pass
_warmup()

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fungidna.model.config import FungiDNAConfig
from fungidna.model.striped_mamba import StripedMambaBackbone
from fungidna.data.tokenizer import DualTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score)
import umap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda:0"); BATCH_SIZE = 32; SEED = 42
ATTN_LAYER = 7; BLOCK_OUTPUT = {}
def hook(m,i,o): BLOCK_OUTPUT[1] = o.detach()

# ═══ Data ═══
MERGED = PROJECT_ROOT.parent / "1kfg_datasets/output/merged"
ELEM = {"cds":"cds.fasta","intron":"intron.fasta","intergenic":"intergenic.fasta",
        "five_prime_utr":"five_prime_utr.fasta","three_prime_utr":"three_prime_utr.fasta"}
NUM = 5; TGT = 20000; tokenizer = DualTokenizer(str(PROJECT_ROOT / "data/processed/bpe_fungi.model"))

# Pass 1: scan lengths
print("Pass 1: scan lengths...")
elens = []
for name, fn in ELEM.items():
    ls = []
    with open(MERGED / fn) as f:
        h = None; sl = 0
        for line in f:
            if line.startswith(">"):
                if h: ls.append(sl)
                h = line.strip(); sl = 0
            else: sl += len(line.strip())
        if h: ls.append(sl)
    elens.append(np.array(ls, dtype=np.int32))
    print(f"  {name}: {len(ls):,}")

# ── REPLICATE ORIGINAL sampling (random first to advance rng, then balanced) ──
rng = np.random.default_rng(SEED)
print("\nStep A: random sampling (advance rng to original state)...")
for lbl in range(NUM):
    n_total = len(elens[lbl]); n_take = min(TGT, n_total)
    _ = rng.choice(n_total, size=n_take, replace=False)

print("Step B: balanced sampling (correct rng state)...")
samples = []
for lbl in range(NUM):
    lens = elens[lbl]; n_total = len(lens); si = np.argsort(lens)
    bs = n_total // 3; pb = TGT // 3
    picks = []
    for b in range(3):
        s = b * bs; e = s + bs if b < 3-1 else n_total
        bp = si[s:e]; nt = min(pb, len(bp))
        picks.append(rng.choice(bp, size=nt, replace=False))
    picks = np.concatenate(picks)
    if len(picks) < TGT:
        rem = np.setdiff1d(np.arange(n_total), picks)
        picks = np.concatenate([picks, rng.choice(rem, size=min(TGT - len(picks), len(rem)), replace=False)])
    samples.append(picks[:TGT])

# Step C: load seqs
print("Step C: load sampled seqs...")
all_seqs = []
for lbl, (name, fn) in enumerate(ELEM.items()):
    target = set(samples[lbl])
    with open(MERGED / fn) as f:
        h = None; sp = []; si = -1
        for line in f:
            if line.startswith(">"):
                if h and (si in target): all_seqs.append(("".join(sp), lbl, h.split("|")[0][1:]))
                h = line.strip(); sp = []; si += 1
            elif si in target: sp.append(line.strip())
        if h and (si in target): all_seqs.append(("".join(sp), lbl, h.split("|")[0][1:]))
ML = 100000; all_seqs = [(s[:ML], l, g) for s, l, g in all_seqs]
seqs = [s for s, _, _ in all_seqs]; labels = np.array([l for _, l, _ in all_seqs])
print(f"  Loaded: {len(seqs):,}")

# ── Run 5k/10k/15k ──
CKPTS = [(PROJECT_ROOT / "checkpoints/phase2_joint/backbone-5000.pt", 5000),
         (PROJECT_ROOT / "checkpoints/phase2_joint/backbone-10000.pt", 10000),
         (PROJECT_ROOT / "checkpoints/phase2_joint/backbone-15000.pt", 15000)]
results = []

for cpath, step in CKPTS:
    print(f"\n=== backbone-{step} ===")
    model_cfg = FungiDNAConfig(); backbone = StripedMambaBackbone(model_cfg)
    ckpt = torch.load(cpath, map_location=DEVICE)
    backbone.load_state_dict(ckpt, strict=True)
    backbone = backbone.to(DEVICE).bfloat16(); backbone.eval()
    backbone.layers[ATTN_LAYER].register_forward_hook(hook)

    t0 = time.time(); n = len(seqs); feats = []
    MIN_TOKENS = 16
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n); batch = seqs[start:end]
        all_ids = []; ml = 0
        for s in batch:
            ids = tokenizer.encode_bpe(s, add_cls=False)
            if len(ids) < MIN_TOKENS: ids = ids + [0] * (MIN_TOKENS - len(ids))
            all_ids.append(ids); ml = max(ml, len(ids))
        if ml % 2 != 0: ml += 1
        iids = torch.zeros(len(batch), ml, dtype=torch.long)
        am = torch.zeros(len(batch), ml, dtype=torch.long)
        for i, ids in enumerate(all_ids):
            iids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long); am[i, :len(ids)] = 1
        iids = iids.to(DEVICE); am = am.to(DEVICE)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = backbone(iids, token_type=0, add_cls=False)
        h = BLOCK_OUTPUT[1].float(); mf = am.unsqueeze(-1).float().to(h.device)
        feats.append(((h * mf).sum(1) / mf.sum(1).clamp(min=1e-9)).cpu().numpy())
        if (end) % 10000 == 0: print(f"  {end:,}/{n:,}")
    feats = np.concatenate(feats)
    print(f"  Features: {feats.shape}, {time.time()-t0:.0f}s")

    pca = PCA(n_components=50, random_state=SEED).fit_transform(feats)
    u = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1, verbose=False).fit_transform(pca)
    preds = KMeans(n_clusters=NUM, random_state=SEED, n_init=10).fit_predict(u)
    m = {
        "ari": float(adjusted_rand_score(labels, preds)),
        "nmi": float(normalized_mutual_info_score(labels, preds)),
        "homogeneity": float(homogeneity_score(labels, preds)),
        "completeness": float(completeness_score(labels, preds)),
        "v_measure": float(v_measure_score(labels, preds)),
    }
    print(f"  ARI={m['ari']:.4f} NMI={m['nmi']:.4f}")
    results.append({"checkpoint": f"backbone-{step}.pt", "step": step, "task": "elements", **m})

# ── Merge into CSV ──
CSV = PROJECT_ROOT / "analysis/checkpoint_ablation.csv"
df_orig = pd.read_csv(CSV)
# Keep phylum rows, remove old elements rows, keep final.pt elements (original 0.305)
df_phylum = df_orig[df_orig["task"] == "phylum"]
df_final_orig = df_orig[(df_orig["task"] == "elements") & (df_orig["step"] == 16000)]
# Replace with correct 0.305 from original experiment
df_final_orig = df_final_orig.copy()
# Use the correct backbone_final value from the original elements depth experiment
# (balanced_block1, PCA+UMAP): ARI=0.3052, NMI=0.3602, Hom=0.3555, Com=0.3650, V=0.3602
correct_final = {"ari": 0.3052, "nmi": 0.3602, "homogeneity": 0.3555, "completeness": 0.3650, "v_measure": 0.3602}
for k, v in correct_final.items():
    df_final_orig[k] = v
df_new = pd.DataFrame(results)
df_final = pd.concat([df_phylum, df_final_orig, df_new], ignore_index=True).sort_values(["task", "step"])
df_final.to_csv(CSV, index=False)

print(f"\n=== FINAL CORRECTED RESULTS ===")
print(df_final.to_string(index=False))
