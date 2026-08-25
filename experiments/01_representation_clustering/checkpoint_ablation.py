#!/usr/bin/env python3
"""Checkpoint ablation: Block1 PCA+UMAP clustering vs training steps.

Compares backbone-5000, backbone-10000, backbone-15000, backbone_final
on Phylum and Elements (balanced) tasks. Block 1 only, PCA(50)+UMAP only.
"""
import json, time, numpy as np, pandas as pd
from pathlib import Path
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch; torch.cuda.set_device(0)

def _warmup():
    try:
        from flash_attn import flash_attn_func
        d = torch.device("cuda:0")
        q = torch.randn(1, 64, 12, 64, device=d, dtype=torch.bfloat16)
        flash_attn_func(q, q, q, causal=False)
    except Exception: pass
_warmup()

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fungidna.model.config import FungiDNAConfig
from fungidna.model.striped_mamba import StripedMambaBackbone
from fungidna.data.tokenizer import DualTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score)
import umap

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 32; SEED = 42; DPI = 200
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/phase2_joint"
CHECKPOINTS = [
    (CHECKPOINT_DIR / "backbone-5000.pt", 5000),
    (CHECKPOINT_DIR / "backbone-10000.pt", 10000),
    (CHECKPOINT_DIR / "backbone-15000.pt", 15000),
    (CHECKPOINT_DIR / "backbone_final.pt", 16000),
]
TOKENIZER_PATH = PROJECT_ROOT / "data/processed/bpe_fungi.model"
ATTENTION_LAYER = 7  # Block 1 = layer 7 (0-indexed)
BLOCK_OUTPUT = {}

def make_hook():
    def hook(module, input, output): BLOCK_OUTPUT[1] = output.detach()
    return hook

# ═══════════════════════════════════════
# Phylum data
# ═══════════════════════════════════════
print("=" * 60)
print("PHYLUM: Loading data...")
FOLD_DIR = PROJECT_ROOT / "data/downstream/task0_phylum_five_rank/fold1"
LABEL_MAP = json.load(open(PROJECT_ROOT / "data/downstream/task0_phylum_five_rank/label_map.json"))
idx_to_name = {v: k for k, v in LABEL_MAP.items()}; num_classes = len(LABEL_MAP)

dfs = [pd.read_parquet(FOLD_DIR / f"{s}.parquet") for s in ["train","val","test"]]
df = pd.concat(dfs, ignore_index=True)
total = len(df); target = int(total * 0.25); ideal = target // num_classes
labels_arr = df["label"].values
rng = np.random.default_rng(SEED)
cc = [(labels_arr == l).sum() for l in range(num_classes)]
small = [l for l in range(num_classes) if cc[l] < ideal]
large = [l for l in range(num_classes) if l not in small]
lp = (target - sum(cc[l] for l in small)) // len(large)

phylum_idx = []
for lbl in range(num_classes):
    pool = np.where(labels_arr == lbl)[0]
    n = min(lp if lbl in large else len(pool), len(pool))
    chosen = pool if lbl in small else rng.choice(pool, size=n, replace=False)
    phylum_idx.append(chosen)
phylum_idx = np.concatenate(phylum_idx); rng.shuffle(phylum_idx)
phylum_seqs = df["sequence"].iloc[phylum_idx].tolist()
phylum_labels = labels_arr[phylum_idx]
print(f"  Sampled: {len(phylum_seqs):,}, classes={num_classes}")

# ═══════════════════════════════════════
# Elements data
# ═══════════════════════════════════════
print("ELEMENTS: Loading data...")
MERGED_DIR = PROJECT_ROOT.parent / "1kfg_datasets/output/merged"
ELEMENT_FILES = {"cds": "cds.fasta", "intron": "intron.fasta", "intergenic": "intergenic.fasta",
                 "five_prime_utr": "five_prime_utr.fasta", "three_prime_utr": "three_prime_utr.fasta"}
NUM_ELEM = 5; TARGET_ELEM = 20000

# Pass 1: lengths
elem_lengths = []; elem_gids = []
for lbl, (name, fn) in enumerate(ELEMENT_FILES.items()):
    lengths = []; gids = []
    with open(MERGED_DIR / fn) as f:
        hdr = None; sl = 0
        for line in f:
            if line.startswith(">"):
                if hdr: lengths.append(sl); gids.append(hdr.split("|")[0][1:])
                hdr = line.strip(); sl = 0
            else: sl += len(line.strip())
        if hdr: lengths.append(sl); gids.append(hdr.split("|")[0][1:])
    elem_lengths.append(np.array(lengths, dtype=np.int32))
    elem_gids.append(gids)
    print(f"  {name}: {len(lengths):,}")

# Pass 2: balanced sampling (length-stratified, 3 bins)
rng2 = np.random.default_rng(SEED)
elem_samples = []
for lbl in range(NUM_ELEM):
    lens = elem_lengths[lbl]; n_total = len(lens)
    si = np.argsort(lens); bs = n_total // 3; pb = TARGET_ELEM // 3
    picks = []
    for b in range(3):
        s = b*bs; e = s+bs if b<3-1 else n_total
        bp = si[s:e]; nt = min(pb, len(bp))
        picks.append(rng2.choice(bp, size=nt, replace=False))
    picks = np.concatenate(picks)
    if len(picks) < TARGET_ELEM:
        rem = np.setdiff1d(np.arange(n_total), picks)
        picks = np.concatenate([picks, rng2.choice(rem, size=min(TARGET_ELEM-len(picks), len(rem)), replace=False)])
    elem_samples.append(picks[:TARGET_ELEM])

# Pass 3: load seqs
elem_seqs_all = []
for lbl, (name, fn) in enumerate(ELEMENT_FILES.items()):
    target_set = set(elem_samples[lbl])
    with open(MERGED_DIR / fn) as f:
        hdr = None; sp = []; sidx = -1
        for line in f:
            if line.startswith(">"):
                if hdr and (sidx in target_set):
                    elem_seqs_all.append(("".join(sp), lbl, hdr.split("|")[0][1:]))
                hdr = line.strip(); sp = []; sidx += 1
            elif sidx in target_set: sp.append(line.strip())
        if hdr and (sidx in target_set): elem_seqs_all.append(("".join(sp), lbl, hdr.split("|")[0][1:]))
# Truncate long seqs
MAX_LEN = 100000
elem_seqs_all = [(s[:MAX_LEN], l, g) for s, l, g in elem_seqs_all]
elem_seqs_str = [s for s, _, _ in elem_seqs_all]
elem_labels = np.array([l for _, l, _ in elem_seqs_all])
print(f"  Sampled: {len(elem_seqs_str):,}, classes={NUM_ELEM}")

# ═══════════════════════════════════════
# Run ablation
# ═══════════════════════════════════════
tokenizer = DualTokenizer(str(TOKENIZER_PATH))
results = []

for ckpt_path, step in CHECKPOINTS:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path.name} (step {step})")
    print(f"{'='*60}")

    # Load model
    model_cfg = FungiDNAConfig()
    backbone = StripedMambaBackbone(model_cfg)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    backbone.load_state_dict(ckpt, strict=True)
    backbone = backbone.to(DEVICE).bfloat16()
    backbone.eval()
    backbone.layers[ATTENTION_LAYER].register_forward_hook(make_hook())

    def extract(seqs):
        n = len(seqs); feats = []
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n); batch = seqs[start:end]
            all_ids = []; ml = 0
            for s in batch:
                ids = tokenizer.encode_bpe(s, add_cls=False); all_ids.append(ids); ml = max(ml, len(ids))
            input_ids = torch.zeros(len(batch), ml, dtype=torch.long)
            am = torch.zeros(len(batch), ml, dtype=torch.long)
            for i, ids in enumerate(all_ids):
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long); am[i, :len(ids)] = 1
            input_ids = input_ids.to(DEVICE); am = am.to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = backbone(input_ids, token_type=0, add_cls=False)
            hidden = BLOCK_OUTPUT[1].float()
            mf = am.unsqueeze(-1).float().to(hidden.device)
            pooled = ((hidden * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1e-9)).cpu().numpy()
            feats.append(pooled)
            if (end) % 5000 == 0 or end == n: print(f"    {end:,}/{n:,}")
        return np.concatenate(feats, axis=0)

    # ── Phylum ──
    print("  Phylum feature extraction...")
    t0 = time.time()
    pf = extract(phylum_seqs)
    print(f"    Features: {pf.shape}, {time.time()-t0:.0f}s")
    pf_pca = PCA(n_components=50, random_state=SEED).fit_transform(pf)
    umap_p = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1, verbose=False).fit_transform(pf_pca)
    preds_p = KMeans(n_clusters=num_classes, random_state=SEED, n_init=10).fit_predict(umap_p)
    phylum_metrics = {
        "ari": float(adjusted_rand_score(phylum_labels, preds_p)),
        "nmi": float(normalized_mutual_info_score(phylum_labels, preds_p)),
        "homogeneity": float(homogeneity_score(phylum_labels, preds_p)),
        "completeness": float(completeness_score(phylum_labels, preds_p)),
        "v_measure": float(v_measure_score(phylum_labels, preds_p)),
    }
    print(f"    Phylum ARI={phylum_metrics['ari']:.4f} NMI={phylum_metrics['nmi']:.4f}")
    results.append({"checkpoint": ckpt_path.name, "step": step, "task": "phylum", **phylum_metrics})

    # ── Elements ──
    print("  Elements feature extraction...")
    t0 = time.time()
    ef = extract(elem_seqs_str)
    print(f"    Features: {ef.shape}, {time.time()-t0:.0f}s")
    ef_pca = PCA(n_components=50, random_state=SEED).fit_transform(ef)
    umap_e = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1, verbose=False).fit_transform(ef_pca)
    preds_e = KMeans(n_clusters=NUM_ELEM, random_state=SEED, n_init=10).fit_predict(umap_e)
    elem_metrics = {
        "ari": float(adjusted_rand_score(elem_labels, preds_e)),
        "nmi": float(normalized_mutual_info_score(elem_labels, preds_e)),
        "homogeneity": float(homogeneity_score(elem_labels, preds_e)),
        "completeness": float(completeness_score(elem_labels, preds_e)),
        "v_measure": float(v_measure_score(elem_labels, preds_e)),
    }
    print(f"    Elements ARI={elem_metrics['ari']:.4f} NMI={elem_metrics['nmi']:.4f}")
    results.append({"checkpoint": ckpt_path.name, "step": step, "task": "elements", **elem_metrics})

# ═══════════════════════════════════════
# Save results
# ═══════════════════════════════════════
df_results = pd.DataFrame(results)
OUT = PROJECT_ROOT / "analysis/checkpoint_ablation.csv"
df_results.to_csv(OUT, index=False)
print(f"\n{'='*60}")
print("Results saved to", OUT)
print(df_results.to_string(index=False))

# Quick summary
print(f"\n=== SUMMARY ===")
for task in ["phylum", "elements"]:
    sub = df_results[df_results["task"] == task].sort_values("step")
    print(f"\n{task.upper()}:")
    print(f"  {'Step':>8s}  {'ARI':>8s}  {'NMI':>8s}  {'V-measure':>8s}")
    for _, r in sub.iterrows():
        print(f"  {r['step']:>8d}  {r['ari']:>8.4f}  {r['nmi']:>8.4f}  {r['v_measure']:>8.4f}")
