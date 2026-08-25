#!/usr/bin/env python3
"""Phase 1 (MLM only) checkpoint ablation: Block1 PCA+UMAP on Phylum + Elements."""
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

# ═══ Load Phase 1 checkpoint (handles dict format + backbone. prefix) ═══
def load_p1_backbone(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model_cfg = FungiDNAConfig()
    backbone = StripedMambaBackbone(model_cfg)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = {}
        for k, v in ckpt["model"].items():
            if k.startswith("backbone."): state[k[9:]] = v  # strip backbone. prefix
            elif k.startswith("lm_head"): pass  # skip LM head
            else: state[k] = v
        backbone.load_state_dict(state, strict=False)
    else:
        backbone.load_state_dict(ckpt, strict=True)
    return backbone.to(DEVICE).bfloat16().eval()

tokenizer = DualTokenizer(str(PROJECT_ROOT / "data/processed/bpe_fungi.model"))

# ═══ Phylum data ═══
print("="*60 + "\nPHYLUM data...")
FOLD = PROJECT_ROOT / "data/downstream/task0_phylum_five_rank/fold1"
LM = json.load(open(PROJECT_ROOT / "data/downstream/task0_phylum_five_rank/label_map.json"))
NC = len(LM)
df = pd.concat([pd.read_parquet(FOLD / f"{s}.parquet") for s in ["train","val","test"]], ignore_index=True)
total = len(df); target = int(total * 0.25); ideal = target // NC
labels_arr = df["label"].values
rng = np.random.default_rng(SEED)
cc = [(labels_arr == l).sum() for l in range(NC)]
small = [l for l in range(NC) if cc[l] < ideal]
large = [l for l in range(NC) if l not in small]
lp = (target - sum(cc[l] for l in small)) // len(large)
p_idx = []
for lbl in range(NC):
    pool = np.where(labels_arr == lbl)[0]
    n = min(lp if lbl in large else len(pool), len(pool))
    chosen = pool if lbl in small else rng.choice(pool, size=n, replace=False)
    p_idx.append(chosen)
p_idx = np.concatenate(p_idx); rng.shuffle(p_idx)
p_seqs = df["sequence"].iloc[p_idx].tolist(); p_labels = labels_arr[p_idx]
print(f"  {len(p_seqs):,} seqs, {NC} classes")

# ═══ Elements data ═══
print("ELEMENTS data...")
MERGED = PROJECT_ROOT.parent / "1kfg_datasets/output/merged"
ELEM = {"cds":"cds.fasta","intron":"intron.fasta","intergenic":"intergenic.fasta",
        "five_prime_utr":"five_prime_utr.fasta","three_prime_utr":"three_prime_utr.fasta"}
NE = 5; TE = 20000

# Scan lengths
elens = []
for name, fn in ELEM.items():
    ls = []; f=open(MERGED/fn); h=None; sl=0
    for line in f:
        if line.startswith(">"):
            if h: ls.append(sl)
            h=line.strip(); sl=0
        else: sl+=len(line.strip())
    if h: ls.append(sl); f.close()
    elens.append(np.array(ls, dtype=np.int32))

# Replicate exact original sampling (random first to advance rng, then balanced)
rng2 = np.random.default_rng(SEED)
for lbl in range(NE):
    _ = rng2.choice(len(elens[lbl]), size=min(TE, len(elens[lbl])), replace=False)
e_samples = []
for lbl in range(NE):
    lens = elens[lbl]; si = np.argsort(lens)
    bs = len(lens)//3; pb = TE//3
    picks = []
    for b in range(3):
        s = b*bs; e = s+bs if b<3-1 else len(lens)
        bp = si[s:e]; nt = min(pb, len(bp))
        picks.append(rng2.choice(bp, size=nt, replace=False))
    picks = np.concatenate(picks)
    if len(picks) < TE:
        rem = np.setdiff1d(np.arange(len(lens)), picks)
        picks = np.concatenate([picks, rng2.choice(rem, size=min(TE-len(picks), len(rem)), replace=False)])
    e_samples.append(picks[:TE])

# Load seqs
e_seqs = []
for lbl, (name, fn) in enumerate(ELEM.items()):
    target = set(e_samples[lbl])
    f=open(MERGED/fn); h=None; sp=[]; si=-1
    for line in f:
        if line.startswith(">"):
            if h and (si in target): e_seqs.append(("".join(sp), lbl, h.split("|")[0][1:]))
            h=line.strip(); sp=[]; si+=1
        elif si in target: sp.append(line.strip())
    if h and (si in target): e_seqs.append(("".join(sp), lbl, h.split("|")[0][1:])); f.close()
ML=100000; e_seqs = [(s[:ML], l, g) for s,l,g in e_seqs]
e_seqs_str = [s for s,_,_ in e_seqs]; e_labels = np.array([l for _,l,_ in e_seqs])
print(f"  {len(e_seqs_str):,} seqs, {NE} classes")

# ═══ Run ablation ═══
CKPTS = [(PROJECT_ROOT / "checkpoints/phase1_mlm/checkpoint-40000.pt", 40000),
         (PROJECT_ROOT / "checkpoints/phase1_mlm/checkpoint-50000.pt", 50000),
         (PROJECT_ROOT / "checkpoints/phase1_mlm/checkpoint-60000.pt", 60000)]
results = []

MIN_TOKENS = 16

for cpath, step in CKPTS:
    print(f"\n{'='*60}\nPhase1 checkpoint-{step}")
    backbone = load_p1_backbone(cpath)
    backbone.layers[ATTN_LAYER].register_forward_hook(hook)

    for task_name, seqs, labels, n_clusters in [
        ("phylum", p_seqs, p_labels, NC),
        ("elements", e_seqs_str, e_labels, NE)]:

        print(f"  {task_name} feature extraction...")
        t0 = time.time(); n = len(seqs); feats = []
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
            if (end) % 10000 == 0: print(f"    {end:,}/{n:,}")
        feats = np.concatenate(feats)
        print(f"    Features: {feats.shape}, {time.time()-t0:.0f}s")

        pca = PCA(n_components=50, random_state=SEED).fit_transform(feats)
        u = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1, verbose=False).fit_transform(pca)
        preds = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit_predict(u)
        m = {
            "checkpoint": f"phase1-{step}", "step": step, "task": task_name,
            "ari": float(adjusted_rand_score(labels, preds)),
            "nmi": float(normalized_mutual_info_score(labels, preds)),
            "homogeneity": float(homogeneity_score(labels, preds)),
            "completeness": float(completeness_score(labels, preds)),
            "v_measure": float(v_measure_score(labels, preds)),
        }
        results.append(m)
        print(f"    ARI={m['ari']:.4f} NMI={m['nmi']:.4f}")

# ═══ Merge with existing ablation CSV ═══
df_new = pd.DataFrame(results)
CSV = PROJECT_ROOT / "analysis/checkpoint_ablation.csv"
df_old = pd.read_csv(CSV)
df_all = pd.concat([df_old, df_new], ignore_index=True).sort_values(["task", "step"])
df_all.to_csv(CSV, index=False)

print(f"\n{'='*60}\n=== ALL RESULTS ===")
print(df_all.to_string(index=False))

for task in ["phylum", "elements"]:
    sub = df_all[df_all["task"] == task].sort_values("step")
    print(f"\n{task.upper()}:")
    print(f"  {'Step':>10s}  {'ARI':>8s}  {'NMI':>8s}  {'V-measure':>8s}")
    for _, r in sub.iterrows():
        print(f"  {r['step']:>10d}  {r['ari']:>8.4f}  {r['nmi']:>8.4f}  {r['v_measure']:>8.4f}")
