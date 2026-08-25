#!/usr/bin/env python3
"""Phylum depth-wise UMAP clustering: 6 blocks × dual UMAP. GPU 7."""
import os; os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch; torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=7 maps to device 0

def _warmup_flash_attn():
    try:
        from flash_attn import flash_attn_func
        _dev = torch.device("cuda:7")
        _q = torch.randn(1, 64, 12, 64, device=_dev, dtype=torch.bfloat16)
        _k = torch.randn(1, 64, 12, 64, device=_dev, dtype=torch.bfloat16)
        _v = torch.randn(1, 64, 12, 64, device=_dev, dtype=torch.bfloat16)
        flash_attn_func(_q, _k, _v, causal=False)
    except Exception:
        pass
_warmup_flash_attn()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fungidna.model.config import FungiDNAConfig
from fungidna.model.striped_mamba import StripedMambaBackbone
from fungidna.data.tokenizer import DualTokenizer

FOLD_DIR = PROJECT_ROOT / "data/downstream/task0_phylum_five_rank/fold1"
LABEL_MAP_JSON = PROJECT_ROOT / "data/downstream/task0_phylum_five_rank/label_map.json"
BACKBONE_PATH = PROJECT_ROOT / "checkpoints/phase2_joint/backbone_final.pt"
TOKENIZER_PATH = PROJECT_ROOT / "data/processed/bpe_fungi.model"
OUT_DIR = PROJECT_ROOT / "analysis/umap_phylum_depth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=7
BATCH_SIZE = 32; SEED = 42; TARGET_FRAC = 0.25; DPI = 200
COLORS = ["#7B1FA2", "#2E7D32", "#C62828", "#1565C0", "#F9A825", "#EF6C00"]
NUM_DEPTHS = 6; DEPTH_NAMES = [f"block{i}" for i in range(NUM_DEPTHS)]
ATTENTION_LAYER_INDICES = [3, 7, 11, 15, 19, 23]

try: (OUT_DIR / ".write_test").touch(); (OUT_DIR / ".write_test").unlink()
except (OSError, PermissionError) as e: sys.exit(f"Not writeable: {OUT_DIR}\n{e}")
print(f"Device: {DEVICE}\nOutput: {OUT_DIR}")

# ── Load data ──
with open(LABEL_MAP_JSON) as f: label_map = json.load(f)
idx_to_name = {v: k for k, v in label_map.items()}
num_classes = len(label_map)
print(f"Phyla ({num_classes}): {list(label_map.keys())}")

dfs = []
for split_name in ["train", "val", "test"]:
    df_split = pd.read_parquet(FOLD_DIR / f"{split_name}.parquet")
    dfs.append(df_split)
    print(f"  {split_name}: {len(df_split):,}")
df = pd.concat(dfs, ignore_index=True)
total_seqs = len(df)
print(f"Total: {total_seqs:,}")

# ── Balanced sampling ──
target_total = int(total_seqs * TARGET_FRAC)
ideal = target_total // num_classes
labels_arr = df["label"].values; gids_arr = df["genome_id"].values
rng = np.random.default_rng(SEED)
class_counts = [(labels_arr == lbl).sum() for lbl in range(num_classes)]
small_classes = [l for l in range(num_classes) if class_counts[l] < ideal]
small_total = sum(class_counts[l] for l in small_classes)
remaining = target_total - small_total
large_classes = [l for l in range(num_classes) if l not in small_classes]
large_per = remaining // len(large_classes)
print(f"Target: {target_total:,}, ideal={ideal:,}, large_per={large_per:,}")

sample_indices = []
for lbl in range(num_classes):
    pool = np.where(labels_arr == lbl)[0]
    n = min(large_per if lbl in large_classes else len(pool), len(pool))
    if lbl in small_classes: chosen = pool
    else: chosen = rng.choice(pool, size=n, replace=False)
    sample_indices.append(chosen)
    print(f"  {idx_to_name[lbl]:25s}: {len(pool):>7,} -> {n:>7,}")
sample_idx = np.concatenate(sample_indices)
rng.shuffle(sample_idx)
print(f"Sampled: {len(sample_idx):,}")

sampled_seqs = df["sequence"].iloc[sample_idx].tolist()
sampled_labels = labels_arr[sample_idx]
sampled_gids = gids_arr[sample_idx]

# ── Load model with hooks ──
print("Loading backbone...")
model_cfg = FungiDNAConfig()
backbone = StripedMambaBackbone(model_cfg)
ckpt = torch.load(BACKBONE_PATH, map_location=DEVICE)
backbone.load_state_dict(ckpt, strict=True)
backbone = backbone.to(DEVICE).bfloat16()
backbone.eval()
tokenizer = DualTokenizer(str(TOKENIZER_PATH))
block_outputs = {}
def make_hook(idx):
    def hook(module, input, output): block_outputs[idx] = output.detach()
    return hook
for block_idx, layer_idx in enumerate(ATTENTION_LAYER_INDICES):
    backbone.layers[layer_idx].register_forward_hook(make_hook(block_idx))
print(f"Backbone loaded: {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M params, hooks registered")

# ── Feature extraction ──
print(f"Extracting features for {len(sampled_seqs):,} seqs...")
all_features = {f"block{i}": [] for i in range(NUM_DEPTHS)}
t0 = time.time()
n = len(sampled_seqs)
for start in range(0, n, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n)
    batch_seqs = sampled_seqs[start:end]
    all_ids = []; max_len = 0
    for seq in batch_seqs:
        ids = tokenizer.encode_bpe(seq, add_cls=False)
        all_ids.append(ids); max_len = max(max_len, len(ids))
    input_ids = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
    attn_mask = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
    for i, ids in enumerate(all_ids):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i, :len(ids)] = 1
    input_ids = input_ids.to(DEVICE); attn_mask = attn_mask.to(DEVICE)
    with torch.no_grad():
        with torch.amp.autocast(DEVICE.type, dtype=torch.bfloat16):
            _ = backbone(input_ids, token_type=0, add_cls=False)
    for depth_name in all_features:
        hidden = block_outputs[int(depth_name.replace("block", ""))].float()
        m = attn_mask.unsqueeze(-1).float().to(hidden.device)
        pooled = ((hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)).cpu().numpy()
        all_features[depth_name].append(pooled)
    if (end) % (BATCH_SIZE * 100) == 0 or end == n:
        print(f"  Feature extraction: {end:,}/{n:,} ({100*end/n:.0f}%)")
all_features = {k: np.concatenate(v, axis=0) for k, v in all_features.items()}
print(f"  Done in {(time.time()-t0)/60:.1f} min")

# Save
(feats_d := OUT_DIR / "features").mkdir(exist_ok=True)
(lbls_d := OUT_DIR / "labels").mkdir(exist_ok=True)
for k, v in all_features.items(): np.save(feats_d / f"{k}.npy", v)
np.save(lbls_d / "labels.npy", sampled_labels)
np.save(lbls_d / "genome_ids.npy", sampled_gids)

# ── Per-depth analysis ──
import umap; from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import seaborn as sns

def _clean_ax(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4); ax.set_axisbelow(True)

all_metrics = {}
for depth_name in DEPTH_NAMES:
    print(f"\n=== {depth_name} ===")
    out_path = OUT_DIR / depth_name; out_path.mkdir(parents=True, exist_ok=True)
    feat = all_features[depth_name]

    # UMAP Direct
    t0 = time.time()
    umap_direct = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1, verbose=False).fit_transform(feat)
    print(f"  Direct UMAP: {time.time()-t0:.0f}s")

    # PCA+UMAP
    t0 = time.time()
    feat_pca = PCA(n_components=50, random_state=SEED).fit_transform(feat)
    umap_pca = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1, verbose=False).fit_transform(feat_pca)
    print(f"  PCA+UMAP: {time.time()-t0:.0f}s")

    np.save(out_path / "umap_direct.npy", umap_direct)
    np.save(out_path / "umap_pca.npy", umap_pca)

    # K-Means
    k = num_classes
    for path_name, coords in [("direct", umap_direct), ("pca", umap_pca)]:
        preds = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(coords)
        m = {
            f"{path_name}_ari": float(adjusted_rand_score(sampled_labels, preds)),
            f"{path_name}_nmi": float(normalized_mutual_info_score(sampled_labels, preds)),
            f"{path_name}_homogeneity": float(homogeneity_score(sampled_labels, preds)),
            f"{path_name}_completeness": float(completeness_score(sampled_labels, preds)),
            f"{path_name}_v_measure": float(v_measure_score(sampled_labels, preds)),
        }
        all_metrics[f"{depth_name}_{path_name}"] = m
        if path_name == "direct":  # save once
            with open(out_path / "metrics.json", "w") as f: json.dump(m, f, indent=2); f.write("\n")

    # UMAP plots
    xlim = tuple(np.percentile(np.concatenate([umap_direct[:,0], umap_pca[:,0]]), [1,99]))
    ylim = tuple(np.percentile(np.concatenate([umap_direct[:,1], umap_pca[:,1]]), [1,99]))
    for coords, suffix in [(umap_direct, "direct"), (umap_pca, "pca")]:
        fig, ax = plt.subplots(figsize=(13,10))
        for lbl in sorted(range(k), key=lambda l: (sampled_labels==l).sum()):
            mask = sampled_labels == lbl
            ax.scatter(coords[mask,0], coords[mask,1], c=COLORS[lbl],
                       label=f"{idx_to_name[lbl]} ({mask.sum():,})",
                       s=3, alpha=0.70, rasterized=True, edgecolors="none")
        _clean_ax(ax); ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.legend(markerscale=14, fontsize=11, loc="upper right", frameon=True, framealpha=0.92, edgecolor="#ccc")
        ax.set_title(f"Phylum UMAP — {depth_name} ({suffix.upper()})", fontsize=16, fontweight="bold", pad=12)
        fig.tight_layout(); fig.savefig(out_path / f"umap_by_phylum_{suffix}.png", dpi=DPI, bbox_inches="tight"); plt.close()

    # Metrics bar
    dm = all_metrics[f"{depth_name}_direct"]; pm = all_metrics[f"{depth_name}_pca"]
    mnames = ["ARI","NMI","Homogeneity","Completeness","V-measure"]
    bm = {"ARI":"ari","NMI":"nmi","Homogeneity":"homogeneity","Completeness":"completeness","V-measure":"v_measure"}
    dv = [dm[f"direct_{v}"] for v in bm.values()]; pv = [pm[f"pca_{v}"] for v in bm.values()]
    x = np.arange(5); w = 0.35
    fig, ax = plt.subplots(figsize=(10,6))
    for bars, vals, color, label in [(x-w/2,dv,COLORS[3],"Direct"),(x+w/2,pv,COLORS[0],"PCA+UMAP")]:
        b = ax.bar(bars, vals, w, label=label, color=color, edgecolor="white", linewidth=0.5)
        for bar in b: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012, f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold", color=color)
    _clean_ax(ax); ax.set_xticks(x); ax.set_xticklabels(mnames); ax.legend(); ax.set_ylim(0,1.08)
    fig.tight_layout(); fig.savefig(out_path / "metrics_comparison.png", dpi=DPI, bbox_inches="tight"); plt.close()

    # Metadata CSV
    pd.DataFrame({"idx": np.arange(len(sampled_labels)), "phylum": [idx_to_name[l] for l in sampled_labels],
                  "label": sampled_labels, "umap_direct_x": umap_direct[:,0], "umap_direct_y": umap_direct[:,1],
                  "umap_pca_x": umap_pca[:,0], "umap_pca_y": umap_pca[:,1]}).to_csv(out_path / "sample_metadata.csv", index=False)
    print(f"  ARI: direct={dm['direct_ari']:.4f}, pca={pm['pca_ari']:.4f}")

# ── Save all_metrics ──
with open(OUT_DIR / "all_metrics.json", "w") as f: json.dump(all_metrics, f, indent=2); f.write("\n")

# ── Comparison chart: ARI vs Block ──
comp_dir = OUT_DIR / "comparison"; comp_dir.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(10,6))
for umap_type, ls_style, color in [("direct","-",COLORS[3]),("pca","--",COLORS[0])]:
    ari_vals = [all_metrics[f"{d}_{umap_type}"][f"{umap_type}_ari"] for d in DEPTH_NAMES]
    ax.plot(range(NUM_DEPTHS), ari_vals, linestyle=ls_style, label=f"{umap_type.upper()}", linewidth=2, markersize=8, color=color)
_clean_ax(ax)
ax.set_xticks(range(NUM_DEPTHS)); ax.set_xticklabels([f"Block {i}\n(layer {(i+1)*4})" for i in range(NUM_DEPTHS)])
ax.set_ylabel("ARI"); ax.set_title("Phylum Clustering Quality vs Model Depth", fontsize=14, fontweight="bold")
ax.legend(); fig.tight_layout(); fig.savefig(comp_dir / "ari_vs_block.png", dpi=DPI, bbox_inches="tight"); plt.close()
print(f"\nDone -> {OUT_DIR}")
