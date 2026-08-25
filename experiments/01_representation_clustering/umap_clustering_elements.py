#!/usr/bin/env python3
"""Sequence elements UMAP clustering: 2 sampling × 7 depths × dual UMAP.

Analyzes 5 fungal sequence element types (cds, intron, intergenic, 5'UTR,
3'UTR) across two sampling strategies (random, length-balanced) and 7 feature
depths (overall + 6 blocks).

Usage:
    conda activate fungi
    python scripts/umap_clustering_elements.py
"""
import sys, json, time, random as py_random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Warmup FlashAttention ──
def _warmup_flash_attn():
    try:
        from flash_attn import flash_attn_func
        _dev = torch.device("cuda:0")
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

# ── Config ──
MERGED_DIR = PROJECT_ROOT.parent / "1kfg_datasets/output/merged"
ELEMENT_FILES = {
    "cds":              MERGED_DIR / "cds.fasta",
    "intron":           MERGED_DIR / "intron.fasta",
    "intergenic":       MERGED_DIR / "intergenic.fasta",
    "five_prime_utr":   MERGED_DIR / "five_prime_utr.fasta",
    "three_prime_utr":  MERGED_DIR / "three_prime_utr.fasta",
}
BACKBONE_PATH = PROJECT_ROOT / "checkpoints/phase2_joint/backbone_final.pt"
TOKENIZER_PATH = PROJECT_ROOT / "data/processed/bpe_fungi.model"
OUT_DIR = PROJECT_ROOT / "analysis/umap_elements"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
SEED = 42
TARGET_PER_CLASS = 20000
DPI = 200
ELEMENT_COLORS = ["#7B1FA2", "#2E7D32", "#C62828", "#1565C0", "#EF6C00"]
ELEMENT_NAMES = ["cds", "intron", "intergenic", "five_prime_utr", "three_prime_utr"]

NUM_CLASSES = 5
NUM_DEPTHS = 6  # block0..block5
DEPTH_NAMES = [f"block{i}" for i in range(NUM_DEPTHS)]

rng = np.random.default_rng(SEED)
py_random.seed(SEED)

# Early writeability check
try:
    (OUT_DIR / ".write_test").touch(); (OUT_DIR / ".write_test").unlink()
except (OSError, PermissionError) as e:
    sys.exit(f"Not writeable: {OUT_DIR}\n{e}")

print(f"Device: {DEVICE}\nOutput: {OUT_DIR}")
print(f"Elements ({NUM_CLASSES}): {ELEMENT_NAMES}")
print(f"Depths ({NUM_DEPTHS}): {DEPTH_NAMES}")

# ═══════════════════════════════════════════════════
# 1. Two-pass FASTA parsing (memory-efficient for 63M seqs)
#    Pass 1: scan lengths + genome IDs only (no sequences)
#    Pass 2: re-read only sampled sequences
# ═══════════════════════════════════════════════════

print("\n=== Pass 1: scanning FASTA files (lengths + genome IDs only) ===")
element_lengths = []   # list of numpy arrays, one per element type
element_gids = []      # list of list of genome_id strings
element_counts = []

for lbl, (name, path) in enumerate(ELEMENT_FILES.items()):
    t0 = time.time()
    lengths = []
    gids = []
    with open(path) as f:
        header = None
        seq_len = 0
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    lengths.append(seq_len)
                    gids.append(header.split("|")[0][1:])  # strip '>'
                header = line.strip()
                seq_len = 0
            else:
                seq_len += len(line.strip())
        if header is not None:
            lengths.append(seq_len)
            gids.append(header.split("|")[0][1:])
    lengths_arr = np.array(lengths, dtype=np.int32)
    element_lengths.append(lengths_arr)
    element_gids.append(gids)
    element_counts.append(len(lengths_arr))
    t1 = time.time()
    print(f"  {name:20s}: {len(lengths_arr):>10,} seqs, "
          f"len min={lengths_arr.min()} median={np.median(lengths_arr):.0f} max={lengths_arr.max()}, "
          f"scan={t1-t0:.0f}s")

total_seqs = sum(element_counts)
print(f"Total: {total_seqs:,} sequences")

# ═══════════════════════════════════════════════════
# 2. Sampling (from length arrays, no sequences in memory)
# ═══════════════════════════════════════════════════

print("\n=== Sampling ===")
rng = np.random.default_rng(SEED)

def sample_random(element_lengths, per_class):
    """Simple random sampling per class. Returns per-class index arrays."""
    result = []
    for lbl in range(NUM_CLASSES):
        n_total = len(element_lengths[lbl])
        n_take = min(per_class, n_total)
        chosen = rng.choice(n_total, size=n_take, replace=False)
        result.append(chosen)
        print(f"  [random] {ELEMENT_NAMES[lbl]:20s}: {n_total:>10,} -> {n_take:>7,}")
    return result

def sample_length_balanced(element_lengths, per_class):
    """Length-stratified: 3 equal bins per class, sample equally."""
    result = []
    for lbl in range(NUM_CLASSES):
        lengths = element_lengths[lbl]
        n_total = len(lengths)
        # Sort indices by length
        sorted_idx = np.argsort(lengths)
        n_bins = 3
        bin_size = n_total // n_bins
        per_bin = per_class // n_bins

        class_chosen = []
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size if b < n_bins - 1 else n_total
            bin_pool = sorted_idx[start:end]
            n_take = min(per_bin, len(bin_pool))
            chosen = rng.choice(bin_pool, size=n_take, replace=False)
            class_chosen.append(chosen)

        class_chosen = np.concatenate(class_chosen)
        if len(class_chosen) < per_class:
            # top up
            remaining = np.setdiff1d(np.arange(n_total), class_chosen)
            extra = rng.choice(remaining, size=min(per_class - len(class_chosen), len(remaining)), replace=False)
            class_chosen = np.concatenate([class_chosen, extra])

        class_chosen = class_chosen[:per_class]
        chosen_lengths = lengths[class_chosen]
        print(f"  [balanced] {ELEMENT_NAMES[lbl]:20s}: {n_total:>10,} -> {per_class:>7,}  "
              f"len: min={chosen_lengths.min()} median={np.median(chosen_lengths):.0f} max={chosen_lengths.max()}")
        result.append(class_chosen)
    return result

random_samples = sample_random(element_lengths, TARGET_PER_CLASS)
balanced_samples = sample_length_balanced(element_lengths, TARGET_PER_CLASS)

# ═══════════════════════════════════════════════════
# 3. Pass 2: extract only sampled sequences from FASTA
# ═══════════════════════════════════════════════════

print("\n=== Pass 2: extracting sampled sequences from FASTA ===")

def extract_sampled_sequences(element_samples):
    """Re-read FASTA files and extract only the sampled sequences for each element type.
    Returns list of (sequence, label, genome_id).
    """
    # Build lookup: label -> set of indices to extract
    sample_sets = [set(samples) for samples in element_samples]

    all_seqs = []
    for lbl, (name, path) in enumerate(ELEMENT_FILES.items()):
        t0 = time.time()
        target_set = sample_sets[lbl]
        extracted = 0
        with open(path) as f:
            header = None
            seq_parts = []
            seq_idx = -1
            for line in f:
                if line.startswith(">"):
                    if header is not None and (seq_idx in target_set):
                        seq_str = "".join(seq_parts)
                        genome_id = header.split("|")[0][1:]
                        all_seqs.append((seq_str, lbl, genome_id, len(seq_str)))
                        extracted += 1
                    header = line.strip()
                    seq_parts = []
                    seq_idx += 1
                elif (seq_idx in target_set):
                    seq_parts.append(line.strip())
            # last sequence
            if header is not None and (seq_idx in target_set):
                seq_str = "".join(seq_parts)
                genome_id = header.split("|")[0][1:]
                all_seqs.append((seq_str, lbl, genome_id, len(seq_str)))
                extracted += 1
        t1 = time.time()
        print(f"  {name:20s}: extracted {extracted:,} / {len(target_set):,} seqs, {t1-t0:.0f}s")
    return all_seqs

all_seqs_random = extract_sampled_sequences(random_samples)
all_seqs_balanced = extract_sampled_sequences(balanced_samples)

# Truncate sequences exceeding model's positional encoding limit (131,072 BPE tokens → ~262K bp safe limit: 100K bp)
MAX_SEQ_LEN = 100000
all_seqs_random = [(s[:MAX_SEQ_LEN], l, g, min(ln, MAX_SEQ_LEN)) for s, l, g, ln in all_seqs_random]
all_seqs_balanced = [(s[:MAX_SEQ_LEN], l, g, min(ln, MAX_SEQ_LEN)) for s, l, g, ln in all_seqs_balanced]

print(f"Random sampling: {len(all_seqs_random):,} sequences loaded")
print(f"Balanced sampling: {len(all_seqs_balanced):,} sequences loaded")

# ═══════════════════════════════════════════════════
# 4. Load model with hooks
# ═══════════════════════════════════════════════════

print("\n=== Loading backbone with hooks ===")
model_cfg = FungiDNAConfig()
backbone = StripedMambaBackbone(model_cfg)
ckpt = torch.load(BACKBONE_PATH, map_location=DEVICE)
backbone.load_state_dict(ckpt, strict=True)
backbone = backbone.to(DEVICE).bfloat16()
backbone.eval()
print(f"Backbone: {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M params")

# Register hooks on attention layers (every 4th layer: 3, 7, 11, 15, 19, 23)
# Layer indices: 0-2 Mamba2, 3 Attention, 4-6 Mamba2, 7 Attention, ...
ATTENTION_LAYER_INDICES = [3, 7, 11, 15, 19, 23]  # 0-indexed
block_outputs = {}

def make_hook(block_idx):
    def hook(module, input, output):
        block_outputs[block_idx] = output.detach()
    return hook

handles = []
for block_idx, layer_idx in enumerate(ATTENTION_LAYER_INDICES):
    h = backbone.layers[layer_idx].register_forward_hook(make_hook(block_idx))
    handles.append(h)

# Also register on final_norm for "overall"
final_output = {}
def final_hook(module, input, output):
    final_output["overall"] = output.detach()
h_final = backbone.final_norm.register_forward_hook(final_hook)
handles.append(h_final)

tokenizer = DualTokenizer(str(TOKENIZER_PATH))
print(f"Tokenizer: vocab_size={tokenizer.bpe_vocab_size}")
print(f"Hooks: {len(handles)} registered (6 blocks + final_norm)")

# ═══════════════════════════════════════════════════
# 5. Feature extraction
# ═══════════════════════════════════════════════════

def extract_features(seq_list, desc="features"):
    """Extract features at all 7 depths for all sequences."""
    n = len(seq_list)
    all_features = {f"block{i}": [] for i in range(NUM_DEPTHS)}

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_seqs = [seq_list[i][0] for i in range(start, end)]

        # Tokenize; enforce min_length to avoid RoPE dimension issues on tiny seqs
        MIN_TOKENS = 16
        all_ids = []; max_len = 0
        for seq in batch_seqs:
            ids = tokenizer.encode_bpe(seq, add_cls=False)
            if len(ids) < MIN_TOKENS:
                ids = ids + [0] * (MIN_TOKENS - len(ids))  # pad with PAD tokens
            all_ids.append(ids)
            max_len = max(max_len, len(ids))
        # Ensure even max_len for RoPE (avoids odd-dimension mismatch)
        if max_len % 2 != 0:
            max_len += 1

        input_ids = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
        for i, ids in enumerate(all_ids):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1

        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        with torch.no_grad():
            amp_device = DEVICE.type if DEVICE.type == "cuda" else "cpu"
            amp_dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
            with torch.amp.autocast(amp_device, dtype=amp_dtype):
                _ = backbone(input_ids, token_type=0, add_cls=False)

        # Mean pool each depth's output
        for depth_name in all_features:
            hidden = block_outputs[int(depth_name.replace("block", ""))].float()
            mask_f32 = attention_mask.unsqueeze(-1).float().to(hidden.device)
            sum_hidden = (hidden * mask_f32).sum(dim=1)
            valid_len = mask_f32.sum(dim=1).clamp(min=1e-9)
            pooled = (sum_hidden / valid_len).cpu().numpy()
            all_features[depth_name].append(pooled)

        if (end) % (BATCH_SIZE * 100) == 0 or end == n:
            print(f"  [{desc}] Feature extraction: {end:,}/{n:,} ({100*end/n:.0f}%)")

    return {k: np.concatenate(v, axis=0) for k, v in all_features.items()}

# ── Extract for both sampling strategies ──
print(f"\nExtracting features for random sampling ({len(all_seqs_random):,} seqs)...")
t0 = time.time()
random_features = extract_features(all_seqs_random, "random")
t1 = time.time()
print(f"  Done in {(t1-t0)/60:.1f} min")

print(f"\nExtracting features for balanced sampling ({len(all_seqs_balanced):,} seqs)...")
t0 = time.time()
balanced_features = extract_features(all_seqs_balanced, "balanced")
t1 = time.time()
print(f"  Done in {(t1-t0)/60:.1f} min")

# Clean up hooks
for h in handles:
    h.remove()

# ── Build labels per strategy ──
random_labels = np.array([s[1] for s in all_seqs_random])
random_gids = np.array([s[2] for s in all_seqs_random])
balanced_labels = np.array([s[1] for s in all_seqs_balanced])
balanced_gids = np.array([s[2] for s in all_seqs_balanced])

# Free sequence strings to save memory
del all_seqs_random, all_seqs_balanced

# ═══════════════════════════════════════════════════
# 5. Save features + labels
# ═══════════════════════════════════════════════════

print("\n=== Saving features ===")
features_dir = OUT_DIR / "features"
labels_dir = OUT_DIR / "labels"
features_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

for depth_name in DEPTH_NAMES:
    np.save(features_dir / f"random_{depth_name}.npy", random_features[depth_name])
    np.save(features_dir / f"balanced_{depth_name}.npy", balanced_features[depth_name])

np.save(labels_dir / "random_labels.npy", random_labels)
np.save(labels_dir / "balanced_labels.npy", balanced_labels)
np.save(labels_dir / "random_genome_ids.npy", random_gids)
np.save(labels_dir / "balanced_genome_ids.npy", balanced_gids)
print("  Features + labels saved.")

# ═══════════════════════════════════════════════════
# 6. K-mer baseline (shared)
# ═══════════════════════════════════════════════════

def compute_kmer_freq(sequences, k=4):
    vocab = {}
    bases = ["A", "C", "G", "T"]
    def gen_kmers(k, prefix=""):
        if k == 0: vocab[prefix] = len(vocab); return
        for b in bases: gen_kmers(k - 1, prefix + b)
    gen_kmers(k)
    freqs = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        su = seq.upper()
        for j in range(len(su) - k + 1):
            kmer = su[j:j+k]
            if kmer in vocab: freqs[i, vocab[kmer]] += 1
        total = freqs[i].sum()
        if total > 0: freqs[i] /= total
        if (i + 1) % 20000 == 0:
            print(f"  K-mer: {i+1:,}/{len(sequences):,}")
    return freqs

print("\n=== K-mer baseline (5K subset) ===")
# Quick scan to get first 1000 seqs per element for k-mer baseline
kmer_seqs = []; kmer_labels_list = []
kmer_target = 1000
for lbl, (name, path) in enumerate(ELEMENT_FILES.items()):
    extracted = 0
    with open(path) as f:
        header = None; seq_parts = []
        for line in f:
            if line.startswith(">"):
                if header is not None and extracted < kmer_target:
                    kmer_seqs.append("".join(seq_parts)); kmer_labels_list.append(lbl)
                    extracted += 1
                header = line.strip(); seq_parts = []
            elif extracted < kmer_target:
                seq_parts.append(line.strip())
        if header is not None and extracted < kmer_target:
            kmer_seqs.append("".join(seq_parts)); kmer_labels_list.append(lbl)

kmer_features = compute_kmer_freq(kmer_seqs, k=4)
kmer_labels = np.array(kmer_labels_list)
print(f"K-mer features: {kmer_features.shape}")

# ═══════════════════════════════════════════════════
# 7. Per-analysis pipeline
# ═══════════════════════════════════════════════════

import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

def run_one_analysis(feat, labels, gids, out_subdir, element_names, depth_label):
    """Run dual UMAP + K-Means + all figures for one feature set.

    Returns dict of metrics.
    """
    out_path = OUT_DIR / out_subdir
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{out_subdir}] Running analysis...")

    # UMAP Direct
    t0 = time.time()
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1, verbose=False)
    umap_direct = reducer.fit_transform(feat)
    t1 = time.time()
    print(f"    Direct UMAP: {t1-t0:.0f}s")

    # PCA(50) + UMAP
    t0 = time.time()
    pca = PCA(n_components=50, random_state=SEED)
    feat_pca = pca.fit_transform(feat)
    reducer2 = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1, verbose=False)
    umap_pca = reducer2.fit_transform(feat_pca)
    t1 = time.time()
    print(f"    PCA+UMAP: {t1-t0:.0f}s")

    np.save(out_path / "umap_direct.npy", umap_direct)
    np.save(out_path / "umap_pca.npy", umap_pca)

    # K-Means on UMAP coords
    k = NUM_CLASSES
    km_direct = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    preds_direct = km_direct.fit_predict(umap_direct)
    km_pca = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    preds_pca = km_pca.fit_predict(umap_pca)

    metrics = {}
    for path_name, preds in [("direct", preds_direct), ("pca", preds_pca)]:
        metrics[f"{path_name}_ari"] = float(adjusted_rand_score(labels, preds))
        metrics[f"{path_name}_nmi"] = float(normalized_mutual_info_score(labels, preds))
        metrics[f"{path_name}_homogeneity"] = float(homogeneity_score(labels, preds))
        metrics[f"{path_name}_completeness"] = float(completeness_score(labels, preds))
        metrics[f"{path_name}_v_measure"] = float(v_measure_score(labels, preds))

    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    # UMAP plots
    xlim = tuple(np.percentile(np.concatenate([umap_direct[:, 0], umap_pca[:, 0]]), [1, 99]))
    ylim = tuple(np.percentile(np.concatenate([umap_direct[:, 1], umap_pca[:, 1]]), [1, 99]))

    for coords, suffix in [(umap_direct, "direct"), (umap_pca, "pca")]:
        # By element type
        fig, ax = plt.subplots(figsize=(13, 10))
        plot_order = sorted(range(k), key=lambda l: (labels == l).sum())
        for lbl in plot_order:
            mask = labels == lbl; n = mask.sum()
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=ELEMENT_COLORS[lbl], label=f"{element_names[lbl]} ({n:,})",
                       s=3, alpha=0.70, rasterized=True, edgecolors="none",
                       zorder=2 if n < 5000 else 1)
        _clean_ax(ax)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.legend(markerscale=14, fontsize=12, loc="upper right",
                  frameon=True, framealpha=0.92, edgecolor="#cccccc")
        ax.set_title(f"UMAP by Element — {depth_label} ({suffix.upper()})", fontsize=16, fontweight="bold", pad=12)
        ax.set_xlabel("UMAP 1", fontsize=13); ax.set_ylabel("UMAP 2", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_path / f"umap_by_element_{suffix}.png", dpi=DPI, bbox_inches="tight")
        plt.close()

    # Metrics comparison bar chart
    mnames = ["ARI", "NMI", "Homogeneity", "Completeness", "V-measure"]
    bm = {"ARI": "ari", "NMI": "nmi", "Homogeneity": "homogeneity",
          "Completeness": "completeness", "V-measure": "v_measure"}
    dv = [metrics[f"direct_{v}"] for v in bm.values()]
    pv = [metrics[f"pca_{v}"] for v in bm.values()]

    x = np.arange(len(mnames)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, dv, width, label="Direct UMAP", color=ELEMENT_COLORS[3])
    b2 = ax.bar(x + width/2, pv, width, label="PCA+UMAP", color=ELEMENT_COLORS[0])
    for bars, color in [(b1, ELEMENT_COLORS[3]), (b2, ELEMENT_COLORS[0])]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold", color=color)
    _clean_ax(ax)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"Metrics — {depth_label}", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(mnames, fontsize=12)
    ax.legend(fontsize=11, frameon=True, framealpha=0.92)
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    fig.savefig(out_path / "metrics_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Metadata CSV
    meta = pd.DataFrame({
        "sequence_idx": np.arange(len(labels)),
        "element": [element_names[l] for l in labels],
        "label": labels,
        "umap_direct_x": umap_direct[:, 0],
        "umap_direct_y": umap_direct[:, 1],
        "umap_pca_x": umap_pca[:, 0],
        "umap_pca_y": umap_pca[:, 1],
    })
    meta.to_csv(out_path / "sample_metadata.csv", index=False)

    return metrics, umap_direct, umap_pca

# ── Run all 14 analyses ──
print("\n=== Running all analyses ===")
all_metrics = {}

for strategy, features_dict, labels_arr, gids_arr in [
    ("random", random_features, random_labels, random_gids),
    ("balanced", balanced_features, balanced_labels, balanced_gids),
]:
    for depth_name in DEPTH_NAMES:
        key = f"{strategy}_{depth_name}"
        out_subdir = f"{strategy}/{depth_name}"
        depth_label = f"{strategy} / {depth_name}"
        print(f"\n{'='*60}")
        print(f"Analysis: {key}")

        feats = features_dict[depth_name]
        metrics, _, _ = run_one_analysis(feats, labels_arr, gids_arr, out_subdir,
                                         ELEMENT_NAMES, depth_label)
        all_metrics[key] = metrics

        print(f"  ARI: direct={metrics['direct_ari']:.4f}, pca={metrics['pca_ari']:.4f}")

# ── K-mer baseline metrics ──
print("\n=== K-mer baseline ===")
km = KMeans(n_clusters=NUM_CLASSES, random_state=SEED, n_init=10)
kmer_preds = km.fit_predict(kmer_features)
kmer_metrics = {
    "kmer_ari": float(adjusted_rand_score(kmer_labels, kmer_preds)),
    "kmer_nmi": float(normalized_mutual_info_score(kmer_labels, kmer_preds)),
    "kmer_homogeneity": float(homogeneity_score(kmer_labels, kmer_preds)),
    "kmer_completeness": float(completeness_score(kmer_labels, kmer_preds)),
    "kmer_v_measure": float(v_measure_score(kmer_labels, kmer_preds)),
}
all_metrics["kmer_baseline"] = kmer_metrics
print(f"  K-mer ARI: {kmer_metrics['kmer_ari']:.4f}")

# Save all metrics
with open(OUT_DIR / "all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)
    f.write("\n")
print(f"\nAll metrics saved to all_metrics.json")

# ═══════════════════════════════════════════════════
# 8. Cross-analysis comparison figures
# ═══════════════════════════════════════════════════

print("\n=== Cross-analysis comparison ===")
comp_dir = OUT_DIR / "comparison"
comp_dir.mkdir(parents=True, exist_ok=True)

# 8a. ARI vs Block depth
fig, ax = plt.subplots(figsize=(12, 7))
for strategy, color, marker in [("random", ELEMENT_COLORS[3], "o"), ("balanced", ELEMENT_COLORS[0], "s")]:
    for umap_type, ls_style in [("direct", "-"), ("pca", "--")]:
        ari_vals = []
        for depth_name in DEPTH_NAMES:
            key = f"{strategy}_{depth_name}"
            ari_vals.append(all_metrics[key][f"{umap_type}_ari"])
        label = f"{strategy} / {umap_type.upper()}"
        ax.plot(range(NUM_DEPTHS), ari_vals, marker=marker, linestyle=ls_style,
                label=label, linewidth=2, markersize=8,
                color=color if umap_type == "direct" else None, alpha=0.7 if umap_type == "pca" else 1.0)

# Add k-mer baseline
ax.axhline(y=kmer_metrics["kmer_ari"], color="gray", linestyle=":", linewidth=2, label=f"K-mer baseline (ARI={kmer_metrics['kmer_ari']:.3f})")

_clean_ax(ax)
ax.set_xticks(range(NUM_DEPTHS))
ax.set_xticklabels([f"Block {i}\n(layer {(i+1)*4})" for i in range(NUM_DEPTHS)], fontsize=10)
ax.set_ylabel("ARI", fontsize=14)
ax.set_title("Clustering Quality vs Model Depth", fontsize=16, fontweight="bold")
ax.legend(fontsize=10, frameon=True, framealpha=0.92)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(comp_dir / "ari_vs_block.png", dpi=DPI, bbox_inches="tight")
plt.close()

# 8b. Heatmap: strategy × depth, best ARI
heatmap_data = np.zeros((2, NUM_DEPTHS))
for i, strategy in enumerate(["random", "balanced"]):
    for j, depth_name in enumerate(DEPTH_NAMES):
        key = f"{strategy}_{depth_name}"
        heatmap_data[i, j] = max(all_metrics[key]["direct_ari"], all_metrics[key]["pca_ari"])

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
for i in range(2):
    for j in range(NUM_DEPTHS):
        ax.text(j, i, f"{heatmap_data[i, j]:.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
ax.set_xticks(range(NUM_DEPTHS))
ax.set_xticklabels([f"Block {i}" for i in range(NUM_DEPTHS)], fontsize=11)
ax.set_yticks(range(2))
ax.set_yticklabels(["Random", "Length-Balanced"], fontsize=11)
ax.set_title("Best ARI: Sampling Strategy × Model Depth", fontsize=15, fontweight="bold")
fig.colorbar(im, ax=ax, label="ARI")
fig.tight_layout()
fig.savefig(comp_dir / "sampling_vs_block_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.close()

# 8c. Random vs Balanced scatter
fig, ax = plt.subplots(figsize=(8, 8))
for j, depth_name in enumerate(DEPTH_NAMES):
    r_ari = max(all_metrics[f"random_{depth_name}"]["direct_ari"], all_metrics[f"random_{depth_name}"]["pca_ari"])
    b_ari = max(all_metrics[f"balanced_{depth_name}"]["direct_ari"], all_metrics[f"balanced_{depth_name}"]["pca_ari"])
    ax.scatter(r_ari, b_ari, s=120, label=f"Block {j}", zorder=3)
    ax.annotate(f"B{j}", (r_ari, b_ari), textcoords="offset points", xytext=(8, 4), fontsize=9)

lims = [0, 1.05]
ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
_clean_ax(ax)
ax.set_xlabel("Random Sampling ARI", fontsize=13)
ax.set_ylabel("Length-Balanced Sampling ARI", fontsize=13)
ax.set_title("Sampling Strategy Comparison by Depth", fontsize=15, fontweight="bold")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(comp_dir / "length_vs_random_scatter.png", dpi=DPI, bbox_inches="tight")
plt.close()

print("  -> comparison/ari_vs_block.png")
print("  -> comparison/sampling_vs_block_heatmap.png")
print("  -> comparison/length_vs_random_scatter.png")

print(f"\n{'='*60}")
print(f"Done! All outputs in {OUT_DIR}")
print(f"Total analyses: 14 (2 sampling × 7 depths)")
