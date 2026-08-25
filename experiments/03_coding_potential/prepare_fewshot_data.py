"""
Prepare few-shot training subsets from fold 0 of CDS vs Intergenic dataset.
Stratified nested sampling: 10% ⊂ 20% ⊂ 30% ⊂ 40%.
"""

import os, json, numpy as np, pandas as pd

SEED = 42
np.random.seed(SEED)

DATA_PATH = "/home/lty/yy_projects/fungi_project/fungi_dna_model/data/downstream/task_cds_intergenic/dataset.parquet"
OUTPUT_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/checkpoints/cds_intergenic_fewshot/data"
RATIOS = [0.10, 0.20, 0.30, 0.40]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)

# Fold 0: train pool (fold != 0), test set (fold == 0)
train_pool = df[df["fold"] != 0].copy()
test_set = df[df["fold"] == 0].copy()
print(f"Train pool: {len(train_pool):,} (pos={train_pool.label.sum():,})")
print(f"Test set:  {len(test_set):,} (pos={test_set.label.sum():,})")

pos_pool = train_pool[train_pool["label"] == 1]
neg_pool = train_pool[train_pool["label"] == 0]
print(f"Positive pool: {len(pos_pool):,}")
print(f"Negative pool: {len(neg_pool):,}")

# Generate nested samples using positional indexing
pos_indices = pos_pool.index.tolist()
neg_indices = neg_pool.index.tolist()
prev_pos_set = set()
prev_neg_set = set()

for ratio in RATIOS:
    n_per_class = int(len(pos_pool) * ratio)
    print(f"\n--- {ratio*100:.0f}%: {n_per_class:,} per class ---")

    n_new_pos = n_per_class - len(prev_pos_set)
    n_new_neg = n_per_class - len(prev_neg_set)

    # Sample from remaining pool
    remaining_pos = [i for i in pos_indices if i not in prev_pos_set]
    remaining_neg = [i for i in neg_indices if i not in prev_neg_set]

    new_pos = np.random.choice(remaining_pos, size=n_new_pos, replace=False)
    new_neg = np.random.choice(remaining_neg, size=n_new_neg, replace=False)

    prev_pos_set.update(new_pos)
    prev_neg_set.update(new_neg)

    # Build subset (all accumulated indices)
    subset_idx = list(prev_pos_set) + list(prev_neg_set)
    subset = train_pool.loc[subset_idx]
    print(f"  Total: {len(subset):,} (pos={subset.label.sum():,}, neg={len(subset)-subset.label.sum():,})")

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"train_{int(ratio*100)}pct.parquet")
    subset.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

# Save test set copy for reference
test_path = os.path.join(OUTPUT_DIR, "test_fold0.parquet")
test_set.to_parquet(test_path, index=False)
print(f"\nTest set saved: {test_path}")
print("Done.")
