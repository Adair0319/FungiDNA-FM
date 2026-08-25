"""Prepare few-shot training subsets for folds 1-4 (fold 0 already done)."""
import os, numpy as np, pandas as pd

SEED = 42
np.random.seed(SEED)

DATA_PATH = "/home/lty/yy_projects/fungi_project/fungi_dna_model/data/downstream/task_cds_intergenic/dataset.parquet"
OUTPUT_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/checkpoints/cds_intergenic_fewshot/data"
RATIOS = [0.10, 0.20, 0.30, 0.40]
FOLDS = [1, 2, 3, 4]

df = pd.read_parquet(DATA_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fold in FOLDS:
    train_pool = df[df["fold"] != fold]
    pos_pool = train_pool[train_pool["label"] == 1]
    neg_pool = train_pool[train_pool["label"] == 0]

    pos_indices = pos_pool.index.tolist()
    neg_indices = neg_pool.index.tolist()

    pos_set, neg_set = set(), set()

    for ratio in RATIOS:
        n_per_class = int(len(pos_pool) * ratio)
        n_new_pos = n_per_class - len(pos_set)
        n_new_neg = n_per_class - len(neg_set)

        remaining_pos = [i for i in pos_indices if i not in pos_set]
        remaining_neg = [i for i in neg_indices if i not in neg_set]

        new_pos = np.random.choice(remaining_pos, size=n_new_pos, replace=False)
        new_neg = np.random.choice(remaining_neg, size=n_new_neg, replace=False)

        pos_set.update(new_pos)
        neg_set.update(new_neg)

        subset = train_pool.loc[list(pos_set) + list(neg_set)]
        out_path = os.path.join(OUTPUT_DIR, f"train_fold{fold}_{int(ratio*100)}pct.parquet")
        subset.to_parquet(out_path, index=False)
        print(f"Fold {fold} {int(ratio*100)}%: {len(subset):,} saved")

    # Also save test set for this fold
    test_set = df[df["fold"] == fold]
    test_path = os.path.join(OUTPUT_DIR, f"test_fold{fold}.parquet")
    test_set.to_parquet(test_path, index=False)
    print(f"Fold {fold} test: {len(test_set):,} saved")

print("Done.")
