"""
Streaming combine: write final parquets using PyArrow ParquetWriter.
Memory-efficient: processes one species at a time.
"""
import os, random
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TEMP_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/data/downstream/task2_splice_site/_temp_species"
OUT_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/data/downstream/task2_splice_site"
SEED = 42
LABEL_NONSITE = 2

random.seed(SEED)
np.random.seed(SEED)

def split_species(species_list, train_r=0.70, val_r=0.10, test_r=0.20, seed=42):
    rng = random.Random(seed)
    sp = sorted(species_list)
    rng.shuffle(sp)
    n = len(sp)
    n_tr = round(n * train_r)
    n_vl = round(n * val_r)
    return set(sp[:n_tr]), set(sp[n_tr:n_tr+n_vl]), set(sp[n_tr+n_vl:])

temp_species = sorted([f.replace(".parquet","") for f in os.listdir(TEMP_DIR) if f.endswith(".parquet")])
print(f"Found {len(temp_species)} species temp files")

train_sp, val_sp, test_sp = split_species(temp_species, seed=SEED)
print(f"Split: {len(train_sp)} train / {len(val_sp)} val / {len(test_sp)} test")

for split_name, sp_set in [("val", val_sp), ("test", test_sp), ("train", train_sp)]:
    sp_list = sorted(sp_set)
    print(f"\n{'='*50}")

    # ── Pass 1: count ──
    print(f"{split_name}: counting...")
    total_pos = 0
    total_neg = 0
    species_info = {}
    for sp in sp_list:
        tp = os.path.join(TEMP_DIR, f"{sp}.parquet")
        if not os.path.exists(tp):
            continue
        df = pd.read_parquet(tp, columns=["label"])
        n_pos = int((df["label"] != LABEL_NONSITE).sum())
        n_neg = int((df["label"] == LABEL_NONSITE).sum())
        species_info[sp] = (n_pos, n_neg)
        total_pos += n_pos
        total_neg += n_neg

    target_neg = max(total_pos // 2, 100)
    keep_neg_ratio = min(target_neg / total_neg, 1.0) if total_neg > 0 else 0
    print(f"  {total_pos} pos, {total_neg} neg → target {target_neg} neg (ratio: {keep_neg_ratio:.3f})")

    # ── Pass 2: write ──
    print(f"{split_name}: writing...")
    out_path = os.path.join(OUT_DIR, f"{split_name}.parquet")
    writer = None
    written = 0
    neg_written = 0

    rng = random.Random(SEED)
    sp_order = list(sp_list)
    rng.shuffle(sp_order)

    for sp in sp_order:
        if sp not in species_info:
            continue
        tp = os.path.join(TEMP_DIR, f"{sp}.parquet")
        df = pd.read_parquet(tp)
        # Ensure consistent types: transcript_id should be string, not null
        df["transcript_id"] = df["transcript_id"].fillna("").astype(str)

        pos_df = df[df["label"] != LABEL_NONSITE]
        neg_df = df[df["label"] == LABEL_NONSITE]

        # Write all positives
        if len(pos_df) > 0:
            tbl = pa.Table.from_pandas(pos_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema)
            writer.write_table(tbl)
            written += len(pos_df)

        # Write proportion of negatives
        n_sp_neg, sp_neg_total = species_info[sp]
        sp_neg_keep = max(0, int(n_sp_neg * keep_neg_ratio))
        if sp_neg_keep > 0 and len(neg_df) > 0:
            if sp_neg_keep < len(neg_df):
                neg_sample = neg_df.sample(n=sp_neg_keep, random_state=SEED + hash(sp) % 10000)
            else:
                neg_sample = neg_df
            tbl = pa.Table.from_pandas(neg_sample, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl.schema)
            writer.write_table(tbl)
            written += len(neg_sample)
            neg_written += len(neg_sample)

        del df, pos_df, neg_df
        if (len(sp_order) > 10 and sp_list.index(sp) % 50 == 0):
            print(f"  ... {sp} ({written} samples so far)")

    if writer:
        writer.close()

    # Verify
    actual = len(pd.read_parquet(out_path, columns=["label"]))
    print(f"  {split_name}: {actual} samples → {out_path}")

print("\nDone!")
