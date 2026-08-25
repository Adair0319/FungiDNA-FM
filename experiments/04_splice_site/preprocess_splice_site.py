"""
Pre-tokenize splice site parquet files into memory-mapped numpy arrays.
Reads parquet in chunks, BPE-tokenizes, writes int16 numpy memmap.
Memory usage: constant (~2GB).
Output:
  data/downstream/task2_splice_site/tokenized/
    train_input_ids.npy  (int16 memmap)
    train_labels.npy     (int8)
    val_input_ids.npy
    val_labels.npy
    test_input_ids.npy
    test_labels.npy
    train_lengths.npy    (int32, actual token count per sequence)
"""

import os, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fungidna.data.tokenizer import DualTokenizer

DATA_DIR = "data/downstream/task2_splice_site"
OUT_DIR = os.path.join(DATA_DIR, "tokenized")
TOKENIZER_PATH = "data/processed/bpe_fungi.model"
MAX_TOKENS_PER_SEQ = 200  # 401bp BPE → rarely exceeds 150 tokens

os.makedirs(OUT_DIR, exist_ok=True)
tokenizer = DualTokenizer(TOKENIZER_PATH)


def preprocess_split(split_name: str):
    pq_path = os.path.join(DATA_DIR, f"{split_name}.parquet")
    pf = pq.ParquetFile(pq_path)
    n_rows = pf.metadata.num_rows
    print(f"\n{split_name}: {n_rows:,} rows")

    # Read row groups one at a time
    input_ids_list = []
    labels_list = []
    lengths_list = []

    t0 = time.time()
    batch_size = 10000
    n_batches = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=["sequence", "label"]):
        sequences = batch.column("sequence").to_pylist()
        labels = batch.column("label").to_numpy()

        batch_ids = []
        batch_lengths = []
        for seq in sequences:
            ids = tokenizer.encode_bpe(seq, add_cls=True)
            # Truncate to MAX_TOKENS_PER_SEQ (shouldn't happen for 401bp)
            ids = ids[:MAX_TOKENS_PER_SEQ]
            batch_lengths.append(len(ids))
            # Pad to MAX_TOKENS_PER_SEQ
            ids = ids + [0] * (MAX_TOKENS_PER_SEQ - len(ids))
            batch_ids.append(ids)

        input_ids_list.append(np.array(batch_ids, dtype=np.int16))
        labels_list.append(labels.astype(np.int8))
        lengths_list.append(np.array(batch_lengths, dtype=np.int32))

        n_batches += 1
        if n_batches % 100 == 0:
            processed = n_batches * batch_size
            elapsed = time.time() - t0
            print(f"  {processed:,}/{n_rows:,} ({elapsed:.0f}s)")

        # Free memory periodically
        del sequences, batch_ids, batch_lengths

    # Concatenate and save as memmap
    all_ids = np.concatenate(input_ids_list, axis=0)
    all_labels = np.concatenate(labels_list)
    all_lengths = np.concatenate(lengths_list)

    print(f"  Saving... shapes: ids={all_ids.shape}, labels={all_labels.shape}, lengths={all_lengths.shape}")

    ids_path = os.path.join(OUT_DIR, f"{split_name}_input_ids.npy")
    labels_path = os.path.join(OUT_DIR, f"{split_name}_labels.npy")
    lengths_path = os.path.join(OUT_DIR, f"{split_name}_lengths.npy")

    np.save(ids_path, all_ids)
    np.save(labels_path, all_labels)
    np.save(lengths_path, all_lengths)

    # Convert to memory-mapped versions
    ids_mmap = np.load(ids_path, mmap_mode="r")
    labels_mmap = np.load(labels_path, mmap_mode="r")
    lengths_mmap = np.load(lengths_path, mmap_mode="r")

    elapsed = time.time() - t0
    print(f"  Done: {all_ids.shape} in {elapsed:.0f}s "
          f"({os.path.getsize(ids_path)/1e9:.1f}GB ids, "
          f"{os.path.getsize(labels_path)/1e6:.1f}MB labels)")


if __name__ == "__main__":
    for split in ["test", "val", "train"]:
        preprocess_split(split)
    print("\nAll done!")
