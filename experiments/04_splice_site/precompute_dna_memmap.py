"""
Pre-compute DNA sequences as uint8 numpy memmap (one-time).
Stores each 401bp sequence as a row of uint8 bytes: A=0, C=1, G=2, T=3, N=4.
Output: 68M × 401 bytes ≈ 27 GB per split (train), ~4 GB val, ~8 GB test.
"""
import os, sys, time
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = "data/downstream/task2_splice_site"
OUT_DIR = os.path.join(DATA_DIR, "dna_memmap")
os.makedirs(OUT_DIR, exist_ok=True)

NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

def precompute(split_name):
    pq_path = f"{DATA_DIR}/{split_name}.parquet"
    out_path = f"{OUT_DIR}/{split_name}_dna.npy"

    if os.path.exists(out_path):
        print(f"{split_name}: already exists, skip")
        return

    print(f"{split_name}: loading parquet...", flush=True)
    pf = pq.ParquetFile(pq_path)
    n_rows = pf.metadata.num_rows
    seq_len = 401

    # Create memmap
    arr = np.memmap(out_path + ".tmp", dtype=np.uint8, mode="w+", shape=(n_rows, seq_len))

    batch_size = 500000
    offset = 0
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=batch_size, columns=["sequence"]):
        seqs = batch.column("sequence").to_pylist()
        for i, s in enumerate(seqs):
            for j, c in enumerate(s):
                arr[offset + i, j] = NUC_MAP.get(c, 4)  # N → 4
        offset += len(seqs)
        if offset % 5000000 == 0:
            print(f"  {offset:,}/{n_rows:,} ({time.time()-t0:.0f}s)", flush=True)

    arr.flush()
    os.rename(out_path + ".tmp", out_path)
    print(f"  Done: {out_path} ({os.path.getsize(out_path)/1e9:.1f} GB)", flush=True)

if __name__ == "__main__":
    for split in ["test", "val", "train"]:
        precompute(split)
    print("All done!")
