"""Pre-compute per-species file lists and cumulative counts to avoid filesystem contention during DDP init."""
import os, json, pyarrow.parquet as pq

DATA_DIR = "data/downstream/task2_splice_site"
TEMP_DIR = os.path.join(DATA_DIR, "_temp_species")

for split in ["train", "val", "test"]:
    with open(os.path.join(DATA_DIR, f"{split}_species.json")) as f:
        species_list = json.load(f)

    files = []
    cum_counts = []
    total = 0
    for sp in species_list:
        fpath = os.path.join(TEMP_DIR, f"{sp}.parquet")
        if os.path.exists(fpath):
            pf = pq.ParquetFile(fpath)
            total += pf.metadata.num_rows
            files.append(fpath)
            cum_counts.append(total)

    out = {"files": files, "cum_counts": cum_counts, "total": total}
    out_path = os.path.join(DATA_DIR, f"{split}_index.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"{split}: {len(files)} files, {total:,} rows → {out_path}")
