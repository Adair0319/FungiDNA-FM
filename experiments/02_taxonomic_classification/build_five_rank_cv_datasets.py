#!/usr/bin/env python3
"""Build five-rank 5-fold CV datasets from 735 genome assemblies.

One sampling pass -> per-genome 5-fold split -> five independent datasets
(Phylum/Subphylum/Class/Order/Family), each with Incertae sedis filtered.
"""
import sys, os, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fungidna.data.taxonomy import TaxonomyDB
from fungidna.data.qc_filter import _read_fasta

# -- Config --
WINDOW_SIZE = 10_000
STRIDE = 10_000
MAX_N_FRAC = 0.05
MAX_PER_GENOME = 600
SEED = 42
ASSEMBLY_DIR = PROJECT_ROOT.parent / "1kfg_datasets" / "assembly_genome"
OUTPUT_BASE = PROJECT_ROOT / "data" / "downstream"
TAXONOMY_XLSX = PROJECT_ROOT / "fungi_classification.xlsx"

RANKS = ["phylum", "subphylum", "class", "order", "family"]
INCERTAE_SEDIS = "Incertae sedis"


def sample_genome_windows(fasta_path: Path, rng: np.random.Generator) -> list[str]:
    """Proportional-allocation sampling of 10Kbp windows. Returns up to 600."""
    valid_scaffolds = []
    for header, seq in _read_fasta(str(fasta_path)):
        if len(seq) >= WINDOW_SIZE:
            valid_scaffolds.append((header, seq))
    if not valid_scaffolds:
        return []

    total_len = sum(len(s) for _, s in valid_scaffolds)
    pool = []
    for header, seq in valid_scaffolds:
        weight = len(seq) / total_len
        target = int(weight * MAX_PER_GENOME)
        possible_starts = list(range(0, len(seq) - WINDOW_SIZE + 1, STRIDE))
        if not possible_starts:
            continue
        n_sample = min(target, len(possible_starts))
        selected = rng.choice(possible_starts, size=n_sample, replace=False)
        for start in selected:
            window = seq[start: start + WINDOW_SIZE]
            if window.count("N") / WINDOW_SIZE <= MAX_N_FRAC:
                pool.append(window)
    if len(pool) > MAX_PER_GENOME:
        pool = rng.choice(pool, size=MAX_PER_GENOME, replace=False).tolist()
    return pool


def assign_folds(n_windows: int, seed: int = SEED) -> np.ndarray:
    """Assign each window to a fold (0-4), shuffled within genome."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_windows)
    folds = np.empty(n_windows, dtype=int)
    for fold_id in range(5):
        start = fold_id * n_windows // 5
        end = (fold_id + 1) * n_windows // 5
        folds[indices[start:end]] = fold_id
    return folds


def build_fold_parquet(rows: list[dict], fold_id: int, output_dir: Path):
    """Write train/val/test parquet for a specific fold."""
    test_rows = [r for r in rows if r["fold_id"] == fold_id]
    train_pool = [r for r in rows if r["fold_id"] != fold_id]

    rng = np.random.default_rng(SEED + fold_id)
    rng.shuffle(train_pool)
    n_val = max(1, len(train_pool) // 10)
    val_rows = train_pool[:n_val]
    train_rows = train_pool[n_val:]

    fold_dir = output_dir / f"fold{fold_id + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    cols = ["sequence", "genome_id", "fold_id", "label"]
    pd.DataFrame([{c: r[c] for c in cols} for r in train_rows]).to_parquet(
        fold_dir / "train.parquet", index=False
    )
    pd.DataFrame([{c: r[c] for c in cols} for r in val_rows]).to_parquet(
        fold_dir / "val.parquet", index=False
    )
    pd.DataFrame([{c: r[c] for c in cols} for r in test_rows]).to_parquet(
        fold_dir / "test.parquet", index=False
    )
    return train_rows, val_rows, test_rows


def build_stats(rows_by_fold: dict, label_map: dict, rank: str, output_dir: Path):
    """Write stats.csv for all folds."""
    idx_to_name = {v: k for k, v in label_map.items()}
    stats_rows = []
    for fold_id in range(5):
        train_rows, val_rows, test_rows = rows_by_fold[fold_id]
        for lbl in sorted(label_map.values()):
            name = idx_to_name[lbl]
            stats_rows.append({
                "fold": fold_id + 1,
                "rank": rank,
                "class_name": name,
                "train_count": sum(1 for r in train_rows if r["label"] == lbl),
                "val_count": sum(1 for r in val_rows if r["label"] == lbl),
                "test_count": sum(1 for r in test_rows if r["label"] == lbl),
            })
    pd.DataFrame(stats_rows).to_csv(output_dir / "stats.csv", index=False)
    print(f"  stats.csv ({len(stats_rows)} rows)")


def build_rank_dataset(rank: str, all_rows: list[dict], taxdb: TaxonomyDB, output_dir: Path):
    """Build 5-fold dataset for one taxonomic rank."""
    print(f"\n{'='*60}")
    print(f"Building task0_{rank}_five_rank")
    print(f"{'='*60}")

    # Identify Incertae sedis label for this rank
    genome_to_group, _ = taxdb._rank_maps[rank]
    is_genomes = {g for g in taxdb.all_genomes if genome_to_group.get(g) == INCERTAE_SEDIS}
    print(f"  Incertae sedis genomes to exclude: {len(is_genomes)}")

    # Filter rows: remove IS genomes
    filtered_rows = [r for r in all_rows if r["genome_id"] not in is_genomes]
    n_dropped = len(all_rows) - len(filtered_rows)
    print(f"  Dropped {n_dropped:,} windows from {len(is_genomes)} IS genomes")
    print(f"  Remaining: {len(filtered_rows):,} windows from {len(set(r['genome_id'] for r in filtered_rows))} genomes")

    # Build label map
    all_labels = sorted(set(
        genome_to_group.get(r["genome_id"], "Unknown")
        for r in filtered_rows
    ))
    label_map = {lbl: i for i, lbl in enumerate(all_labels)}
    print(f"  Classes: {len(label_map)}")

    # Assign labels
    for r in filtered_rows:
        r["label"] = label_map[genome_to_group[r["genome_id"]]]

    # Save label_map
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    # Build per-fold parquets
    rows_by_fold = {}
    for fold_id in range(5):
        print(f"  Fold {fold_id + 1}...")
        train_rows, val_rows, test_rows = build_fold_parquet(filtered_rows, fold_id, output_dir)
        rows_by_fold[fold_id] = (train_rows, val_rows, test_rows)
        print(f"    train={len(train_rows):,}  val={len(val_rows):,}  test={len(test_rows):,}")

    build_stats(rows_by_fold, label_map, rank, output_dir)


def main():
    rng = np.random.default_rng(SEED)
    taxdb = TaxonomyDB(str(TAXONOMY_XLSX))
    print(f"Taxonomy: {len(taxdb)} genomes")

    # -- Phase 1: Sample all genomes --
    print(f"\n{'='*60}")
    print("Phase 1: Sampling all 735 genomes")
    print(f"{'='*60}")

    genome_to_group = {}
    for rank in RANKS:
        g2g, _ = taxdb._rank_maps[rank]
        genome_to_group[rank] = g2g

    all_rows = []
    genome_window_counts = {}
    fasta_files = sorted(ASSEMBLY_DIR.glob("*.fasta"))

    for i, fp in enumerate(fasta_files):
        genome_id = fp.stem
        if genome_id not in taxdb.all_genomes:
            continue
        windows = sample_genome_windows(fp, rng)
        for w in windows:
            row = {
                "sequence": w,
                "genome_id": genome_id,
                "phylum": genome_to_group["phylum"].get(genome_id, "Unknown"),
                "subphylum": genome_to_group["subphylum"].get(genome_id, "Unknown"),
                "class": genome_to_group["class"].get(genome_id, "Unknown"),
                "order": genome_to_group["order"].get(genome_id, "Unknown"),
                "family": genome_to_group["family"].get(genome_id, "Unknown"),
            }
            all_rows.append(row)
        genome_window_counts[genome_id] = len(windows)
        if (i + 1) % 100 == 0:
            print(f"  Sampled {i+1}/{len(fasta_files)} genomes, {len(all_rows):,} windows so far")

    print(f"  Total: {len(all_rows):,} windows from {len(genome_window_counts)} genomes")
    print(f"  Mean windows/genome: {np.mean(list(genome_window_counts.values())):.0f}")
    print(f"  Genomes at 600 cap: {sum(1 for v in genome_window_counts.values() if v >= 600)}")

    # -- Phase 2: Assign folds per genome --
    print(f"\n{'='*60}")
    print("Phase 2: Assigning 5-fold splits per genome")
    print(f"{'='*60}")

    genome_rows = defaultdict(list)
    for r in all_rows:
        genome_rows[r["genome_id"]].append(r)

    fold_assignments = {}
    for gid, rows in genome_rows.items():
        folds = assign_folds(len(rows))
        fold_assignments[gid] = folds
        for idx, r in enumerate(rows):
            r["fold_id"] = int(folds[idx])

    for fold_id in range(5):
        count = sum(1 for r in all_rows if r["fold_id"] == fold_id)
        print(f"  Fold {fold_id}: {count:,} windows ({100*count/len(all_rows):.1f}%)")

    # -- Phase 3: Build rank datasets --
    print(f"\n{'='*60}")
    print("Phase 3: Building rank-specific datasets")
    print(f"{'='*60}")

    for rank in RANKS:
        output_dir = OUTPUT_BASE / f"task0_{rank}_five_rank"
        output_dir.mkdir(parents=True, exist_ok=True)
        build_rank_dataset(rank, all_rows, taxdb, output_dir)

    print(f"\n{'='*60}")
    print("Done. Outputs in data/downstream/task0_*_five_rank/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
