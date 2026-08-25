#!/usr/bin/env python3
"""Build genome-isolated Class, Order, Family 10Kbp datasets.

Same methodology as build_phylum_subphylum_datasets.py:
  - Stratified per-label genome-level 80/20 split
  - 10Kbp non-overlapping windows, N% ≤ 5%, max 600 per genome
  - Incertae sedis genomes excluded

Output: data/downstream/task0_class/  task0_order/  task0_family/
"""
import sys, os, json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fungidna.data.taxonomy import TaxonomyDB
from fungidna.data.qc_filter import _read_fasta

# ── Config ───────────────────────────────────────────────────
WINDOW_SIZE = 10_000
STRIDE = 10_000
MAX_N_FRAC = 0.05
MAX_PER_GENOME = 600
SEED = 42
ASSEMBLY_DIR = PROJECT_ROOT.parent / "1kfg_datasets" / "assembly_genome"
OUTPUT_BASE = PROJECT_ROOT / "data" / "downstream"
TAXONOMY_XLSX = str(PROJECT_ROOT / "fungi_classification.xlsx")


def sample_genome_windows(fasta_path: Path, rng: np.random.Generator) -> list[str]:
    """Proportional-allocation sampling of 10Kbp windows from a genome FASTA.

    Returns up to MAX_PER_GENOME non-overlapping windows with N% ≤ MAX_N_FRAC.
    """
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
            window = seq[start : start + WINDOW_SIZE]
            n_count = window.count("N")
            if n_count / WINDOW_SIZE <= MAX_N_FRAC:
                pool.append(window)

    if len(pool) > MAX_PER_GENOME:
        pool = rng.choice(pool, size=MAX_PER_GENOME, replace=False).tolist()

    return pool


def build_dataset(
    task_name: str,
    rank: str,
    exclude_labels: set[str] | None,
    output_dir: Path,
    rng: np.random.Generator,
):
    """Full pipeline: split → sample → save for one dataset."""
    print(f"\n{'='*60}")
    print(f"Building {task_name} ({rank})")
    print(f"{'='*60}")

    taxdb = TaxonomyDB(TAXONOMY_XLSX)
    train_ids, test_ids, ignore_labels = taxdb.split_stratified(
        rank=rank, seed=SEED, exclude_labels=exclude_labels
    )
    print(f"Split: {len(train_ids)} train, {len(test_ids)} test genomes")

    # Build label map
    genome_to_group, _ = taxdb._rank_maps[rank]
    all_labels = sorted(set(
        genome_to_group.get(g, "Unknown")
        for g in taxdb.all_genomes
        if not (exclude_labels and genome_to_group.get(g) in exclude_labels)
    ))
    label_map = {lbl: i for i, lbl in enumerate(all_labels)}
    print(f"Labels: {len(label_map)} classes")
    for lbl, idx in label_map.items():
        print(f"  [{idx}] {lbl}")

    # Sample and save each split
    os.makedirs(output_dir, exist_ok=True)

    for split_name, genome_list in [("train", train_ids), ("test", test_ids)]:
        rows = []
        stats = {}
        for genome_id in genome_list:
            fasta_path = (ASSEMBLY_DIR / f"{genome_id}.fasta").resolve()
            if not fasta_path.exists():
                print(f"  WARN: missing FASTA for {genome_id}, skipping")
                stats[genome_id] = 0
                continue
            windows = sample_genome_windows(fasta_path, rng)
            label = label_map[genome_to_group.get(genome_id, "Unknown")]
            for w in windows:
                rows.append({"sequence": w, "genome_id": genome_id, "label": label})
            stats[genome_id] = len(windows)

        df = pd.DataFrame(rows)
        out_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        total_windows = len(df)
        print(f"  {split_name}: {len(genome_list)} genomes → {total_windows:,} windows → {out_path}")

    # Save metadata
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    with open(output_dir / "genome_split.json", "w") as f:
        json.dump({"train": sorted(train_ids), "test": sorted(test_ids)}, f, indent=2)
    with open(output_dir / "sampling_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(output_dir / "ignore_labels.json", "w") as f:
        json.dump(list(ignore_labels.keys()), f)

    # Summary
    train_df = pd.read_parquet(output_dir / "train.parquet")
    test_df = pd.read_parquet(output_dir / "test.parquet")
    print(f"Summary: {len(train_df):,} train windows, {len(test_df):,} test windows")
    print(f"Genomes reaching 600 cap: {sum(1 for v in stats.values() if v >= 600)}/{len(stats)}")
    print(f"Ignored labels (all-train, no test): {len(ignore_labels)}")


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    # ── Class dataset ──
    build_dataset(
        task_name="task0_class",
        rank="class",
        exclude_labels={"Incertae sedis"},
        output_dir=OUTPUT_BASE / "task0_class",
        rng=rng,
    )

    # ── Order dataset ──
    build_dataset(
        task_name="task0_order",
        rank="order",
        exclude_labels={"Incertae sedis"},
        output_dir=OUTPUT_BASE / "task0_order",
        rng=rng,
    )

    # ── Family dataset ──
    build_dataset(
        task_name="task0_family",
        rank="family",
        exclude_labels={"Incertae sedis"},
        output_dir=OUTPUT_BASE / "task0_family",
        rng=rng,
    )

    print(f"\n{'='*60}")
    print("Done. Outputs:")
    print(f"  {OUTPUT_BASE / 'task0_class'}")
    print(f"  {OUTPUT_BASE / 'task0_order'}")
    print(f"  {OUTPUT_BASE / 'task0_family'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
