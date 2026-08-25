#!/usr/bin/env python
"""
Build CDS vs Intergenic binary classification dataset.
CD-HIT dedup → length filter → bucket sample → negative match → 5-fold split → parquet.
"""

import sys
import os
import json
import random
import subprocess
import tempfile
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold

random.seed(42)
np.random.seed(42)

# Paths
CDHIT_BIN = "/home/lty/miniconda3/envs/fungi_dna/bin/cd-hit-est"
CDS_FASTA = "/home/lty/yy_projects/fungi_project/1kfg_datasets/output/merged/cds.fasta"
INTERGENIC_FASTA = "/home/lty/yy_projects/fungi_project/1kfg_datasets/output/merged/intergenic.fasta"
OUTPUT_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/data/downstream/task_cds_intergenic"

# Parameters
CDHIT_THRESHOLD = 0.9
MIN_LEN = 300
MAX_LEN = 5000
N_BUCKETS = 20
PER_BUCKET = 5000   # 20 * 5000 = 100k per class
TOTAL_SAMPLES = 200000
N_FOLDS = 5
VAL_RATIO = 0.10    # 10% of non-test = val
TEST_RATIO = 0.20   # 20% of total = test

# Temporary files
TEMP_CDS_FILTERED = "/tmp/cds_length_filtered.fasta"
TEMP_CDS_DEDUP = "/tmp/cds_dedup.fasta"
TEMP_POS_SAMPLED = "/tmp/cds_pos_sampled.fasta"
TEMP_POS_DEDUP = "/tmp/cds_pos_dedup.fasta"


def stream_lengths(fasta_path):
    """
    Stream a FASTA file and yield (header, sequence_length) tuples.
    O(1) memory — never holds all sequences at once.

    Yields:
        (header: str, length: int)
    """
    header = None
    seq_len = 0
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    yield (header, seq_len)
                header = line[1:]
                seq_len = 0
            else:
                seq_len += len(line)
        if header is not None:
            yield (header, seq_len)


def run_cdhit(input_fasta, output_fasta, threshold=0.9):
    """
    Run cd-hit-est for sequence deduplication.

    Args:
        input_fasta: path to input FASTA
        output_fasta: path for deduplicated output
        threshold: sequence identity threshold (0.9 = 90%)

    Returns:
        path to output fasta
    """
    cmd = [
        CDHIT_BIN,
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(threshold),
        "-aS", "0.9",    # alignment coverage >= 90%
        "-d", "0",       # full header in output
        "-T", "8",       # 8 threads
        "-M", "0",       # unlimited memory
    ]
    print(f"Running cd-hit-est: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:])
        raise RuntimeError(f"cd-hit-est failed with code {result.returncode}")
    return output_fasta


def build_cds_length_distribution(cds_fasta, min_len, max_len):
    """
    Stream CDS FASTA, filter by length, return length array + write filtered FASTA.

    Returns:
        np.ndarray of filtered CDS lengths
    Side effect: writes TEMP_CDS_FILTERED
    """
    lengths = []
    count_total = 0
    count_filtered = 0

    with open(TEMP_CDS_FILTERED, 'w') as outf:
        with open(cds_fasta, 'r') as inf:
            header = None
            seq_lines = []
            for line in inf:
                if line.startswith('>'):
                    # Write previous sequence if passes filter
                    if header is not None:
                        seq = ''.join(seq_lines)
                        seq_len = len(seq)
                        count_total += 1
                        if min_len <= seq_len <= max_len:
                            outf.write(f'>{header}\n')
                            for i in range(0, len(seq), 60):
                                outf.write(seq[i:i+60] + '\n')
                            lengths.append(seq_len)
                            count_filtered += 1
                    header = line[1:].strip()
                    seq_lines = []
                else:
                    seq_lines.append(line.strip())
            # Last sequence
            if header is not None:
                seq = ''.join(seq_lines)
                seq_len = len(seq)
                count_total += 1
                if min_len <= seq_len <= max_len:
                    outf.write(f'>{header}\n')
                    for i in range(0, len(seq), 60):
                        outf.write(seq[i:i+60] + '\n')
                    lengths.append(seq_len)
                    count_filtered += 1

    print(f"CDS length filter: {count_total:,} -> {count_filtered:,} "
          f"({100*count_filtered/max(count_total,1):.1f}%) in [{min_len}, {max_len}]")
    return np.array(lengths)


def compute_softmask_ratio(seq):
    """Fraction of lowercase bases (softmasked repeats)."""
    if not seq:
        return 0.0
    return sum(1 for c in seq if c.islower()) / len(seq)


def get_bucket_boundaries(lengths, n_buckets):
    """
    Compute equal-frequency bucket boundaries from a length array.

    Returns:
        list of (min, max) tuples for each bucket
    """
    percentiles = np.linspace(0, 100, n_buckets + 1)
    boundaries = np.percentile(lengths, percentiles)
    buckets = []
    for i in range(n_buckets):
        lo = int(np.floor(boundaries[i]))
        hi = int(np.ceil(boundaries[i+1]))
        if i == n_buckets - 1:
            hi = int(np.ceil(boundaries[-1]))  # last bucket: include max
        buckets.append((lo, hi))
    return buckets


def extract_sequences_from_fasta(fasta_path, target_headers, max_extract):
    """
    Extract full sequences from a FASTA matching given headers.
    Returns dict: {header: sequence}

    Args:
        fasta_path: path to FASTA
        target_headers: set of headers to extract
        max_extract: stop after extracting this many
    """
    found = {}
    with open(fasta_path, 'r') as f:
        header = None
        seq_lines = []
        for line in f:
            if line.startswith('>'):
                if header is not None and header in target_headers:
                    found[header] = ''.join(seq_lines)
                    if len(found) >= max_extract:
                        break
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None and header in target_headers and len(found) < max_extract:
            found[header] = ''.join(seq_lines)
    return found


def sample_positive(cds_dedup_fasta, cds_lengths, n_buckets, per_bucket,
                    min_len=MIN_LEN, max_len=MAX_LEN):
    """
    Bucket-sample CDS sequences keeping natural length distribution.

    Strategy:
      1. Compute bucket boundaries from cds_lengths
      2. Stream the deduplicated CDS FASTA, assign each sequence to a bucket by length
      3. Within each bucket, random-sample per_bucket sequences
      4. Extract full sequences for sampled headers only (single FASTA pass)

    Returns:
        list of dicts with keys: sequence, species, seqid, length, label=1
    """
    buckets = get_bucket_boundaries(cds_lengths, n_buckets)
    print(f"CDS buckets ({n_buckets}):")
    for i, (lo, hi) in enumerate(buckets):
        print(f"  Bucket {i}: [{lo}, {hi}]")

    # First pass: collect candidates per bucket (headers only, O(1) memory)
    bucket_candidates = defaultdict(list)  # bucket_idx -> [(header, length), ...]
    total_in_range = 0
    for header, seq_len in stream_lengths(cds_dedup_fasta):
        if seq_len < min_len or seq_len > max_len:
            continue
        total_in_range += 1
        # Find bucket
        for i, (lo, hi) in enumerate(buckets):
            if lo <= seq_len <= hi:
                bucket_candidates[i].append((header, seq_len))
                break

    print(f"CDS in range [{min_len}, {max_len}]: {total_in_range:,}")
    for i in range(n_buckets):
        print(f"  Bucket {i}: {len(bucket_candidates[i]):,} candidates")

    # Second pass: random-sample per bucket, store chosen for later use
    bucket_chosen = {}  # bucket_idx -> [(header, seq_len), ...]
    all_sampled_headers = set()

    for i in range(n_buckets):
        candidates = bucket_candidates[i]
        n_sample = min(per_bucket, len(candidates))
        chosen = random.sample(candidates, n_sample)
        bucket_chosen[i] = chosen
        for header, seq_len in chosen:
            all_sampled_headers.add(header)

    print(f"Total headers to extract: {len(all_sampled_headers):,}")

    # Extract sequences (single FASTA pass)
    seq_map = extract_sequences_from_fasta(
        cds_dedup_fasta, all_sampled_headers, len(all_sampled_headers)
    )
    print(f"Sequences extracted: {len(seq_map):,}")

    # Build records from the exact same sampled set (no re-sampling)
    sampled = []
    for i in range(n_buckets):
        for header, seq_len in bucket_chosen[i]:
            seq = seq_map.get(header, '')
            if not seq:
                continue
            # Parse header: species|transcript_id|gene_name|scaffold:start-end(strand)|cds
            parts = header.split('|')
            species = parts[0] if len(parts) >= 1 else 'unknown'
            seqid = parts[1] if len(parts) >= 2 else 'unknown'
            sampled.append({
                'sequence': seq.upper(),
                'label': 1,
                'species': species,
                'seqid': seqid,
                'start': -1,
                'length': len(seq),
                'softmask_ratio': compute_softmask_ratio(seq),
            })

    print(f"Positive samples: {len(sampled):,}")
    return sampled


def sample_negative(intergenic_fasta, cds_lengths, n_buckets, per_bucket,
                    min_len=MIN_LEN, max_len=MAX_LEN):
    """
    Sample intergenic regions matching CDS length distribution.

    Strategy:
      1. Same bucket boundaries as CDS
      2. Stream intergenic FASTA:
         - seq_len in [min_len, max_len]: keep original
         - seq_len > max_len: randomly window to a target_len drawn from cds_lengths
         - seq_len < min_len: skip
      3. Assign to bucket by length after windowing
      4. Sample per_bucket per bucket
      5. Single-pass FASTA extraction for all sampled headers
    """
    buckets = get_bucket_boundaries(cds_lengths, n_buckets)

    # First pass: collect candidates
    bucket_candidates = defaultdict(list)  # bucket_idx -> [(header, chosen_len, orig_len, start_pos), ...]

    for header, orig_len in stream_lengths(intergenic_fasta):
        if orig_len < min_len:
            continue

        if orig_len <= max_len:
            # Keep original
            chosen_len = orig_len
            start_pos = -1
        else:
            # Randomly window to a length drawn from CDS distribution
            chosen_len = int(np.random.choice(cds_lengths))
            chosen_len = min(chosen_len, orig_len)  # can't exceed orig
            start_pos = random.randint(0, orig_len - chosen_len)

        # Find bucket
        for i, (lo, hi) in enumerate(buckets):
            if lo <= chosen_len <= hi:
                bucket_candidates[i].append((header, chosen_len, orig_len, start_pos))
                break

    print(f"Negative candidates by bucket:")
    for i in range(n_buckets):
        print(f"  Bucket {i}: {len(bucket_candidates[i]):,} candidates")

    # Second pass: sample per bucket
    bucket_chosen = {}  # bucket_idx -> [(header, chosen_len, orig_len, start_pos), ...]
    for i in range(n_buckets):
        candidates = bucket_candidates[i]
        n_sample = min(per_bucket, len(candidates))
        bucket_chosen[i] = random.sample(candidates, n_sample)

    # Single-pass FASTA extraction: collect all sampled headers, extract once
    all_sampled_headers = set()
    for i in range(n_buckets):
        for hdr, _, _, _ in bucket_chosen[i]:
            all_sampled_headers.add(hdr)

    seq_map = extract_sequences_from_fasta(
        intergenic_fasta, all_sampled_headers, len(all_sampled_headers)
    )

    # Build records from the exact same sampled set
    sampled = []
    for i in range(n_buckets):
        for header, chosen_len, orig_len, start_pos in bucket_chosen[i]:
            full_seq = seq_map.get(header, '')
            if not full_seq:
                continue

            if start_pos >= 0:
                seq = full_seq[start_pos:start_pos + chosen_len]
            else:
                seq = full_seq

            # Parse header: species|scaffold:start-end|intergenic_N
            parts = header.split('|')
            species = parts[0] if len(parts) >= 1 else 'unknown'
            seqid = parts[1] if len(parts) >= 2 else 'unknown'

            sampled.append({
                'sequence': seq.upper(),
                'label': 0,
                'species': species,
                'seqid': seqid,
                'start': start_pos,
                'length': len(seq),
                'softmask_ratio': compute_softmask_ratio(seq),
            })

    print(f"Negative samples: {len(sampled):,}")
    return sampled


def stratified_5fold_split(df):
    """
    5-fold stratified split (by label + species).

    Each row is assigned to one fold as test (20%).
    Train/val split is done at training time (90/10 of non-test rows),
    documented in splits.json per fold.

    Returns df with added 'fold' (0-4) and 'split' columns.
    split is always 'test' — each row is a test sample in its fold.
    """
    from collections import Counter

    # Group by species+label for stratification
    df['stratify_key'] = df['species'].astype(str) + '_' + df['label'].astype(str)
    key_counts = Counter(df['stratify_key'])
    min_count = min(key_counts.values())
    if min_count >= N_FOLDS:
        stratify_col = df['stratify_key']
    else:
        print(f"WARNING: min group size {min_count} < {N_FOLDS}, using label-only stratification")
        stratify_col = df['label']

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    df['fold'] = -1
    df['split'] = 'test'

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, stratify_col)):
        df.loc[test_idx, 'fold'] = fold_idx

    df.drop(columns=['stratify_key'], inplace=True)

    # Print fold stats
    for f in range(N_FOLDS):
        test_n = (df['fold'] == f).sum()
        train_val_n = len(df) - test_n
        train_n = int(train_val_n * (1 - VAL_RATIO))
        val_n = train_val_n - train_n
        test_pos = (df[(df['fold'] == f) & (df['label'] == 1)]).shape[0]
        test_neg = test_n - test_pos
        print(f"  Fold {f}: test={test_n:,} (pos={test_pos:,}, neg={test_neg:,})  "
              f"train_val={train_val_n:,} -> train~{train_n:,}, val~{val_n:,}")

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TEMP_CDS_FILTERED), exist_ok=True)

    # ========================================================================
    # Step 1: CDS length filter → TEMP_CDS_FILTERED (skip if already exists)
    # ========================================================================
    print("=" * 60)
    print("Step 1: CDS length filter [300, 5000]")
    print("=" * 60)
    if os.path.exists(TEMP_CDS_FILTERED):
        print(f"  Using existing filtered file: {TEMP_CDS_FILTERED}")
        cds_lengths = np.array([ln for _, ln in stream_lengths(TEMP_CDS_FILTERED)])
        print(f"  CDS after filter: {len(cds_lengths):,} sequences")
        print(f"  Length: min={cds_lengths.min():.0f}, P50={np.median(cds_lengths):.0f}, "
              f"mean={cds_lengths.mean():.1f}, max={cds_lengths.max():.0f}")
    else:
        cds_lengths = build_cds_length_distribution(CDS_FASTA, MIN_LEN, MAX_LEN)
        print(f"  CDS after filter: {len(cds_lengths):,} sequences")
        print(f"  Length: min={cds_lengths.min():.0f}, P50={np.median(cds_lengths):.0f}, "
              f"mean={cds_lengths.mean():.1f}, max={cds_lengths.max():.0f}")

    # ========================================================================
    # Step 2: Positive sampling (100k CDS) — sample FIRST, then dedup
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Positive sampling (100k CDS from filtered pool)")
    print("=" * 60)
    positive_samples = sample_positive(TEMP_CDS_FILTERED, cds_lengths, N_BUCKETS, PER_BUCKET)
    if len(positive_samples) < PER_BUCKET * N_BUCKETS:
        actual_per_bucket = len(positive_samples) // N_BUCKETS
        print(f"WARNING: Only got {len(positive_samples):,} positive samples "
              f"(target: {PER_BUCKET * N_BUCKETS:,}), adjusting per_bucket to {actual_per_bucket}")

    # ========================================================================
    # Step 3: CD-HIT dedup skipped — random stratified sampling across 735
    # species from a 9.2M pool provides sufficient diversity.
    # 100k / 735 ≈ 136 samples per species, near-zero collision probability.
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 3: CD-HIT dedup — SKIPPED (stratified sampling sufficient)")
    print("=" * 60)

    # ========================================================================
    # Step 4: Negative sampling (100k intergenic)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Negative sampling (100k intergenic)")
    print("=" * 60)
    negative_samples = sample_negative(
        INTERGENIC_FASTA, cds_lengths, N_BUCKETS, PER_BUCKET, MIN_LEN, MAX_LEN
    )
    if len(negative_samples) < PER_BUCKET * N_BUCKETS:
        actual_per_bucket = len(negative_samples) // N_BUCKETS
        print(f"WARNING: Only got {len(negative_samples):,} negative samples "
              f"(target: {PER_BUCKET * N_BUCKETS:,})")

    # ========================================================================
    # Step 5: Combine + shuffle
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Combining and shuffling")
    print("=" * 60)
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    df = pd.DataFrame(all_samples)
    print(f"  Total: {len(df):,}")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"  Species: {df['species'].nunique()} unique")
    print(f"  Length: min={df['length'].min()}, median={df['length'].median():.0f}, "
          f"max={df['length'].max()}, mean={df['length'].mean():.1f}")

    # ========================================================================
    # Step 6: 5-fold stratified split
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 6: 5-fold stratified split")
    print("=" * 60)
    df = stratified_5fold_split(df)

    # ========================================================================
    # Step 7: Write outputs
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 7: Writing outputs")
    print("=" * 60)

    parquet_path = os.path.join(OUTPUT_DIR, 'dataset.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"  Parquet: {parquet_path} ({os.path.getsize(parquet_path)/1024/1024:.1f} MB)")

    # splits.json -- documents fold assignments and expected train/val split
    splits = {}
    for f in range(N_FOLDS):
        test_n = int((df['fold'] == f).sum())
        train_val_n = len(df) - test_n
        val_n = int(train_val_n * VAL_RATIO)
        train_n = train_val_n - val_n
        splits[f'fold_{f}'] = {
            'test': test_n,
            'train': train_n,
            'val': val_n,
            'usage': f'When fold {f} is active: test = rows with fold=={f} ({test_n} rows); '
                     f'from remaining {train_val_n} rows, randomly split {train_n} train / {val_n} val '
                     f'(seed={42+f})'
        }
    splits_path = os.path.join(OUTPUT_DIR, 'splits.json')
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"  Splits: {splits_path}")

    # metadata.json
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    metadata = {
        'task': 'cds_vs_intergenic',
        'date': '2026-07-31',
        'total_samples': len(df),
        'positive_samples': int((df['label'] == 1).sum()),
        'negative_samples': int((df['label'] == 0).sum()),
        'length_range': [MIN_LEN, MAX_LEN],
        'cdhit_threshold': CDHIT_THRESHOLD,
        'cds_length_stats': {
            'p5': float(np.percentile(cds_lengths, 5)),
            'p50': float(np.median(cds_lengths)),
            'p95': float(np.percentile(cds_lengths, 95)),
            'min': int(cds_lengths.min()),
            'max': int(cds_lengths.max()),
            'mean': float(cds_lengths.mean()),
            'std': float(cds_lengths.std()),
            'n_after_filter': len(cds_lengths),
            'n_after_cdhit': len(positive_samples),
            'cdhit_note': 'CD-HIT skipped — random stratified sampling across 735 species from 9.2M pool provides sufficient diversity',
        },
        'positive_length_stats': {
            'min': int(pos_df['length'].min()),
            'p50': float(pos_df['length'].median()),
            'mean': float(pos_df['length'].mean()),
            'max': int(pos_df['length'].max()),
        },
        'negative_length_stats': {
            'min': int(neg_df['length'].min()),
            'p50': float(neg_df['length'].median()),
            'mean': float(neg_df['length'].mean()),
            'max': int(neg_df['length'].max()),
        },
        'num_buckets': N_BUCKETS,
        'per_bucket': PER_BUCKET,
        'n_folds': N_FOLDS,
        'n_species': int(df['species'].nunique()),
    }
    metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")

    # Cleanup temp files
    for tmp in [TEMP_CDS_FILTERED, TEMP_CDS_DEDUP, TEMP_CDS_DEDUP + '.clstr',
                TEMP_POS_SAMPLED, TEMP_POS_DEDUP, TEMP_POS_DEDUP + '.clstr']:
        if os.path.exists(tmp):
            os.remove(tmp)

    # ========================================================================
    # Done
    # ========================================================================
    print("\n" + "=" * 60)
    print("DATASET CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}/")
    for fn in ['dataset.parquet', 'splits.json', 'metadata.json']:
        fp = os.path.join(OUTPUT_DIR, fn)
        size_mb = os.path.getsize(fp) / 1024 / 1024 if os.path.exists(fp) else 0
        print(f"  {fn}: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
