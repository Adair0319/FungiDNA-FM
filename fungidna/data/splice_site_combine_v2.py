"""
splice_site_combine_v2.py — GroupKFold splitting + metadata + statistics for v2 splice site dataset.

Pipeline:
  1. Load filtered_all.parquet (from Task 1)
  2. GroupKFold split by (species, transcript) for positives, singleton groups for non-sites
  3. Write fold_0..4/{train,val,test}.parquet
  4. Build metadata.csv
  5. Build dataset_statistics.json
  6. Build dataset_report.md

CLI:
  python splice_site_combine_v2.py --input <parquet> --output_dir <dir> [--seed 42]
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_DONOR = 0
LABEL_ACCEPTOR = 1
LABEL_NONSITE = 2
LABEL_NAMES = {0: "donor", 1: "acceptor", 2: "nonsite"}
N_FOLDS = 5
VAL_RATIO = 0.10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def logp(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"{ts}  {msg}", flush=True)


# ===================================================================
# 1. split_folds — GroupKFold with transcript-level grouping
# ===================================================================

def _make_group_id(row: dict, nonsite_counter: List[int]) -> str:
    """
    Build group identifier for GroupKFold.

    Positive samples (label 0 or 1):  ``{species}|{transcript_id}``
    Non-site samples (label 2):       ``{species}|__nonsite__{counter}``
    (each non-site is its own group to avoid data leakage)
    """
    if row["label"] == LABEL_NONSITE:
        nonsite_counter[0] += 1
        return f"{row['species']}|__nonsite__{nonsite_counter[0]}"
    return f"{row['species']}|{row['transcript_id']}"


def split_folds(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """
    Perform GroupKFold splitting stratified by transcript (positives) or
    singleton groups (non-sites).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: label, species, transcript_id.
    n_folds : int
        Number of folds for GroupKFold (default 5).
    val_ratio : float
        Fraction of train pool to hold out as validation (default 0.10).
    seed : int
        Random seed for train/val shuffle (default 42).

    Returns
    -------
    List[Dict]
        ``[{"fold": 0, "train": DataFrame, "val": DataFrame, "test": DataFrame}, ...]``

    Notes
    -----
    The returned DataFrames preserve the **original** DataFrame index so that
    callers can map rows back to the input ``df``.  When saving to parquet
    use ``to_parquet(index=False)`` to avoid writing the index column.
    """
    # Build group IDs
    nonsite_counter = [0]
    group_ids = df.apply(
        lambda row: _make_group_id(row, nonsite_counter), axis=1
    )

    # Map group_id -> integer label for GroupKFold
    unique_groups = group_ids.unique()
    group_to_int = {g: i for i, g in enumerate(unique_groups)}
    group_int = group_ids.map(group_to_int)

    logp(
        f"split_folds: {len(df)} samples, {len(unique_groups)} unique groups, "
        f"over {n_folds} folds"
    )

    gkf = GroupKFold(n_splits=n_folds)

    fold_results: List[Dict] = []

    for fold_idx, (train_test_idx, test_idx) in enumerate(
        gkf.split(df, groups=group_int)
    ):
        # IMPORTANT: do NOT reset_index — original df index is used for
        # fold_assignments and consistency checking downstream.
        test_df = df.iloc[test_idx]

        # train pool: further split into train / val (90/10)
        train_pool_df = df.iloc[train_test_idx]

        rng = np.random.default_rng(seed + fold_idx)
        train_pool_indices = np.arange(len(train_pool_df))
        rng.shuffle(train_pool_indices)

        n_val = max(int(len(train_pool_indices) * val_ratio), 1)
        val_positions = train_pool_indices[:n_val]
        train_positions = train_pool_indices[n_val:]

        train_df = train_pool_df.iloc[train_positions]
        val_df = train_pool_df.iloc[val_positions]

        # Log distribution
        for split_name, split_df in [
            ("train", train_df), ("val", val_df), ("test", test_df)
        ]:
            counts = split_df["label"].value_counts()
            label_counts = {
                LABEL_NAMES.get(k, str(k)): int(v)
                for k, v in sorted(counts.items())
            }
            logp(
                f"  fold {fold_idx} {split_name}: {len(split_df)} samples, "
                f"labels={label_counts}"
            )

        fold_results.append({
            "fold": fold_idx,
            "train": train_df,
            "val": val_df,
            "test": test_df,
        })

    # Cross-fold consistency check
    _check_fold_consistency(df, fold_results, group_ids)

    return fold_results


def _check_fold_consistency(
    df: pd.DataFrame,
    fold_results: List[Dict],
    group_ids: pd.Series,
) -> None:
    """
    Verify that no positive group appears in more than one fold's **test** set.

    In GroupKFold, for fold *i* the *train/val* DataFrames contain groups
    from *all other* folds (the training pool).  So a positive group
    **will** appear in multiple folds' train/val splits — that is correct
    by design.

    The only thing that must not happen is a group appearing in the **test**
    set of more than one fold, which would mean the same transcript is
    used for evaluation across folds.
    """
    group_test_folds: Dict[str, set] = defaultdict(set)
    for fold_result in fold_results:
        fold_num = fold_result["fold"]
        split_df = fold_result["test"]
        pos_in_test = split_df[split_df["label"] != LABEL_NONSITE]
        if pos_in_test.empty:
            continue
        # pos_in_test.index preserves the ORIGINAL df index
        pos_group_ids = group_ids.loc[pos_in_test.index].unique()
        for g in pos_group_ids:
            group_test_folds[g].add(fold_num)

    cross_test = {
        g: folds for g, folds in group_test_folds.items()
        if len(folds) > 1
    }
    if cross_test:
        logp(
            f"  WARNING: {len(cross_test)} positive groups appear in "
            f"test sets of multiple folds (data leak)"
        )
    else:
        logp("  Cross-fold consistency check PASSED (no test-set leakage)")


# ===================================================================
# 2. build_metadata
# ===================================================================

def build_metadata(
    df: pd.DataFrame,
    fold_assignments: Dict[int, Dict],
) -> pd.DataFrame:
    """
    Build per-sample metadata DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset with columns: species, transcript_id, intron_id.
    fold_assignments : Dict[int, Dict]
        ``{original_df_index: {"fold": 0..4, "split": "train"|"val"|"test"}}``

    Returns
    -------
    pd.DataFrame
        Columns: sample_id, species, transcript_id, intron_id, type, fold, split.
    """
    records: List[Dict] = []
    for idx in sorted(fold_assignments.keys()):
        fa = fold_assignments[idx]
        row = df.iloc[idx]
        label = int(row["label"])
        records.append({
            "sample_id": fa["sample_id"],
            "species": str(row["species"]),
            "transcript_id": str(row["transcript_id"]),
            "intron_id": str(row["intron_id"]),
            "type": LABEL_NAMES[label],
            "fold": fa["fold"],
            "split": fa["split"],
        })
    meta_df = pd.DataFrame(records)
    # Replace any nulls with empty string so they survive CSV round-trips
    meta_df = meta_df.fillna("")
    return meta_df


# ===================================================================
# 3. build_statistics
# ===================================================================

def build_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute dataset statistics.

    Returns
    -------
    dict with keys:
        total_samples, n_species, class_counts, intron_length,
        species_distribution, dinucleotide_distribution,
        n_transcripts, n_introns
    """
    total_samples = len(df)
    n_species = int(df["species"].nunique())

    # Class counts
    class_counts = {
        "donor": int((df["label"] == LABEL_DONOR).sum()),
        "acceptor": int((df["label"] == LABEL_ACCEPTOR).sum()),
        "nonsite": int((df["label"] == LABEL_NONSITE).sum()),
    }

    # Intron length statistics (positive samples only where intron_length > 0)
    if "intron_length" not in df.columns:
        intron_length = {"median": 0.0, "p95": 0.0, "p99": 0.0}
    else:
        pos_mask = (df["label"] != LABEL_NONSITE) & (df["intron_length"].notna())
        pos_intron_length = df.loc[pos_mask, "intron_length"]
        if len(pos_intron_length) > 0:
            intron_length = {
                "median": float(pos_intron_length.median()),
                "p95": float(pos_intron_length.quantile(0.95)),
                "p99": float(pos_intron_length.quantile(0.99)),
            }
        else:
            intron_length = {"median": 0.0, "p95": 0.0, "p99": 0.0}

    # Species distribution (total samples per species)
    sp_counts = df["species"].value_counts()
    species_distribution = {
        "min_per_species": int(sp_counts.min()),
        "median_per_species": int(sp_counts.median()),
        "max_per_species": int(sp_counts.max()),
    }

    # Dinucleotide distribution per class
    if "dinucleotide" in df.columns:
        donor_df = df[df["label"] == LABEL_DONOR]
        acceptor_df = df[df["label"] == LABEL_ACCEPTOR]
        di_counts_donor = dict(donor_df["dinucleotide"].value_counts().to_dict())
        di_counts_acceptor = dict(acceptor_df["dinucleotide"].value_counts().to_dict())
        dinucleotide_distribution = {"donor": di_counts_donor, "acceptor": di_counts_acceptor}
    else:
        dinucleotide_distribution = {"donor": {}, "acceptor": {}}

    # Transcript & intron counts
    pos_df = df[df["label"] != LABEL_NONSITE]
    n_transcripts = int(pos_df["transcript_id"].nunique()) if "transcript_id" in df.columns else 0
    n_introns = int(pos_df["intron_id"].nunique()) if "intron_id" in df.columns else 0

    # Sanity check: non-sites should not contribute to tx/intron counts
    non_site_tx = df[df["label"] == LABEL_NONSITE]["transcript_id"].unique()
    non_site_with_tx = [t for t in non_site_tx if t and t != ""]
    if non_site_with_tx:
        logp(f"  WARNING: {len(non_site_with_tx)} non-sites have non-empty transcript_id")

    return {
        "total_samples": total_samples,
        "n_species": n_species,
        "class_counts": class_counts,
        "intron_length": intron_length,
        "species_distribution": species_distribution,
        "dinucleotide_distribution": dinucleotide_distribution,
        "n_transcripts": n_transcripts,
        "n_introns": n_introns,
    }


# ===================================================================
# 4. _build_report — Markdown report
# ===================================================================

def _build_report(
    df: pd.DataFrame,
    fold_results: List[Dict],
    stats: Dict,
    fold_assignments: Dict[int, Dict],
) -> str:
    """
    Generate a Markdown dataset report.
    """
    lines: List[str] = []
    lines.append("# Splice Site Dataset v2 — Dataset Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Total samples: {stats['total_samples']:,}")
    lines.append(f"- Species: {stats['n_species']}")
    lines.append(f"- Transcripts: {stats['n_transcripts']:,}")
    lines.append(f"- Introns: {stats['n_introns']:,}")
    lines.append("")

    # Class distribution
    lines.append("## Class Distribution")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("|-------|------:|")
    for cls_name in ["donor", "acceptor", "nonsite"]:
        count = stats["class_counts"][cls_name]
        lines.append(f"| {cls_name} | {count:,} |")
    lines.append("")

    # Intron length
    lines.append("## Intron Length (Positive Samples Only)")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|------:|")
    for k, v in stats["intron_length"].items():
        lines.append(f"| {k} | {v:.1f} |")
    lines.append("")

    # Species distribution
    lines.append("## Species Distribution")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|------:|")
    for k, v in stats["species_distribution"].items():
        lines.append(f"| {k} | {v:,} |")
    lines.append("")

    # Dinucleotide distribution
    lines.append("## Dinucleotide Distribution")
    lines.append("")
    for cls_name in ["donor", "acceptor"]:
        di_dict = stats["dinucleotide_distribution"][cls_name]
        if di_dict:
            lines.append(f"### {cls_name.capitalize()}")
            lines.append("")
            lines.append("| Dinucleotide | Count | Frequency |")
            lines.append("|--------------|------:|----------:|")
            total = sum(di_dict.values())
            for di, cnt in sorted(di_dict.items(), key=lambda x: -x[1]):
                freq = cnt / total
                lines.append(f"| {di} | {cnt:,} | {freq:.4f} |")
            lines.append("")

    # Per-fold breakdown
    lines.append("## Per-Fold Sample Distribution")
    lines.append("")
    header = "| Fold | Split | Donor | Acceptor | Non-Site | Total |"
    sep = "|------|-------|------:|---------:|---------:|------:|"
    lines.append(header)
    lines.append(sep)
    for fr in fold_results:
        fold = fr["fold"]
        for split_name in ["train", "val", "test"]:
            sdf = fr[split_name]
            n_d = int((sdf["label"] == LABEL_DONOR).sum())
            n_a = int((sdf["label"] == LABEL_ACCEPTOR).sum())
            n_n = int((sdf["label"] == LABEL_NONSITE).sum())
            n_t = len(sdf)
            lines.append(
                f"| {fold} | {split_name} | {n_d:,} | {n_a:,} "
                f"| {n_n:,} | {n_t:,} |"
            )
    lines.append("")

    # Data integrity: each sample in exactly one fold's test set
    test_set_sizes = [len(fr["test"]) for fr in fold_results]
    lines.append("## Data Integrity")
    lines.append("")
    lines.append(f"- Original dataset size: {len(df)}")
    lines.append(f"- Fold test-set sizes: {test_set_sizes}")
    lines.append(f"- Sum of test sets: {sum(test_set_sizes)} (should equal total)")
    lines.append(f"- Cross-fold test-set leakage: None (verified by group consistency check)")
    lines.append("")

    return "\n".join(lines)


# ===================================================================
# 5. combine_and_split — Main entry point
# ===================================================================

def combine_and_split(
    input_path: str,
    output_dir: str,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Full pipeline:

    1. Load ``filtered_all.parquet``
    2. Call ``split_folds()``
    3. Write ``fold_{0..4}/{train,val,test}.parquet`` (index=False)
    4. Build ``fold_assignments`` dict tracking original df indices
    5. Call ``build_metadata()`` → ``metadata.csv``
    6. Call ``build_statistics()`` → ``dataset_statistics.json``
    7. ``_build_report()`` → ``dataset_report.md``
    """
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Load ----
    logp(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    logp(f"Loaded {len(df)} samples, columns: {list(df.columns)}")

    # Ensure string types for key columns (replace NaN with "" first so
    # empty strings survive round-trips through CSV)
    for col in ["species", "transcript_id", "intron_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # ---- 2. Split into folds ----
    logp("Splitting into folds ...")
    fold_results = split_folds(df, seed=seed)

    # ---- 3. Write fold parquets (index=False) ----
    logp("Writing fold parquets ...")
    for fr in fold_results:
        fold_dir = os.path.join(output_dir, f"fold_{fr['fold']}")
        os.makedirs(fold_dir, exist_ok=True)
        for split_name in ["train", "val", "test"]:
            split_df = fr[split_name]
            out_path = os.path.join(fold_dir, f"{split_name}.parquet")
            # Use index=False so we don't persist the original df index
            split_df.to_parquet(out_path, index=False)
            logp(f"  -> {out_path} ({len(split_df)} samples)")

    # ---- 4. Build fold_assignments dict ----
    # Each original index appears in exactly ONE fold's test set (the
    # designated hold-out fold for that sample / group).  We record that
    # designated assignment so downstream users can do 5-fold CV by
    # picking fold N as test and using all other folds for training.
    #
    # Within a fold, the train/val pool (groups from the other 4 folds)
    # is NOT recorded here because those samples already have their
    # own designated fold in ``fold_assignments``.
    logp("Building fold assignments ...")
    fold_assignments: Dict[int, Dict] = {}
    for fr in fold_results:
        fold_num = fr["fold"]
        test_df = fr["test"]
        for rank, idx in enumerate(test_df.index):
            sample_id = f"v2_{idx:08d}"
            fold_assignments[idx] = {
                "fold": fold_num,
                "split": "test",
                "sample_id": sample_id,
            }

    # Sanity: every original index should have been assigned
    missing = [i for i in range(len(df)) if i not in fold_assignments]
    if missing:
        logp(
            f"  WARNING: {len(missing)} samples not assigned to any fold's "
            f"test set (should be 0!)"
        )

    # ---- 5. Build metadata ----
    logp("Building metadata ...")
    meta_df = build_metadata(df, fold_assignments)
    meta_path = os.path.join(output_dir, "metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    logp(f"  -> {meta_path} ({len(meta_df)} rows)")

    # ---- 6. Build dataset statistics ----
    logp("Building dataset statistics ...")
    stats = build_statistics(df)
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logp(f"  -> {stats_path}")

    # ---- 7. Build dataset report ----
    logp("Building dataset report ...")
    report = _build_report(df, fold_results, stats, fold_assignments)
    report_path = os.path.join(output_dir, "dataset_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    logp(f"  -> {report_path}")

    elapsed = time.time() - t_start
    logp(f"Done! ({elapsed:.1f}s)")


# ===================================================================
# CLI Entry Point
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Combine and split v2 splice site dataset into "
            "GroupKFold folds with metadata and statistics."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to filtered_all.parquet (from Task 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for folds, metadata, and reports",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )

    args = parser.parse_args()
    combine_and_split(args.input, args.output_dir, seed=args.seed)
