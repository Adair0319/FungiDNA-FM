"""
Splice Site Prediction Dataset Builder v2.

v2 improvements over v1:
  - Perfect-match only transcript QC (no stop-codon-drop or N-wildcard exemptions)
  - Additional metadata per splice site (intron_id, intron_length, exon_count)
  - Dinucleotide filtering for donor/acceptor
  - N-content filtering (reject windows with ambiguous bases)
  - Cross-label conflict resolution and within-class deduplication
  - Species-level cap and global stratified downsampling to ~1:1:1

Pipeline:
  1. Parse GFF -> exon groups
  2. Load assembly + transcript FASTA
  3. validate_transcript_strict() -> Perfect Match QC
  4. derive_splice_sites_v2() -> intron length & exon count filtering
  5. extract_window() -> 401bp windows
  6. Dinucleotide & N filters
  7. Conflict resolution & dedup
  8. Non-site sampling
  9. Species cap & global stratified downsampling
  10. Output filtered_all.parquet + filter_stats.json

Reference: xxxxxx.docx v2.0
"""

import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from splice_site_dataset import (
    detect_gff_format,
    parse_gff3_exons,
    parse_gff_exons,
    load_assembly_index,
    extract_window,
    build_transcript_index,
    sample_non_sites,
    reverse_complement,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = "/home/lty/yy_projects/fungi_project/fungi_dna_model/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "splice_site_build_v2.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def logp(msg: str) -> None:
    """Print progress with immediate flush."""
    ts = time.strftime("%H:%M:%S")
    print(f"{ts}  {msg}", flush=True)
    logger.info(msg)


# ---------------------------------------------------------------------------
# Constants (use verbatim from spec)
# ---------------------------------------------------------------------------

WINDOW_HALF = 200
LABEL_DONOR = 0
LABEL_ACCEPTOR = 1
LABEL_NONSITE = 2
ALLOWED_DONOR_DI = {"GT", "GG", "TG", "AG", "CG"}
ALLOWED_ACC_DI = {"AG", "GG", "GA", "GC", "GT"}
INTRON_LEN_MIN = 50
INTRON_LEN_MAX = 500
MIN_EXONS = 3
TARGET_TOTAL = 200_000
SPECIES_CAP_FACTOR = 3
RANDOM_SEED = 42


# ===================================================================
# Strict Transcript QC -- Perfect Match only
# ===================================================================

def validate_transcript_strict(
    tx_exons: Dict[str, List[dict]],
    asm_seqs: Dict[str, str],
    tx_index: Dict[str, str],
) -> Dict[str, bool]:
    """
    Validate each transcript by exact match of spliced assembly vs gold FASTA.

    Unlike v1 ``validate_transcript``, this function does NOT allow:
      - Stop-codon-drop (spliced 3 bp longer than gold)
      - N-wildcard equivalence

    Returns ``{transcript_id: is_perfect_match}``.
    """
    result: Dict[str, bool] = {}
    for t_id, exons in tx_exons.items():
        if t_id not in tx_index:
            result[t_id] = False
            continue

        gold = tx_index[t_id]

        # sort + dedup exons
        exons_sorted = sorted(exons, key=lambda e: (e["seqid"], e["start"]))
        strand = exons_sorted[0]["strand"]
        seen: Set[Tuple[str, int, int]] = set()
        exons_dedup: List[dict] = []
        for e in exons_sorted:
            k = (e["seqid"], e["start"], e["end"])
            if k not in seen:
                seen.add(k)
                exons_dedup.append(e)

        # splice assembly exons
        parts: List[str] = []
        for e in exons_dedup:
            if e["seqid"] not in asm_seqs:
                continue
            s = asm_seqs[e["seqid"]]
            s0, e0 = e["start"] - 1, e["end"]
            if s0 < 0 or e0 > len(s):
                continue
            parts.append(s[s0:e0])

        if not parts:
            result[t_id] = False
            continue

        spliced = "".join(parts).upper()
        if strand == "-":
            spliced = reverse_complement(spliced)

        result[t_id] = (spliced == gold)

    return result


# ===================================================================
# Splice Site Derivation v2 -- with intron_id / intron_length / exon_count
# ===================================================================

def derive_splice_sites_v2(
    tx_exons: Dict[str, List[dict]],
    valid_tx: Dict[str, bool],
    min_exons: int = MIN_EXONS,
    intron_len_min: int = INTRON_LEN_MIN,
    intron_len_max: int = INTRON_LEN_MAX,
) -> List[dict]:
    """
    Derive donor/acceptor positions with inline filters.

    Only processes transcripts where ``valid_tx[t_id]`` is ``True``.
    Each returned site dict includes ``intron_id``, ``intron_length``,
    and ``exon_count``.

    Filters applied:
      - Transcript must have at least ``min_exons`` exons.
      - Each intron gap must be in ``[intron_len_min, intron_len_max]``.
    """
    sites: List[dict] = []
    for t_id, exons in tx_exons.items():
        if not valid_tx.get(t_id, False):
            continue
        if len(exons) < min_exons:
            continue

        exons_sorted = sorted(exons, key=lambda e: (e["seqid"], e["start"]))
        # dedup overlapping exon entries
        seen: Set[Tuple[str, int, int]] = set()
        exons_dedup: List[dict] = []
        for e in exons_sorted:
            key = (e["seqid"], e["start"], e["end"])
            if key not in seen:
                seen.add(key)
                exons_dedup.append(e)

        if len(exons_dedup) < min_exons:
            continue

        strand = exons_dedup[0]["strand"]
        for i in range(len(exons_dedup) - 1):
            a, b = exons_dedup[i], exons_dedup[i + 1]

            # skip if exons are on different seqids or adjacent/overlapping
            if a["seqid"] != b["seqid"]:
                continue
            if b["start"] - a["end"] <= 1:
                continue

            intron_len = b["start"] - a["end"] - 1
            if intron_len < intron_len_min or intron_len > intron_len_max:
                continue

            intron_start = a["end"] + 1
            intron_end = b["start"] - 1
            intron_id = f"{t_id}_intron_{i}"

            if strand == "+":
                donor, acceptor = intron_start, intron_end
            else:
                donor, acceptor = intron_end, intron_start

            sites.append({
                "seqid": a["seqid"],
                "center_pos": donor,
                "label": LABEL_DONOR,
                "strand": strand,
                "transcript_id": t_id,
                "intron_id": intron_id,
                "intron_length": intron_len,
                "exon_count": len(exons_dedup),
            })
            sites.append({
                "seqid": a["seqid"],
                "center_pos": acceptor,
                "label": LABEL_ACCEPTOR,
                "strand": strand,
                "transcript_id": t_id,
                "intron_id": intron_id,
                "intron_length": intron_len,
                "exon_count": len(exons_dedup),
            })

    return sites


# ===================================================================
# Per-Species Pipeline (filters 1-6)
# ===================================================================

def process_species_v2(
    sp: str,
    gff_path: str,
    asm_path: str,
    tx_path: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full v2 pipeline for a single species.

    Steps
    -----
    1. Parse GFF -> exon groups
    2. Load assembly + transcript FASTA
    3. ``validate_transcript_strict()`` -> Perfect Match QC
    4. ``derive_splice_sites_v2()`` -> intron 50-500bp + >=3 exons
    5. ``extract_window()`` -> 401 bp windows
    6. Filter (a): dinucleotide check
    7. Filter (b): reject windows containing 'N'
    8. Build DataFrame
    9. Filter (c): cross-label conflict discard + intra-class dedup
    10. ``sample_non_sites()`` (target = n_pos // 2), N-reject + dedup

    Returns
    -------
    (DataFrame, stats_dict)
        DataFrame columns: ``[sequence, label, species, seqid, transcript_id,
        intron_id, center_pos, dinucleotide, intron_length, exon_count]``
    """
    # ---- 1. Parse GFF -> exon groups ----
    fmt = detect_gff_format(gff_path)
    tx_exons = (
        parse_gff3_exons(gff_path)
        if fmt == "gff3"
        else parse_gff_exons(gff_path)
    )

    # ---- 2. Load assembly + transcript FASTA ----
    asm_seqs = load_assembly_index(asm_path)
    tx_index = build_transcript_index(tx_path)

    # ---- 3. Perfect Match QC ----
    valid_tx = validate_transcript_strict(tx_exons, asm_seqs, tx_index)

    # ---- 4. Derive splice sites (intron 50-500bp + >=3 exons) ----
    sites = derive_splice_sites_v2(
        tx_exons,
        valid_tx,
        min_exons=MIN_EXONS,
        intron_len_min=INTRON_LEN_MIN,
        intron_len_max=INTRON_LEN_MAX,
    )

    raw_n_donor = sum(1 for s in sites if s["label"] == LABEL_DONOR)
    raw_n_acceptor = sum(1 for s in sites if s["label"] == LABEL_ACCEPTOR)

    # ---- 5. Extract 401 bp windows ----
    extracted: List[dict] = []
    for site in sites:
        res = extract_window(
            asm_seqs, site["seqid"], site["center_pos"],
            site["label"], site["strand"],
        )
        if res is None:
            continue
        seq, di = res
        extracted.append({
            "seqid": site["seqid"],
            "center_pos": site["center_pos"],
            "label": site["label"],
            "strand": site["strand"],
            "transcript_id": site["transcript_id"],
            "sequence": seq,
            "dinucleotide": di,
            "intron_id": site["intron_id"],
            "intron_length": site["intron_length"],
            "exon_count": site["exon_count"],
        })

    after_window = len(extracted)

    # ---- 6. Filter (a): dinucleotide check ----
    di_pass: List[dict] = []
    for s in extracted:
        if s["label"] == LABEL_DONOR and s["dinucleotide"] not in ALLOWED_DONOR_DI:
            continue
        if s["label"] == LABEL_ACCEPTOR and s["dinucleotide"] not in ALLOWED_ACC_DI:
            continue
        di_pass.append(s)

    after_di = len(di_pass)

    # ---- 7. Filter (b): reject windows with 'N' ----
    n_pass = [s for s in di_pass if "N" not in s["sequence"]]
    after_n = len(n_pass)

    n_valid_tx = sum(1 for v in valid_tx.values() if v)

    # early exit if nothing survived
    if not n_pass:
        return pd.DataFrame(), {
            "raw_donor": raw_n_donor,
            "raw_acceptor": raw_n_acceptor,
            "after_window": after_window,
            "after_di": after_di,
            "after_n": after_n,
            "donor": 0,
            "acceptor": 0,
            "non_sites": 0,
            "tx_total": len(tx_exons),
            "tx_valid": n_valid_tx,
            "format": fmt,
        }

    # ---- 8. Build DataFrame ----
    df = pd.DataFrame(n_pass)
    df["species"] = sp

    # ---- 9. Filter (c): cross-label conflict + intra-class dedup ----
    # Cross-label: same (seqid, center_pos) with >1 label -> discard all
    label_counts = df.groupby(["seqid", "center_pos"])["label"].nunique()
    conflict_keys = label_counts[label_counts > 1].index
    if len(conflict_keys) > 0:
        idx = df.set_index(["seqid", "center_pos"]).index
        df = df[~idx.isin(conflict_keys)].copy()

    # Intra-class dedup: keep unique (seqid, center_pos, label)
    df = df.drop_duplicates(subset=["seqid", "center_pos", "label"])

    after_dedup_pos = len(df)

    # ---- 10. Non-site sampling ----
    pos_positions: Set[Tuple[str, int]] = set(
        zip(df["seqid"], df["center_pos"])
    )
    n_pos = len(df)
    n_neg_target = max(n_pos // 2, 1)

    neg_positions = sample_non_sites(
        asm_seqs, pos_positions, n_neg_target,
        seed=seed,
    )

    neg_records: List[dict] = []
    for ns in neg_positions:
        res = extract_window(
            asm_seqs, ns["seqid"], ns["center_pos"], LABEL_NONSITE, ".",
        )
        if res is None:
            continue
        seq, di = res
        if "N" in seq:
            continue
        neg_records.append({
            "sequence": seq,
            "label": LABEL_NONSITE,
            "species": sp,
            "seqid": ns["seqid"],
            "transcript_id": None,
            "center_pos": ns["center_pos"],
            "dinucleotide": di,
            "intron_id": None,
            "intron_length": None,
            "exon_count": None,
        })

    if neg_records:
        neg_df = pd.DataFrame(neg_records)
        neg_df = neg_df.drop_duplicates(subset=["seqid", "center_pos", "label"])
        df = pd.concat([df, neg_df], ignore_index=True)

    # Final counts
    n_final_donor = int((df["label"] == LABEL_DONOR).sum())
    n_final_acceptor = int((df["label"] == LABEL_ACCEPTOR).sum())
    n_final_nonsite = int((df["label"] == LABEL_NONSITE).sum())

    stats = {
        "raw_donor": raw_n_donor,
        "raw_acceptor": raw_n_acceptor,
        "after_window": after_window,
        "after_di": after_di,
        "after_n": after_n,
        "after_dedup_pos": after_dedup_pos,
        "donor": n_final_donor,
        "acceptor": n_final_acceptor,
        "non_sites": n_final_nonsite,
        "tx_total": len(tx_exons),
        "tx_valid": n_valid_tx,
        "format": fmt,
    }
    return df, stats


# ===================================================================
# Main Pipeline
# ===================================================================

def build_dataset_v2(
    assembly_dir: str,
    gff_dir: str,
    transcript_dir: str,
    output_dir: str,
    seed: int = 42,
) -> None:
    """
    Build the v2 splice site dataset end-to-end.

    Phases
    ------
    1. Discover species (intersection of assembly / GFF / transcript dirs)
    2. Per-species processing -> ``_temp_v2/{species}.parquet`` (resume-safe)
    3. Species-level positive cap (large species downsampled)
    4. Global stratified downsampling to ~1:1:1

    Outputs
    -------
    - ``output_dir/filtered_all.parquet``
    - ``output_dir/filter_stats.json``
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_temp_v2")
    os.makedirs(temp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Discover species
    # ------------------------------------------------------------------
    logp("Phase 1: Discovering species...")

    asm_files = {
        f.replace(".fasta", ""): os.path.join(assembly_dir, f)
        for f in os.listdir(assembly_dir)
        if f.endswith(".fasta")
    }
    gff_files: Dict[str, str] = {}
    for f in os.listdir(gff_dir):
        if f.endswith(".gff3"):
            gff_files[f.replace(".gff3", "")] = os.path.join(gff_dir, f)
        elif f.endswith(".gff"):
            gff_files[f.replace(".gff", "")] = os.path.join(gff_dir, f)
    tx_files = {
        f.replace(".fasta", ""): os.path.join(transcript_dir, f)
        for f in os.listdir(transcript_dir)
        if f.endswith(".fasta")
    }

    species_list = sorted(set(asm_files) & set(gff_files) & set(tx_files))
    logp(f"Found {len(species_list)} species with all three files")

    if not species_list:
        logp("No species found -- check input directories.")
        return

    # ------------------------------------------------------------------
    # Phase 2: Per-species -> temp parquet (with resume support)
    # ------------------------------------------------------------------
    logp("Phase 2: Processing species -> temp files...")

    per_species_stats: Dict[str, Dict] = {}
    t0 = time.time()
    local_pos = 0
    local_neg = 0

    for i, sp in enumerate(species_list):
        temp_path = os.path.join(temp_dir, f"{sp}.parquet")

        # Resume: skip if temp file already exists
        if os.path.exists(temp_path):
            per_species_stats[sp] = {"resumed": True}
            if (i + 1) % 100 == 0 or i < 3:
                logp(f"  {i + 1}/{len(species_list)}: {sp} (resumed)")
            continue

        if (i + 1) % 100 == 0 or i < 3:
            elapsed = time.time() - t0
            logp(
                f"  {i + 1}/{len(species_list)}: {sp} "
                f"({elapsed:.0f}s, {local_pos} pos, {local_neg} neg)"
            )

        try:
            df, stats = process_species_v2(
                sp, gff_files[sp], asm_files[sp], tx_files[sp], seed=seed + i,
            )

            if not df.empty:
                df.to_parquet(temp_path, index=False)
                local_pos += stats["donor"] + stats["acceptor"]
                local_neg += stats["non_sites"]
            else:
                # Write empty placeholder to mark as processed
                pd.DataFrame().to_parquet(temp_path, index=False)

            per_species_stats[sp] = stats

        except Exception as e:
            logp(f"  ERROR {sp}: {e}")
            import traceback
            traceback.print_exc()
            per_species_stats[sp] = {"error": str(e)}

    elapsed = time.time() - t0
    n_new = sum(
        1 for v in per_species_stats.values()
        if not v.get("resumed") and not v.get("error")
    )
    n_resumed = sum(1 for v in per_species_stats.values() if v.get("resumed"))
    logp(
        f"Phase 2 done ({elapsed:.0f}s): {local_pos} pos + {local_neg} neg "
        f"from {n_new} new species, {n_resumed} resumed"
    )

    # ------------------------------------------------------------------
    # Phase 3: Species-level capping
    # ------------------------------------------------------------------
    logp("Phase 3: Species-level capping...")

    cap = max(
        int(SPECIES_CAP_FACTOR * TARGET_TOTAL / max(len(species_list), 1)),
        10,
    )
    small_threshold = cap * 0.1
    logp(f"  cap={cap}, small_species_threshold={small_threshold:.1f}")

    capped_dfs: List[pd.DataFrame] = []
    total_raw_positive = 0
    total_before_nonsite = 0
    total_with_nonsite = 0

    for sp in species_list:
        temp_path = os.path.join(temp_dir, f"{sp}.parquet")
        if not os.path.exists(temp_path):
            continue

        df = pd.read_parquet(temp_path)
        if df.empty:
            continue

        n_donor = int((df["label"] == LABEL_DONOR).sum())
        n_acceptor = int((df["label"] == LABEL_ACCEPTOR).sum())
        n_nonsite = int((df["label"] == LABEL_NONSITE).sum())
        n_pos = n_donor + n_acceptor

        # Accumulate pre-cap stats
        sp_stats = per_species_stats.get(sp, {})
        if "raw_donor" in sp_stats:
            total_raw_positive += sp_stats["raw_donor"] + sp_stats["raw_acceptor"]
        else:
            total_raw_positive += n_pos  # approximate for resumed species
        total_before_nonsite += n_pos
        total_with_nonsite += n_pos + n_nonsite

        # Small species: keep everything
        if n_pos <= small_threshold:
            capped_dfs.append(df)
            continue

        # Large species: cap positives, match non-sites
        donor_part = df[df["label"] == LABEL_DONOR]
        acceptor_part = df[df["label"] == LABEL_ACCEPTOR]
        non_site_part = df[df["label"] == LABEL_NONSITE]

        if n_pos > cap:
            target_donor = max(round(cap * n_donor / n_pos), 0)
            target_acceptor = cap - target_donor
            if len(donor_part) > target_donor > 0:
                donor_part = donor_part.sample(n=target_donor, random_state=seed)
            if len(acceptor_part) > target_acceptor > 0:
                acceptor_part = acceptor_part.sample(n=target_acceptor, random_state=seed)

        n_pos_capped = len(donor_part) + len(acceptor_part)
        target_neg = max(n_pos_capped // 2, 1)
        if len(non_site_part) > target_neg:
            non_site_part = non_site_part.sample(n=target_neg, random_state=seed)

        cap_df = pd.concat(
            [donor_part, acceptor_part, non_site_part], ignore_index=True,
        )
        capped_dfs.append(cap_df)

    logp(
        f"  Stats: raw_positive={total_raw_positive}, "
        f"before_non_site={total_before_nonsite}, "
        f"with_non_site={total_with_nonsite}"
    )

    # ------------------------------------------------------------------
    # Phase 4: Global stratified downsampling -> ~1:1:1
    # ------------------------------------------------------------------
    logp("Phase 4: Global stratified downsampling...")

    if not capped_dfs:
        logp("  No data after capping -- nothing to output.")
        return

    merged = pd.concat(capped_dfs, ignore_index=True)
    n_donor = int((merged["label"] == LABEL_DONOR).sum())
    n_acceptor = int((merged["label"] == LABEL_ACCEPTOR).sum())
    n_nonsite = int((merged["label"] == LABEL_NONSITE).sum())
    logp(
        f"  Pre-downsampling: donor={n_donor}, acceptor={n_acceptor}, "
        f"non-site={n_nonsite}"
    )

    per_class_target = min(n_donor, n_acceptor, n_nonsite)
    if per_class_target == 0:
        logp("  One or more classes have 0 samples -- cannot balance!")
        return

    donor_sample = merged[merged["label"] == LABEL_DONOR].sample(
        n=per_class_target, random_state=seed,
    )
    acceptor_sample = merged[merged["label"] == LABEL_ACCEPTOR].sample(
        n=per_class_target, random_state=seed,
    )
    non_site_sample = merged[merged["label"] == LABEL_NONSITE].sample(
        n=per_class_target, random_state=seed,
    )

    final_df = pd.concat(
        [donor_sample, acceptor_sample, non_site_sample], ignore_index=True,
    )
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logp(
        f"  Post-downsampling: donor={len(donor_sample)}, "
        f"acceptor={len(acceptor_sample)}, non-site={len(non_site_sample)}"
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    # Select and order columns per spec
    desired_columns = [
        "sequence", "label", "species", "seqid", "transcript_id",
        "intron_id", "center_pos", "dinucleotide", "intron_length",
        "exon_count",
    ]
    final_df = final_df[desired_columns]

    output_path = os.path.join(output_dir, "filtered_all.parquet")
    final_df.to_parquet(output_path, index=False)
    logp(f"  -> {output_path} ({len(final_df)} total samples)")

    # filter_stats.json
    filter_stats = {
        "raw_positive": total_raw_positive,
        "total_before_non_site": total_before_nonsite,
        "total_with_non_site": total_with_nonsite,
        "species_cap": cap,
        "species_cap_value": cap,
        "balanced_sampling": per_class_target,
        "n_species": len(species_list),
    }
    stats_path = os.path.join(output_dir, "filter_stats.json")
    with open(stats_path, "w") as f:
        json.dump(filter_stats, f, indent=2)
    logp(f"  -> {stats_path}")

    logp("=" * 50)
    logp(f"Dataset v2 build complete! Output: {output_dir}")


# ===================================================================
# CLI Entry Point
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build v2 splice site prediction dataset",
    )
    parser.add_argument(
        "--assembly_dir",
        default="/home/lty/yy_projects/fungi_project/1kfg_datasets/assembly_genome",
    )
    parser.add_argument(
        "--gff_dir",
        default="/home/lty/yy_projects/fungi_project/1kfg_datasets/gff3",
    )
    parser.add_argument(
        "--transcript_dir",
        default="/home/lty/yy_projects/fungi_project/1kfg_datasets/transcript",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/lty/yy_projects/fungi_project/"
            "fungi_dna_model/data/downstream/task2_splice_site_v2"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    build_dataset_v2(
        args.assembly_dir,
        args.gff_dir,
        args.transcript_dir,
        args.output_dir,
        args.seed,
    )
