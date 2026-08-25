#!/usr/bin/env python
"""End-to-end data preparation for FungiDNA pretraining.

Usage:
    python scripts/prepare_pretrain_data.py \\
        --raw_dir ~/yy_projects/fungi_project/1kfg_datasets \\
        --taxonomy_xlsx fungi_classification.xlsx \\
        --output_dir data/processed \\
        --phase1_min_len 5000 \\
        --phase2_min_len 50000

Steps:
    1. Scan raw .fasta.gz files → filtered .fasta (Phase 1: ≥5Kb, Phase 2: ≥50Kb)
    2. Train BPE tokenizer on filtered data
    3. Report dataset statistics
"""
import argparse, sys, os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fungidna.data.qc_filter import filter_contigs, iter_contigs, collect_long_contigs
from fungidna.data.taxonomy import TaxonomyDB
from fungidna.data.tokenizer import train_bpe


def main():
    parser = argparse.ArgumentParser(description="Prepare FungiDNA pretraining data")
    parser.add_argument("--raw_dir", required=True, help="Directory with 735 genome subfolders (each containing *.fasta.gz)")
    parser.add_argument("--taxonomy_xlsx", required=True, help="fungi_classification.xlsx path")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    parser.add_argument("--phase1_min_len", type=int, default=5000)
    parser.add_argument("--phase2_min_len", type=int, default=50000)
    parser.add_argument("--max_n_frac", type=float, default=0.10)
    parser.add_argument("--bpe_vocab_size", type=int, default=4096)
    parser.add_argument("--bpe_sample", type=float, default=0.05)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Load taxonomy ──
    print(f"[0/4] Loading taxonomy from {args.taxonomy_xlsx} ...")
    taxdb = TaxonomyDB(args.taxonomy_xlsx)
    print(f"      {len(taxdb)} genomes loaded")

    # ── Step 1: QC Filter ──
    print(f"[1/4] Filtering genomes from {args.raw_dir} ...")
    phase1_out = str(out / "phase1_filtered")
    totals = filter_contigs(
        args.raw_dir, phase1_out,
        min_length=args.phase1_min_len,
        max_n_frac=args.max_n_frac,
    )
    total_bases = sum(totals.values())
    print(f"      Phase 1: {len(totals)} genomes, {total_bases / 1e9:.2f} Gb total (min contig={args.phase1_min_len}bp)")

    # Also copy long contigs for Phase 2
    print(f"[2/4] Collecting long contigs (≥{args.phase2_min_len}bp) ...")
    long = collect_long_contigs(phase1_out, min_length=args.phase2_min_len)
    phase2_out = out / "phase2_filtered"
    phase2_out.mkdir(exist_ok=True)
    n_long = 0
    total_long_bases = 0
    for genome_id, contigs in long.items():
        with open(phase2_out / f"{genome_id}.fasta", "w") as f:
            for cid, seq in contigs:
                f.write(f">{cid}\n")
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i + 80] + "\n")
                n_long += 1
                total_long_bases += len(seq)
    print(f"      Phase 2: {len(long)} genomes, {n_long} contigs, {total_long_bases / 1e9:.2f} Gb total")

    # ── Step 3: Train BPE ──
    print(f"[3/4] Training BPE tokenizer (vocab={args.bpe_vocab_size}) ...")
    tokenizer_path = str(out / "bpe_fungi.model")
    train_bpe(
        phase1_out,
        tokenizer_path,
        vocab_size=args.bpe_vocab_size,
        sample_fraction=args.bpe_sample,
    )

    # ── Step 4: Summary ──
    print(f"\n[4/4] Data preparation complete!")
    print(f"      Phase 1 filtered FASTA: {phase1_out}/")
    print(f"      Phase 2 filtered FASTA: {str(phase2_out)}/")
    print(f"      BPE tokenizer:         {tokenizer_path}")
    print(f"      Total bases (Phase1):  {total_bases / 1e9:.2f} Gb across {len(totals)} genomes")
    print(f"      Long contigs (Phase2): {total_long_bases / 1e9:.2f} Gb across {len(long)} genomes")

    # Print taxonomy distribution
    print(f"\n      Phylum distribution:")
    for phylum in sorted(taxdb.phylum_to_genomes, key=lambda p: -len(taxdb.phylum_to_genomes[p])):
        gids = taxdb.phylum_to_genomes[phylum]
        print(f"        {phylum}: {len(gids)}")
    print(f"\n      Ready for pretraining! Next: scripts/train_phase1.sh")


if __name__ == "__main__":
    main()
