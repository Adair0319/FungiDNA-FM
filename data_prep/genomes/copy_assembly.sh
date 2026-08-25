#!/bin/bash
set -euo pipefail

SRC_BASE="/home/lty/yy_projects/fungi_project/1kfg_datasets/genomes_unmasked"
DST_DIR="/home/lty/yy_projects/fungi_project/1kfg_datasets/assembly_genome"

mkdir -p "$DST_DIR"

LOG_FILE="$DST_DIR/copy_log.txt"
> "$LOG_FILE"

total=0
success=0
skipped=0

for species_dir in "$SRC_BASE"/*/; do
    species=$(basename "$species_dir")
    total=$((total + 1))

    fasta_file=$(find "$species_dir" -maxdepth 1 -name "*.fasta.gz" 2>/dev/null | head -1)

    if [ -z "$fasta_file" ]; then
        echo "[SKIP] $species — No .fasta.gz found" | tee -a "$LOG_FILE"
        skipped=$((skipped + 1))
        continue
    fi

    out_file="$DST_DIR/${species}.fasta"

    if gunzip -c "$fasta_file" > "$out_file" 2>/dev/null; then
        echo "[OK] $species — $(basename "$fasta_file") → ${species}.fasta" | tee -a "$LOG_FILE"
        success=$((success + 1))
    else
        echo "[FAIL] $species — decompress failed: $fasta_file" | tee -a "$LOG_FILE"
        skipped=$((skipped + 1))
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "Done. Total: $total | Success: $success | Skipped: $skipped" | tee -a "$LOG_FILE"
echo "Output directory: $DST_DIR" | tee -a "$LOG_FILE"
