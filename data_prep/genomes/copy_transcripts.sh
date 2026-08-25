#!/bin/bash
set -euo pipefail

SRC_BASE="/home/lty/yy_projects/fungi_project/1kfg_datasets/annotation_filtered"
DST_DIR="/home/lty/yy_projects/fungi_project/1kfg_datasets/transcript"
TRANSCRIPTS_SUBPATH="Filtered_Models___best__/Transcripts"

mkdir -p "$DST_DIR"

LOG_FILE="$DST_DIR/copy_log.txt"
> "$LOG_FILE"

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

pick_latest() {
    local best_file=""
    local best_date=""
    while IFS= read -r f; do
        date_str=$(echo "$f" | grep -oP '\d{8}(?!\d)|\d{4}-\d{2}-\d{2}' | head -1)
        if [ -z "$date_str" ]; then
            best_file="$f"
            break
        fi
        local cmp_date
        cmp_date=$(echo "$date_str" | tr -d '-')
        if [ -z "$best_date" ] || [ "$cmp_date" -gt "$best_date" ]; then
            best_date="$cmp_date"
            best_file="$f"
        fi
    done
    echo "$best_file"
}

total=0
success=0
skipped=0

for species_dir in "$SRC_BASE"/*/; do
    species=$(basename "$species_dir")
    transcripts_dir="$species_dir/$TRANSCRIPTS_SUBPATH"
    total=$((total + 1))

    if [ ! -d "$transcripts_dir" ]; then
        log "[SKIP] $species — Transcripts directory not found"
        skipped=$((skipped + 1))
        continue
    fi

    selected=""
    selected_type=""

    # Priority 1: _GeneCatalog_transcripts_*.nt.fasta.gz (latest date, arbitrary prefix)
    gc_files=$(find "$transcripts_dir" -maxdepth 1 -name "*_GeneCatalog_transcripts_*.nt.fasta.gz" 2>/dev/null)
    if [ -n "$gc_files" ]; then
        selected=$(echo "$gc_files" | pick_latest)
        selected_type="GeneCatalog_nt"
    fi

    # Priority 2: fallback to .fasta.gz, excluding primary/sec alleles
    if [ -z "$selected" ]; then
        fb_files=$(find "$transcripts_dir" -maxdepth 1 -name "*.fasta.gz" \
            ! -name "*_primary_alleles_*" \
            ! -name "*_secondary_alleles_*" 2>/dev/null)
        if [ -n "$fb_files" ]; then
            selected=$(echo "$fb_files" | pick_latest)
            selected_type="fasta_fallback"
        fi
    fi

    if [ -z "$selected" ]; then
        log "[SKIP] $species — No matching fasta file found"
        skipped=$((skipped + 1))
        continue
    fi

    out_file="$DST_DIR/${species}.fasta"

    if gunzip -c "$selected" > "$out_file" 2>/dev/null; then
        log "[OK] $species — $selected_type: $(basename "$selected") → ${species}.fasta"
        success=$((success + 1))
    else
        log "[FAIL] $species — decompress failed: $selected"
        skipped=$((skipped + 1))
    fi
done

log ""
log "============================================"
log "Done. Total: $total | Success: $success | Skipped: $skipped"
log "Output directory: $DST_DIR"
log "Log file: $LOG_FILE"
