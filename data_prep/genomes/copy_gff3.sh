#!/bin/bash
set -euo pipefail

SRC_BASE="/home/lty/yy_projects/fungi_project/1kfg_datasets/annotation_filtered"
DST_DIR="/home/lty/yy_projects/fungi_project/1kfg_datasets/gff3"
GENES_SUBPATH="Filtered_Models___best__/Genes"

mkdir -p "$DST_DIR"

LOG_FILE="$DST_DIR/copy_log.txt"
> "$LOG_FILE"

total=0
success=0
skipped=0

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Find latest by date from a list of filenames (newest YYYYMMDD or YYYY-MM-DD)
# Input: newline-separated filenames
# Output: single filename (latest)
pick_latest() {
    local best_file=""
    local best_date=""
    while IFS= read -r f; do
        # Extract date string: YYYYMMDD or YYYY-MM-DD
        # Match patterns like 20160228 or 2024-05-29
        date_str=$(echo "$f" | grep -oP '\d{8}(?!\d)|\d{4}-\d{2}-\d{2}' | head -1)
        if [ -z "$date_str" ]; then
            # No date found, just use this file
            best_file="$f"
            break
        fi
        # Normalize to YYYYMMDD for comparison
        local cmp_date
        cmp_date=$(echo "$date_str" | tr -d '-')
        if [ -z "$best_date" ] || [ "$cmp_date" -gt "$best_date" ]; then
            best_date="$cmp_date"
            best_file="$f"
        fi
    done
    echo "$best_file"
}

for species_dir in "$SRC_BASE"/*/; do
    species=$(basename "$species_dir")
    genes_dir="$species_dir/$GENES_SUBPATH"
    total=$((total + 1))

    if [ ! -d "$genes_dir" ]; then
        log "[SKIP] $species — Genes directory not found: $genes_dir"
        skipped=$((skipped + 1))
        continue
    fi

    selected=""
    selected_type=""

    # Priority 1: _FilteredModels1_deflines.gff3.gz
    deflines_file=$(find "$genes_dir" -maxdepth 1 -name "*_FilteredModels1_deflines.gff3.gz" 2>/dev/null | head -1)
    if [ -n "$deflines_file" ]; then
        selected="$deflines_file"
        selected_type="deflines"
    fi

    # Priority 2: _FilteredModels1_YYYY-MM-DD.gff3.gz (latest date)
    if [ -z "$selected" ]; then
        fm1_files=$(find "$genes_dir" -maxdepth 1 -name "*_FilteredModels1_*-*-*.gff3.gz" 2>/dev/null)
        if [ -n "$fm1_files" ]; then
            selected=$(echo "$fm1_files" | pick_latest)
            selected_type="FilteredModels1_dated"
        fi
    fi

    # Priority 3: _GeneCatalog_*.gff3.gz (latest date, arbitrary prefix)
    if [ -z "$selected" ]; then
        gc_files=$(find "$genes_dir" -maxdepth 1 -name "*_GeneCatalog_*.gff3.gz" 2>/dev/null)
        if [ -n "$gc_files" ]; then
            selected=$(echo "$gc_files" | pick_latest)
            selected_type="GeneCatalog"
        fi
    fi

    # Priority 4: fallback to .gff.gz (latest date)
    if [ -z "$selected" ]; then
        gff_files=$(find "$genes_dir" -maxdepth 1 -name "*.gff.gz" 2>/dev/null)
        if [ -n "$gff_files" ]; then
            selected=$(echo "$gff_files" | pick_latest)
            selected_type="gff_fallback"
        fi
    fi

    if [ -z "$selected" ]; then
        log "[SKIP] $species — No matching gff/gff3 file found"
        skipped=$((skipped + 1))
        continue
    fi

    # Determine output extension
    if [[ "$selected" == *.gff3.gz ]]; then
        out_ext=".gff3"
    else
        out_ext=".gff"
    fi

    out_file="$DST_DIR/${species}${out_ext}"

    # Copy and decompress
    if gunzip -c "$selected" > "$out_file" 2>/dev/null; then
        log "[OK] $species — $selected_type: $(basename "$selected") → ${species}${out_ext}"
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
