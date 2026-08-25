#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/yy_projects/fungi_project/1kfg_datasets/genomes_unmasked}"
BACKUP_DIR="$ROOT/_backup_multiple_fasta"
STATS="$ROOT/random_pick_stats.tsv"

if [[ ! -d "$ROOT" ]]; then
  echo "Root directory not found: $ROOT" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
printf "sample\tn_fasta_gz\tkept_file\tstatus\tbackup_files\n" > "$STATS"

shopt -s nullglob

for sample_dir in "$ROOT"/*; do
  [[ -d "$sample_dir" ]] || continue
  sample="$(basename "$sample_dir")"

  # Only consider fasta.gz files directly under the sample folder.
  mapfile -d '' files < <(find "$sample_dir" -maxdepth 1 -type f -name '*.fasta.gz' -print0 | sort -z)
  n=${#files[@]}

  if (( n == 0 )); then
    printf "%s\t0\t\tNO_FASTA_GZ\t\n" "$sample" >> "$STATS"
    continue
  fi

  if (( n == 1 )); then
    printf "%s\t1\t%s\tSINGLE_FILE\t\n" "$sample" "$(basename "${files[0]}")" >> "$STATS"
    continue
  fi

  # Randomly keep one file.
  kept_file="$(printf '%s\n' "${files[@]}" | shuf -n 1)"
  backup_sample_dir="$BACKUP_DIR/$sample"
  mkdir -p "$backup_sample_dir"

  backup_list=()
  for f in "${files[@]}"; do
    if [[ "$f" == "$kept_file" ]]; then
      continue
    fi
    mv "$f" "$backup_sample_dir/$(basename "$f")"
    backup_list+=("$(basename "$f")")
  done

  printf "%s\t%d\t%s\tRANDOM_KEEP\t%s\n" \
    "$sample" "$n" "$(basename "$kept_file")" "$(IFS=';'; echo "${backup_list[*]}")" >> "$STATS"
done

echo "Done. Stats saved to: $STATS"
echo "Backups saved under: $BACKUP_DIR"
