#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/yy_projects/fungi_project/1kfg_datasets/genomes_unmasked}"
STATS="$ROOT/genomes_unmasked_fasta_stats.txt"

if [[ ! -d "$ROOT" ]]; then
  echo "Root directory not found: $ROOT" >&2
  exit 1
fi

extract_date() {
  local filename="$1"
  if [[ "$filename" =~ ([0-9]{4}-[0-9]{2}-[0-9]{2}) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  fi
}

# Statistics table
printf "sample\tstatus\tselected_file\toriginal_path\tdestination_path\tfile_size_bytes\tnote\n" > "$STATS"

shopt -s nullglob

for sample_dir in "$ROOT"/*; do
  [[ -d "$sample_dir" ]] || continue
  sample="$(basename "$sample_dir")"
  src_dir="$sample_dir/Assembly/Mycocosm/Assembly/Genome_Assembly__unmasked_"
  dest_dir="$ROOT/$sample"

  if [[ ! -d "$src_dir" ]]; then
    printf "%s\tMISSING_UNMASKED_FOLDER\t\t%s\t%s\t\t\n" "$sample" "$src_dir" "$dest_dir" >> "$STATS"
    continue
  fi

  preferred=("$src_dir"/*_AssemblyScaffolds.fasta.gz)
  selected_files=()
  status=""
  note=""

  if (( ${#preferred[@]} > 0 )); then
    selected_files=("${preferred[@]}")
    status="MOVED_ASSEMBLY_SCAFFOLDS"
  else
    all_fasta=("$src_dir"/*.fasta.gz)
    if (( ${#all_fasta[@]} == 0 )); then
      printf "%s\tNO_FASTA_GZ_FOUND\t\t%s\t%s\t\t\n" "$sample" "$src_dir" "$dest_dir" >> "$STATS"
      continue
    fi

    best_file=""
    best_date=""
    first_file="${all_fasta[0]}"
    for f in "${all_fasta[@]}"; do
      [[ -f "$f" ]] || continue
      base="$(basename "$f")"
      date="$(extract_date "$base" || true)"
      if [[ -n "$date" ]]; then
        if [[ -z "$best_date" || "$date" > "$best_date" ]]; then
          best_date="$date"
          best_file="$f"
        fi
      fi
    done

    if [[ -n "$best_file" ]]; then
      selected_files=("$best_file")
      status="FALLBACK_LATEST_DATED_FASTA_GZ"
      note="No *_AssemblyScaffolds.fasta.gz found; selected the .fasta.gz file with the latest date in its name"
    else
      selected_files=("$first_file")
      status="FALLBACK_NO_DATE_PATTERN"
      note="No *_AssemblyScaffolds.fasta.gz found and no date pattern detected; selected the first .fasta.gz file"
    fi
  fi

  mkdir -p "$dest_dir"

  for f in "${selected_files[@]}"; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f")"
    target="$dest_dir/$base"
    size=$(stat -c '%s' "$f")

    if [[ -e "$target" ]]; then
      printf "%s\tALREADY_EXISTS\t%s\t%s\t%s\t%s\t%s\n" "$sample" "$base" "$f" "$target" "$size" "$note" >> "$STATS"
    else
      mv "$f" "$target"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$sample" "$status" "$base" "$f" "$target" "$size" "$note" >> "$STATS"
    fi
  done

done

echo "Done. Statistics saved to: $STATS"
