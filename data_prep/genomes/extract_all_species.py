#!/usr/bin/env python
"""
Batch extract 5 genomic region types (CDS, Intron, Intergenic, 5'UTR, 3'UTR)
from 736 fungal species with GFF/GFF3 annotations.
Output BED + FASTA per species + merged FASTA + summary.json.
Pure Python standard library.
"""

import sys
import os
import re
import json
import traceback
from collections import defaultdict

# ============================================================================
# Base utilities
# ============================================================================

COMPLEMENT = str.maketrans('ATCGatcgRYSWKMryswkmBDHVbdhv',
                           'TAGCtagcYRSWMKyrswmkVHDBvhdb')


def reverse_complement(seq):
    """Return reverse complement of a DNA sequence."""
    return seq.translate(COMPLEMENT)[::-1]


def parse_fasta(filepath):
    """
    Parse a FASTA file.

    Supports two header formats:
      - JGI:  >jgi|Species|ID|gene_name  → ID = 3rd field
      - Generic: >scaffold_name           → ID = first space-delimited token

    Returns:
        {header_id: sequence}
    """
    seqs = {}
    current_id = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    seqs[current_id] = ''.join(current_seq)
                header = line[1:]
                parts = header.split('|')
                if len(parts) >= 3:
                    current_id = parts[2]
                else:
                    current_id = header.split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            seqs[current_id] = ''.join(current_seq)

    return seqs


def parse_genome_fasta(filepath):
    """
    Parse a genome FASTA file.

    Returns:
        {scaffold_name: sequence}
    """
    genome = {}
    current_id = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    genome[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            genome[current_id] = ''.join(current_seq)

    return genome


def extract_from_genome(genome, scaffold, coords, strand):
    """
    Extract and concatenate sequence from genome by coordinates.

    Args:
        genome: {scaffold_name: sequence}
        scaffold: scaffold name
        coords: [(start, end), ...] — 1-based closed intervals
        strand: '+' or '-'

    Returns:
        Concatenated uppercase sequence string, or None if out of range.
    """
    if scaffold not in genome:
        return None

    seq_parts = []
    for start, end in coords:
        s = start - 1   # 1-based → 0-based
        e = end          # closed → Python slice right-open
        if s < 0 or e > len(genome[scaffold]):
            return None
        seq_parts.append(genome[scaffold][s:e])

    full_seq = ''.join(seq_parts)

    if strand == '-':
        full_seq = reverse_complement(full_seq)

    return full_seq.upper()


def write_bed(filepath, entries):
    """
    Write 6-column BED file (tab-separated).

    Args:
        entries: [(scaffold, start_0based, end, name, score, strand), ...]
    """
    with open(filepath, 'w') as f:
        for scaffold, start, end, name, score, strand in entries:
            f.write(f'{scaffold}\t{start}\t{end}\t{name}\t{score}\t{strand}\n')


def write_fasta(filepath, entries, line_width=60):
    """
    Write a FASTA file.

    Args:
        entries: [(header, sequence), ...]
        line_width: bases per line
    """
    with open(filepath, 'w') as f:
        for header, seq in entries:
            f.write(f'>{header}\n')
            for i in range(0, len(seq), line_width):
                f.write(seq[i:i+line_width] + '\n')


# ============================================================================
# GFF format detection
# ============================================================================

def detect_gff_format(filepath):
    """
    Detect GFF format by reading the first line.

    Returns:
        'gff3' if file starts with '##gff-version 3', otherwise 'gff'.
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    if first_line.startswith('##gff-version 3'):
        return 'gff3'
    return 'gff'


# ============================================================================
# GFF3 parser (standard format, Achstr1-style)
# ============================================================================

def parse_gff3(filepath):
    """
    Parse standard GFF3 format.

    Hierarchy: gene → mRNA → exon / CDS / five_prime_UTR / three_prime_UTR
    Parent= attribute establishes parent-child relationships.

    Returns:
        {transcript_id: {
            'scaffold': str,
            'strand': '+'/'-',
            'gene_name': str,
            'exons': [(start, end), ...],
            'cdss': [(start, end, phase), ...],
            'utr5': [(start, end), ...],
            'utr3': [(start, end), ...],
            'protein_id': str | None,
        }}
    """
    mrna_info = {}
    mrna_to_exons = defaultdict(list)
    mrna_to_cdss = defaultdict(list)
    mrna_to_utr5 = defaultdict(list)
    mrna_to_utr3 = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('##'):
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                continue

            scaffold, source, feature, start, end, score, strand, phase, attrs = parts
            start, end = int(start), int(end)

            if feature == 'mRNA':
                mrna_id_match = re.search(r'ID=([^;]+)', attrs)
                tid_match = re.search(r'transcriptId=(\d+)', attrs)
                pid_match = re.search(r'proteinId=(\d+)', attrs)
                if mrna_id_match and tid_match:
                    mrna_info[mrna_id_match.group(1)] = {
                        'transcript_id': tid_match.group(1),
                        'protein_id': pid_match.group(1) if pid_match else None,
                        'strand': strand,
                        'scaffold': scaffold,
                    }

            elif feature == 'exon':
                parent_match = re.search(r'Parent=([^;]+)', attrs)
                if parent_match:
                    for pid in parent_match.group(1).split(','):
                        mrna_to_exons[pid].append((start, end))

            elif feature == 'CDS':
                parent_match = re.search(r'Parent=([^;]+)', attrs)
                if parent_match:
                    phase_val = int(phase) if phase != '.' else 0
                    for pid in parent_match.group(1).split(','):
                        mrna_to_cdss[pid].append((start, end, phase_val))

            elif feature == 'five_prime_UTR':
                parent_match = re.search(r'Parent=([^;]+)', attrs)
                if parent_match:
                    for pid in parent_match.group(1).split(','):
                        mrna_to_utr5[pid].append((start, end))

            elif feature == 'three_prime_UTR':
                parent_match = re.search(r'Parent=([^;]+)', attrs)
                if parent_match:
                    for pid in parent_match.group(1).split(','):
                        mrna_to_utr3[pid].append((start, end))

    gene_models = {}
    for mrna_id, info in mrna_info.items():
        tid = info['transcript_id']
        gene_models[tid] = {
            'scaffold': info['scaffold'],
            'strand': info['strand'],
            'gene_name': mrna_id,
            'exons': sorted(mrna_to_exons.get(mrna_id, [])),
            'cdss': sorted(mrna_to_cdss.get(mrna_id, [])),
            'utr5': sorted(mrna_to_utr5.get(mrna_id, [])),
            'utr3': sorted(mrna_to_utr3.get(mrna_id, [])),
            'protein_id': info['protein_id'],
        }

    return gene_models


# ============================================================================
# Simplified GFF parser (Acain1-style)
# ============================================================================

def parse_simple_gff(filepath):
    """
    Parse simplified GFF format (Acain1-style).

    exon lines → transcriptId attribute
    CDS lines  → proteinId attribute
    Both linked to the same gene via the name attribute.

    Returns:
        {transcript_id: {
            'scaffold': str,
            'strand': '+'/'-',
            'gene_name': str,
            'exons': [(start, end), ...],
            'cdss': [(start, end, phase), ...],
            'protein_id': str | None,
        }}
    """
    name_to_exons = defaultdict(list)
    name_to_cdss = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 9:
                continue

            scaffold, source, feature, start, end, score, strand, phase, attrs = parts
            start, end = int(start), int(end)

            name_match = re.search(r'name "([^"]+)"', attrs)
            if not name_match:
                continue
            name = name_match.group(1)

            if feature == 'exon':
                tid_match = re.search(r'transcriptId (\d+)', attrs)
                if tid_match:
                    name_to_exons[name].append(
                        (start, end, strand, scaffold, tid_match.group(1))
                    )

            elif feature == 'CDS':
                pid_match = re.search(r'proteinId (\d+)', attrs)
                if pid_match:
                    phase_val = int(phase) if phase != '.' else 0
                    name_to_cdss[name].append(
                        (start, end, phase_val, strand, scaffold, pid_match.group(1))
                    )

    gene_models = {}
    for name, exons in name_to_exons.items():
        if not exons:
            continue
        tid = exons[0][4]
        strand = exons[0][2]
        scaffold = exons[0][3]
        exons_sorted = sorted([(s, e) for s, e, *_ in exons])

        cdss = name_to_cdss.get(name, [])
        cdss_sorted = sorted([(s, e, p) for s, e, p, *_ in cdss])
        protein_id = cdss[0][5] if cdss else None

        # If same transcript_id maps to multiple names, keep the one with more exons
        if tid in gene_models:
            if len(exons_sorted) > len(gene_models[tid]['exons']):
                gene_models[tid] = {
                    'scaffold': scaffold, 'strand': strand, 'gene_name': name,
                    'exons': exons_sorted, 'cdss': cdss_sorted, 'protein_id': protein_id,
                }
        else:
            gene_models[tid] = {
                'scaffold': scaffold, 'strand': strand, 'gene_name': name,
                'exons': exons_sorted, 'cdss': cdss_sorted, 'protein_id': protein_id,
            }

    return gene_models


# ============================================================================
# Module 1: CDS extraction + FASTA validation
# ============================================================================

def extract_cds(species, gene_models, genome, cds_fasta_path, out_dir):
    """
    Extract CDS sequences and validate against CDS FASTA.
    Only CDS with perfect match are retained.

    Returns:
        stats dict
    """
    cds_fasta_seqs = parse_fasta(cds_fasta_path)

    bed_entries = []
    fasta_entries = []
    matched = 0
    discarded = 0
    no_cds = 0

    for tid, fasta_seq in cds_fasta_seqs.items():
        fasta_seq = fasta_seq.upper()

        if tid not in gene_models:
            discarded += 1
            continue

        model = gene_models[tid]
        cdss = model['cdss']
        if not cdss:
            no_cds += 1
            continue

        cds_coords = [(s, e) for s, e, p in cdss]
        genomic_seq = extract_from_genome(genome, model['scaffold'], cds_coords, model['strand'])

        if genomic_seq is None:
            discarded += 1
            continue

        if genomic_seq == fasta_seq:
            matched += 1
            gene_name = model.get('gene_name', tid)

            # BED: one row per CDS fragment
            for i, (s, e, _) in enumerate(cdss, 1):
                name = f'{tid}|{gene_name}|cds_exon_{i}'
                bed_entries.append((model['scaffold'], s - 1, e, name, '.', model['strand']))

            # FASTA: full concatenated CDS
            total_start = min(s for s, e, _ in cdss)
            total_end = max(e for s, e, _ in cdss)
            fa_header = (
                f'{tid}|{gene_name}|'
                f'{model["scaffold"]}:{total_start}-{total_end}({model["strand"]})|cds'
            )
            fasta_entries.append((fa_header, genomic_seq))
        else:
            discarded += 1

    write_bed(os.path.join(out_dir, 'cds.bed'), bed_entries)
    write_fasta(os.path.join(out_dir, 'cds.fasta'), fasta_entries)

    return {
        'total_in_fasta': len(cds_fasta_seqs),
        'matched': matched,
        'discarded': discarded,
        'no_cds': no_cds,
        'bed_rows': len(bed_entries),
    }


# ============================================================================
# Module 2: Intron extraction
# ============================================================================

def extract_introns(species, gene_models, genome, out_dir):
    """
    Infer introns from adjacent exons of each transcript.

    intron_i = (exon_i.end + 1, exon_{i+1}.start - 1)
    Negative-strand introns are reverse-complemented (follow gene strand).
    """
    bed_entries = []
    fasta_entries = []
    intron_count = 0
    skipped_single_exon = 0

    for tid, model in gene_models.items():
        exons = model['exons']
        if len(exons) < 2:
            skipped_single_exon += 1
            continue

        gene_name = model.get('gene_name', tid)

        for i in range(len(exons) - 1):
            intron_start = exons[i][1] + 1
            intron_end = exons[i+1][0] - 1

            if intron_start > intron_end:
                continue

            intron_count += 1
            coords = [(intron_start, intron_end)]
            intron_seq = extract_from_genome(genome, model['scaffold'], coords, model['strand'])

            if intron_seq is None:
                continue

            name = f'{tid}|{gene_name}|intron_{i+1}'
            bed_entries.append((
                model['scaffold'], intron_start - 1, intron_end,
                name, '.', model['strand']
            ))
            fa_header = (
                f'{tid}|{gene_name}|'
                f'{model["scaffold"]}:{intron_start}-{intron_end}({model["strand"]})|intron_{i+1}'
            )
            fasta_entries.append((fa_header, intron_seq))

    write_bed(os.path.join(out_dir, 'intron.bed'), bed_entries)
    write_fasta(os.path.join(out_dir, 'intron.fasta'), fasta_entries)

    return {
        'total_transcripts': len(gene_models),
        'intron_count': intron_count,
        'single_exon_skipped': skipped_single_exon,
        'bed_rows': len(bed_entries),
    }


# ============================================================================
# Module 3: Intergenic extraction
# ============================================================================

def build_gene_clusters(gene_models):
    """
    Build gene clusters by merging overlapping gene intervals.

    Each gene interval = min(exon starts) to max(exon ends).
    Overlapping intervals are merged into clusters.

    Returns:
        {scaffold: [(cluster_start, cluster_end), ...]}  — sorted, non-overlapping
    """
    scaffold_spans = defaultdict(list)
    for tid, model in gene_models.items():
        exons = model['exons']
        if not exons:
            continue
        min_s = min(s for s, e in exons)
        max_e = max(e for s, e in exons)
        scaffold_spans[model['scaffold']].append((min_s, max_e))

    clusters = {}
    for scaffold, spans in scaffold_spans.items():
        spans.sort()
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        clusters[scaffold] = merged

    return clusters


def extract_intergenic(species, gene_models, genome, out_dir):
    """
    Extract intergenic regions.

    Regions between gene clusters + scaffold ends = intergenic.
    Always positive strand (no directionality).
    """
    clusters = build_gene_clusters(gene_models)

    bed_entries = []
    fasta_entries = []
    total_intergenic = 0

    for scaffold, seq in genome.items():
        scaffold_clusters = clusters.get(scaffold, [])
        all_intervals = []

        # Before first gene cluster
        if scaffold_clusters:
            if scaffold_clusters[0][0] > 1:
                all_intervals.append((1, scaffold_clusters[0][0] - 1))
        else:
            # Entire scaffold has no genes → all intergenic
            all_intervals.append((1, len(seq)))

        # Between gene clusters
        for i in range(len(scaffold_clusters) - 1):
            gap_start = scaffold_clusters[i][1] + 1
            gap_end = scaffold_clusters[i+1][0] - 1
            if gap_start <= gap_end:
                all_intervals.append((gap_start, gap_end))

        # After last gene cluster
        if scaffold_clusters:
            last_end = scaffold_clusters[-1][1]
            if last_end < len(seq):
                all_intervals.append((last_end + 1, len(seq)))

        for idx, (start, end) in enumerate(all_intervals, 1):
            seq_extracted = extract_from_genome(genome, scaffold, [(start, end)], '+')
            if seq_extracted is None:
                continue
            total_intergenic += 1

            name = f'{scaffold}:{start}-{end}|intergenic_{idx}'
            bed_entries.append((scaffold, start - 1, end, name, '.', '.'))
            fa_header = f'{scaffold}:{start}-{end}|intergenic_{idx}'
            fasta_entries.append((fa_header, seq_extracted))

    write_bed(os.path.join(out_dir, 'intergenic.bed'), bed_entries)
    write_fasta(os.path.join(out_dir, 'intergenic.fasta'), fasta_entries)

    return {
        'total_scaffolds': len(genome),
        'intergenic_count': total_intergenic,
        'total_bases': sum(len(s) for _, s in fasta_entries),
        'bed_rows': len(bed_entries),
    }


# ============================================================================
# Module 4+5: UTR extraction (GFF3 only)
# ============================================================================

def extract_utrs(species, gene_models, genome, out_dir):
    """
    Extract 5'UTR and 3'UTR (only applicable for GFF3-format species).

    Aggregates UTR fragments per mRNA/transcript.
    BED: one row per fragment; FASTA: one concatenated sequence.
    """
    stats = {}

    for utr_type, utr_key in [('five_prime_utr', 'utr5'), ('three_prime_utr', 'utr3')]:
        bed_entries = []
        fasta_entries = []
        count = 0

        for tid, model in gene_models.items():
            utr_coords = model.get(utr_key, [])
            if not utr_coords:
                continue

            gene_name = model.get('gene_name', tid)
            utr_seq = extract_from_genome(genome, model['scaffold'], utr_coords, model['strand'])
            if utr_seq is None:
                continue

            count += 1

            # BED: one row per UTR fragment
            for i, (s, e) in enumerate(utr_coords, 1):
                name = f'{tid}|{gene_name}|{utr_type}_{i}'
                bed_entries.append((model['scaffold'], s - 1, e, name, '.', model['strand']))

            # FASTA: full concatenated UTR
            total_start = min(s for s, e in utr_coords)
            total_end = max(e for s, e in utr_coords)
            fa_header = (
                f'{tid}|{gene_name}|'
                f'{model["scaffold"]}:{total_start}-{total_end}({model["strand"]})|{utr_type}'
            )
            fasta_entries.append((fa_header, utr_seq))

        write_bed(os.path.join(out_dir, f'{utr_type}.bed'), bed_entries)
        write_fasta(os.path.join(out_dir, f'{utr_type}.fasta'), fasta_entries)
        stats[utr_type] = count

    return stats


# ============================================================================
# Species mapping
# ============================================================================

def build_species_map(genome_dir, gff_dir, cds_dir):
    """
    Scan input directories and build a species → file paths mapping.

    Only species with all three files (genome + gff/gff3 + cds) are included.

    Returns:
        {
            species_name: {
                'genome': '/path/to/genome.fasta',
                'gff': '/path/to/annotation.gff' or '.gff3',
                'cds': '/path/to/cds.fasta',
            }
        },
        skipped: {species_name: reason}
    """
    # Scan genome files
    genome_files = {}
    for f in os.listdir(genome_dir):
        if f.endswith('.fasta'):
            species = f[:-6]  # strip .fasta
            genome_files[species] = os.path.join(genome_dir, f)

    # Scan GFF files (both .gff and .gff3)
    gff_files = {}
    for f in os.listdir(gff_dir):
        if f.endswith('.gff3'):
            species = f[:-5]  # strip .gff3
            gff_files[species] = os.path.join(gff_dir, f)
        elif f.endswith('.gff'):
            species = f[:-4]  # strip .gff
            gff_files[species] = os.path.join(gff_dir, f)

    # Scan CDS files
    cds_files = {}
    for f in os.listdir(cds_dir):
        if f.endswith('.fasta'):
            species = f[:-6]  # strip .fasta
            cds_files[species] = os.path.join(cds_dir, f)

    # Cross-reference: species must have all three
    all_species = set(genome_files.keys()) & set(gff_files.keys()) & set(cds_files.keys())
    species_map = {}
    skipped = {}

    for sp in sorted(all_species):
        species_map[sp] = {
            'genome': genome_files[sp],
            'gff': gff_files[sp],
            'cds': cds_files[sp],
        }

    # Record species missing one or more files
    all_candidates = set(genome_files.keys()) | set(gff_files.keys()) | set(cds_files.keys())
    for sp in sorted(all_candidates - all_species):
        missing = []
        if sp not in genome_files:
            missing.append('genome')
        if sp not in gff_files:
            missing.append('gff/gff3')
        if sp not in cds_files:
            missing.append('cds')
        skipped[sp] = f'missing: {", ".join(missing)}'

    return species_map, skipped


# ============================================================================
# Single-species orchestrator
# ============================================================================

def process_species(species, paths, out_base):
    """
    Run all five extraction modules for a single species.

    Args:
        species: species name
        paths: {'genome': str, 'gff': str, 'cds': str}
        out_base: base output directory

    Returns:
        stats dict on success, or error string on failure
    """
    out_dir = os.path.join(out_base, species)
    os.makedirs(out_dir, exist_ok=True)

    # Load genome
    genome = parse_genome_fasta(paths['genome'])

    # Detect GFF format and parse
    gff_format = detect_gff_format(paths['gff'])
    if gff_format == 'gff3':
        gene_models = parse_gff3(paths['gff'])
        has_utr = True
    else:
        gene_models = parse_simple_gff(paths['gff'])
        has_utr = False

    if not gene_models:
        print(f"WARNING: zero gene models parsed from {paths['gff']} "
              f"(flat annotation — no mRNA/exon/CDS features). Skipping extraction.")
        return {
            'gff_format': gff_format,
            'models_note': 'zero gene models parsed from GFF',
            'cds': {'total_in_fasta': 0, 'matched': 0, 'discarded': 0, 'no_cds': 0, 'bed_rows': 0},
            'intron': {'total_transcripts': 0, 'intron_count': 0, 'single_exon_skipped': 0, 'bed_rows': 0},
            'intergenic': {'total_scaffolds': 0, 'intergenic_count': 0, 'total_bases': 0, 'bed_rows': 0},
            'five_prime_utr': 0,
            'three_prime_utr': 0,
        }

    # Extract all regions
    stats = {'gff_format': gff_format}

    stats['cds'] = extract_cds(species, gene_models, genome, paths['cds'], out_dir)
    stats['intron'] = extract_introns(species, gene_models, genome, out_dir)
    stats['intergenic'] = extract_intergenic(species, gene_models, genome, out_dir)

    if has_utr:
        utr_stats = extract_utrs(species, gene_models, genome, out_dir)
        stats['five_prime_utr'] = utr_stats.get('five_prime_utr', 0)
        stats['three_prime_utr'] = utr_stats.get('three_prime_utr', 0)
    else:
        stats['five_prime_utr'] = 0
        stats['three_prime_utr'] = 0

    return stats


# ============================================================================
# Merged output
# ============================================================================

# Region types and their output filenames
REGION_TYPES = ['cds', 'intron', 'intergenic', 'five_prime_utr', 'three_prime_utr']


def merge_fastas(output_dir, processed_species=None):
    """
    Collect per-species FASTA files and merge by region type into merged/.

    Args:
        output_dir: base output directory containing per-species subdirectories
        processed_species: optional set/list of species names to include.
                           If None, all subdirectories (except 'merged') are included.

    Header format for merged files:
        >species|original_header
    """
    merged_dir = os.path.join(output_dir, 'merged')
    os.makedirs(merged_dir, exist_ok=True)

    counts = {}

    for region in REGION_TYPES:
        merged_path = os.path.join(merged_dir, f'{region}.fasta')
        total_entries = 0

        with open(merged_path, 'w') as outf:
            for species_dir in sorted(os.listdir(output_dir)):
                if processed_species is not None and species_dir not in processed_species:
                    continue
                species_path = os.path.join(output_dir, species_dir)
                if not os.path.isdir(species_path) or species_dir == 'merged':
                    continue

                region_fasta = os.path.join(species_path, f'{region}.fasta')
                if not os.path.isfile(region_fasta):
                    continue

                with open(region_fasta, 'r') as inf:
                    for line in inf:
                        line = line.rstrip('\n')
                        if line.startswith('>'):
                            original_header = line[1:]  # strip '>'
                            outf.write(f'>{species_dir}|{original_header}\n')
                            total_entries += 1
                        else:
                            outf.write(line + '\n')

        counts[region] = total_entries

    return counts


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    genome_dir = os.path.join(base_dir, 'assembly_genome')
    gff_dir = os.path.join(base_dir, 'gff3')
    cds_dir = os.path.join(base_dir, 'cds')
    output_dir = os.path.join(base_dir, 'output')

    # Build species map
    print("Building species map...")
    species_map, skipped = build_species_map(genome_dir, gff_dir, cds_dir)
    print(f"  {len(species_map)} species with complete data")
    print(f"  {len(skipped)} species skipped")

    total = len(species_map)
    processed = 0
    failed = 0
    summary = {
        'total_species': total,
        'processed': 0,
        'skipped': len(skipped),
        'failed': 0,
        'by_species': {},
        'skipped_species': skipped,
        'failed_species': {},
    }

    # Process each species
    for i, (species, paths) in enumerate(sorted(species_map.items()), 1):
        print(f"[{i}/{total}] {species}...", end=' ', flush=True)

        try:
            stats = process_species(species, paths, output_dir)
            summary['by_species'][species] = stats
            processed += 1
            cds_m = stats['cds']['matched']
            cds_d = stats['cds']['discarded']
            intr = stats['intron']['intron_count']
            inter = stats['intergenic']['intergenic_count']
            u5 = stats.get('five_prime_utr', 0)
            u3 = stats.get('three_prime_utr', 0)
            print(f"OK  CDS:{cds_m}/{cds_m+cds_d}  Intron:{intr}  Intergenic:{inter}  5'UTR:{u5}  3'UTR:{u3}")
        except Exception as e:
            failed += 1
            error_msg = f"{type(e).__name__}: {e}"
            summary['failed_species'][species] = error_msg
            print(f"FAIL  {error_msg}")
            traceback.print_exc()

    # Update summary counts
    summary['processed'] = processed
    summary['failed'] = failed

    # Write summary
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to {summary_path}")

    # Merge FASTAs
    print("Merging FASTA files...")
    merge_counts = merge_fastas(output_dir, processed_species=set(species_map.keys()))
    merged_dir = os.path.join(output_dir, 'merged')
    print(f"Merged FASTA files written to {merged_dir}/")
    for region, count in merge_counts.items():
        print(f"  {region}.fasta: {count:,} entries")

    # Final report
    print(f"\n{'='*60}")
    print(f"BATCH EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total species:  {total}")
    print(f"Processed:      {processed}")
    print(f"Failed:         {failed}")
    print(f"Skipped:        {len(skipped)}")
    print(f"Output:         {output_dir}/")


if __name__ == '__main__':
    main()
