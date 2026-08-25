"""FASTA QC filtering — read .fasta.gz files, filter contigs, output cleaned FASTA."""
import gzip
from pathlib import Path
from typing import Iterator, NamedTuple


class ContigStats(NamedTuple):
    genome_id: str
    contig_id: str
    length: int
    n_frac: float
    gc_content: float


def _read_fasta_gz(filepath: str) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) from a .fasta.gz file."""
    with gzip.open(filepath, "rt") as f:
        header = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield (header, "".join(seq_parts))
                header = line[1:].split()[0]  # ID up to first space
                seq_parts = []
            else:
                seq_parts.append(line.upper())
        if header is not None:
            yield (header, "".join(seq_parts))


def iter_contigs(fasta_dir: str) -> Iterator[ContigStats]:
    """Scan all *.fasta.gz files under fasta_dir (one subfolder per genome), yield stats."""
    base = Path(fasta_dir)
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        genome_id = folder.name
        for gz_file in folder.glob("*.fasta.gz"):
            for header, seq in _read_fasta_gz(str(gz_file)):
                length = len(seq)
                n_count = seq.count("N")
                gc = seq.count("G") + seq.count("C")
                at = seq.count("A") + seq.count("T")
                n_frac = n_count / length if length > 0 else 1.0
                gc_content = gc / max(gc + at, 1)
                yield ContigStats(
                    genome_id=genome_id,
                    contig_id=header,
                    length=length,
                    n_frac=n_frac,
                    gc_content=gc_content,
                )


def filter_contigs(
    fasta_dir: str,
    output_dir: str,
    min_length: int = 5000,
    max_n_frac: float = 0.10,
) -> dict[str, int]:
    """Filter .fasta.gz files: remove short contigs and high-N scaffolds.

    Writes cleaned .fasta (uncompressed) to output_dir/{genome_id}.fasta.
    Returns genome_id → total_bases_after_filtering dict.
    """
    base = Path(fasta_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    genome_totals = {}

    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        genome_id = folder.name
        kept_records = []
        for gz_file in folder.glob("*.fasta.gz"):
            for header, seq in _read_fasta_gz(str(gz_file)):
                n_frac = seq.count("N") / len(seq) if len(seq) > 0 else 1.0
                if len(seq) >= min_length and n_frac <= max_n_frac:
                    kept_records.append((header, seq))

        if kept_records:
            out_path = out / f"{genome_id}.fasta"
            with open(out_path, "w") as f:
                for header, seq in kept_records:
                    f.write(f">{header}\n")
                    # Wrap at 80 chars
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i + 80] + "\n")
            genome_totals[genome_id] = sum(len(s) for _, s in kept_records)

    return genome_totals


def collect_long_contigs(
    filtered_dir: str,
    min_length: int = 50000,
) -> dict[str, list[tuple[str, str]]]:
    """Collect contigs >= min_length from filtered FASTA files for Phase 2.

    Returns genome_id → list of (contig_id, sequence).
    """
    result = {}
    for fasta_path in Path(filtered_dir).glob("*.fasta"):
        genome_id = fasta_path.stem
        long_contigs = []
        for header, seq in _read_fasta(str(fasta_path)):
            if len(seq) >= min_length:
                long_contigs.append((header, seq))
        if long_contigs:
            result[genome_id] = long_contigs
    return result


def _read_fasta(filepath: str) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) from a plain .fasta file."""
    with open(filepath) as f:
        header = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield (header, "".join(seq_parts))
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
        if header is not None:
            yield (header, "".join(seq_parts))
