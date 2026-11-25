"""Sliding window slicing for Phase 1 (BPE, short-range) and Phase 2 (single-nt, long-range)."""
from pathlib import Path
from typing import Iterator, Optional
import numpy as np

from fungidna.data.qc_filter import _read_fasta


def slice_windows(
    fasta_path: str,
    window_size: int,
    stride: int,
    min_contig_length: int,
    rng: np.random.Generator,
    max_windows_per_contig: int = 500,
    max_n_frac: float = 0.05,
) -> Iterator[str]:
    """Yield random sliding windows from a FASTA file.

    Random sub-sampling per contig prevents giant contigs from dominating.
    """
    for header, seq in _read_fasta(fasta_path):
        if len(seq) < window_size:
            continue
        n_possible = (len(seq) - window_size) // stride + 1
        indices = list(range(0, len(seq) - window_size + 1, stride))
        if len(indices) > max_windows_per_contig:
            indices = sorted(rng.choice(indices, size=max_windows_per_contig, replace=False).tolist())

        for start in indices:
            window = seq[start:start + window_size]
            if window.count("N") / len(window) > max_n_frac:
                continue
            yield window


def slice_windows_all(
    fasta_path: str,
    window_size: int,
    stride: int,
    min_contig_length: int,
    rng: np.random.Generator,
    max_n_frac: float = 0.05,
) -> Iterator[str]:
    """Yield ALL sliding windows from a FASTA file (no per-contig cap).

    Unlike slice_windows, this preserves every valid window from every contig,
    including ultra-long chromosomes. Use with an external shuffle buffer to
    prevent consecutive windows from the same contig.
    """
    for header, seq in _read_fasta(fasta_path):
        if len(seq) < window_size:
            continue
        for start in range(0, len(seq) - window_size + 1, stride):
            window = seq[start:start + window_size]
            if window.count("N") / len(window) > max_n_frac:
                continue
            yield window


def slice_windows_interleaved(
    fasta_path: str,
    window_size: int,
    stride: int,
    min_contig_length: int,
    rng: np.random.Generator,
    max_n_frac: float = 0.05,
) -> Iterator[str]:
    """Yield windows from a FASTA file with round-robin interleaving across contigs.

    Prevents ultra-long contigs from flooding the output by yielding one window
    per contig per round. Contigs are processed in random order each round.
    """
    # Read all contigs into memory
    contigs = [(header, seq) for header, seq in _read_fasta(fasta_path)
               if len(seq) >= window_size]

    if not contigs:
        return

    # Build start-position lists for each contig
    all_starts = []
    for cid, seq in contigs:
        starts = list(range(0, len(seq) - window_size + 1, stride))
        rng.shuffle(starts)  # shuffle starts within each contig
        all_starts.append((cid, seq, starts, 0))  # (cid, seq, start_list, cursor)

    # Round-robin yield until all contigs exhausted
    active = list(range(len(all_starts)))
    while active:
        rng.shuffle(active)  # shuffle contig order each round
        next_active = []
        for idx in active:
            cid, seq, starts, cursor = all_starts[idx]
            if cursor >= len(starts):
                continue
            start = starts[cursor]
            all_starts[idx] = (cid, seq, starts, cursor + 1)
            next_active.append(idx)
            window = seq[start:start + window_size]
            if window.count("N") / len(window) <= max_n_frac:
                yield window
        active = next_active


def slice_windows_from_long_contigs(
    contigs: list[tuple[str, str]],
    window_size: int,
    stride: int,
    rng: np.random.Generator,
    max_windows_per_contig: int = 200,
    max_n_frac: float = 0.05,
) -> Iterator[str]:
    """Same as slice_windows but operating on pre-loaded long contigs (Phase 2)."""
    for cid, seq in contigs:
        if len(seq) < window_size:
            continue
        indices = list(range(0, len(seq) - window_size + 1, stride))
        if len(indices) > max_windows_per_contig:
            indices = sorted(rng.choice(indices, size=max_windows_per_contig, replace=False).tolist())
        for start in indices:
            window = seq[start:start + window_size]
            if window.count("N") / len(window) > max_n_frac:
                continue
            yield window


def sample_pair_from_contig(
    contig_seq: str,
    window_size: int,
    max_distance: int,
    rng: np.random.Generator,
    n_pairs_per_contig: int = 5,
) -> list[tuple[str, str]]:
    """Sample positive pairs (two nearby windows) from a single contig.

    Two windows within max_distance bp are considered a positive pair.
    """
    if len(contig_seq) < window_size * 2:
        return []
    pairs = []
    for _ in range(n_pairs_per_contig):
        pos1 = int(rng.integers(0, len(contig_seq) - window_size))
        max_offset = min(max_distance, len(contig_seq) - pos1 - window_size)
        if max_offset < 1:
            continue
        offset = int(rng.integers(1, max_offset + 1))
        pos2 = pos1 + offset
        if pos2 + window_size > len(contig_seq):
            pos2 = len(contig_seq) - window_size
        w1 = contig_seq[pos1:pos1 + window_size]
        w2 = contig_seq[pos2:pos2 + window_size]
        if w1.count("N") / len(w1) <= 0.05 and w2.count("N") / len(w2) <= 0.05:
            pairs.append((w1, w2))
    return pairs


def sample_negative_from_other_genome(
    contigs_dict: dict[str, list[tuple[str, str]]],
    exclude_genome: str,
    window_size: int,
    rng: np.random.Generator,
) -> Optional[str]:
    """Sample a single window from a randomly chosen *other* genome."""
    other_genomes = [g for g in contigs_dict if g != exclude_genome]
    if not other_genomes:
        return None
    gid = rng.choice(other_genomes)
    contigs = contigs_dict[gid]
    if not contigs:
        return None
    cid, seq = contigs[int(rng.integers(0, len(contigs)))]
    if len(seq) < window_size:
        return None
    start = int(rng.integers(0, len(seq) - window_size))
    window = seq[start:start + window_size]
    return window if window.count("N") / len(window) <= 0.05 else None


def build_window_index(
    filtered_fasta_dir: str,
    window_size: int,
    stride: int,
    min_length: int = 50000,
    max_n_frac: float = 0.05,
) -> list[dict]:
    """Build a flat index of all valid non-overlapping windows for Phase 2 Map-style dataset.

    Each index entry: {"fasta_path": str, "start": int, "genome_id": str, "species_idx": int}

    The returned list can be used as the samples list for a torch Dataset.
    Assigns a contiguous species_idx (0..N-1) to each genome for use as the
    class label in supervised contrastive loss.
    """
    base = Path(filtered_fasta_dir)
    fasta_files = sorted(base.glob("*.fasta"))
    genome_ids = sorted([f.stem for f in fasta_files])
    genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

    index = []
    for fasta_path in fasta_files:
        genome_id = fasta_path.stem
        species_idx = genome_to_idx[genome_id]
        for header, seq in _read_fasta(str(fasta_path)):
            if len(seq) < window_size:
                continue
            for start in range(0, len(seq) - window_size + 1, stride):
                window = seq[start:start + window_size]
                if window.count("N") / len(window) > max_n_frac:
                    continue
                index.append({
                    "fasta_path": str(fasta_path),
                    "start": start,
                    "genome_id": genome_id,
                    "species_idx": species_idx,
                })
    return index
