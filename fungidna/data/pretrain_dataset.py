"""Pretraining datasets: Phase 1 MLM (BPE + span masking) and Phase 2 (Contrastive + Joint) datasets."""
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset, Sampler

from fungidna.data.tokenizer import DualTokenizer, MASK_ID, CLS_ID, PAD_ID
from fungidna.data.slicing import slice_windows_from_long_contigs, sample_pair_from_contig, sample_negative_from_other_genome
from fungidna.data.qc_filter import collect_long_contigs, _read_fasta


class Phase1MLMDataset(IterableDataset):
    """Phase 1: BPE-tokenized MLM with span-aware masking (no [CLS]).

    Uses interleaved slicing + per-worker shuffle buffer to:
    - Preserve ALL contig windows (no max_windows cap)
    - Prevent consecutive batches from same contig/species
    - Decouple shuffling from filesystem order
    """

    def __init__(
        self,
        filtered_fasta_dir: str,
        tokenizer: DualTokenizer,
        window_sizes: list = None,
        stride_fraction: float = 0.25,
        mask_rate: float = 0.15,
        span_mask_fraction: float = 0.30,
        span_lambda: float = 3.0,
        max_span_len: int = 10,
        shuffle_buffer_size: int = 8000,
        use_interleave: bool = True,
        seed: int = 42,
    ):
        self.fasta_dir = Path(filtered_fasta_dir)
        self.tokenizer = tokenizer
        self.window_sizes = window_sizes or [1024, 2048, 4096]
        self.stride_fraction = stride_fraction
        self.mask_rate = mask_rate
        self.span_mask_fraction = span_mask_fraction
        self.span_lambda = span_lambda
        self.max_span_len = max_span_len
        self.shuffle_buffer_size = shuffle_buffer_size
        self.use_interleave = use_interleave
        self.seed = seed
        self.fasta_files = sorted(self.fasta_dir.glob("*.fasta"))
        if not self.fasta_files:
            raise FileNotFoundError(f"No .fasta files found in {filtered_fasta_dir}")

    def _mask_tokens(self, token_ids: torch.Tensor, rng: np.random.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Span-aware masking (no [CLS] — all positions are DNA tokens).

        Args:
            rng: Optional numpy Generator for deterministic masking.
                When None (default), falls back to global np.random state.
        """
        n = len(token_ids)
        labels = torch.full((n,), -100, dtype=torch.long)
        num_tokens = n

        num_to_mask = max(1, int(num_tokens * self.mask_rate))
        num_span_mask = max(1, int(num_to_mask * self.span_mask_fraction))
        num_independent = num_to_mask - num_span_mask

        _geometric = (lambda p: rng.geometric(p)) if rng is not None else (lambda p: np.random.geometric(p))

        mask_set = set()
        i = 0  # start from position 0 (no [CLS] to skip)
        while len(mask_set) < num_span_mask and i < n:
            span_len = min(_geometric(1.0 / self.span_lambda), self.max_span_len)
            span_len = min(span_len, n - i, num_span_mask - len(mask_set))
            for j in range(i, min(i + span_len, n)):
                mask_set.add(j)
            i += span_len + _geometric(1.0 / self.span_lambda)

        remaining = sorted(set(range(n)) - mask_set)
        if remaining and num_independent > 0:
            if rng is not None:
                chosen = rng.choice(remaining, size=min(num_independent, len(remaining)), replace=False)
            else:
                chosen = np.random.choice(remaining, size=min(num_independent, len(remaining)), replace=False)
            for c in chosen:
                mask_set.add(c)

        masked_ids = token_ids.clone()
        for pos in mask_set:
            labels[pos] = token_ids[pos]
            if rng is not None:
                rand = rng.uniform()
            else:
                rand = torch.rand(1).item()
            if rand < 0.80:
                masked_ids[pos] = MASK_ID
            elif rand < 0.90:
                if rng is not None:
                    masked_ids[pos] = int(rng.integers(5, self.tokenizer.bpe_vocab_size, size=1)[0])
                else:
                    masked_ids[pos] = torch.randint(5, self.tokenizer.bpe_vocab_size, (1,)).item()

        return masked_ids, labels

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = max(1, len(self.fasta_files) // worker_info.num_workers)
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.fasta_files)
            my_files = self.fasta_files[start:end]
            rng = np.random.default_rng(self.seed + worker_info.id)
        else:
            my_files = self.fasta_files

        buffer = []
        file_order = rng.permutation(len(my_files))

        for file_idx in file_order:
            fasta_path = str(my_files[file_idx])
            window_size = int(rng.choice(self.window_sizes))
            stride = max(1, int(window_size * self.stride_fraction))

            if self.use_interleave:
                from fungidna.data.slicing import slice_windows_interleaved
                window_iter = slice_windows_interleaved(fasta_path, window_size, stride, 5000, rng)
            else:
                from fungidna.data.slicing import slice_windows_all
                window_iter = slice_windows_all(fasta_path, window_size, stride, 5000, rng)

            for window in window_iter:
                # BPE tokenize — NO [CLS]
                token_ids = torch.tensor(
                    self.tokenizer.encode_bpe(window, add_cls=False),
                    dtype=torch.long,
                )
                if len(token_ids) < 10:
                    continue
                masked_ids, labels = self._mask_tokens(token_ids, rng=rng)

                buffer.append({"input_ids": masked_ids, "labels": labels})

                if len(buffer) >= self.shuffle_buffer_size:
                    # Deep shuffle the buffer
                    perm = rng.permutation(len(buffer))
                    for idx in perm:
                        yield buffer[idx]
                    buffer = []

        # Drain remaining buffer
        if buffer:
            perm = rng.permutation(len(buffer))
            for idx in perm:
                yield buffer[idx]


class Phase2ContrastiveDataset(IterableDataset):
    """Phase 2: Contrastive + NSP + Species-ID pretraining with single-nt tokens.

    Yields batches of (anchor, positive, negative) triplets for contrastive learning,
    plus NSP and species-ID auxiliary labels.
    """

    def __init__(
        self,
        filtered_fasta_dir: str,
        tokenizer: DualTokenizer,
        window_sizes: list = None,
        positive_distance: int = 50000,
        seed: int = 42,
    ):
        self.fasta_dir = Path(filtered_fasta_dir)
        self.tokenizer = tokenizer
        self.window_sizes = window_sizes or [8192, 16384, 32768]
        self.positive_distance = positive_distance
        self.seed = seed
        self.fasta_files = sorted(self.fasta_dir.glob("*.fasta"))

        # Pre-load long contigs (>= 50Kb) for all genomes
        self.long_contigs = collect_long_contigs(filtered_fasta_dir, min_length=50000)
        self.genome_ids = sorted(self.long_contigs.keys())
        if not self.genome_ids:
            raise RuntimeError(f"No genomes with contigs >= 50000 bp found in {filtered_fasta_dir}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rng = np.random.default_rng(self.seed + worker_info.id)

        # Iterate over randomly ordered genomes
        genome_order = rng.permutation(len(self.genome_ids))
        for gidx in genome_order:
            gid = self.genome_ids[gidx]
            contigs = self.long_contigs[gid]
            if not contigs:
                continue

            window_size = int(rng.choice(self.window_sizes))
            for cid, seq in contigs:
                if len(seq) < window_size * 2:
                    continue
                # Positive pairs from same contig
                pairs = sample_pair_from_contig(seq, window_size, self.positive_distance, rng, n_pairs_per_contig=3)
                for w1, w2 in pairs:
                    # Negative from other genome
                    w3 = sample_negative_from_other_genome(self.long_contigs, gid, window_size, rng)
                    if w3 is None:
                        continue

                    # NSP: consecutive segments from same contig
                    nsp_start1 = int(rng.integers(0, len(seq) - window_size * 2))
                    s1 = seq[nsp_start1:nsp_start1 + window_size]
                    s2 = seq[nsp_start1 + window_size:nsp_start1 + window_size * 2]

                    yield {
                        "anchor_ids": self.tokenizer.single_nt_to_tensor(w1),
                        "positive_ids": self.tokenizer.single_nt_to_tensor(w2),
                        "negative_ids": self.tokenizer.single_nt_to_tensor(w3),
                        "nsp_seg1": self.tokenizer.single_nt_to_tensor(s1),
                        "nsp_seg2": self.tokenizer.single_nt_to_tensor(s2),
                        "nsp_label": 0,   # contiguous
                        "species_id": gidx,
                    }


def apply_span_masking(
    token_ids: torch.Tensor,
    mask_rate: float,
    span_mask_fraction: float,
    span_lambda: float = 3.0,
    max_span_len: int = 10,
    bpe_vocab_size: int = 4096,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stateless span-aware masking. Returns (masked_ids, labels). No [CLS] handling.

    Args:
        rng: Optional numpy Generator for deterministic masking. When provided,
            uses rng.geometric() instead of np.random.geometric() and
            torch.tensor(rng.integers(...)) for random token replacement.
            When None (default), falls back to global np.random state.
    """
    n = len(token_ids)
    labels = torch.full((n,), -100, dtype=torch.long)
    num_to_mask = max(1, int(n * mask_rate))
    num_span_mask = max(1, int(num_to_mask * span_mask_fraction))
    num_independent = num_to_mask - num_span_mask

    _geometric = (lambda p: rng.geometric(p)) if rng is not None else (lambda p: np.random.geometric(p))

    mask_set = set()
    i = 0
    while len(mask_set) < num_span_mask and i < n:
        span_len = min(_geometric(1.0 / span_lambda), max_span_len)
        span_len = min(span_len, n - i, num_span_mask - len(mask_set))
        for j in range(i, min(i + span_len, n)):
            mask_set.add(j)
        i += span_len + _geometric(1.0 / span_lambda)

    remaining = sorted(set(range(n)) - mask_set)
    if remaining and num_independent > 0:
        if rng is not None:
            chosen = rng.choice(remaining, size=min(num_independent, len(remaining)), replace=False)
        else:
            chosen = np.random.choice(remaining, size=min(num_independent, len(remaining)), replace=False)
        for c in chosen:
            mask_set.add(c)

    masked_ids = token_ids.clone()
    for pos in mask_set:
        labels[pos] = token_ids[pos]
        if rng is not None:
            rand = rng.uniform()
        else:
            rand = torch.rand(1).item()
        if rand < 0.80:
            masked_ids[pos] = MASK_ID
        elif rand < 0.90:
            if rng is not None:
                masked_ids[pos] = int(rng.integers(5, bpe_vocab_size, size=1)[0])
            else:
                masked_ids[pos] = torch.randint(5, bpe_vocab_size, (1,)).item()

    return masked_ids, labels


class Phase2JointDataset(Dataset):
    """Phase 2 Joint Training: Map-style dataset with fixed window size.

    Pre-builds a flat index of all valid windows across all genomes.
    Supports NxK structured sampling via NKBatchSampler.

    Each sample: window of ``window_size`` bp -> BPE tokenize -> padding/truncation
    to ``max_seq_len`` tokens -> clean_ids (no masking). Masking is applied
    at collate time to produce the MLM path.
    """

    def __init__(
        self,
        filtered_fasta_dir: str,
        tokenizer: DualTokenizer,
        window_size: int = 8192,
        stride: int = 8192,
        max_seq_len: int = 4096,
        min_contig_length: int = 50000,
        seed: int = 42,
    ):
        self.fasta_dir = Path(filtered_fasta_dir)
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.max_seq_len = max_seq_len
        self.seed = seed

        # Build flat index
        from fungidna.data.slicing import build_window_index
        self.index = build_window_index(
            filtered_fasta_dir, window_size, stride, min_contig_length
        )
        if not self.index:
            raise RuntimeError(f"No valid windows found in {filtered_fasta_dir}")

        # Build species -> indices mapping
        self.species_to_indices: dict[int, list[int]] = {}
        for i, entry in enumerate(self.index):
            sp = entry["species_idx"]
            if sp not in self.species_to_indices:
                self.species_to_indices[sp] = []
            self.species_to_indices[sp].append(i)

        self.num_species = len(self.species_to_indices)
        self._contig_cache: dict[str, list[tuple[str, str]]] = {}
        print(f"Phase2JointDataset: {len(self.index)} windows across {self.num_species} species, "
              f"window={window_size}bp, max_seq_len={max_seq_len} tokens")

    def __len__(self) -> int:
        return len(self.index)

    def _get_contigs(self, fasta_path: str) -> list[tuple[str, str]]:
        """Load and cache contigs for a FASTA file. Returns list of (header, seq)."""
        if fasta_path not in self._contig_cache:
            contigs = list(_read_fasta(fasta_path))
            if not contigs:
                raise RuntimeError(f"No contigs found in {fasta_path}")
            self._contig_cache[fasta_path] = contigs
        return self._contig_cache[fasta_path]

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        fasta_path = entry["fasta_path"]
        start = entry["start"]
        species_idx = entry["species_idx"]

        # Read the window from cached contigs
        contigs = self._get_contigs(fasta_path)
        window = None
        for header, seq in contigs:
            if len(seq) >= start + self.window_size:
                window = seq[start:start + self.window_size]
                break
        if window is None:
            raise RuntimeError(
                f"No contig long enough in {fasta_path} for start={start}, "
                f"window_size={self.window_size}. Contig lengths: "
                f"{[len(s) for _, s in contigs]}"
            )

        # BPE tokenize, NO [CLS]
        token_ids = self.tokenizer.encode_bpe(window, add_cls=False)
        token_ids = token_ids[:self.max_seq_len]  # truncate if too long

        # Convert to tensor and pad
        clean_ids = torch.full((self.max_seq_len,), PAD_ID, dtype=torch.long)
        clean_ids[:len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)

        return {
            "clean_ids": clean_ids,
            "species_idx": species_idx,
        }


class NKBatchSampler(Sampler):
    """NxK structured batch sampler for supervised contrastive learning.

    Each batch: N distinct species x K samples each = N*K total.

    Args:
        species_to_indices: dict mapping species_idx -> list of sample indices
        n_species: number of species per batch (N)
        k_per_species: number of samples per species (K)
        seed: random seed
    """

    def __init__(self, species_to_indices: dict, n_species: int = 8,
                 k_per_species: int = 4, seed: int = 42):
        self.species_to_indices = species_to_indices
        self.species_list = sorted(species_to_indices.keys())
        self.n_species = n_species
        self.k = k_per_species
        self.seed = seed
        if len(self.species_list) < n_species:
            raise ValueError(f"Only {len(self.species_list)} species available, need {n_species}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        n_batches = 100000  # effectively infinite for training
        for _ in range(n_batches):
            chosen_species = rng.choice(self.species_list, size=self.n_species, replace=False)
            batch_indices = []
            for sp in chosen_species:
                pool = self.species_to_indices[sp]
                if len(pool) < self.k:
                    sampled = rng.choice(pool, size=self.k, replace=True).tolist()
                else:
                    sampled = rng.choice(pool, size=self.k, replace=False).tolist()
                batch_indices.extend(sampled)
            rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return 100000  # arbitrary; used by DataLoader for epoch boundary


def phase2_collate_fn(batch: list[dict], rng: np.random.Generator | None = None) -> dict:
    """Collate Phase 2 batch: masked_ids (model input), mlm_labels, species labels, attention_mask.

    Each item in batch is from Phase2JointDataset.__getitem__:
        {"clean_ids": Tensor[max_seq_len], "species_idx": int}

    Applies span-aware masking to generate masked input_ids,
    which serve both MLM and CL heads in a single backbone forward.

    Args:
        rng: Optional numpy Generator passed through to apply_span_masking for
            deterministic masking. When None, uses global random state.
    """
    clean_ids = torch.stack([item["clean_ids"] for item in batch])  # [batch, max_seq_len]
    species_labels = torch.tensor([item["species_idx"] for item in batch], dtype=torch.long)

    # Generate masked versions on-the-fly
    masked_ids_list = []
    labels_list = []
    for i in range(len(batch)):
        tok_ids = batch[i]["clean_ids"]
        real_len = (tok_ids != PAD_ID).sum().item()
        masked, lbls = apply_span_masking(tok_ids[:real_len], mask_rate=0.15, span_mask_fraction=0.30, rng=rng)
        # Pad masked_ids and labels back to max_seq_len
        masked_padded = torch.full_like(tok_ids, PAD_ID)
        masked_padded[:real_len] = masked
        labels_padded = torch.full_like(tok_ids, -100)
        labels_padded[:real_len] = lbls
        masked_ids_list.append(masked_padded)
        labels_list.append(labels_padded)

    masked_ids = torch.stack(masked_ids_list)
    mlm_labels = torch.stack(labels_list)

    # attention_mask: 1 for real (non-pad) tokens, 0 for padding — computed from masked_ids
    attention_mask = (masked_ids != PAD_ID).to(dtype=torch.long)

    return {
        "input_ids": masked_ids,
        "mlm_labels": mlm_labels,
        "species_labels": species_labels,
        "attention_mask": attention_mask,
    }
