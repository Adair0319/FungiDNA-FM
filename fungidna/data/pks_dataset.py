"""P-K-S structured batch sampling for SupCon training."""
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Iterator


class PKSDataset(Dataset):
    """Map-style dataset wrapping a parquet DataFrame for BPE tokenization.

    Each item returns (input_ids, label, genome_id).
    input_ids is a pre-tokenized list[int] (tokenized on __getitem__ for memory).
    """

    def __init__(self, df, tokenizer):
        self.sequences = df["sequence"].tolist()
        self.labels = df["label"].tolist()
        self.genome_ids = df["genome_id"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode_bpe(self.sequences[idx], add_cls=True)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": self.labels[idx],
            "genome_id": self.genome_ids[idx],
        }


class PKSBatchSampler:
    """Yield batches satisfying P-K-S: P classes, K genomes/class, S windows/genome.

    Each batch = P x K x S samples. Small classes (genomes < K) use replacement.
    """

    def __init__(
        self,
        labels: list[int],
        genome_ids: list[str],
        P: int,
        K: int,
        S: int,
        seed: int = 42,
    ):
        self.P = P
        self.K = K
        self.S = S
        self.rng = np.random.default_rng(seed)

        # label -> list of unique genome IDs
        self.label_to_genomes = defaultdict(set)
        for lbl, gid in zip(labels, genome_ids):
            self.label_to_genomes[lbl].add(gid)

        # genome -> list of sample indices
        self.genome_to_indices = defaultdict(list)
        for idx, gid in enumerate(genome_ids):
            self.genome_to_indices[gid].append(idx)

        self.all_labels = sorted(self.label_to_genomes.keys())
        assert len(self.all_labels) >= P, f"Need at least {P} classes, got {len(self.all_labels)}"

    def __iter__(self) -> Iterator[list[int]]:
        return self

    def __next__(self) -> list[int]:
        # 1. Select P classes randomly
        chosen_labels = self.rng.choice(self.all_labels, size=self.P, replace=False)

        batch_indices = []
        for lbl in chosen_labels:
            genome_pool = list(self.label_to_genomes[lbl])
            # 2. Select K genomes (with replacement if needed)
            chosen_genomes = self.rng.choice(genome_pool, size=self.K, replace=True)

            for gid in chosen_genomes:
                sample_pool = self.genome_to_indices[gid]
                # 3. Select S windows per genome
                chosen = self.rng.choice(sample_pool, size=self.S, replace=len(sample_pool) < self.S)
                batch_indices.extend(chosen.tolist())

        self.rng.shuffle(batch_indices)
        return batch_indices

    def __len__(self):
        # Approximate: ~(n_labels choose P) combinations
        return 10000  # Per-epoch steps (set high, controlled by steps_per_epoch)


def pks_collate_fn(batch: list[dict]) -> dict:
    """Pad input_ids to batch max length."""
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0
    )
    return {
        "input_ids": input_ids,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "genome_id": [item["genome_id"] for item in batch],
    }
