"""Species classification dataset: BPE windows from genomes with 5-rank taxonomy labels."""
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from torch.utils.data import IterableDataset

from fungidna.data.tokenizer import DualTokenizer
from fungidna.data.slicing import slice_windows
from fungidna.data.taxonomy import TaxonomyDB


class SpeciesClassificationDataset(IterableDataset):
    """Iterable dataset for hierarchical species classification.

    Each genome contributes ~N windows, each labeled with the genome's
    Phylum, Subphylum, Class, Order, and Family indices.
    """

    def __init__(
        self,
        filtered_fasta_dir: str,
        tokenizer: DualTokenizer,
        taxonomy_db: TaxonomyDB,
        genome_ids: list[str],
        windows_per_genome: int = 100,
        window_sizes: list = None,
        fixed_window_size: Optional[int] = 2048,
        stride_fraction: float = 0.25,
        seed: int = 42,
    ):
        self.fasta_dir = Path(filtered_fasta_dir)
        self.tokenizer = tokenizer
        self.taxdb = taxonomy_db
        self.genome_ids = genome_ids
        self.windows_per_genome = windows_per_genome
        self.window_sizes = window_sizes or [512, 1024, 2048]
        self.fixed_window_size = fixed_window_size
        self.stride_fraction = stride_fraction
        self.seed = seed

        # Build label encoders from taxonomy data (all 735 genomes)
        self._build_label_encoders()

    def _build_label_encoders(self):
        """Build string→int mappings for each taxonomic rank."""
        all_genomes = self.taxdb.all_genomes

        ranks = ["phylum", "subphylum", "class", "order", "family"]
        getter = {
            "phylum": lambda g: self.taxdb.genome_to_phylum.get(g, "Unknown"),
            "subphylum": lambda g: self.taxdb.genome_to_subphylum.get(g, "Unknown"),
            "class": lambda g: self.taxdb.genome_to_class.get(g, "Unknown"),
            "order": lambda g: self.taxdb.genome_to_order.get(g, "Unknown"),
            "family": lambda g: self.taxdb.genome_to_family.get(g, "Unknown"),
        }

        self.label_maps = {}
        self.num_classes = {}
        self.incertae_sedis_ids = {}  # rank → label index for "Incertae sedis"
        for rank in ranks:
            unique_vals = sorted(set(getter[rank](g) for g in all_genomes))
            self.label_maps[rank] = {v: i for i, v in enumerate(unique_vals)}
            self.num_classes[rank] = len(unique_vals)
            # Record the index of "Incertae sedis" for loss masking
            if "Incertae sedis" in self.label_maps[rank]:
                self.incertae_sedis_ids[rank] = self.label_maps[rank]["Incertae sedis"]

    def _get_labels(self, genome_id: str) -> dict[str, int]:
        """Get integer labels for a genome at all 5 ranks."""
        return {
            "phylum": self.label_maps["phylum"][self.taxdb.genome_to_phylum.get(genome_id, "Unknown")],
            "subphylum": self.label_maps["subphylum"][self.taxdb.genome_to_subphylum.get(genome_id, "Unknown")],
            "class": self.label_maps["class"][self.taxdb.genome_to_class.get(genome_id, "Unknown")],
            "order": self.label_maps["order"][self.taxdb.genome_to_order.get(genome_id, "Unknown")],
            "family": self.label_maps["family"][self.taxdb.genome_to_family.get(genome_id, "Unknown")],
        }

    @property
    def num_genomes(self) -> int:
        return len(self.genome_ids)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split genomes across workers
            per_worker = max(1, len(self.genome_ids) // worker_info.num_workers)
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.genome_ids)
            my_genomes = self.genome_ids[start:end]
            rng = np.random.default_rng(self.seed + worker_info.id)
        else:
            my_genomes = self.genome_ids

        # Yield multiple windows per genome for balanced sampling
        for genome_id in my_genomes:
            fasta_path = self.fasta_dir / f"{genome_id}.fasta"
            if not fasta_path.exists():
                continue

            labels = self._get_labels(genome_id)
            window_size = self.fixed_window_size if self.fixed_window_size else int(rng.choice(self.window_sizes))
            stride = max(1, int(window_size * self.stride_fraction))

            count = 0
            for window in slice_windows(
                str(fasta_path), window_size, stride,
                min_contig_length=window_size,
                rng=rng,
                max_windows_per_contig=self.windows_per_genome,
            ):
                token_ids = self.tokenizer.encode_bpe(window, add_cls=True)
                if len(token_ids) < 10:
                    continue
                yield {
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "phylum":    labels["phylum"],
                    "subphylum": labels["subphylum"],
                    "class":     labels["class"],
                    "order":     labels["order"],
                    "family":    labels["family"],
                }
                count += 1
                if count >= self.windows_per_genome:
                    break
