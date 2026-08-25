"""Taxonomy metadata parsing and phylogeny-aware data splits."""
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional
import pandas as pd


class TaxonomyDB:
    """Build taxonomy hierarchy from fungi_classification.xlsx.

    Columns: portal, Phylum name, Subphylum name, Class name, Order name, Family name
    'portal' is the genome/folder identifier (e.g. 'Aaoar1', 'Entmec1').
    """

    def __init__(self, xlsx_path: str):
        df = pd.read_excel(xlsx_path)
        self.genome_to_phylum = {}
        self.genome_to_subphylum = {}
        self.genome_to_class = {}
        self.genome_to_order = {}
        self.genome_to_family = {}

        self.phylum_to_genomes = defaultdict(list)
        self.subphylum_to_genomes = defaultdict(list)
        self.class_to_genomes = defaultdict(list)
        self.order_to_genomes = defaultdict(list)
        self.family_to_genomes = defaultdict(list)

        for _, row in df.iterrows():
            gid = str(row["portal"])
            phylum = str(row.get("Phylum name", "Unknown"))
            subphylum = str(row.get("Subphylum name", "Unknown"))
            cls = str(row.get("Class name", "Unknown"))
            order = str(row.get("Order name", "Unknown"))
            family = str(row.get("Family name", "Unknown"))

            self.genome_to_phylum[gid] = phylum
            self.genome_to_subphylum[gid] = subphylum
            self.genome_to_class[gid] = cls
            self.genome_to_order[gid] = order
            self.genome_to_family[gid] = family

            self.phylum_to_genomes[phylum].append(gid)
            self.subphylum_to_genomes[subphylum].append(gid)
            self.class_to_genomes[cls].append(gid)
            self.order_to_genomes[order].append(gid)
            self.family_to_genomes[family].append(gid)

        self.all_genomes = list(self.genome_to_phylum.keys())
        self._rank_maps = {
            "phylum": (self.genome_to_phylum, self.phylum_to_genomes),
            "subphylum": (self.genome_to_subphylum, self.subphylum_to_genomes),
            "class": (self.genome_to_class, self.class_to_genomes),
            "order": (self.genome_to_order, self.order_to_genomes),
            "family": (self.genome_to_family, self.family_to_genomes),
        }

    def __len__(self):
        return len(self.all_genomes)

    def split_by_rank(
        self,
        genome_ids: list[str],
        rank: str = "order",
        train_frac: float = 0.70,
        val_frac: float = 0.10,
        test_frac: float = 0.20,
        seed: int = 42,
    ) -> tuple[list[str], list[str], list[str]]:
        """Split genome IDs ensuring no rank-level group straddles train/val/test.

        For genus-level split when no genus column exists, use family as closest proxy.
        Recommended primary split: rank='order' (best balance for 735 genomes).
        """
        if rank not in self._rank_maps:
            raise ValueError(f"Unknown rank '{rank}'. Choose from: {list(self._rank_maps.keys())}")
        genome_to_group, group_map = self._rank_maps[rank]

        rng = random.Random(seed)
        groups = sorted(set(genome_to_group[g] for g in genome_ids if g in genome_to_group))
        rng.shuffle(groups)

        n = len(groups)
        train_end = max(1, int(n * train_frac))
        val_end = train_end + max(1, int(n * val_frac))

        train_groups = set(groups[:train_end])
        val_groups = set(groups[train_end:val_end])
        test_groups = set(groups[val_end:])

        train = [g for g in genome_ids if g in genome_to_group and genome_to_group[g] in train_groups]
        val = [g for g in genome_ids if g in genome_to_group and genome_to_group[g] in val_groups]
        test = [g for g in genome_ids if g in genome_to_group and genome_to_group[g] in test_groups]
        return train, val, test

    def split_random(
        self,
        genome_ids: list[str],
        train_frac: float = 0.70,
        val_frac: float = 0.10,
        test_frac: float = 0.20,
        seed: int = 42,
    ) -> tuple[list[str], list[str], list[str]]:
        """Random genome-level split (no taxonomic constraint)."""
        rng = random.Random(seed)
        shuffled = list(genome_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = max(1, int(n * train_frac))
        val_end = train_end + max(1, int(n * val_frac))
        return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]

    def split_stratified(
        self,
        rank: str = "phylum",
        train_frac: float = 0.80,
        seed: int = 42,
        exclude_labels: set[str] | None = None,
    ) -> tuple[list[str], list[str], dict[int, int | None]]:
        """Stratified genome-level split with special handling for tiny categories.

        Rules:
          - N = 1: all genomes → train, label → ignore_index in evaluation
          - N = 2–4: guarantee ≥1 genome in test, rest in train
          - N ≥ 5: standard train_frac / (1-train_frac) random split.
            Non-integer test counts: <0.5 floor, >=0.5 ceil.

        Args:
            rank: taxonomic rank to split by ("phylum" or "subphylum")
            train_frac: fraction for training (default 0.80)
            seed: random seed
            exclude_labels: set of label strings to exclude entirely (e.g. {"Incertae sedis"})

        Returns:
            train_ids, test_ids, ignore_labels dict: label_index → None for
            labels with no test representation (use ignore_index in evaluation)
        """
        if rank not in self._rank_maps:
            raise ValueError(f"Unknown rank '{rank}'. Choose from: {list(self._rank_maps.keys())}")
        genome_to_group, group_map = self._rank_maps[rank]

        rng = random.Random(seed)

        # 1. Build per-label genome lists, excluding unwanted labels
        label_to_genomes = {}
        for g in self.all_genomes:
            label = genome_to_group.get(g, "Unknown")
            if exclude_labels and label in exclude_labels:
                continue
            label_to_genomes.setdefault(label, []).append(g)

        # 2. Build label map (sorted for deterministic ordering)
        all_labels = sorted(label_to_genomes.keys())
        label_map = {lbl: i for i, lbl in enumerate(all_labels)}

        # 3. Split per label
        train_ids, test_ids = [], []
        ignore_labels = {}  # label_index → None (for labels with no test presence)

        for label in all_labels:
            genomes = label_to_genomes[label]
            n = len(genomes)
            rng.shuffle(genomes)
            label_idx = label_map[label]

            if n == 1:
                # Singleton: all train, mark as ignore in test eval
                train_ids.extend(genomes)
                ignore_labels[label_idx] = None
            elif n <= 4:
                # Tiny: guarantee 1 test
                test_ids.append(genomes[0])
                train_ids.extend(genomes[1:])
            else:
                # Normal: stratified 80/20
                n_test_raw = n * (1 - train_frac)
                if n_test_raw < 0.5:
                    n_test = 0
                    ignore_labels[label_idx] = None
                elif n_test_raw < 1.5:
                    n_test = 1
                else:
                    n_test = int(round(n_test_raw))
                n_test = max(1, n_test) if n_test_raw >= 0.5 else 0
                if n_test == 0:
                    train_ids.extend(genomes)
                    ignore_labels[label_idx] = None
                else:
                    test_ids.extend(genomes[:n_test])
                    train_ids.extend(genomes[n_test:])

        # Sort for deterministic output
        train_ids.sort()
        test_ids.sort()

        return train_ids, test_ids, ignore_labels

    def get_taxonomy_str(self, genome_id: str) -> str:
        """Return a compact taxonomy string for knowledge context."""
        phy = self.genome_to_phylum.get(genome_id, "Unknown")
        sub = self.genome_to_subphylum.get(genome_id, "Unknown")
        cls = self.genome_to_class.get(genome_id, "Unknown")
        order = self.genome_to_order.get(genome_id, "Unknown")
        family = self.genome_to_family.get(genome_id, "Unknown")
        return f"Phylum:{phy} | Subphylum:{sub} | Class:{cls} | Order:{order} | Family:{family}"
