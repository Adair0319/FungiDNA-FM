"""Minimal self-contained tests that run without downloaded weights or CUDA.

Covers:
  1. single-nucleotide tokenizer mapping and special-token IDs;
  2. model config (vocab sizes);
  3. task-name normalization (six canonical tasks);
  4. checkpoint loading with explicit shape checking (mock model).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_single_nt_mapping():
    from fungidna.data.tokenizer import NT_TO_ID, PAD_ID, CLS_ID, MASK_ID, UNK_ID, SEP_ID
    assert NT_TO_ID == {"A": 5, "C": 6, "G": 7, "T": 8, "N": 9}
    assert (PAD_ID, CLS_ID, MASK_ID, UNK_ID, SEP_ID) == (0, 1, 2, 3, 4)
    # "G" must not collide with [MASK]=2
    assert NT_TO_ID["G"] != MASK_ID


def test_config_vocab():
    from fungidna.model.config import FungiDNAConfig
    cfg = FungiDNAConfig()
    assert cfg.bpe_vocab_size == 4096
    assert cfg.nt_vocab_size == 10
    assert cfg.hidden_size == 768


def test_task_names():
    from fungidna.invoke.tasks import normalize_task, CANONICAL_TASKS
    assert set(CANONICAL_TASKS) == {"embedding", "taxonomy", "cds", "short_cds",
                                    "splice_site", "bgc_boundary"}
    assert normalize_task("representation") == "embedding"
    assert normalize_task("short CDS") == "short_cds"


def test_load_checkpoint_shape_check():
    from fungidna.invoke.handlers.base import load_checkpoint, WeightsNotFoundError

    class Mock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Module()
            self.embedding.nt_embed = torch.nn.Embedding(10, 8)   # (10, 8)
            self.embedding.bpe_embed = torch.nn.Embedding(4, 8)   # (4, 8)
            self.linear = torch.nn.Linear(8, 2)

    # checkpoint with nt_embed (5,8) — the known Unigram-vestigial mismatch
    ckpt = {"embedding.nt_embed.weight": torch.randn(5, 8),
            "embedding.bpe_embed.weight": torch.randn(4, 8),
            "linear.weight": torch.randn(2, 8),
            "linear.bias": torch.randn(2)}
    torch.save(ckpt, "/tmp/_mock_ckpt.pt")

    model = Mock()
    loaded, skipped, missing, unexpected = load_checkpoint(
        model, "/tmp/_mock_ckpt.pt", unwrap=None, skip_size_mismatch=("nt_embed",)
    )
    assert loaded == 3, loaded           # bpe_embed + linear.weight + linear.bias
    assert len(skipped) == 1             # nt_embed skipped
    assert skipped[0][0] == "embedding.nt_embed.weight"

    # without the allowed-skip, an unexpected shape mismatch must raise
    model2 = Mock()
    try:
        load_checkpoint(model2, "/tmp/_mock_ckpt.pt", unwrap=None, skip_size_mismatch=())
        raise AssertionError("expected WeightsNotFoundError for nt_embed mismatch")
    except WeightsNotFoundError:
        pass
    os.remove("/tmp/_mock_ckpt.pt")


def test_label_map():
    from fungidna.invoke.handlers.taxonomy import load_label_map
    counts = {"phylum": 6, "subphylum": 11, "class": 26, "order": 98, "family": 281}
    for rank, n in counts.items():
        name_to_id, id_to_name = load_label_map(rank)
        assert len(name_to_id) == n, (rank, len(name_to_id))
        assert len(id_to_name) == n
        assert set(name_to_id.values()) == set(range(n))  # IDs are contiguous 0..n-1
    _, id_to_name = load_label_map("phylum")
    assert id_to_name[0] == "Ascomycota"
    assert id_to_name.get(999) is None  # invalid id -> None


if __name__ == "__main__":
    test_single_nt_mapping()
    test_config_vocab()
    test_task_names()
    test_load_checkpoint_shape_check()
    test_label_map()
    print("all tests passed")
