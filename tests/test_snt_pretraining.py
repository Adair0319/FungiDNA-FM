"""Variant support for pretraining: model ties, dataset encoding and configs.

Note S2 describes two variants that share the backbone but differ in
tokenization. These tests pin the parts that must not drift:

* the MLM head vocabulary and its tied embedding table,
* the dataset's encoder and random-replacement range,
* the effective batch size implied by each Phase 2 config.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungidna.data.pretrain_dataset import Phase1MLMDataset, apply_span_masking
from fungidna.data.tokenizer import NT_VOCAB_SIZE
from fungidna.model.config import FungiDNAConfig
from fungidna.model.fungi_dna import FungiDNAForJointTraining, FungiDNAForMLM

REPO = Path(__file__).resolve().parent.parent


class TestMlmHeadTying(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = FungiDNAConfig()

    def test_unigram_head_ties_bpe_embedding(self):
        model = FungiDNAForMLM(self.config, token_type=0)
        self.assertEqual(model.vocab_size, self.config.bpe_vocab_size)
        self.assertEqual(model.lm_head.out_features, self.config.bpe_vocab_size)
        self.assertIs(model.lm_head.weight, model.backbone.embedding.bpe_embed.weight)

    def test_snt_head_ties_nt_embedding(self):
        model = FungiDNAForMLM(self.config, token_type=1)
        self.assertEqual(model.vocab_size, NT_VOCAB_SIZE)
        self.assertEqual(model.lm_head.out_features, NT_VOCAB_SIZE)
        self.assertIs(model.lm_head.weight, model.backbone.embedding.nt_embed.weight)

    def test_joint_head_follows_variant(self):
        uni = FungiDNAForJointTraining(self.config, token_type=0)
        snt = FungiDNAForJointTraining(self.config, token_type=1)
        self.assertEqual(uni.vocab_size, 4096)
        self.assertIs(uni.lm_head.weight, uni.backbone.embedding.bpe_embed.weight)
        self.assertEqual(snt.vocab_size, NT_VOCAB_SIZE)
        self.assertIs(snt.lm_head.weight, snt.backbone.embedding.nt_embed.weight)


class TestDatasetVariant(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        (Path(self.tmp.name) / "dummy.fasta").write_text(">x\nACGT\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, token_type):
        class _Tok:
            bpe_vocab_size = 4096

        return Phase1MLMDataset(filtered_fasta_dir=self.tmp.name, tokenizer=_Tok(),
                                token_type=token_type)

    def test_unigram_dataset_uses_bpe_range(self):
        ds = self._dataset(0)
        self.assertEqual(ds.vocab_size, 4096)
        self.assertEqual(ds.token_type, 0)

    def test_snt_dataset_restricts_replacement_to_nt_vocab(self):
        ds = self._dataset(1)
        self.assertEqual(ds.vocab_size, NT_VOCAB_SIZE)

    def test_span_masking_respects_vocab_size(self):
        rng = np.random.default_rng(0)
        ids = torch.arange(100) % 5 + 5          # valid nucleotide ids 5-9
        for _ in range(50):
            masked, labels = apply_span_masking(
                ids.clone(), mask_rate=0.15, span_mask_fraction=0.30,
                vocab_size=NT_VOCAB_SIZE, rng=rng,
            )
            self.assertLess(int(masked.max()), NT_VOCAB_SIZE)


class TestPhase2Configs(unittest.TestCase):
    """Effective batch = GPUs (8) × n_species × k_per_species × grad_accum."""

    def _load(self, name):
        return yaml.safe_load((REPO / "configs" / name).read_text())

    def _effective(self, cfg, gpus=8):
        return gpus * cfg["n_species"] * cfg["k_per_species"] * cfg["gradient_accumulation_steps"]

    def test_unigram_released_run_is_2048(self):
        cfg = self._load("pretrain_phase2_unigram.yaml")
        self.assertEqual(cfg["variant"], "unigram")
        self.assertEqual(cfg["window_size"], 8192)
        self.assertEqual(self._effective(cfg), 2048)
        self.assertEqual(cfg["resume_step"], 10000)
        self.assertTrue(str(cfg["resume_backbone"]).endswith("backbone-10000.pt"))

    def test_snt_final_run_is_2048(self):
        cfg = self._load("pretrain_phase2_snt.yaml")
        self.assertEqual(cfg["variant"], "snt")
        self.assertEqual(cfg["window_size"], 4096)
        self.assertEqual(self._effective(cfg), 2048)
        self.assertEqual(cfg["resume_step"], 5500)

    def test_snt_first_stage_is_3328(self):
        cfg = self._load("pretrain_phase2_snt_stage1.yaml")
        self.assertEqual(cfg["variant"], "snt")
        self.assertEqual(self._effective(cfg), 3328)
        self.assertIn(cfg["resume_step"], (0, None))

    def test_phase1_configs_declare_variant(self):
        for name, variant in (("pretrain_phase1.yaml", "unigram"),
                              ("pretrain_phase1_snt.yaml", "snt")):
            with self.subTest(name=name):
                cfg = self._load(name)
                self.assertEqual(cfg["variant"], variant)
                self.assertEqual(cfg["window_sizes"], [1024, 2048, 4096])
                self.assertEqual(cfg["mask_rate"], 0.15)
                self.assertEqual(cfg["span_mask_fraction"], 0.30)
                self.assertEqual(cfg["total_steps"], 200000)


if __name__ == "__main__":
    unittest.main()
