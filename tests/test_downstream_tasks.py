"""End-to-end inference tests for the five downstream tasks, both variants.

Each task must load its own variant's backbone plus the released task
checkpoint and return a prediction for a single sequence. The expected head
and pooling per variant were established from the checkpoints and the training
code:

============  =====================================  ==========================
task          Unigram                                SNT
============  =====================================  ==========================
cds           ClassificationHead(256), hidden[:,0]   MeanPoolHead(128), hidden[:,1:].mean
short_cds     same as cds                            same as cds
splice_site   same as cds                            same as cds
bgc_boundary  MeanPoolHead(128), hidden[:,1:].mean   MeanPoolHead(128), hidden[:,1:].mean
taxonomy      Linear+BN MLP over mean(all)           LayerNorm+Linear MLP over mean(1:)
============  =====================================  ==========================

Real checkpoints are required; set FUNGIDNA_TEST_UNIGRAM_TASKS / _SNT_TASKS
(FUNGIDNA_TEST_WEIGHTS dir) or rely on the local defaults. Without them the
tests are skipped rather than run against random weights.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

_ROOT = Path(os.environ.get(
    "FUNGIDNA_TEST_WEIGHTS",
    "/home/lty/yy_projects/fungi_project/fungi_dna_model",
))
TOKENIZER = Path(os.environ.get("FUNGIDNA_TEST_TOKENIZER", _ROOT / "data/processed/bpe_fungi.model"))

CHECKPOINTS = {
    "unigram": {
        "backbone": _ROOT / "checkpoints/phase2_joint/backbone_final.pt",
        "cds": _ROOT / "checkpoints/extended_finetune_cds_intergenic_v2_50k/fungidna_bpe/full/fold_0/best_model.pt",
        "short_cds": _ROOT / "checkpoints/sorf_v3/sorf_bpe_finetune/best_model.pt",
        "splice_site": _ROOT / "checkpoints/splice_v3/bpe_ft/best_model.pt",
        "bgc_boundary": _ROOT / "checkpoints/bgc_boundary_v2/funbgc10_bpe_ft/best_model.pt",
        "taxonomy_dir": _ROOT / "checkpoints/phase2_joint_eval_final",
    },
    "snt": {
        "backbone": _ROOT / "checkpoints/phase2_joint_single_nt/backbone_final.pt",
        "cds": _ROOT / "checkpoints/extended_finetune_cds_intergenic_v2_50k/fungidna_snt/full/fold_0/best_model.pt",
        "short_cds": _ROOT / "checkpoints/sorf_v3/sorf_snt_finetune/best_model.pt",
        "splice_site": _ROOT / "checkpoints/splice_v3/snt_ft/best_model.pt",
        "bgc_boundary": _ROOT / "checkpoints/bgc_boundary_v2/funbgc10_snt_ft/best_model.pt",
        "taxonomy_dir": _ROOT / "checkpoints/taxonomy_single_nt_retrain",
    },
}

RANKS = ("phylum", "subphylum", "class", "order", "family")
AVAILABLE = TOKENIZER.exists() and all(
    (c["backbone"]).exists() and c["cds"].exists() and c["taxonomy_dir"].exists()
    for c in CHECKPOINTS.values()
)
CUDA = torch.cuda.is_available()

SEQ = "ATGGCTAGCTAGCTAGGCTAACCGGTTACGTAGCTAGCATCGATCGATCGTAGCTAGCTAGCTA"


def _write_weights_yaml() -> str:
    lines = [f"tokenizer: {TOKENIZER}"]
    for variant, c in CHECKPOINTS.items():
        lines.append(f"{variant}:")
        for key in ("backbone", "cds", "short_cds", "splice_site", "bgc_boundary"):
            lines.append(f"  {key}: {c[key]}")
        for rank in RANKS:
            lines.append(f"  taxonomy_{rank}: {c['taxonomy_dir']}/{rank}/fold1/mlp_best.pt")
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="fungidna_weights_")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


@unittest.skipUnless(CUDA, "model requires CUDA (flash-attn / mamba-ssm kernels)")
@unittest.skipUnless(AVAILABLE, "real checkpoints not available")
class TestDownstreamTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _write_weights_yaml()
        os.environ["FUNGIDNA_WEIGHTS_CONFIG"] = cls.config
        f = tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False)
        f.write(f">seq1\n{SEQ}\n")
        f.close()
        cls.fasta = f.name
        from fungidna.invoke import run_task

        cls.run_task = staticmethod(run_task)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.config)
        os.unlink(cls.fasta)
        os.environ.pop("FUNGIDNA_WEIGHTS_CONFIG", None)

    def _run(self, task, variant, rank="phylum"):
        return self.run_task(file_path=self.fasta, task=task, variant=variant, rank=rank)

    def test_cds_both_variants(self):
        for variant in ("unigram", "snt"):
            pred = self._run("cds", variant).results["predictions"][">seq1"]
            self.assertIn(pred["label"], ("cds", "intergenic"))
            self.assertGreaterEqual(pred["probability"], 0.0)
            self.assertLessEqual(pred["probability"], 1.0)

    def test_short_cds_both_variants(self):
        for variant in ("unigram", "snt"):
            pred = self._run("short_cds", variant).results["predictions"][">seq1"]
            self.assertIn(pred["label"], ("short_cds", "intergenic_orf"))

    def test_splice_site_both_variants(self):
        for variant in ("unigram", "snt"):
            hits = self._run("splice_site", variant).results["predictions"][">seq1"]
            self.assertIsInstance(hits, list)
            for hit in hits:
                self.assertIn(hit["class"], (0, 1, 2))
                self.assertIn(hit["motif"], ("GT", "GC", "AG"))

    def test_bgc_boundary_both_variants(self):
        for variant in ("unigram", "snt"):
            out = self._run("bgc_boundary", variant).results["boundaries"][">seq1"]
            self.assertTrue(all(isinstance(out[k], float) for k in ("left_bp", "right_bp")))

    def test_taxonomy_both_variants(self):
        for variant in ("unigram", "snt"):
            res = self._run("taxonomy", variant, rank="phylum").results
            self.assertIsInstance(res["labels"][">seq1"], int)
            self.assertIsInstance(res["labels_named"][">seq1"], str)

    def test_taxonomy_all_ranks_load(self):
        """Every rank's head must load with the variant's head architecture."""
        for variant in ("unigram", "snt"):
            for rank in RANKS:
                with self.subTest(variant=variant, rank=rank):
                    res = self._run("taxonomy", variant, rank=rank).results
                    self.assertEqual(res["rank"], rank)
                    self.assertIsNotNone(res["labels"][">seq1"])

    def test_variant_selects_distinct_backbone(self):
        from fungidna.invoke.handlers.base import WeightLoader

        loader = WeightLoader()
        uni = loader.resolve_backbone(variant="unigram")
        snt = loader.resolve_backbone(variant="snt")
        self.assertNotEqual(uni, snt)

    def test_variant_selects_distinct_task_checkpoints(self):
        """A handler that forgets ``variant=`` would silently load Unigram weights."""
        from fungidna.invoke.handlers.base import WeightLoader

        loader = WeightLoader()
        for task in ("cds", "short_cds", "splice_site", "bgc_boundary"):
            with self.subTest(task=task):
                self.assertNotEqual(
                    loader.resolve(task, variant="unigram"),
                    loader.resolve(task, variant="snt"),
                )


if __name__ == "__main__":
    unittest.main()
