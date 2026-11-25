"""Natural-language terminal (Figure 1D): request -> task, plus CLI flags.

Six window-level tasks must be reachable from free text, ``short CDS`` must not
be parsed as ``cds``, and the CLI must honour ``--variant`` / ``--rank``.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fungidna.__main__ import _build_parser
from fungidna.invoke.parser import match_task, parse_prompt

_ROOT = Path(os.environ.get(
    "FUNGIDNA_TEST_WEIGHTS",
    "/home/lty/yy_projects/fungi_project/fungi_dna_model",
))
TOKENIZER = Path(os.environ.get("FUNGIDNA_TEST_TOKENIZER", _ROOT / "data/processed/bpe_fungi.model"))
UNIGRAM_CKPT = _ROOT / "checkpoints/phase2_joint/backbone_final.pt"
SNT_CKPT = _ROOT / "checkpoints/phase2_joint_single_nt/backbone_final.pt"
CUDA = torch.cuda.is_available()

REQUESTS = [
    ("extract the representation of /tmp/x.fasta", "embedding"),
    ("classify the taxonomy of /tmp/x.fasta", "taxonomy"),
    ("tell me the taxonomic classification of /tmp/x.fasta", "taxonomy"),
    ("classify this CDS versus intergenic sequence in /tmp/x.fasta", "cds"),
    ("discriminate CDS from intergenic in /tmp/x.fasta", "cds"),
    ("classify short CDS in /tmp/x.fasta", "short_cds"),
    ("is this a short-CDS sequence, /tmp/x.fasta", "short_cds"),
    ("predict splice sites in /tmp/x.fasta", "splice_site"),
    ("classify splice sites in /tmp/x.fasta", "splice_site"),
    ("predict BGC boundaries in /tmp/x.fasta", "bgc_boundary"),
]


class TestPromptParsing(unittest.TestCase):
    def test_all_six_tasks_are_recognised(self):
        seen = set()
        for text, expected in REQUESTS:
            with self.subTest(text=text):
                parsed = parse_prompt(text)
                self.assertEqual(parsed["task"], expected)
                self.assertEqual(parsed["file_path"], "/tmp/x.fasta")
                seen.add(expected)
        self.assertEqual(seen, {"embedding", "taxonomy", "cds", "short_cds",
                                "splice_site", "bgc_boundary"})

    def test_short_cds_is_not_parsed_as_cds(self):
        self.assertEqual(match_task("classify short CDS in a window"), "short_cds")
        self.assertEqual(match_task("short-coding sequence"), "short_cds")
        self.assertEqual(match_task("CDS versus intergenic"), "cds")

    def test_unknown_request_returns_none(self):
        self.assertIsNone(match_task("do something unrelated"))


class TestCliFlags(unittest.TestCase):
    def test_variant_and_rank_defaults(self):
        args = _build_parser().parse_args(["extract the representation of /tmp/x.fasta"])
        self.assertEqual(args.variant, "unigram")
        self.assertEqual(args.rank, "phylum")

    def test_variant_and_rank_override(self):
        args = _build_parser().parse_args(
            ["expand /tmp/x.fasta", "--variant", "snt", "--rank", "family"]
        )
        self.assertEqual(args.variant, "snt")
        self.assertEqual(args.rank, "family")

    def test_rejects_unknown_variant(self):
        with self.assertRaises(SystemExit):
            _build_parser().parse_args(["x", "--variant", "bpe"])


@unittest.skipUnless(CUDA, "model requires CUDA")
@unittest.skipUnless(TOKENIZER.exists() and UNIGRAM_CKPT.exists() and SNT_CKPT.exists(),
                     "real checkpoints not available")
class TestCliEndToEnd(unittest.TestCase):
    """One real run: the CLI must dispatch the parsed task to the right model."""

    def setUp(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False)
        tmp.write(">seq1\nATGGCTAGCTAGCTAGGCTAACCGGTTACGTAGCTAGCATCGAT\n")
        tmp.close()
        self.fasta = tmp.name
        cfg = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        cfg.write(
            f"tokenizer: {TOKENIZER}\n"
            f"unigram:\n  backbone: {UNIGRAM_CKPT}\n"
            f"snt:\n  backbone: {SNT_CKPT}\n"
        )
        cfg.close()
        self.config = cfg.name
        self.env = dict(os.environ, FUNGIDNA_WEIGHTS_CONFIG=self.config)

    def tearDown(self):
        os.unlink(self.fasta)
        os.unlink(self.config)

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "fungidna", *args],
            capture_output=True, text=True, env=self.env,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    def test_snt_representation_request(self):
        result = self._run(f"extract the representation of {self.fasta}", "--variant", "snt")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["task"], "embedding")
        self.assertEqual(payload["results"]["variant"], "snt")
        self.assertEqual(len(payload["results"]["embeddings"][">seq1"]), 768)

    def test_unknown_request_errors_cleanly(self):
        result = self._run(f"please do something odd with {self.fasta}")
        self.assertEqual(result.returncode, 1)
        self.assertIn("TASK_UNSUPPORTED", result.stderr)


if __name__ == "__main__":
    unittest.main()
