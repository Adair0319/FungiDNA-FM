"""Variant helpers: ``token_type_for`` / ``encode_window`` / ``mlm_vocab_size``.

These are additive: the integer ``token_type`` arguments used by the models and
datasets stay exactly as they are, and no token ID, vocabulary size or embedding
row changes.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungidna.data.tokenizer import (
    NT_VOCAB_SIZE,
    DualTokenizer,
    encode_window,
    mlm_vocab_size,
    token_type_for,
)

_ROOT = Path(os.environ.get(
    "FUNGIDNA_TEST_WEIGHTS",
    "/home/lty/yy_projects/fungi_project/fungi_dna_model",
))
TOKENIZER = Path(os.environ.get("FUNGIDNA_TEST_TOKENIZER", _ROOT / "data/processed/bpe_fungi.model"))
SEQ = "ATGGCTAGCTAGCTAGGCTAACCGGTTACGTAGCTAGCATCGAT"


class TestTokenTypeFor(unittest.TestCase):
    def test_canonical_values(self):
        self.assertEqual(token_type_for("unigram"), 0)
        self.assertEqual(token_type_for("snt"), 1)

    def test_case_insensitive(self):
        self.assertEqual(token_type_for("SNT"), 1)
        self.assertEqual(token_type_for("Unigram"), 0)

    def test_whitespace_tolerated(self):
        self.assertEqual(token_type_for("  snt  "), 1)
        self.assertEqual(token_type_for("\tUnigram\n"), 0)

    def test_invalid_values_raise(self):
        for bad in ("bpe", "", "  ", None, "single-nucleotide"):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    token_type_for(bad)


class TestMlmVocabSize(unittest.TestCase):
    def test_values(self):
        class _Tok:
            bpe_vocab_size = 4096

        self.assertEqual(mlm_vocab_size(_Tok(), "unigram"), 4096)
        self.assertEqual(mlm_vocab_size(_Tok(), "snt"), NT_VOCAB_SIZE)
        self.assertEqual(NT_VOCAB_SIZE, 10)


@unittest.skipUnless(TOKENIZER.exists(), "tokenizer file not available")
class TestEncodeWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = DualTokenizer(str(TOKENIZER))

    def test_unigram_matches_encode_bpe(self):
        self.assertEqual(encode_window(self.tok, "unigram", SEQ),
                         self.tok.encode_bpe(SEQ, add_cls=False))

    def test_snt_matches_encode_single_nt(self):
        self.assertEqual(encode_window(self.tok, "snt", SEQ),
                         self.tok.encode_single_nt(SEQ, add_cls=False))

    def test_variant_case_insensitive(self):
        self.assertEqual(encode_window(self.tok, "SNT", SEQ),
                         self.tok.encode_single_nt(SEQ, add_cls=False))

    def test_no_cls_token(self):
        from fungidna.data.tokenizer import CLS_ID
        self.assertNotIn(CLS_ID, encode_window(self.tok, "unigram", SEQ))
        self.assertNotIn(CLS_ID, encode_window(self.tok, "snt", SEQ))


class TestSingleSourceOfTruth(unittest.TestCase):
    def test_representation_uses_shared_map(self):
        from fungidna import representation
        from fungidna.data.tokenizer import VARIANT_TOKEN_TYPE

        self.assertIs(representation._VARIANT_TOKEN_TYPE, VARIANT_TOKEN_TYPE)

    def test_representation_tokens_unchanged(self):
        from fungidna import representation
        from fungidna.data.tokenizer import token_type_for

        for name, tt in representation._VARIANT_TOKEN_TYPE.items():
            with self.subTest(variant=name):
                self.assertEqual(tt, token_type_for(name))


class TestIntegerApiUnchanged(unittest.TestCase):
    """The existing integer ``token_type`` API must keep working identically."""

    @classmethod
    def setUpClass(cls):
        from fungidna.model.config import FungiDNAConfig
        cls.config = FungiDNAConfig()

    def test_model_head_tie_unchanged(self):
        from fungidna.model.fungi_dna import FungiDNAForMLM

        uni = FungiDNAForMLM(self.config, token_type=0)
        snt = FungiDNAForMLM(self.config, token_type=1)
        self.assertEqual(uni.vocab_size, 4096)
        self.assertIs(uni.lm_head.weight, uni.backbone.embedding.bpe_embed.weight)
        self.assertEqual(snt.vocab_size, 10)
        self.assertIs(snt.lm_head.weight, snt.backbone.embedding.nt_embed.weight)

    def test_dataset_vocab_size_unchanged(self):
        import tempfile

        from fungidna.data.pretrain_dataset import Phase1MLMDataset

        class _Tok:
            bpe_vocab_size = 4096

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "dummy.fasta").write_text(">x\nACGT\n")
            ds = Phase1MLMDataset(filtered_fasta_dir=tmp, tokenizer=_Tok(), token_type=1)
            self.assertEqual(ds.vocab_size, 10)


if __name__ == "__main__":
    unittest.main()
