"""Parser: variant/rank extraction with explicit ambiguity and unsupported-rank errors.

Backward compatibility is part of the contract: ``parse_prompt(text, llm_extractor)``
must keep returning the original ``task`` / ``file_path`` keys with the same
semantics, and ``match_task`` keeps its behaviour.

Supported taxonomy ranks are the five the project actually ships label maps for:
phylum, subphylum, class, order, family (paper: "five ranks from phylum to
family"). ``genus`` is **not** supported by the heads, label maps or handler, so
it must raise instead of silently defaulting.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungidna.invoke.parser import (
    AmbiguousRequestError,
    UnsupportedRankError,
    match_task,
    parse_prompt,
)

RANKS = ("phylum", "subphylum", "class", "order", "family")


class TestVariantParsing(unittest.TestCase):
    def test_explicit_unigram(self):
        self.assertEqual(parse_prompt("unigram embedding of /tmp/x.fasta")["variant"], "unigram")

    def test_explicit_snt(self):
        self.assertEqual(parse_prompt("embedding of /tmp/x.fasta using SNT")["variant"], "snt")

    def test_single_nucleotide_forms(self):
        for text in ("single-nucleotide representation of /tmp/x.fasta",
                     "single nucleotide representation of /tmp/x.fasta",
                     "single_nucleotide representation of /tmp/x.fasta"):
            with self.subTest(text=text):
                self.assertEqual(parse_prompt(text)["variant"], "snt")

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(parse_prompt("  SNT   embedding of /tmp/x.fasta")["variant"], "snt")
        self.assertEqual(parse_prompt("Unigram embedding of /tmp/x.fasta")["variant"], "unigram")

    def test_defaults_to_unigram(self):
        self.assertEqual(parse_prompt("embedding of /tmp/x.fasta")["variant"], "unigram")

    def test_conflicting_variants_are_rejected(self):
        with self.assertRaises(AmbiguousRequestError) as ctx:
            parse_prompt("compare unigram and snt embeddings of /tmp/x.fasta")
        msg = str(ctx.exception).lower()
        self.assertIn("unigram", msg)
        self.assertIn("snt", msg)

    def test_filenames_do_not_trigger_variant(self):
        # "x.snt.fasta" is a file name, not a variant request.
        parsed = parse_prompt("embedding of /tmp/x.snt.fasta")
        self.assertEqual(parsed["variant"], "unigram")


class TestRankParsing(unittest.TestCase):
    def test_all_five_supported_ranks(self):
        for rank in RANKS:
            with self.subTest(rank=rank):
                parsed = parse_prompt(f"classify the taxonomy at {rank} of /tmp/x.fasta")
                self.assertEqual(parsed["task"], "taxonomy")
                self.assertEqual(parsed["rank"], rank)

    def test_rank_case_insensitive(self):
        self.assertEqual(parse_prompt("taxonomy at FAMILY of /tmp/x.fasta")["rank"], "family")

    def test_taxonomy_default_rank_matches_handler_default(self):
        self.assertEqual(parse_prompt("classify the taxonomy of /tmp/x.fasta")["rank"], "phylum")

    def test_non_taxonomy_task_has_no_rank(self):
        for text in ("embedding of /tmp/x.fasta",
                     "classify short CDS in /tmp/x.fasta",
                     "predict splice sites in /tmp/x.fasta"):
            with self.subTest(text=text):
                self.assertIsNone(parse_prompt(text)["rank"])

    def test_conflicting_ranks_are_rejected(self):
        with self.assertRaises(AmbiguousRequestError) as ctx:
            parse_prompt("classify the taxonomy at phylum and class of /tmp/x.fasta")
        msg = str(ctx.exception).lower()
        self.assertIn("phylum", msg)
        self.assertIn("class", msg)

    def test_unsupported_rank_is_rejected(self):
        """genus/species/kingdom are not supported by the shipped heads."""
        for word in ("genus", "species", "kingdom"):
            with self.subTest(word=word):
                with self.assertRaises(UnsupportedRankError) as ctx:
                    parse_prompt(f"classify the taxonomy at {word} of /tmp/x.fasta")
                self.assertIn(word, str(ctx.exception).lower())

    def test_genus_mentioned_outside_taxonomy_is_ignored(self):
        parsed = parse_prompt("extract the representation of /tmp/x.fasta for a genus-level study")
        self.assertIsNone(parsed["rank"])


class TestPathParsing(unittest.TestCase):
    def test_absolute_path(self):
        self.assertEqual(parse_prompt("embedding of /tmp/x.fasta")["file_path"], "/tmp/x.fasta")

    def test_relative_forms(self):
        for text, expected in (("embedding of x.fasta", "x.fasta"),
                               ("embedding of ./x.fasta", "./x.fasta"),
                               ("embedding of data/x.fasta", "data/x.fasta"),
                               ("embedding of ../data/x.fa", "../data/x.fa")):
            with self.subTest(text=text):
                self.assertEqual(parse_prompt(text)["file_path"], expected)

    def test_quoted_path_with_space(self):
        self.assertEqual(parse_prompt('embedding of "/tmp/my data/x.fasta"')["file_path"],
                         "/tmp/my data/x.fasta")

    def test_mixed_prose(self):
        parsed = parse_prompt("please predict splice sites in data/sample.fastq and explain")
        self.assertEqual(parsed["task"], "splice_site")
        self.assertEqual(parsed["file_path"], "data/sample.fastq")

    def test_no_path(self):
        self.assertIsNone(parse_prompt("embedding please")["file_path"])

    def test_plain_prose_not_mistaken_for_path(self):
        self.assertIsNone(parse_prompt("this is a fasta-like discussion")["file_path"])
        self.assertIsNone(parse_prompt("a fastq-like sentence")["file_path"])

    def test_extension_case_insensitive(self):
        self.assertEqual(parse_prompt("embedding of /tmp/X.FASTA")["file_path"], "/tmp/X.FASTA")

    def test_unsupported_extension_is_not_a_path(self):
        for text in ("embedding of /tmp/x.txt", "embedding of data/x.csv"):
            with self.subTest(text=text):
                self.assertIsNone(parse_prompt(text)["file_path"])

    def test_unquoted_space_path_is_not_fully_matched(self):
        """A path with spaces must be quoted; unquoted input only yields the last
        token. Documented limitation, asserted so behaviour cannot drift silently."""
        parsed = parse_prompt("embedding of /tmp/my data/x.fasta")
        self.assertEqual(parsed["file_path"], "data/x.fasta")


class TestBackwardCompatibility(unittest.TestCase):
    def test_original_keys_and_semantics(self):
        parsed = parse_prompt("classify short CDS in /tmp/x.fasta")
        self.assertEqual(set(parsed) >= {"task", "file_path"}, True)
        self.assertEqual(parsed["task"], "short_cds")          # precedence unchanged
        self.assertEqual(parsed["file_path"], "/tmp/x.fasta")

    def test_match_task_still_public(self):
        self.assertEqual(match_task("classify CDS"), "cds")
        self.assertEqual(match_task("short CDS"), "short_cds")
        self.assertIsNone(match_task("do something odd"))

    def test_llm_extractor_still_used_as_fallback(self):
        calls = []

        def fake(text):
            calls.append(text)
            return {"task": "bgc_boundary", "file_path": "/tmp/from_llm.fasta"}

        parsed = parse_prompt("predict clusters please", llm_extractor=fake)
        self.assertEqual(calls, ["predict clusters please"])
        self.assertEqual(parsed["task"], "bgc_boundary")
        self.assertEqual(parsed["file_path"], "/tmp/from_llm.fasta")

    def test_llm_extractor_cannot_override_regex_result(self):
        def fake(text):
            return {"task": "cds", "file_path": "/tmp/other.fasta"}

        parsed = parse_prompt("embedding of /tmp/x.fasta", llm_extractor=fake)
        self.assertEqual(parsed["task"], "embedding")
        self.assertEqual(parsed["file_path"], "/tmp/x.fasta")


class TestCliPrecedence(unittest.TestCase):
    """CLI flags win over text parsing; text wins over defaults; non-taxonomy
    requests never carry a taxonomy rank."""

    @staticmethod
    def _resolve(argv, text):
        from fungidna.__main__ import _build_parser, _resolve_options

        args = _build_parser().parse_args(argv)
        return _resolve_options(args, parse_prompt(text))

    def test_embedding_has_no_rank(self):
        self.assertEqual(self._resolve(["embedding of /tmp/x.fasta"],
                                       "embedding of /tmp/x.fasta"), ("unigram", None))

    def test_cds_has_no_rank(self):
        self.assertEqual(self._resolve(["classify CDS in /tmp/x.fasta"],
                                       "classify CDS in /tmp/x.fasta"), ("unigram", None))

    def test_taxonomy_default_rank(self):
        self.assertEqual(self._resolve(["classify the taxonomy of /tmp/x.fasta"],
                                       "classify the taxonomy of /tmp/x.fasta"),
                         ("unigram", "phylum"))

    def test_taxonomy_rank_from_text(self):
        text = "classify the taxonomy at family of /tmp/x.fasta"
        self.assertEqual(self._resolve([text], text), ("unigram", "family"))

    def test_cli_rank_wins_over_text(self):
        text = "classify the taxonomy at family of /tmp/x.fasta"
        self.assertEqual(self._resolve([text, "--rank", "order"], text),
                         ("unigram", "order"))

    def test_cli_variant_wins_over_text(self):
        text = "embedding of /tmp/x.fasta using unigram"
        self.assertEqual(self._resolve([text, "--variant", "snt"], text),
                         ("snt", None))

    def test_text_variant_used_when_flag_absent(self):
        text = "embedding of /tmp/x.fasta using SNT"
        self.assertEqual(self._resolve([text], text), ("snt", None))

    def test_cli_rank_on_non_taxonomy_is_ignored_not_applied(self):
        text = "embedding of /tmp/x.fasta"
        self.assertEqual(self._resolve([text, "--rank", "family"], text),
                         ("unigram", None))


if __name__ == "__main__":
    unittest.main()
