import unittest
import tempfile
from pathlib import Path

from fungidna.invoke.sequence import (
    read_sequences,
    first_record_valid,
    first_record_wellformed,
    SUPPORTED_EXTENSIONS,
)


def _write(content, suffix=".fasta"):
    f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


class TestReadSequences(unittest.TestCase):
    def test_fasta_two_records(self):
        p = _write(">seq1\nACGT\n>seq2 description\nTTTT\n")
        recs = read_sequences(p)
        self.assertEqual([r[0] for r in recs], [">seq1", ">seq2 description"])
        self.assertEqual([r[1] for r in recs], ["ACGT", "TTTT"])

    def test_fastq(self):
        p = _write("@r1\nACGT\n+\nIIII\n", suffix=".fastq")
        recs = read_sequences(p)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0][1], "ACGT")

    def test_invalid_first_line(self):
        p = _write("not a sequence file\n")
        with self.assertRaises(ValueError):
            read_sequences(p)

    def test_first_record_valid(self):
        self.assertTrue(first_record_valid(_write(">x\nACGT\n")))
        self.assertTrue(first_record_valid(_write("@x\nACGT\n+\nIIII\n", suffix=".fastq")))
        self.assertFalse(first_record_valid(_write("junk\n")))

    def test_supported_extensions(self):
        self.assertEqual(SUPPORTED_EXTENSIONS, {".fa", ".fasta", ".fq", ".fastq"})

    def test_first_record_wellformed_fasta(self):
        self.assertTrue(first_record_wellformed(_write(">x\nACGT\n")))
        # header with no sequence is not a well-formed record
        self.assertFalse(first_record_wellformed(_write(">x\n")))
        # header-only followed by a second header is still empty
        self.assertFalse(first_record_wellformed(_write(">x\n>y\nACGT\n")))

    def test_first_record_wellformed_fastq(self):
        self.assertTrue(first_record_wellformed(_write("@x\nACGT\n+\nIIII\n", suffix=".fastq")))
        # missing '+' line
        self.assertFalse(first_record_wellformed(_write("@x\nACGT\n", suffix=".fastq")))
        # empty sequence
        self.assertFalse(first_record_wellformed(_write("@x\n\n+\nIIII\n", suffix=".fastq")))

    def test_first_record_wellformed_non_sequence(self):
        self.assertFalse(first_record_wellformed(_write("junk\n")))
