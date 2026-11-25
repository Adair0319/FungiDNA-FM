import os
import tempfile
import unittest
from pathlib import Path

from fungidna.invoke.validation import validate, ValidationError
from fungidna.invoke.request import Task


def _fasta():
    f = tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False)
    f.write(">s1\nACGT\n")
    f.close()
    return Path(f.name)


class TestValidate(unittest.TestCase):
    def test_valid_request(self):
        p = _fasta()
        req = validate("splice site", str(p))
        self.assertEqual(req.task, Task.SPLICE_SITE)
        self.assertTrue(req.file_path.is_absolute())

    def test_unsupported_task(self):
        p = _fasta()
        with self.assertRaises(ValidationError) as cm:
            validate("promoter", str(p))
        self.assertEqual(cm.exception.code, "TASK_UNSUPPORTED")

    def test_missing_file_path(self):
        with self.assertRaises(ValidationError) as cm:
            validate("taxonomy", None)
        self.assertEqual(cm.exception.code, "MISSING_FILE_PATH")

    def test_file_not_found(self):
        with self.assertRaises(ValidationError) as cm:
            validate("taxonomy", "/no/such/file.fasta")
        self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")

    def test_unsupported_format(self):
        p = _fasta()
        gz = Path(str(p) + ".gz")
        gz.write_text(">s\nACGT\n")
        with self.assertRaises(ValidationError) as cm:
            validate("taxonomy", str(gz))
        self.assertEqual(cm.exception.code, "UNSUPPORTED_FORMAT")

    def test_invalid_sequence_format(self):
        p = _fasta()
        bad = Path(p.parent / "bad.fasta")
        bad.write_text("this is not a sequence\n")
        with self.assertRaises(ValidationError) as cm:
            validate("taxonomy", str(bad))
        self.assertEqual(cm.exception.code, "INVALID_SEQUENCE_FORMAT")
