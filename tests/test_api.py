import tempfile
import unittest
from pathlib import Path

from fungidna.invoke import run_task
from fungidna.invoke.validation import ValidationError


class TestRunTask(unittest.TestCase):
    def test_invalid_task_raises(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False)
        f.write(">s\nACGT\n")
        f.close()
        with self.assertRaises(ValidationError) as cm:
            run_task(file_path=f.name, task="promoter")
        self.assertEqual(cm.exception.code, "TASK_UNSUPPORTED")

    def test_missing_file_raises(self):
        with self.assertRaises(ValidationError) as cm:
            run_task(file_path="/nope.fasta", task="taxonomy")
        self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")
