import json
import unittest
from pathlib import Path

from fungidna.invoke.report import Report
from fungidna.invoke.request import Task


class TestReport(unittest.TestCase):
    def test_to_json_roundtrip(self):
        r = Report(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"),
                   status="ok", results={"rank": "phylum", "label": "Ascomycota"})
        d = json.loads(r.to_json())
        self.assertEqual(d["task"], "taxonomy")
        self.assertEqual(d["file_path"], "/tmp/x.fasta")
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["results"]["label"], "Ascomycota")
