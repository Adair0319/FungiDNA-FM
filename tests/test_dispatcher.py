import unittest
from pathlib import Path

from fungidna.invoke.dispatcher import dispatch, register, _HANDLERS
from fungidna.invoke.report import Report
from fungidna.invoke.request import Task, StructuredRequest


class FakeHandler:
    def __init__(self, label):
        self.label = label

    def run(self, request):
        return Report(task=request.task, file_path=request.file_path,
                      status="ok", results={"handler": self.label})


class TestDispatcher(unittest.TestCase):
    def setUp(self):
        _HANDLERS.clear()

    def test_dispatch_routes_to_registered_handler(self):
        register(Task.TAXONOMY, FakeHandler("tax"))
        req = StructuredRequest(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"))
        r = dispatch(req)
        self.assertEqual(r.results["handler"], "tax")

    def test_dispatch_unregistered_raises(self):
        req = StructuredRequest(task=Task.SPLICE_SITE, file_path=Path("/tmp/x.fasta"))
        with self.assertRaises(RuntimeError):
            dispatch(req)
