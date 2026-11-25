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
        # ``fungidna.invoke`` registers the real handlers at import time; this
        # test swaps in fakes, so the live registry must be restored afterwards
        # or every later test that dispatches would see an empty registry.
        self._saved_handlers = dict(_HANDLERS)
        _HANDLERS.clear()

    def tearDown(self):
        _HANDLERS.clear()
        _HANDLERS.update(self._saved_handlers)

    def test_dispatch_routes_to_registered_handler(self):
        register(Task.TAXONOMY, FakeHandler("tax"))
        req = StructuredRequest(task=Task.TAXONOMY, file_path=Path("/tmp/x.fasta"))
        r = dispatch(req)
        self.assertEqual(r.results["handler"], "tax")

    def test_dispatch_unregistered_raises(self):
        req = StructuredRequest(task=Task.SPLICE_SITE, file_path=Path("/tmp/x.fasta"))
        with self.assertRaises(RuntimeError):
            dispatch(req)
