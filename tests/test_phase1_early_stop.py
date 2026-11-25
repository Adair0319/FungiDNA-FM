"""Phase 1 early stopping: evaluation nodes and cross-rank stop synchronisation.

The training loop must reach the stop-decision broadcast at every evaluation
node on *every* rank, so no rank can leave the loop while another keeps
training. These tests cover the two helpers that implement that contract,
including a real two-process Gloo run.
"""
import socket
import sys
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungidna.training.distributed import should_evaluate, sync_stop_decision

GLOO = dist.is_available() and dist.is_gloo_available()
EPOCH = 100_000


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sync_worker(rank, world_size, port, rank0_stop, stop_step, step, results):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        stopped = bool(rank0_stop) if rank == 0 else False
        stopped, out_step = sync_stop_decision(stopped, stop_step, step,
                                               torch.device("cpu"))
        results.put((rank, stopped, out_step))
    finally:
        dist.destroy_process_group()


class TestEvaluationNodes(unittest.TestCase):
    def test_only_at_or_after_min_steps(self):
        self.assertFalse(should_evaluate(999, 1000, 100, 5))
        self.assertTrue(should_evaluate(1000, 1000, 100, 5))

    def test_only_on_eval_every_boundaries(self):
        self.assertFalse(should_evaluate(1050, 1000, 100, 5))
        self.assertTrue(should_evaluate(1100, 1000, 100, 5))
        self.assertFalse(should_evaluate(1101, 1000, 100, 5))

    def test_disabled_when_patience_is_not_positive(self):
        for patience in (0, -1):
            with self.subTest(patience=patience):
                self.assertFalse(should_evaluate(10_000, 1000, 100, patience))

    def test_disabled_when_eval_every_is_not_positive(self):
        self.assertFalse(should_evaluate(10_000, 1000, 0, 5))


class TestNonDistributedPassthrough(unittest.TestCase):
    def test_decision_unchanged_without_process_group(self):
        stopped, step = sync_stop_decision(False, EPOCH, 1500, torch.device("cpu"))
        self.assertFalse(stopped)
        self.assertEqual(step, EPOCH)

        stopped, step = sync_stop_decision(True, 1500, 1500, torch.device("cpu"))
        self.assertTrue(stopped)
        self.assertEqual(step, 1500)


@unittest.skipUnless(GLOO, "torch.distributed with Gloo is not available")
class TestTwoProcessStopSync(unittest.TestCase):
    """Real two-process Gloo run: the decision must agree on every rank."""

    def _run(self, rank0_stop):
        ctx = mp.get_context("spawn")
        results = ctx.Queue()
        port = _free_port()
        mp.spawn(
            _sync_worker,
            args=(2, port, rank0_stop, EPOCH, 1500, results),
            nprocs=2,
            join=True,
        )
        return sorted([results.get(timeout=60), results.get(timeout=60)])

    def test_stop_is_broadcast_to_other_ranks(self):
        results = self._run(rank0_stop=True)
        ranks = [r[0] for r in results]
        self.assertEqual(ranks, [0, 1])
        for rank, stopped, step in results:
            with self.subTest(rank=rank):
                self.assertTrue(stopped)
                self.assertEqual(step, 1500)

    def test_continue_when_rank0_does_not_stop(self):
        results = self._run(rank0_stop=False)
        for rank, stopped, step in results:
            with self.subTest(rank=rank):
                self.assertFalse(stopped)
                self.assertEqual(step, EPOCH)


if __name__ == "__main__":
    unittest.main()
