"""Pins the known paper/code deviations so they cannot change silently.

README.md ("Paper–code correspondence") states three deviations between the
submitted manuscript and this repository. Two of them live in the model code and
are covered here:

* the rotary embedding rotates by **head index**, not by token position, so the
  attention scores carry no positional signal;
* the input embedding is followed by a **LayerNorm**, while the rest of the
  backbone uses RMSNorm.

These tests assert the current behaviour, not the paper's wording. If someone
changes either implementation, these tests fail and the change has to be
accompanied by an explicit retraining decision — the released checkpoints were
trained with the behaviour pinned here.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from fungidna.model.config import FungiDNAConfig
from fungidna.model.embedding import DualTokenEmbedding, RotaryPositionEmbedding

HEAD_DIM = 64
N_HEADS = 12
HIDDEN = 768


class TestRotaryEmbeddingIsPerHead(unittest.TestCase):
    """The angle table is indexed by head, not by sequence position."""

    def setUp(self):
        self.config = FungiDNAConfig()
        self.rope = RotaryPositionEmbedding(
            self.config.hidden_size,
            self.config.rope_theta,
            self.config.max_position_embeddings,
            min_seq_len=self.config.num_attention_heads,
        )

    def test_rotation_does_not_depend_on_position(self):
        seq_len = 8
        cos, sin = self.rope(seq_len, torch.device("cpu"))
        constant = torch.randn(1, 1, N_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        x = constant.expand(1, seq_len, N_HEADS, HEAD_DIM).contiguous()

        out = RotaryPositionEmbedding.apply_rotary(x, cos, sin)

        # Same vector at every position -> same rotated vector at every position.
        self.assertEqual(float((out - out[:, :1]).abs().max()), 0.0)

    def test_head_zero_is_not_rotated(self):
        seq_len = 8
        cos, sin = self.rope(seq_len, torch.device("cpu"))
        x = torch.randn(2, seq_len, N_HEADS, HEAD_DIM, dtype=torch.bfloat16)

        out = RotaryPositionEmbedding.apply_rotary(x, cos, sin)

        # The angle for head 0 is 0 for every channel, i.e. an identity rotation.
        self.assertTrue(torch.equal(out[:, :, 0, :], x[:, :, 0, :]))

    def test_min_seq_len_only_changes_short_inputs(self):
        short = RotaryPositionEmbedding(HIDDEN, min_seq_len=1)
        padded = RotaryPositionEmbedding(HIDDEN, min_seq_len=N_HEADS)

        cos_long, sin_long = short(16, torch.device("cpu"))
        cos_pad, sin_pad = padded(16, torch.device("cpu"))
        self.assertTrue(torch.equal(cos_long, cos_pad))
        self.assertTrue(torch.equal(sin_long, sin_pad))

        # Shorter than the head count: the padded table still has 12 rows.
        cos_short, _ = padded(4, torch.device("cpu"))
        self.assertEqual(cos_short.shape[0], N_HEADS)


class TestEmbeddingNormalisation(unittest.TestCase):
    """Documented deviation: LayerNorm after the input embedding."""

    def test_input_embedding_uses_layernorm_with_bias(self):
        emb = DualTokenEmbedding(FungiDNAConfig())

        self.assertIsInstance(emb.layer_norm, nn.LayerNorm)
        self.assertIsNotNone(emb.layer_norm.bias)
        self.assertEqual(emb.layer_norm.weight.shape, (HIDDEN,))
        self.assertEqual(emb.layer_norm.bias.shape, (HIDDEN,))


if __name__ == "__main__":
    unittest.main()
