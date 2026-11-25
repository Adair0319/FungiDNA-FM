"""Representation (embedding) tests for both tokenizer variants.

The pooling contract under test follows Supplementary Note S2 and matches the
development-repository script that produced the paper's clustering numbers
(``scripts/checkpoint_ablation.py``: layer index 7, masked mean, no CLS). That
script is not shipped in this repository.

So *no* [CLS] is involved in representation extraction. (The ``add_cls=True``
conventions belong to downstream task heads, not to the representation.)

Real checkpoints are required. Point the tests at them with:

    export FUNGIDNA_TEST_TOKENIZER=/path/to/bpe_fungi.model
    export FUNGIDNA_TEST_UNIGRAM_BACKBONE=/path/to/unigram/backbone_final.pt
    export FUNGIDNA_TEST_SNT_BACKBONE=/path/to/snt/backbone_final.pt

Without them the checkpoint-backed tests are skipped (they do not silently fall
back to random weights).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fungidna.data.tokenizer import CLS_ID, NT_TO_ID, PAD_ID, DualTokenizer

_DEFAULT_ROOT = Path("/home/lty/yy_projects/fungi_project/fungi_dna_model")


def _path(env, default):
    value = os.environ.get(env)
    if value:
        return Path(value)
    candidate = _DEFAULT_ROOT / default
    return candidate if candidate.exists() else None


TOKENIZER = _path("FUNGIDNA_TEST_TOKENIZER", "data/processed/bpe_fungi.model")
UNIGRAM_CKPT = _path(
    "FUNGIDNA_TEST_UNIGRAM_BACKBONE", "checkpoints/phase2_joint/backbone_final.pt"
)
SNT_CKPT = _path(
    "FUNGIDNA_TEST_SNT_BACKBONE", "checkpoints/phase2_joint_single_nt/backbone_final.pt"
)

CUDA = torch.cuda.is_available()


def manual_layer_hidden(model, input_ids, layer_index=7, add_cls=False):
    """Walk the backbone by hand and return layer ``layer_index`` (before final_norm)."""
    with torch.no_grad(), torch.amp.autocast(device_type=model.device.type,
                                             dtype=torch.bfloat16):
        h, cos, sin = model.backbone.embedding(
            input_ids, token_type=model.token_type, add_cls=add_cls
        )
        for i, layer in enumerate(model.backbone.layers):
            is_attn = (i + 1) % (model.backbone.config.mamba_per_attention + 1) == 0
            h = layer(h, cos, sin) if is_attn else layer(h)
            if i == layer_index:
                break
    return h.float()

SEQ_A = "ATGGCTAGCTAGCTAGGCTAACCGGTTACGTAGCTAGCATCGATCGATCGTAGCTAGCTAGCTA"
SEQ_B = "TTTTAAAACCCCGGGGATCGATCGATCGATCGATCGATCGAAACTGACTG"
SEQ_C = "GCGCGCGCGCGCGCGCGCGC"


class TestTokenizerContract(unittest.TestCase):
    """Token-level behaviour the representation depends on."""

    @classmethod
    def setUpClass(cls):
        if TOKENIZER is None:
            raise unittest.SkipTest("tokenizer file not available")
        cls.tok = DualTokenizer(str(TOKENIZER))

    def test_encode_bpe_add_cls_prepends_cls_id(self):
        with_cls = self.tok.encode_bpe(SEQ_A, add_cls=True)
        without = self.tok.encode_bpe(SEQ_A, add_cls=False)
        self.assertEqual(with_cls[0], CLS_ID)
        self.assertEqual(with_cls[1:], without)

    def test_encode_bpe_default_has_cls(self):
        # Default is add_cls=True; representation code must opt out explicitly.
        self.assertEqual(self.tok.encode_bpe(SEQ_A)[0], CLS_ID)

    def test_encode_single_nt_adds_no_cls_by_default(self):
        ids = self.tok.encode_single_nt(SEQ_A)
        self.assertEqual(len(ids), len(SEQ_A))
        self.assertNotIn(CLS_ID, ids)
        self.assertEqual(ids[:4], [NT_TO_ID[c] for c in SEQ_A[:4]])

    def test_pad_id_is_zero(self):
        self.assertEqual(PAD_ID, 0)


class TestVariantWeightResolution(unittest.TestCase):
    """WeightLoader must resolve paths per variant without cross-talk."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.config = Path(self.tmp.name) / "weights.yaml"
        os.environ["FUNGIDNA_WEIGHTS_CONFIG"] = str(self.config)
        self._env_backup = dict(os.environ)
        for key in list(os.environ):
            if key.startswith("FUNGIDNA_") and key.endswith("_WEIGHTS"):
                del os.environ[key]

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmp.cleanup()

    def _write(self, text):
        self.config.write_text(text)

    def test_backbone_resolves_per_variant_from_yaml(self):
        from fungidna.invoke.handlers.base import WeightLoader

        self._write(
            "tokenizer: /tok/bpe.model\n"
            "unigram:\n  backbone: /w/uni.pt\n  cds: /w/uni_cds.pt\n"
            "snt:\n  backbone: /w/snt.pt\n  cds: /w/snt_cds.pt\n"
        )
        loader = WeightLoader()
        self.assertEqual(loader.resolve_backbone(variant="unigram"), Path("/w/uni.pt"))
        self.assertEqual(loader.resolve_backbone(variant="snt"), Path("/w/snt.pt"))

    def test_task_resolves_per_variant_from_yaml(self):
        from fungidna.invoke.handlers.base import WeightLoader

        self._write(
            "unigram:\n  cds: /w/uni_cds.pt\nsnt:\n  cds: /w/snt_cds.pt\n"
        )
        loader = WeightLoader()
        self.assertEqual(loader.resolve("cds", variant="unigram"), Path("/w/uni_cds.pt"))
        self.assertEqual(loader.resolve("cds", variant="snt"), Path("/w/snt_cds.pt"))

    def test_env_override_is_variant_scoped(self):
        from fungidna.invoke.handlers.base import WeightLoader

        os.environ["FUNGIDNA_UNIGRAM_BACKBONE_WEIGHTS"] = "/env/uni.pt"
        os.environ["FUNGIDNA_SNT_BACKBONE_WEIGHTS"] = "/env/snt.pt"
        loader = WeightLoader()
        self.assertEqual(loader.resolve_backbone(variant="unigram"), Path("/env/uni.pt"))
        self.assertEqual(loader.resolve_backbone(variant="snt"), Path("/env/snt.pt"))

    def test_backbone_does_not_silently_use_flat_key(self):
        """A flat ``backbone:`` entry predates variant support and would make
        ``variant='snt'`` load the Unigram weights. It must be an explicit error."""
        from fungidna.invoke.handlers.base import WeightLoader, WeightsNotFoundError

        self._write("backbone: /w/flat.pt\n")
        with self.assertRaises(WeightsNotFoundError) as ctx:
            WeightLoader().resolve_backbone(variant="snt")
        self.assertIn("snt", str(ctx.exception))

    def test_unknown_variant_is_rejected(self):
        from fungidna.invoke.handlers.base import WeightLoader, WeightsNotFoundError

        self._write("unigram:\n  backbone: /w/uni.pt\n")
        with self.assertRaises(WeightsNotFoundError):
            WeightLoader().resolve_backbone(variant="bpe")


@unittest.skipUnless(CUDA, "model requires CUDA (flash-attn / mamba-ssm kernels)")
@unittest.skipUnless(TOKENIZER and UNIGRAM_CKPT and SNT_CKPT, "real checkpoints not available")
class TestRepresentationModel(unittest.TestCase):
    """End-to-end embedding with the real released checkpoints."""

    @classmethod
    def setUpClass(cls):
        from fungidna.representation import RepresentationModel

        cls.uni = RepresentationModel(
            variant="unigram", tokenizer_path=TOKENIZER, backbone_path=UNIGRAM_CKPT
        )
        cls.snt = RepresentationModel(
            variant="snt", tokenizer_path=TOKENIZER, backbone_path=SNT_CKPT
        )

    def test_single_sequence_shape_and_dtype(self):
        for model in (self.uni, self.snt):
            vec = model.embed(SEQ_A)
            self.assertEqual(vec.shape, (768,))
            self.assertEqual(vec.dtype, np.float32)
            self.assertTrue(np.isfinite(vec).all())

    def test_batch_shape(self):
        for model in (self.uni, self.snt):
            mat = model.embed([SEQ_A, SEQ_B, SEQ_C])
            self.assertEqual(mat.shape, (3, 768))
            self.assertTrue(np.isfinite(mat).all())

    def test_batch_row_matches_single(self):
        for model in (self.uni, self.snt):
            single = model.embed(SEQ_B)
            batched = model.embed([SEQ_A, SEQ_B, SEQ_C])[1]
            np.testing.assert_allclose(single, batched, rtol=1e-4, atol=1e-4)

    def test_pooling_excludes_padding(self):
        """A short sequence must embed the same alone and next to a long one."""
        for model in (self.uni, self.snt):
            alone = model.embed(SEQ_C)  # short
            mixed = model.embed([SEQ_A, SEQ_C])[1]  # padded to len(SEQ_A)
            np.testing.assert_allclose(alone, mixed, rtol=1e-4, atol=1e-4)

    def test_pooling_matches_audited_rule(self):
        """No [CLS]; masked mean over layer-8 tokens (Note S2)."""
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(TOKENIZER))
        for model in (self.uni, self.snt):
            encode = (tok.encode_single_nt if model.variant == "snt"
                      else lambda s: tok.encode_bpe(s, add_cls=False))
            ids = torch.tensor([encode(SEQ_A)], dtype=torch.long, device=model.device)
            reference = manual_layer_hidden(model, ids).mean(dim=1)[0].cpu().numpy()

            np.testing.assert_allclose(
                model.embed(SEQ_A), reference, rtol=1e-4, atol=1e-4,
                err_msg="embed() must be the mean over layer-8 DNA tokens (no CLS)",
            )

    def test_cls_pooling_would_differ(self):
        """Guards against silently re-introducing the [CLS]-inclusive mean."""
        model = self.uni
        ids = torch.tensor(
            [DualTokenizer(str(TOKENIZER)).encode_bpe(SEQ_A, add_cls=True)],
            dtype=torch.long, device=model.device,
        )
        with_cls = manual_layer_hidden(model, ids, add_cls=True).mean(dim=1)[0].cpu().numpy()
        self.assertFalse(np.allclose(model.embed(SEQ_A), with_cls, rtol=1e-3, atol=1e-3))

    def test_variants_use_distinct_backbones(self):
        self.assertEqual(str(self.uni.backbone_path), str(UNIGRAM_CKPT))
        self.assertEqual(str(self.snt.backbone_path), str(SNT_CKPT))

    def test_rejects_unknown_variant(self):
        from fungidna.representation import RepresentationModel

        with self.assertRaises(ValueError):
            RepresentationModel(variant="bpe", tokenizer_path=TOKENIZER,
                                backbone_path=UNIGRAM_CKPT)

    def test_empty_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            self.uni.embed("")


class TestInferenceDoesNotDependOnTraining(unittest.TestCase):
    def test_representation_import_chain_avoids_training_losses(self):
        import subprocess

        code = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "import fungidna.representation\n"
            "import fungidna.invoke.handlers.base\n"
            "assert 'fungidna.training.losses' not in sys.modules, 'training.losses imported'\n"
            "print('ok')\n" % str(Path(__file__).resolve().parent.parent)
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(CUDA, "model requires CUDA (flash-attn / mamba-ssm kernels)")
@unittest.skipUnless(TOKENIZER and UNIGRAM_CKPT and SNT_CKPT, "real checkpoints not available")
class TestLayer8Representation(unittest.TestCase):
    """Note S2: representations come from layer 8 of 24 — the attention layer of
    the second hybrid block — pooled over non-padding positions, before
    ``final_norm``.
    """

    LAYER_INDEX = 7  # 0-based index of paper layer 8

    @classmethod
    def setUpClass(cls):
        from fungidna.representation import RepresentationModel

        cls.models = {
            "unigram": RepresentationModel(
                variant="unigram", tokenizer_path=TOKENIZER, backbone_path=UNIGRAM_CKPT
            ),
            "snt": RepresentationModel(
                variant="snt", tokenizer_path=TOKENIZER, backbone_path=SNT_CKPT
            ),
        }

    def _ids(self, model, seq):
        if model.variant == "snt":
            return model.tokenizer.encode_single_nt(seq, add_cls=False)
        return model.tokenizer.encode_bpe(seq, add_cls=False)

    def _manual_layer8(self, model, seq):
        """Recompute the layer-8 hidden state by walking the backbone by hand."""
        ids = torch.tensor([self._ids(model, seq)], dtype=torch.long, device=model.device)
        return manual_layer_hidden(model, ids, layer_index=self.LAYER_INDEX)

    def test_layer_7_is_an_attention_block(self):
        from fungidna.model.attention_block import AttentionBlock

        for model in self.models.values():
            self.assertIsInstance(model.backbone.layers[self.LAYER_INDEX], AttentionBlock)
            self.assertEqual(model.backbone.config.mamba_per_attention, 3)

    def test_captured_hidden_equals_direct_layer8(self):
        for name, model in self.models.items():
            model.embed(SEQ_A)
            captured = model.last_layer_output.float().cpu().numpy()
            manual = self._manual_layer8(model, SEQ_A).cpu().numpy()
            np.testing.assert_allclose(
                captured, manual, rtol=1e-4, atol=1e-4,
                err_msg=f"{name}: captured hidden state is not layer {self.LAYER_INDEX}",
            )

    def test_embedding_is_masked_mean_of_layer8(self):
        for name, model in self.models.items():
            vec = model.embed(SEQ_A)
            manual = self._manual_layer8(model, SEQ_A).mean(dim=1)[0].cpu().numpy()
            np.testing.assert_allclose(
                vec, manual, rtol=1e-4, atol=1e-4,
                err_msg=f"{name}: embed() is not the mean of layer {self.LAYER_INDEX}",
            )

    def test_does_not_use_final_layer(self):
        for name, model in self.models.items():
            ids = torch.tensor([self._ids(model, SEQ_A)], dtype=torch.long,
                               device=model.device)
            with torch.no_grad(), torch.amp.autocast(device_type=model.device.type,
                                                     dtype=torch.bfloat16):
                final = model.backbone(ids, token_type=model.token_type, add_cls=False)
            final_vec = final.float().mean(dim=1)[0].cpu().numpy()
            self.assertFalse(
                np.allclose(model.embed(SEQ_A), final_vec, rtol=1e-3, atol=1e-3),
                f"{name}: embedding still matches the final-layer output",
            )


if __name__ == "__main__":
    unittest.main()
