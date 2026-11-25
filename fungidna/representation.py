"""Representation extraction (768-d embeddings) for both released variants.

Pooling contract
----------------
Hidden states come from **paper layer 8 of 24**, taken before the backbone's
``final_norm``, and are mean-pooled over real (non-padding) tokens only. No
``[CLS]`` is involved: both the tokenizer call and the backbone call use
``add_cls=False``.

"Layer 8" is the module at index 7, which is an *attention block*: the
attention sub-layer followed by its SwiGLU sub-layer and both residual adds.
The hook captures the block's output, not the output of the attention op alone.

This is the definition used in Supplementary Note S2 and in the development
scripts that produced the paper's clustering numbers (the adjusted Rand indices
0.152 at the end of Phase 1 and 0.550 after Phase 2)::

    # scripts/checkpoint_ablation.py (development repository; not shipped here)
    ATTENTION_LAYER = 7
    hidden = BLOCK_OUTPUT[1].float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

The same pooling is used by ``scripts/umap_clustering_elements.py`` (also a
development script).

The single-nucleotide variant is a different case. The historical SNT
representation script used the **final** backbone output and averaged over all
positions including padding::

    # scripts/contrastive_representation_eval.py (development repository)
    h = bb(p, token_type=1, add_cls=False)
    feats.append(h.float().mean(dim=1))

No paper figure reports SNT layer-8 embeddings. This class deliberately applies
the Note S2 definition to both variants; it does not reproduce that historical
SNT script.

The ``add_cls=True`` conventions belong to downstream task heads (which read
``hidden[:, 0]`` or ``hidden[:, 1:].mean``); they are *not* the representation.

Batching
--------
The backbone's attention layers are bidirectional (``flash_attn_func(..., causal=False)``)
and take no padding mask, so trailing padding leaks into the hidden states of
neighbouring positions. Embeddings are therefore computed in groups of equal
token length, which keeps a sequence's vector independent of the other sequences
in the same call.

Concurrency
-----------
Layer-8 capture uses one instance-level slot (``_layer_output``), so a single
``RepresentationModel`` instance is **not** thread-safe: concurrent ``embed``
calls would overwrite each other's captured hidden state. Serial calls on one
instance are the supported usage; use one instance per thread otherwise.

Device
------
The backbone uses ``flash-attn`` and ``mamba-ssm`` kernels; both are CUDA-only,
so practical inference requires a GPU. ``device`` is accepted and honoured, but a
CPU forward is not supported by the underlying kernels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fungidna.data.tokenizer import PAD_ID, VARIANT_TOKEN_TYPE, DualTokenizer
from fungidna.invoke.handlers.base import (
    VARIANTS,
    WeightLoader,
    default_device,
    load_checkpoint,
)

_VARIANT_TOKEN_TYPE = VARIANT_TOKEN_TYPE   # single source of truth

#: Note S2: representations are taken "at the output of the attention layer of
#: the second hybrid block (layer 8 of 24)". With six blocks of
#: (3 × Mamba2 + 1 × attention), paper layer *L* is ``layers[L - 1]``, so layer 8
#: is the AttentionBlock at index 7 — the last layer of the second block. The
#: output is taken *before* the backbone's ``final_norm``.
REPRESENTATION_LAYER_INDEX = 7


class RepresentationModel:
    """Load one variant's backbone and embed DNA sequences.

    Args:
        variant: ``"unigram"`` or ``"snt"``.
        tokenizer_path: SentencePiece model path. Resolved from the weights
            config / env when omitted.
        backbone_path: Phase-2 backbone checkpoint. Resolved from the weights
            config / env when omitted.
        device: ``torch.device``-compatible string. Defaults to CUDA when available.
        batch_size: maximum number of equal-length sequences per forward pass.
    """

    def __init__(
        self,
        variant: str = "unigram",
        tokenizer_path=None,
        backbone_path=None,
        device=None,
        batch_size: int = 8,
    ):
        name = str(variant).strip().lower()
        if name not in _VARIANT_TOKEN_TYPE:
            raise ValueError(
                f"unknown variant {variant!r}; supported: {', '.join(VARIANTS)}"
            )
        self.variant = name
        self.token_type = _VARIANT_TOKEN_TYPE[name]
        self.batch_size = max(1, int(batch_size))

        import torch

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.striped_mamba import StripedMambaBackbone

        loader = WeightLoader()
        self.tokenizer_path = Path(
            tokenizer_path if tokenizer_path is not None else loader.resolve_tokenizer()
        )
        self.backbone_path = Path(
            backbone_path
            if backbone_path is not None
            else loader.resolve_backbone(variant=self.variant)
        )
        self.device = torch.device(device) if device is not None else default_device()

        self.tokenizer = DualTokenizer(str(self.tokenizer_path))
        self.backbone = StripedMambaBackbone(FungiDNAConfig())
        # The Unigram checkpoint stores a 5-row nt_embed against the shared
        # architecture's 10 rows; it is unused (token_type=0) and skipped.
        skip = () if self.variant == "snt" else ("nt_embed",)
        load_checkpoint(
            self.backbone,
            self.backbone_path,
            unwrap=None,
            skip_size_mismatch=skip,
            device=self.device,
        )
        self.backbone.eval()

        # Capture layer 8 without changing the backbone's public interface. The
        # hook only stores the module output; it is reset before every forward so
        # a later call can never read a stale tensor.
        self._layer_output = None
        self.backbone.layers[REPRESENTATION_LAYER_INDEX].register_forward_hook(
            self._capture_layer_output
        )

    def _capture_layer_output(self, module, inputs, output):
        self._layer_output = output

    @property
    def last_layer_output(self):
        """Hidden state of paper layer 8 from the most recent ``embed`` call."""
        if self._layer_output is None:
            raise RuntimeError("no forward pass has been run yet")
        return self._layer_output

    # ── public API ──────────────────────────────────────────────────────────
    def embed(self, sequences):
        """Embed one sequence (returns ``(768,)``) or several (``(n, 768)``)."""
        single = isinstance(sequences, str)
        seqs = [sequences] if single else list(sequences)
        if not seqs:
            raise ValueError("no sequences provided")
        cleaned = [self._clean(s) for s in seqs]
        vectors = [None] * len(cleaned)
        for indices in self._equal_length_groups(cleaned):
            # Grouping is by length, so results must be written back by original
            # index — otherwise the output order would not match the input order.
            for idx, vec in zip(indices, self._forward_group(
                [cleaned[i] for i in indices]
            )):
                vectors[idx] = vec
        out = np.asarray(vectors, dtype=np.float32)
        return out[0] if single else out

    def _clean(self, seq) -> str:
        if not isinstance(seq, str):
            raise ValueError(f"sequence must be a string, got {type(seq).__name__}")
        text = seq.strip().upper()
        if not text:
            raise ValueError("empty sequence")
        return text

    def _encode(self, seq: str):
        if self.variant == "snt":
            return self.tokenizer.encode_single_nt(seq, add_cls=False)
        return self.tokenizer.encode_bpe(seq, add_cls=False)

    def _equal_length_groups(self, seqs):
        """Group input indices by encoded length, then chunk.

        Padding would perturb neighbouring positions (attention is
        non-causal and unmasked), so only equal-length sequences share a forward.
        """
        by_length = {}
        for index, seq in enumerate(seqs):
            by_length.setdefault(len(self._encode(seq)), []).append(index)
        groups = []
        for length in sorted(by_length):
            indices = by_length[length]
            for start in range(0, len(indices), self.batch_size):
                groups.append(indices[start:start + self.batch_size])
        return groups

    def _forward_group(self, seqs):
        import torch

        ids = [self._encode(s) for s in seqs]
        max_len = max(len(x) for x in ids)
        padded = torch.full((len(ids), max_len), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(ids), max_len), dtype=torch.float32)
        for i, row in enumerate(ids):
            padded[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1.0
        padded = padded.to(self.device)
        mask = mask.to(self.device)

        amp_device = self.device.type
        amp_dtype = torch.bfloat16 if amp_device == "cuda" else torch.float32
        self._layer_output = None
        with torch.no_grad(), torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            self.backbone(padded, token_type=self.token_type, add_cls=False)
        hidden = self._layer_output
        if hidden is None:  # pragma: no cover - hook must fire on every forward
            raise RuntimeError(
                f"layer {REPRESENTATION_LAYER_INDEX} output was not captured"
            )

        hidden = hidden.float()
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-9)
        return pooled.cpu().numpy().astype(np.float32)
