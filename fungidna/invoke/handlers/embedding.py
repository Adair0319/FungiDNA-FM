"""Embedding task: embed each sequence with the pretrained backbone.

Supports both variants:
  * ``variant="unigram"`` — SentencePiece Unigram tokenizer, ``token_type=0``
    (uses the 4096-vocab embedding; the single-nucleotide ``nt_embed`` is unused
    and its size mismatch is explicitly skipped on load);
  * ``variant="snt"`` — single-nucleotide tokenizer, ``token_type=1``.
"""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader, load_checkpoint


class EmbeddingHandler(TaskHandler):
    task = Task.EMBEDDING

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        variant = getattr(request, "variant", "unigram")
        ckpt = self.loader.resolve_backbone()
        tokenizer_path = self.loader.resolve_tokenizer()

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.striped_mamba import StripedMambaBackbone
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(tokenizer_path))
        model = StripedMambaBackbone(FungiDNAConfig())
        token_type, is_snt = (1, True) if variant == "snt" else (0, False)
        # Unigram variant: the checkpoint's nt_embed is (5,768) vs the shared
        # model's (10,768); it is unused (token_type=0), so skip it explicitly.
        skip = () if is_snt else ("nt_embed",)
        load_checkpoint(model, ckpt, unwrap=None, skip_size_mismatch=skip)
        model.eval()
        device = next(model.parameters()).device

        embeddings = {}
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            for header, seq in read_sequences(request.file_path):
                if is_snt:
                    ids = tok.encode_single_nt(seq, add_cls=False)
                else:
                    ids = tok.encode_bpe(seq, add_cls=True)
                x = torch.tensor([ids], dtype=torch.long, device=device)
                hidden = model(x, token_type=token_type)
                embeddings[header] = hidden.mean(dim=1).squeeze(0).float().tolist()
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"embeddings": embeddings, "dim": model.config.hidden_size,
                               "variant": variant})
