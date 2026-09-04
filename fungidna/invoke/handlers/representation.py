"""Representation task: embed each sequence with the pretrained backbone."""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader, load_checkpoint


class RepresentationHandler(TaskHandler):
    task = Task.REPRESENTATION

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        # The pretrained backbone is a bare state dict (no "model" wrapper).
        ckpt = self.loader.resolve_backbone()
        tokenizer_path = self.loader.resolve_tokenizer()

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.striped_mamba import StripedMambaBackbone
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(tokenizer_path))
        model = StripedMambaBackbone(FungiDNAConfig())
        load_checkpoint(model, ckpt, unwrap=None)
        model.eval()

        embeddings = {}
        with torch.no_grad():
            for header, seq in read_sequences(request.file_path):
                ids = torch.tensor([tok.encode_bpe(seq, add_cls=True)], dtype=torch.long)
                hidden = model(ids)
                embeddings[header] = hidden.mean(dim=1).squeeze(0).float().tolist()
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"embeddings": embeddings, "dim": model.config.hidden_size})
