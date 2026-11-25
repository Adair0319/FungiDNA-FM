"""CDS task: probability that each sequence is a coding sequence (300-5000 bp)."""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader, load_checkpoint


class CdsHandler(TaskHandler):
    task = Task.CDS

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        variant = getattr(request, "variant", "unigram")
        ckpt = self.loader.resolve(self.task.value)
        tokenizer_path = self.loader.resolve_tokenizer()

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.fungi_dna import FungiDNAForSequenceClassification
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(tokenizer_path))
        token_type, is_snt = (1, True) if variant == "snt" else (0, False)
        model = FungiDNAForSequenceClassification(FungiDNAConfig(), num_classes=2,
                                                  token_type=token_type)
        skip = () if is_snt else ("nt_embed",)
        load_checkpoint(model, ckpt, unwrap="model", skip_size_mismatch=skip)
        model.eval()
        device = next(model.parameters()).device

        predictions = {}
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            for header, seq in read_sequences(request.file_path):
                if is_snt:
                    ids = tok.encode_single_nt(seq, add_cls=False)
                else:
                    ids = tok.encode_bpe(seq, add_cls=True)
                x = torch.tensor([ids], dtype=torch.long, device=device)
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[0, 1].item()  # class 1 = CDS
                predictions[header] = {
                    "probability": prob,
                    "label": "cds" if prob >= 0.5 else "intergenic",
                }
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"predictions": predictions})
