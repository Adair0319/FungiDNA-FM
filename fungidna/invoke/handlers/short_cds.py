"""Short-CDS task: probability that a short sequence (<= 300 bp) is a short CDS.

Head and pooling differ per variant, matching how each checkpoint was trained:

* Unigram — ``ClassificationHead(hidden_dim=256)`` reading ``hidden[:, 0]``;
* SNT — ``MeanPoolHead(hidden_dim=128)`` averaging ``hidden[:, 1:]``.
"""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader, load_checkpoint

_NUM_CLASSES = 2


class ShortCdsHandler(TaskHandler):
    task = Task.SHORT_CDS

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        variant = getattr(request, "variant", "unigram")
        ckpt = self.loader.resolve(self.task.value, variant=variant)
        tokenizer_path = self.loader.resolve_tokenizer()

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.fungi_dna import FungiDNAForSequenceClassification
        from fungidna.model.heads import MeanPoolHead
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(tokenizer_path))
        token_type, is_snt = (1, True) if variant == "snt" else (0, False)
        config = FungiDNAConfig()
        model = FungiDNAForSequenceClassification(config, num_classes=_NUM_CLASSES,
                                                  token_type=token_type)
        if is_snt:
            model.head = MeanPoolHead(config.hidden_size, _NUM_CLASSES)
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
                prob = torch.softmax(logits, dim=1)[0, 1].item()  # class 1 = short CDS
                predictions[header] = {
                    "probability": prob,
                    "label": "short_cds" if prob >= 0.5 else "intergenic_orf",
                }
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"predictions": predictions})
