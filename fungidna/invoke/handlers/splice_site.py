"""Splice-site task: classify candidate splice motifs as donor/acceptor/non-site.

Each candidate is a GT/GC (donor) or AG (acceptor) dinucleotide and is
classified from its surrounding +/-200 bp window into 3 classes
(0 donor, 1 acceptor, 2 non-site).

Head and pooling differ per variant, matching the checkpoints:

* Unigram — ``ClassificationHead(hidden_dim=256)`` reading ``hidden[:, 0]``;
* SNT — ``MeanPoolHead(hidden_dim=128)`` averaging ``hidden[:, 1:]``.
"""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader, load_checkpoint

_SPLICE_MOTIFS = {"GT", "GC", "AG"}
_WINDOW_HALF = 200           # ±200 bp around the candidate site → 401 bp window
_NUM_CLASSES = 3             # donor / acceptor / non-site


class SpliceSiteHandler(TaskHandler):
    task = Task.SPLICE_SITE

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        variant = getattr(request, "variant", "unigram")
        token_type, is_snt = (1, True) if variant == "snt" else (0, False)
        ckpt = self.loader.resolve(self.task.value, variant=variant)
        tokenizer_path = self.loader.resolve_tokenizer()

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.fungi_dna import FungiDNAForSequenceClassification
        from fungidna.model.heads import MeanPoolHead
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(tokenizer_path))
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
                seq = seq.upper()
                hits = []
                for i in range(len(seq) - 1):
                    motif = seq[i:i + 2]
                    if motif not in _SPLICE_MOTIFS:
                        continue
                    window = seq[max(0, i - _WINDOW_HALF): i + _WINDOW_HALF + 1]
                    if is_snt:
                        ids = tok.encode_single_nt(window, add_cls=False)
                    else:
                        ids = tok.encode_bpe(window, add_cls=True)
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    cls = int(probs.argmax(dim=0).item())
                    hits.append({"pos": i, "motif": motif, "class": cls,
                                 "prob": float(probs[cls].item())})
                predictions[header] = hits
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"predictions": predictions})
