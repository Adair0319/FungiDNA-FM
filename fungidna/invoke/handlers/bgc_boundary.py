"""BGC-boundary task: predict left/right distance (bp) to the BGC boundaries."""
import torch

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader, load_checkpoint


class BGCBoundaryHandler(TaskHandler):
    task = Task.BGC_BOUNDARY

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        variant = getattr(request, "variant", "unigram")
        token_type, is_snt = (1, True) if variant == "snt" else (0, False)
        ckpt = self.loader.resolve(self.task.value)
        tokenizer_path = self.loader.resolve_tokenizer()

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.fungi_dna import FungiDNAForBGCBoundary
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(tokenizer_path))
        model = FungiDNAForBGCBoundary(FungiDNAConfig(), num_outputs=2, token_type=token_type)
        # BGC checkpoints are saved as {"model": state_dict, "epoch": ..., "val_loss": ...}
        # so unwrap="model" extracts the weight dict.
        skip = () if is_snt else ("nt_embed",)
        load_checkpoint(model, ckpt, unwrap="model", skip_size_mismatch=skip)
        model.eval()
        device = next(model.parameters()).device

        boundaries = {}
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            for header, seq in read_sequences(request.file_path):
                if is_snt:
                    ids = tok.encode_single_nt(seq, add_cls=False)
                else:
                    ids = tok.encode_bpe(seq, add_cls=True)
                x = torch.tensor([ids], dtype=torch.long, device=device)
                out = model(x)
                left_bp, right_bp = (float(v) for v in out[0])
                boundaries[header] = {"left_bp": left_bp, "right_bp": right_bp}
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"boundaries": boundaries})
