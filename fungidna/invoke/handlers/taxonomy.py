"""Taxonomy task: classify sequences at the phylum rank (default) using a
mean-pooled backbone embedding followed by the trained per-rank MLP head.

The head is the small MLP saved by ``train_phase2_joint_mlp.py``
(``net.0`` Linear 768→256, ``net.1`` BatchNorm1d, GELU, Dropout,
``net.4`` Linear 256→num_classes). The backbone is the shared pretrained
``backbone_final.pt``. Both are resolved via the weight loader.
"""
import torch
import torch.nn as nn

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader, load_checkpoint

_WINDOW_BP = 10_000          # 10 kb windows, contiguous non-overlapping
_STEP_BP = 10_000
_DEFAULT_RANK = "phylum"
_DEFAULT_NUM_CLASSES = 6     # phylum


class _TaxonomyHead(nn.Module):
    """Mean-pooled-768 → 256 → num_classes MLP (matches train_phase2_joint_mlp.py)."""

    def __init__(self, num_classes=_DEFAULT_NUM_CLASSES, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class TaxonomyHandler(TaskHandler):
    task = Task.TAXONOMY

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        backbone_ckpt = self.loader.resolve_backbone()
        head_ckpt = self.loader.resolve(self.task.value)
        tokenizer_path = self.loader.resolve_tokenizer()

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.striped_mamba import StripedMambaBackbone
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(tokenizer_path))
        config = FungiDNAConfig()
        backbone = StripedMambaBackbone(config)
        load_checkpoint(backbone, backbone_ckpt, unwrap=None)
        backbone.eval()

        head = _TaxonomyHead(num_classes=_DEFAULT_NUM_CLASSES)
        load_checkpoint(head, head_ckpt, unwrap=None)  # bare state dict, keys "net.*"
        head.eval()

        labels = {}
        with torch.no_grad():
            for header, seq in read_sequences(request.file_path):
                votes = []
                for start in range(0, len(seq), _STEP_BP):
                    window = seq[start:start + _WINDOW_BP]
                    if not window:
                        continue
                    ids = torch.tensor([tok.encode_bpe(window, add_cls=True)], dtype=torch.long)
                    pooled = backbone(ids).mean(dim=1)
                    logits = head(pooled)
                    votes.append(int(logits.argmax(dim=1).item()))
                labels[header] = max(range(_DEFAULT_NUM_CLASSES), key=votes.count) if votes else None
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"rank": _DEFAULT_RANK, "labels": labels})
