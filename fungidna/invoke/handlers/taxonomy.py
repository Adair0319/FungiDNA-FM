"""Taxonomy task: classify sequences at a taxonomic rank using a pooled
backbone representation followed by the trained per-rank MLP head.

The two variants were trained with different probes and pooling, and the
handlers reproduce each exactly:

* Unigram — ``Linear(768->256) + BatchNorm1d + GELU + Dropout + Linear`` over
  ``hidden.mean(dim=1)`` (mean over **all** positions, CLS included);
* SNT — ``LayerNorm(768) + Linear(768->256) + GELU + Dropout + Linear`` over
  ``hidden[:, 1:].mean(dim=1)`` (mean over positions after the prepended CLS).
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader, load_checkpoint

_WINDOW_BP = 10_000          # 10 kb windows, contiguous non-overlapping
_STEP_BP = 10_000
_DEFAULT_RANK = "phylum"

#: num_classes confirmed per rank (only phylum is hardcoded); other ranks are
#: inferred from the head checkpoint's output layer at load time.
_RANK_NUM_CLASSES = {"phylum": 6}

#: Directory of the per-rank ``{name: id}`` label maps shipped with the package.
_LABEL_MAP_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "taxonomy"


def load_label_map(rank):
    """Return ``(name_to_id, id_to_name)`` for a taxonomy rank.

    The on-disk file maps category name -> integer id; ``id_to_name`` is the
    inverse used to turn a predicted class id back into a category name.
    """
    path = _LABEL_MAP_DIR / f"{rank}_label_map.json"
    with open(path) as fh:
        name_to_id = json.load(fh)
    id_to_name = {int(v): k for k, v in name_to_id.items()}
    return name_to_id, id_to_name


class _UnigramTaxonomyHead(nn.Module):
    """Mean-of-all-positions 768 -> 256 -> num_classes (BatchNorm MLP)."""

    def __init__(self, num_classes, hidden_dim=256, dropout=0.3):
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


class _SntTaxonomyHead(nn.Module):
    """Mean-of-positions-1: 768 -> 256 -> num_classes (LayerNorm MLP)."""

    def __init__(self, num_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, hidden_dim),
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
        variant = getattr(request, "variant", "unigram")
        token_type, is_snt = (1, True) if variant == "snt" else (0, False)
        rank = getattr(request, "rank", _DEFAULT_RANK)
        backbone_ckpt = self.loader.resolve_backbone(variant=variant)
        head_ckpt = self.loader.resolve(self.task.value, rank=rank, variant=variant)
        tokenizer_path = self.loader.resolve_tokenizer()

        from fungidna.model.config import FungiDNAConfig
        from fungidna.model.striped_mamba import StripedMambaBackbone
        from fungidna.data.tokenizer import DualTokenizer

        tok = DualTokenizer(str(tokenizer_path))
        config = FungiDNAConfig()
        backbone = StripedMambaBackbone(config)
        skip = () if is_snt else ("nt_embed",)
        load_checkpoint(backbone, backbone_ckpt, unwrap=None, skip_size_mismatch=skip)
        backbone.eval()
        device = next(backbone.parameters()).device

        num_classes = _RANK_NUM_CLASSES.get(rank)
        if num_classes is None:
            state = torch.load(head_ckpt, map_location="cpu", weights_only=True)
            num_classes = state["net.4.weight"].shape[0]
        head = (_SntTaxonomyHead if is_snt else _UnigramTaxonomyHead)(num_classes=num_classes)
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
                    if is_snt:
                        ids = tok.encode_single_nt(window, add_cls=False)
                    else:
                        ids = tok.encode_bpe(window, add_cls=True)
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        hidden = backbone(x, token_type=token_type)
                        pooled = (hidden[:, 1:, :] if is_snt else hidden).mean(dim=1)
                    logits = head(pooled.float())  # MLP head is fp32
                    votes.append(int(logits.argmax(dim=1).item()))
                labels[header] = max(range(num_classes), key=votes.count) if votes else None
        _, id_to_name = load_label_map(rank)
        labels_named = {h: id_to_name.get(i) for h, i in labels.items() if i is not None}
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"rank": rank, "labels": labels, "labels_named": labels_named})
