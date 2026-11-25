"""Embedding task: embed each sequence with the pretrained backbone.

Supports both variants:
  * ``variant="unigram"`` — SentencePiece Unigram tokenizer, ``token_type=0``
    (uses the 4096-vocab embedding; the single-nucleotide ``nt_embed`` is unused
    and its size mismatch is explicitly skipped on load);
  * ``variant="snt"`` — single-nucleotide tokenizer, ``token_type=1``.

The actual pooling lives in :class:`fungidna.representation.RepresentationModel`
so the file-based API and the direct Python API cannot drift apart: no ``[CLS]``
is added and the mean is taken over DNA tokens only, padding excluded.
"""
from ..report import Report
from ..request import Task
from ..sequence import read_sequences
from .base import TaskHandler, WeightLoader


class EmbeddingHandler(TaskHandler):
    task = Task.EMBEDDING

    def __init__(self, loader=None):
        self.loader = loader or WeightLoader()

    def run(self, request):
        variant = getattr(request, "variant", "unigram")

        from fungidna.representation import RepresentationModel

        model = RepresentationModel(
            variant=variant,
            tokenizer_path=self.loader.resolve_tokenizer(),
            backbone_path=self.loader.resolve_backbone(variant=variant),
        )
        records = list(read_sequences(request.file_path))
        matrix = model.embed([sequence for _, sequence in records])

        embeddings = {
            header: vector.tolist() for (header, _), vector in zip(records, matrix)
        }
        return Report(task=request.task, file_path=request.file_path, status="ok",
                      results={"embeddings": embeddings, "dim": int(matrix.shape[1]),
                               "variant": variant})
