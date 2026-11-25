"""Typed structured request passed from either invocation mode to the dispatcher."""
import enum
import dataclasses
from pathlib import Path

from .tasks import CANONICAL_TASKS


class Task(str, enum.Enum):
    EMBEDDING = "embedding"
    TAXONOMY = "taxonomy"
    CDS = "cds"
    SHORT_CDS = "short_cds"
    SPLICE_SITE = "splice_site"
    BGC_BOUNDARY = "bgc_boundary"


def task_from_string(name):
    if name not in CANONICAL_TASKS:
        raise ValueError(f"unsupported task: {name!r}")
    return Task(name)


@dataclasses.dataclass(frozen=True)
class StructuredRequest:
    task: Task
    file_path: Path
    variant: str = "unigram"  # "unigram" or "snt"
    rank: str = "phylum"      # taxonomy rank (phylum/subphylum/class/order/family)
