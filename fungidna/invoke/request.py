"""Typed structured request passed from either invocation mode to the dispatcher."""
import enum
import dataclasses
from pathlib import Path

from .tasks import CANONICAL_TASKS


class Task(str, enum.Enum):
    REPRESENTATION = "representation"
    TAXONOMY = "taxonomy"
    CODING_POTENTIAL = "coding_potential"
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
