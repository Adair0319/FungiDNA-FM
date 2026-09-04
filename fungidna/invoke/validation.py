"""Deterministic validation gate — the only gate before task execution."""
import os
from pathlib import Path

from .request import StructuredRequest, task_from_string
from .sequence import SUPPORTED_EXTENSIONS, first_record_wellformed
from .tasks import normalize_task


class ValidationError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def validate(task_raw, file_path_raw):
    task_name = normalize_task(task_raw)
    if task_name is None:
        raise ValidationError(
            "TASK_UNSUPPORTED",
            f"unsupported or missing task: {task_raw!r}; "
            f"supported: representation, taxonomy, coding_potential, splice_site, bgc_boundary",
        )
    if file_path_raw is None or str(file_path_raw).strip() == "":
        raise ValidationError("MISSING_FILE_PATH", "no file path provided")
    path = Path(str(file_path_raw)).expanduser()
    if not path.exists():
        raise ValidationError("FILE_NOT_FOUND", f"file not found: {path}")
    if not path.is_file():
        raise ValidationError("FILE_NOT_FOUND", f"not a regular file: {path}")
    if not os.access(path, os.R_OK):
        raise ValidationError("FILE_UNREADABLE", f"file not readable: {path}")
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            "UNSUPPORTED_FORMAT",
            f"unsupported extension {path.suffix!r}; supported: .fa .fasta .fq .fastq (uncompressed)",
        )
    if not first_record_wellformed(path):
        raise ValidationError(
            "INVALID_SEQUENCE_FORMAT",
            "first record is not a well-formed FASTA ('>') or FASTQ ('@') record",
        )
    return StructuredRequest(task=task_from_string(task_name), file_path=path.resolve())
