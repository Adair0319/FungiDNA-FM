"""Structured output returned by every task handler."""
import dataclasses
import json
from pathlib import Path

from .request import Task


@dataclasses.dataclass
class Report:
    task: Task
    file_path: Path
    status: str
    results: dict

    def to_json(self):
        return json.dumps(
            {
                "task": self.task.value,
                "file_path": str(self.file_path),
                "status": self.status,
                "results": self.results,
            },
            indent=2,
            default=str,
        )
