"""Prompt-driven terminal parsing: regex first, optional LLM fallback."""
import re

from .tasks import normalize_task

_ALIAS_RE = re.compile(
    r"\b(representations?|embeddings?|taxonomic classification|classification|"
    r"coding[- ]?potential|splice[- ]?sites?|splice|bgc[- ]?boundary|bgc)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?:^|\s)(/(?:[^\s'\"()]+)?[A-Za-z0-9_./-]+\.(?:fa|fasta|fq|fastq))(?:\s|$)")


def parse_prompt(text, llm_extractor=None):
    task = None
    file_path = None

    m = _ALIAS_RE.search(text)
    if m:
        task = normalize_task(m.group(0))

    m = _PATH_RE.search(text)
    if m:
        file_path = m.group(1)

    if (task is None or file_path is None) and llm_extractor is not None:
        candidate = llm_extractor(text) or {}
        if task is None and candidate.get("task"):
            task = normalize_task(candidate["task"])
        if file_path is None and candidate.get("file_path"):
            file_path = candidate["file_path"]

    return {"task": task, "file_path": file_path}
