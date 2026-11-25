"""Prompt-driven terminal parsing: regex first, optional LLM fallback.

Figure 1D describes a terminal in which users enter natural-language requests.
The patterns below cover every window-level task evaluated in the paper. They
are checked in priority order, so ``"short CDS"`` is resolved as ``short_cds``
before the generic ``cds`` pattern can match it.
"""
import re

from .tasks import normalize_task

#: (canonical task, pattern), highest priority first.
_TASK_PATTERNS = (
    ("short_cds", r"\bshort[- _]?cds\b|\bshort[- _]?coding\b"),
    ("cds", r"\bcds\b|\bcoding[- _]?sequence"),
    ("splice_site", r"\bsplice[- _]?sites?\b|\bsplice\b"),
    ("bgc_boundary", r"\bbgc\b|\bboundar(?:y|ies)\b"),
    ("embedding", r"\bembeddings?\b|\brepresentations?\b|\brepresent"),
    ("taxonomy", r"\btaxonom(?:y|ic(?:al)?)\b|\bclassif(?:y|ication)\b"),
)

_PATH_RE = re.compile(
    r"(?:^|\s)(/(?:[^\s'\"()]+)?[A-Za-z0-9_./-]+\.(?:fa|fasta|fq|fastq))(?:\s|$)"
)


def match_task(text):
    """Return the canonical task named in ``text``, or ``None``."""
    if not text:
        return None
    for name, pattern in _TASK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return normalize_task(name) or name
    return None


def parse_prompt(text, llm_extractor=None):
    task = match_task(text)
    file_path = None

    m = _PATH_RE.search(text or "")
    if m:
        file_path = m.group(1)

    if (task is None or file_path is None) and llm_extractor is not None:
        candidate = llm_extractor(text) or {}
        if task is None and candidate.get("task"):
            task = normalize_task(candidate["task"])
        if file_path is None and candidate.get("file_path"):
            file_path = candidate["file_path"]

    return {"task": task, "file_path": file_path}
