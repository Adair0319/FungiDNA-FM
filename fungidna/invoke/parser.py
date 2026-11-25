"""Prompt-driven terminal parsing: regex first, optional LLM fallback.

Figure 1D describes a terminal in which users enter natural-language requests.
The patterns below cover every window-level task evaluated in the paper, plus the
variant (Unigram / SNT) and — for taxonomy — the rank. Task patterns are checked
in priority order, so ``"short CDS"`` is resolved as ``short_cds`` before the
generic ``cds`` pattern can match it.

Rank support mirrors what the project actually ships: the five label maps are
phylum, subphylum, class, order and family (paper: "five ranks from phylum to
family"). Asking for any other level (``genus``, ``species``, ``kingdom``, …)
raises :class:`UnsupportedRankError` instead of silently defaulting.
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

#: Absolute paths, relative paths and quoted paths (which may contain spaces).
_PATH_RE = re.compile(
    r'(?P<quote>["\'])(?P<quoted>[^"\']+\.(?:fa|fasta|fq|fastq))(?P=quote)'
    r"|"
    r"(?<![\w/.-])(?P<plain>(?:/|\.{1,2}/)?(?:[\w.-]+/)*[\w.-]+\.(?:fa|fasta|fq|fastq))(?![\w])",
    re.IGNORECASE,
)

_VARIANT_PATTERNS = {
    "unigram": r"\bunigram\b",
    "snt": r"\bsnt\b|\bsingle[-_ ]nucleotide\b",
}

#: Ranks with a shipped label map and taxonomy head.
SUPPORTED_RANKS = ("phylum", "subphylum", "class", "order", "family")
#: Rank-like words the project does not model; an explicit request must fail.
UNSUPPORTED_RANKS = ("genus", "species", "kingdom", "strain", "subspecies")

DEFAULT_VARIANT = "unigram"
DEFAULT_RANK = "phylum"   # matches StructuredRequest and the taxonomy handler


class ParserError(ValueError):
    """Base class for request-level parsing errors."""


class AmbiguousRequestError(ParserError):
    """The request names more than one variant or more than one rank."""


class UnsupportedRankError(ParserError):
    """The request names a taxonomy rank the project does not model."""


def match_task(text):
    """Return the canonical task named in ``text``, or ``None``."""
    if not text:
        return None
    for name, pattern in _TASK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return normalize_task(name) or name
    return None


def _split_path(text):
    """Return ``(file_path, text_without_path)``."""
    m = _PATH_RE.search(text)
    if m is None:
        return None, text
    path = m.group("quoted") or m.group("plain")
    return path, text[: m.start()] + " " + text[m.end():]


def _match_variant(words):
    found = sorted(name for name, pattern in _VARIANT_PATTERNS.items()
                   if re.search(pattern, words, re.IGNORECASE))
    if len(found) > 1:
        raise AmbiguousRequestError(
            f"request names conflicting variants ({', '.join(found)}); "
            f"pick one with --variant"
        )
    return found[0] if found else None


def _match_rank(words, task):
    if task != "taxonomy":
        return None
    unsupported = [r for r in UNSUPPORTED_RANKS
                   if re.search(rf"\b{r}\b", words, re.IGNORECASE)]
    if unsupported:
        raise UnsupportedRankError(
            f"unsupported taxonomy rank(s): {', '.join(unsupported)}; "
            f"supported: {', '.join(SUPPORTED_RANKS)}"
        )
    found = [r for r in SUPPORTED_RANKS
             if re.search(rf"\b{r}\b", words, re.IGNORECASE)]
    if len(found) > 1:
        raise AmbiguousRequestError(
            f"request names conflicting ranks ({', '.join(found)}); pick one with --rank"
        )
    return found[0] if found else DEFAULT_RANK


def _validate_variant(value):
    name = str(value).strip().lower()
    if name not in _VARIANT_PATTERNS:
        raise AmbiguousRequestError(
            f"LLM extractor returned an unknown variant {value!r}; "
            f"expected one of {sorted(_VARIANT_PATTERNS)}"
        )
    return name


def _validate_rank(value):
    name = str(value).strip().lower()
    if name in UNSUPPORTED_RANKS:
        raise UnsupportedRankError(
            f"unsupported taxonomy rank {value!r}; supported: {', '.join(SUPPORTED_RANKS)}"
        )
    if name not in SUPPORTED_RANKS:
        raise AmbiguousRequestError(
            f"LLM extractor returned an unknown rank {value!r}; "
            f"expected one of {SUPPORTED_RANKS}"
        )
    return name


def parse_prompt(text, llm_extractor=None):
    """Parse a natural-language request.

    Returns ``{"task", "file_path", "variant", "rank"}``. ``task`` and
    ``file_path`` keep their original semantics and remain ``None`` when not
    found; ``variant`` falls back to ``DEFAULT_VARIANT`` and ``rank`` is ``None``
    outside the taxonomy task.
    """
    text = text or ""
    file_path, words = _split_path(text)
    task = match_task(words)
    variant = _match_variant(words)
    rank = _match_rank(words, task)

    # The LLM fallback keeps its original trigger condition (missing task or
    # path); it may additionally supply a variant/rank that the regex missed.
    if (task is None or file_path is None) and llm_extractor is not None:
        candidate = llm_extractor(text) or {}
        if task is None and candidate.get("task"):
            task = normalize_task(candidate["task"])
            if task == "taxonomy" and rank is None:
                rank = _match_rank(words, task)
        if file_path is None and candidate.get("file_path"):
            file_path = candidate["file_path"]
        if variant is None and candidate.get("variant"):
            variant = _validate_variant(candidate["variant"])
        if rank is None and task == "taxonomy" and candidate.get("rank"):
            rank = _validate_rank(candidate["rank"])

    if task != "taxonomy":
        rank = None

    return {
        "task": task,
        "file_path": file_path,
        "variant": variant or DEFAULT_VARIANT,
        "rank": rank,
    }
