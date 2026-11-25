"""Centralized task whitelist and alias normalization.

This is the single source of truth for supported tasks. Every other module
imports from here; do not duplicate the list elsewhere.
"""

CANONICAL_TASKS = (
    "embedding",
    "taxonomy",
    "cds",
    "short_cds",
    "splice_site",
    "bgc_boundary",
)

_TASK_ALIASES = {
    # embedding (formerly "representation")
    "embedding": "embedding",
    "embeddings": "embedding",
    "representation": "embedding",
    "representations": "embedding",
    # taxonomy
    "taxonomy": "taxonomy",
    "taxonomic classification": "taxonomy",
    "classification": "taxonomy",
    # cds (formerly "coding_potential", long CDS 300-5000 bp)
    "cds": "cds",
    "coding_potential": "cds",
    "coding potential": "cds",
    "coding-potential": "cds",
    # short_cds (<= 300 bp)
    "short_cds": "short_cds",
    "short cds": "short_cds",
    "short-cds": "short_cds",
    "sorf": "short_cds",
    # splice_site
    "splice_site": "splice_site",
    "splice site": "splice_site",
    "splice sites": "splice_site",
    "splice-site": "splice_site",
    "splice": "splice_site",
    # bgc_boundary
    "bgc_boundary": "bgc_boundary",
    "bgc boundary": "bgc_boundary",
    "BGC boundary": "bgc_boundary",
    "bgc": "bgc_boundary",
}


def normalize_task(raw):
    """Normalize a raw task string to its canonical name, or None if unsupported."""
    if raw is None:
        return None
    key = " ".join(str(raw).strip().lower().split())
    return _TASK_ALIASES.get(key)


TASK_ALIASES = _TASK_ALIASES  # exposed for introspection/tests
