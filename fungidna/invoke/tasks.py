"""Centralized task whitelist and alias normalization.

This is the single source of truth for supported tasks. Every other module
imports from here; do not duplicate the list elsewhere.
"""

CANONICAL_TASKS = (
    "representation",
    "taxonomy",
    "coding_potential",
    "splice_site",
    "bgc_boundary",
)

_TASK_ALIASES = {
    # representation
    "representation": "representation",
    "representations": "representation",
    "embedding": "representation",
    "embeddings": "representation",
    # taxonomy
    "taxonomy": "taxonomy",
    "taxonomic classification": "taxonomy",
    "classification": "taxonomy",
    # coding potential
    "coding_potential": "coding_potential",
    "coding potential": "coding_potential",
    "coding-potential": "coding_potential",
    # splice site
    "splice_site": "splice_site",
    "splice site": "splice_site",
    "splice sites": "splice_site",
    "splice-site": "splice_site",
    "splice": "splice_site",
    # bgc boundary
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
