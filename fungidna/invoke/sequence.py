"""FASTA/FASTQ reading used by the validation gate and task handlers."""
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".fa", ".fasta", ".fq", ".fastq"})


def _first_nonempty_char(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s:
                return s[0]
    return ""


def first_record_valid(path):
    c = _first_nonempty_char(path)
    return c in (">", "@")


def first_record_wellformed(path):
    """Try to parse the first record; True only if it is structurally valid.

    This is stronger than :func:`first_record_valid` (which only checks the
    first character): it actually reads the first record and confirms a FASTA
    record has a header plus at least one non-empty sequence line, and a FASTQ
    record has a header, a non-empty sequence, and a ``+`` line.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline()
    if first_line.startswith(">"):
        seq = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            next(fh, None)  # skip the header line
            for line in fh:
                s = line.strip()
                if s.startswith(">"):  # next record — stop at the first record
                    break
                if s:
                    seq.append(s)
        return bool(seq)
    if first_line.startswith("@"):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            _header = fh.readline()
            seq = fh.readline()
            plus = fh.readline()
        return bool(seq.strip()) and plus.startswith("+")
    return False


def read_sequences(path):
    """Return [(header, sequence), ...] for a FASTA/FASTQ file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    first = raw.lstrip()
    if not first:
        return []
    if first[0] == ">":
        return _read_fasta(raw)
    if first[0] == "@":
        return _read_fastq(raw)
    raise ValueError(f"unrecognized sequence format in {path}")


def _read_fasta(raw):
    records = []
    header = None
    parts = []
    for line in raw.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts)))
            header = line
            parts = []
        else:
            parts.append(line.strip())
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def _read_fastq(raw):
    records = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("@"):
            header = lines[i]
            seq = lines[i + 1].strip() if i + 1 < len(lines) else ""
            records.append((header, seq))
            i += 4  # @, seq, +, qual
        else:
            i += 1
    return records
