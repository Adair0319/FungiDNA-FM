"""python -m fungidna "<natural-language request>" [--variant snt] [--rank family]

Figure 1D: terminal users enter natural-language requests. The request is parsed
into one of the six window-level tasks, then validated and dispatched. The
variant and, for taxonomy, the rank are optional flags.
"""
import argparse
import dataclasses
import sys

from .invoke import parse_prompt, validate, dispatch
from .invoke.handlers.base import WeightsNotFoundError
from .invoke.validation import ValidationError

_RANKS = ("phylum", "subphylum", "class", "order", "family")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m fungidna",
        description="Run a window-level FungiDNA-FM task from a natural-language request.",
    )
    parser.add_argument("text", nargs="*", help="natural-language request, e.g. "
                        '"extract the representation of /path/seq.fasta"')
    parser.add_argument("--variant", choices=("unigram", "snt"), default="unigram",
                        help="FungiDNA-FM-Unigram (default) or FungiDNA-FM-SNT")
    parser.add_argument("--rank", choices=_RANKS, default="phylum",
                        help="taxonomy rank (only used by the taxonomy task)")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    text = " ".join(args.text).strip()
    if not text:
        print('usage: python -m fungidna "<natural-language request>" '
              '[--variant unigram|snt] [--rank phylum|...]', file=sys.stderr)
        return 2
    parsed = parse_prompt(text)
    try:
        req = validate(parsed["task"], parsed["file_path"])
        req = dataclasses.replace(req, variant=args.variant, rank=args.rank)
        report = dispatch(req)
        print(report.to_json())
        return 0
    except ValidationError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1
    except WeightsNotFoundError as exc:
        print(f"error [WEIGHTS_NOT_FOUND]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
