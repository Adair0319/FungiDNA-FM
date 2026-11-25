"""python -m fungidna "<natural-language request>" [--variant snt] [--rank family]

Figure 1D: terminal users enter natural-language requests. The request is parsed
into one of the six window-level tasks, then validated and dispatched. The
variant and, for taxonomy, the rank are optional flags.
"""
import argparse
import dataclasses
import sys

from .invoke import parse_prompt, validate, dispatch
from .invoke.parser import ParserError
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
    parser.add_argument("--variant", choices=("unigram", "snt"), default=None,
                        help="FungiDNA-FM-Unigram (default) or FungiDNA-FM-SNT; "
                             "overrides any variant mentioned in the request")
    parser.add_argument("--rank", choices=_RANKS, default=None,
                        help="taxonomy rank (only used by the taxonomy task); "
                             "overrides any rank mentioned in the request")
    return parser


def _resolve_options(args, parsed):
    """CLI flags win over text parsing; text wins over the documented defaults.

    A taxonomy rank only exists for the taxonomy task: every other request gets
    ``None`` so no meaningless rank is carried into the handler.
    """
    variant = args.variant or parsed.get("variant") or "unigram"
    if parsed.get("task") != "taxonomy":
        return variant, None
    rank = args.rank or parsed.get("rank") or "phylum"
    return variant, rank


def main(argv=None):
    args = _build_parser().parse_args(argv)
    text = " ".join(args.text).strip()
    if not text:
        print('usage: python -m fungidna "<natural-language request>" '
              '[--variant unigram|snt] [--rank phylum|...]', file=sys.stderr)
        return 2
    try:
        parsed = parse_prompt(text)
        variant, rank = _resolve_options(args, parsed)
        if args.rank and parsed.get("task") != "taxonomy":
            print("note: --rank is only used by the taxonomy task; ignored",
                  file=sys.stderr)
        req = validate(parsed["task"], parsed["file_path"])
        req = dataclasses.replace(req, variant=variant, rank=rank)
        report = dispatch(req)
        print(report.to_json())
        return 0
    except ParserError as exc:
        print(f"error [INVALID_REQUEST]: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1
    except WeightsNotFoundError as exc:
        print(f"error [WEIGHTS_NOT_FOUND]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
