"""python -m fungidna "<natural-language request>" """
import sys

from .invoke import parse_prompt, validate, dispatch
from .invoke.handlers.base import WeightsNotFoundError
from .invoke.validation import ValidationError


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = " ".join(argv).strip()
    if not text:
        print('usage: python -m fungidna "<natural-language request>"', file=sys.stderr)
        return 2
    parsed = parse_prompt(text)
    try:
        req = validate(parsed["task"], parsed["file_path"])
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
