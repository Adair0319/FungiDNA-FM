"""Minimal Representation example for FungiDNA-FM.

    python examples/embed.py --variant unigram
    python examples/embed.py --variant snt --sequence ATGGCTAGCTAGCTA

Weights are located through ~/.config/fungidna/weights.yaml (see
configs/weights.example.yaml) or the FUNGIDNA_* environment variables.
"""
import argparse
import sys
from pathlib import Path

# Run from a source checkout (`python examples/embed.py ...`). No-op when the
# package is already importable (e.g. installed with `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungidna.representation import RepresentationModel

DEFAULT_SEQUENCE = "ATGGCTAGCTAGCTAGGCTAACCGGTTACGTAGCTAGCATCGATCGATCGTAGCTAGCTAGCTA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("unigram", "snt"), default="unigram")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--batch", action="store_true",
                        help="also embed two sequences at once to show the batch shape")
    args = parser.parse_args()

    model = RepresentationModel(variant=args.variant)

    vector = model.embed(args.sequence)
    print(f"variant   : {args.variant}")
    print(f"backbone  : {model.backbone_path}")
    print(f"sequence  : {len(args.sequence)} bp")
    print(f"embedding : shape={vector.shape} dtype={vector.dtype}")
    print(f"first 5   : {vector[:5].round(4).tolist()}")

    if args.batch:
        matrix = model.embed([args.sequence, "GCGCGCGCGCGCGCGCGCGC"])
        print(f"batch     : shape={matrix.shape}")


if __name__ == "__main__":
    main()
