"""Minimal downstream-task example for FungiDNA-FM.

    python examples/predict_tasks.py --variant unigram --task cds
    python examples/predict_tasks.py --variant snt --task taxonomy --rank class

Weights are located through ~/.config/fungidna/weights.yaml (see
configs/weights.example.yaml) or the FUNGIDNA_<VARIANT>_<KEY>_WEIGHTS variables.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungidna.invoke import run_task

TASKS = ("taxonomy", "cds", "short_cds", "splice_site", "bgc_boundary")
RANKS = ("phylum", "subphylum", "class", "order", "family")
DEMO = "ATGGCTAGCTAGCTAGGCTAACCGGTTACGTAGCTAGCATCGATCGATCGTAGCTAGCTAGCTA"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("unigram", "snt"), default="unigram")
    parser.add_argument("--task", choices=TASKS, default="cds")
    parser.add_argument("--rank", choices=RANKS, default="phylum")
    parser.add_argument("--fasta", help="input FASTA/FASTQ (default: a demo sequence)")
    args = parser.parse_args()

    if args.fasta:
        fasta = args.fasta
    else:
        handle = tempfile.NamedTemporaryFile("w", suffix=".fasta", delete=False)
        handle.write(f">demo\n{DEMO}\n")
        handle.close()
        fasta = handle.name

    # Rank only exists for taxonomy; other tasks must not carry one.
    options = {"file_path": fasta, "task": args.task, "variant": args.variant}
    if args.task == "taxonomy":
        options["rank"] = args.rank
    report = run_task(**options)
    print(f"variant : {args.variant}")
    print(f"task    : {args.task}" + (f" (rank={args.rank})" if args.task == "taxonomy" else ""))
    print(report.to_json())


if __name__ == "__main__":
    main()
