#!/usr/bin/env python
"""Download the pretrained baseline model used in the paper.

GENA-LM (BERT-base pretrained on the Saccharomyces cerevisiae genome, ~110M
parameters) is the only pretrained baseline reported in the paper. The other
two baselines -- a 3-block 1-D CNN and a single randomly initialised Mamba2
layer -- are defined in `baselines/baseline_models.py` and trained end-to-end,
so they need no download.

CPU-only, no GPU required for download.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download


MODELS = {
    "GENA-LM": {
        "repo": "AIRI-Institute/gena-lm-110M-yeast",
        "local": "gena-lm-110M-yeast",
        "note": "Yeast fungal baseline, CC-BY-NC-SA 4.0"
    },
}


def download_model(name: str, info: dict, cache_dir: str) -> bool:
    """Download a single model from HuggingFace."""
    local_path = Path(cache_dir) / info["local"]
    if local_path.exists() and list(local_path.iterdir()):
        print(f"  [{name}] Already exists at {local_path}, skipping")
        return True

    print(f"  [{name}] Downloading {info['repo']}...")
    print(f"  [{name}] License: {info['note']}")
    try:
        snapshot_download(
            repo_id=info["repo"],
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  [{name}] Done -> {local_path}")
        return True
    except Exception as e:
        print(f"  [{name}] FAILED: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download baseline models")
    parser.add_argument("--cache-dir", default="./baselines/models")
    parser.add_argument("--models", default="all",
                       help="Comma-separated: DNABERT-2,GENA-LM,NT-v2-500M,Evo2-7B or 'all'")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected = args.models.split(",") if args.models != "all" else list(MODELS.keys())
    results = {}

    for name in selected:
        if name not in MODELS:
            print(f"Unknown model: {name}")
            continue
        results[name] = download_model(name, MODELS[name], str(cache_dir))

    print("\n=== Download Summary ===")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
