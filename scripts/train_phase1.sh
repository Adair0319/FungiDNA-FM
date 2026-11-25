#!/bin/bash
# Phase 1 MLM pretraining.
#
#   VARIANT=unigram bash scripts/train_phase1.sh   # default
#   VARIANT=snt     bash scripts/train_phase1.sh
#
# The variant selects the config file; no source edit is needed. Hyperparameters
# live in configs/pretrain_phase1{,_snt}.yaml.
set -euo pipefail
VARIANT=${VARIANT:-unigram}
NPROC=${NPROC:-8}
PORT=${PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

case "$VARIANT" in
    unigram) CONFIG=configs/pretrain_phase1.yaml ;;
    snt)     CONFIG=configs/pretrain_phase1_snt.yaml ;;
    *) echo "Error: unknown VARIANT '$VARIANT' (expected unigram|snt)" >&2; exit 1 ;;
esac

if [ ! -f "data/processed/bpe_fungi.model" ]; then
    echo "Error: tokenizer not found at data/processed/bpe_fungi.model" >&2
    exit 1
fi

OUT=$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("output_dir", "checkpoints/phase1_mlm/"))
PY
)
mkdir -p "$OUT"

echo "=== FungiDNA Phase 1: MLM pretraining (variant=$VARIANT, config=$CONFIG) ==="
torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
    fungidna/training/phase1_mlm.py "$CONFIG"

echo "Phase 1 complete. Checkpoint: ${OUT}final_model.pt"
