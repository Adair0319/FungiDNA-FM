#!/bin/bash
# Phase 1 MLM pretraining (hyperparameters live in configs/pretrain_phase1.yaml).
set -euo pipefail
NPROC=${NPROC:-8}
PORT=${PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [ ! -f "data/processed/bpe_fungi.model" ]; then
    echo "Error: tokenizer not found at data/processed/bpe_fungi.model" >&2
    exit 1
fi
mkdir -p checkpoints/phase1_mlm

echo "=== FungiDNA Phase 1: MLM pretraining (see configs/pretrain_phase1.yaml) ==="
torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
    fungidna/training/phase1_mlm.py configs/pretrain_phase1.yaml

echo "Phase 1 complete. Checkpoint: checkpoints/phase1_mlm/final_model.pt"
