#!/bin/bash
# Phase 2 joint MLM + contrastive pretraining (hyperparameters in configs/pretrain_phase2_joint.yaml).
set -euo pipefail
NPROC=${NPROC:-8}
PORT=${PORT:-29507}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

if [ ! -f "checkpoints/phase1_mlm/final_model.pt" ]; then
    echo "Error: Phase 1 checkpoint not found at checkpoints/phase1_mlm/final_model.pt" >&2
    exit 1
fi
mkdir -p checkpoints/phase2_joint

echo "=== FungiDNA Phase 2: joint MLM + contrastive (see configs/pretrain_phase2_joint.yaml) ==="
torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
    scripts/phase2_launcher.py configs/pretrain_phase2_joint.yaml

echo "Phase 2 complete. Backbone: checkpoints/phase2_joint/backbone_final.pt"
