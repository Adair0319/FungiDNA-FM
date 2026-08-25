#!/bin/bash
# Launch Phase 2 Joint MLM + Contrastive pretraining on 8 GPUs
# Usage: bash scripts/train_phase2_joint.sh
# Requires Phase 1 checkpoint to exist.

set -e

export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export OMP_NUM_THREADS=8
export WANDB_PROJECT="fungi-dna"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "checkpoints/phase1_mlm/final_model.pt" ]; then
    echo "Error: Phase 1 checkpoint not found at checkpoints/phase1_mlm/final_model.pt"
    exit 1
fi

mkdir -p checkpoints/phase2_joint

echo "=== FungiDNA Phase 2: Joint MLM + CL Training ==="
echo "GPUs: 5 | Steps: 40,000 | NxK: 8x4 | GradAccum: 13 | Effective BS: 2080"
echo "Window: 8192bp | Lambda max: 0.5 | Warmup: 5000 | Early-stop: on"

torchrun \
    --nproc_per_node=5 \
    --master_port=29507 \
    scripts/_phase2_launcher.py \
    configs/pretrain_phase2_joint.yaml

echo "Phase 2 complete. Backbone: checkpoints/phase2_joint/backbone_final.pt"
