#!/bin/bash
# Launch Phase 1 MLM pretraining on 8 GPUs (v2 — no [CLS], grad accum)
# Usage: bash scripts/train_phase1.sh

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export WANDB_PROJECT="fungi-dna"
export PYTHONPATH="$PWD:$PYTHONPATH"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Ensure data is prepared first
if [ ! -f "data/processed/bpe_fungi.model" ]; then
    echo "Error: BPE tokenizer not found. Run prepare_pretrain_data.py first."
    exit 1
fi

mkdir -p checkpoints/phase1_mlm

echo "=== FungiDNA Phase 1: MLM Pretraining (v2) ==="
echo "GPUs: 8 | Steps: 200,000 | Batch: 32/GPU | GradAccum: 8 | Effective BS: 2048"
echo "Windows: [1024, 2048, 4096] | No [CLS] | Shuffle buffer: 8000"

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    src/training/phase1_mlm.py \
    configs/pretrain_phase1.yaml

echo "Phase 1 complete. Checkpoint: checkpoints/phase1_mlm/final_model.pt"
