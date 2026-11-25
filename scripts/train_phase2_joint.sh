#!/bin/bash
# Phase 2 joint MLM + supervised-contrastive pretraining.
#
#   VARIANT=unigram  bash scripts/train_phase2_joint.sh  # run behind the released Unigram weights
#   VARIANT=snt      bash scripts/train_phase2_joint.sh  # SNT, resumed from step 5,500
#   VARIANT=snt-stage1 bash scripts/train_phase2_joint.sh # SNT, first 5,500 steps (accum 13)
#
# The variant selects the config file; no source edit is needed.
set -euo pipefail
VARIANT=${VARIANT:-unigram}
NPROC=${NPROC:-8}
PORT=${PORT:-29507}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

case "$VARIANT" in
    unigram)   CONFIG=configs/pretrain_phase2_unigram.yaml ;;
    snt)       CONFIG=configs/pretrain_phase2_snt.yaml ;;
    snt-stage1) CONFIG=configs/pretrain_phase2_snt_stage1.yaml ;;
    *) echo "Error: unknown VARIANT '$VARIANT' (expected unigram|snt|snt-stage1)" >&2; exit 1 ;;
esac

OUT=$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("output_dir", "checkpoints/phase2_joint/"))
PY
)
mkdir -p "$OUT"

echo "=== FungiDNA Phase 2: joint MLM + contrastive (variant=$VARIANT, config=$CONFIG) ==="
torchrun --nproc_per_node="$NPROC" --master_port="$PORT" \
    scripts/phase2_launcher.py "$CONFIG"

echo "Phase 2 complete. Backbone: ${OUT}backbone_final.pt"
