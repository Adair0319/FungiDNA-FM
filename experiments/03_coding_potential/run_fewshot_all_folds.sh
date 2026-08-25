#!/bin/bash
# Sequential fold launcher for few-shot experiments
# Each fold: 8 experiments (Ours 10/20/30/40% + GENA 10/20/30/40%), 1 per GPU

set -e
BASE="/home/lty/yy_projects/fungi_project/fungi_dna_model"
D="${BASE}/checkpoints/cds_intergenic_fewshot/data"
O="${BASE}/checkpoints/cds_intergenic_fewshot"
PY="/home/lty/miniconda3/envs/fungi/bin/python"
S="${BASE}/scripts/train_cds_intergenic.py"

for FOLD in 2 3 4; do
  echo "========== Starting Fold ${FOLD} at $(date) =========="
  pids=()
  for i in 0 1 2 3; do
    pct=$(( (i+1) * 10 ))

    $PY -u $S --model ours --mode finetune --gpu $i \
      --train-data ${D}/train_fold${FOLD}_${pct}pct.parquet \
      --test-data ${D}/test_fold${FOLD}.parquet \
      --output-dir ${O}/ours_${pct}pct_fold${FOLD} \
      2>&1 | tee ${BASE}/logs/cds_fewshot_ours_${pct}pct_fold${FOLD}.log &
    pids+=($!)

    gpu=$(( i + 4 ))
    $PY -u $S --model gena --mode finetune --gpu $gpu \
      --train-data ${D}/train_fold${FOLD}_${pct}pct.parquet \
      --test-data ${D}/test_fold${FOLD}.parquet \
      --output-dir ${O}/gena_${pct}pct_fold${FOLD} \
      2>&1 | tee ${BASE}/logs/cds_fewshot_gena_${pct}pct_fold${FOLD}.log &
    pids+=($!)
  done

  echo "Fold ${FOLD}: 8 experiments launched (PIDs: ${pids[*]})"
  echo "Waiting for all to complete..."
  wait "${pids[@]}"
  echo "========== Fold ${FOLD} DONE at $(date) =========="
done

echo "========== ALL FOLDS COMPLETE at $(date) =========="
