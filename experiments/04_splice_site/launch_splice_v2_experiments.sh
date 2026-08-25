#!/bin/bash
# Launch 5 splice v2 experiments on 5 GPUs using the fungi conda environment.
# Each GPU runs one experiment type across all 5 folds sequentially.
# Usage: bash scripts/launch_splice_v2_experiments.sh

cd /home/lty/yy_projects/fungi_project/fungi_dna_model
mkdir -p logs/experiments checkpoints/splice_v2

source /home/lty/miniconda3/etc/profile.d/conda.sh
conda activate fungi

echo "Launching 5 experiments on GPUs 0-4..."
echo "Logs: logs/experiments/exp*.log"

nohup python3 -u scripts/run_splice_v2_experiments.py --exp randfull --fold all --gpu 0 > logs/experiments/exp1_randfull.log 2>&1 &
echo "  Exp1 RandFull  (GPU 0) PID: $!"

nohup python3 -u scripts/run_splice_v2_experiments.py --exp frozen   --fold all --gpu 1 > logs/experiments/exp2_frozen.log 2>&1 &
echo "  Exp2 Frozen    (GPU 1) PID: $!"

nohup python3 -u scripts/run_splice_v2_experiments.py --exp fullft   --fold all --gpu 2 > logs/experiments/exp3_fullft.log 2>&1 &
echo "  Exp3 FullFT    (GPU 2) PID: $!"

nohup python3 -u scripts/run_splice_v2_experiments.py --exp onehot   --fold all --gpu 3 > logs/experiments/exp4_onehot.log 2>&1 &
echo "  Exp4 OneHot    (GPU 3) PID: $!"

nohup python3 -u scripts/run_splice_v2_experiments.py --exp cnn      --fold all --gpu 4 > logs/experiments/exp5_cnn.log 2>&1 &
echo "  Exp5 CNN       (GPU 4) PID: $!"

echo ""
echo "All launched. Monitor with:"
echo "  tail -f logs/experiments/exp1_randfull.log"
echo "  watch -n 5 nvidia-smi"
