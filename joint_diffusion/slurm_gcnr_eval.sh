#!/bin/bash
#SBATCH --job-name=gcnr_eval
#SBATCH --time=2:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/gcnr_eval_%N_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=agpu
#SBATCH --constraint=1a100
#SBATCH --exclude=c2110

# Phase 5: gCNR evaluation on PICMUS contrast_speckle val frames.
# Accepts inference config as first argument (default: v4).
# Usage:
#   sbatch slurm_gcnr_eval.sh configs/inference/paper/picmus_dehaze_v4.yaml
#   sbatch slurm_gcnr_eval.sh configs/inference/paper/picmus_dehaze_v5.yaml
#   # v6/v7 runtime configs:
#   sbatch slurm_gcnr_eval.sh configs/inference/paper/picmus_dehaze_v6_runtime.yaml

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

ROOT_DIR="/scrfs/storage/tp030/home"
SCRIPT_DIR="$ROOT_DIR/dehazing/joint_diffusion"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source /home/tp030/.conda/etc/profile.d/conda.sh
conda activate dehazing 2>/dev/null || true
PYTHON=/home/tp030/.conda/envs/dehazing/bin/python

cd "$SCRIPT_DIR"
export WANDB_MODE=disabled

INF_CONFIG="${1:-configs/inference/paper/picmus_dehaze_v4.yaml}"
echo "Inference config: $INF_CONFIG"
echo ""

echo "=== Phase 5: gCNR evaluation (13 contrast_speckle val frames) ==="
$PYTHON -u gcnr_eval.py \
    --inf_config "$INF_CONFIG" \
    --n_batch 4

echo ""
echo "Job finished at: $(date)"
