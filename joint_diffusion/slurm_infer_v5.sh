#!/bin/bash
#SBATCH --job-name=infer_picmus_v5
#SBATCH --time=4:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/infer_picmus_v5_%N_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=agpu
#SBATCH --constraint=1a100
#SBATCH --exclude=c2110

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p /scrfs/storage/tp030/home/dehazing/slurm_logs

ROOT_DIR="/scrfs/storage/tp030/home"
SCRIPT_DIR="$ROOT_DIR/dehazing/joint_diffusion"
PYTHON=/home/tp030/.conda/envs/dehazing/bin/python

cd "$SCRIPT_DIR"
export WANDB_MODE=disabled

# Use static v5 inference config (avoids /tmp issues and NFS race conditions)
V5_CFG="configs/inference/paper/picmus_dehaze_v5.yaml"
echo "Inference config: $V5_CFG"

echo "=== Running PICMUS dehazing inference with v5 score model ==="
$PYTHON -u inference.py -e paper/picmus_dehaze_v5 -t denoise

echo "Job finished at: $(date)"
