#!/bin/bash
#SBATCH --job-name=infer_picmus_v4
#SBATCH --time=4:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/infer_picmus_v4_%N_%j.out
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
V4_CKPT_DIR="$SCRIPT_DIR/wandb/run--cs2128ri/files/training_checkpoints"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source /home/tp030/.conda/etc/profile.d/conda.sh
conda activate dehazing 2>/dev/null || true
PYTHON=/home/tp030/.conda/envs/dehazing/bin/python

echo "=== Environment ==="
$PYTHON --version
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')"
$PYTHON -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

cd "$SCRIPT_DIR"
export WANDB_MODE=disabled

echo ""
echo "=== Available v4 checkpoints ==="
ls -la "$V4_CKPT_DIR" 2>/dev/null || echo "No checkpoints found yet"

LAST_CKPT=$(ls "$V4_CKPT_DIR"/ckpt-*.pt 2>/dev/null | sort -t'-' -k2 -n | tail -1)
echo "Using checkpoint: $LAST_CKPT"

echo ""
echo "=== Running PICMUS dehazing inference with v4 score model ==="
$PYTHON -u inference.py -e paper/picmus_dehaze_v4 -t denoise

echo "Job finished at: $(date)"
