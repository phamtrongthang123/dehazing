#!/bin/bash
#SBATCH --job-name=train_picmus_v5
#SBATCH --time=2-00:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/train_picmus_v5_%N_%j.out
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

echo "=== Environment ==="
$PYTHON --version
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')"
$PYTHON -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
$PYTHON -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Training PICMUS tissue v5 (GroupNorm, ELU, batch=8, channels=64, cosine LR, 1000 epochs) ==="
$PYTHON -u train.py \
    -c "configs/training/score_picmus_tissue_v5.yaml" \
    --data_root "$ROOT_DIR/f2f_ldm/data"

echo "Job finished at: $(date)"
