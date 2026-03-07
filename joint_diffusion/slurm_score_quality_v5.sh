#!/bin/bash
#SBATCH --job-name=score_qual_v5
#SBATCH --time=0:30:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/score_quality_v5_%N_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=agpu
#SBATCH --constraint=1a100
#SBATCH --exclude=c2110

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

DATA_ROOT="$ROOT_DIR/f2f_ldm/data"

# v5 run dir is hardcoded since it's known (avoids picking v6 if it starts first)
V5_FILES="$SCRIPT_DIR/wandb/run--n1kp0qeb/files"
echo "V5 run dir: $V5_FILES"

echo ""
echo "=== PICMUS v2 score quality (baseline) ==="
$PYTHON -u score_quality_check.py \
    --run_dir wandb/run--xnhklvch/files \
    --data_root "$DATA_ROOT" \
    --t 0.01 \
    --n_batches 20

echo ""
echo "=== PICMUS v4 score quality (channels=64, batch norm) ==="
$PYTHON -u score_quality_check.py \
    --run_dir wandb/run--cs2128ri/files \
    --data_root "$DATA_ROOT" \
    --t 0.01 \
    --n_batches 20

echo ""
echo "=== PICMUS v5 score quality (GroupNorm, ELU, batch=8) ==="
$PYTHON -u score_quality_check.py \
    --run_dir "$V5_FILES" \
    --data_root "$DATA_ROOT" \
    --t 0.01 \
    --n_batches 20

echo ""
echo "=== ZEA tissue score quality (reference) ==="
$PYTHON -u score_quality_check.py \
    --run_dir wandb/run--jm8ljbms/files \
    --data_root "$DATA_ROOT" \
    --t 0.01 \
    --n_batches 20

echo "Job finished at: $(date)"
