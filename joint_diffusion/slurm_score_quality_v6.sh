#!/bin/bash
#SBATCH --job-name=score_qual_v6
#SBATCH --time=0:30:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/score_quality_v6_%N_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=agpu
#SBATCH --constraint=1a100
#SBATCH --exclude=c2110,c2008

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

# Read v6 run dir saved by slurm_train_picmus_v6.sh (shared filesystem)
if [ -f "$SCRIPT_DIR/v6_run_dir.txt" ]; then
    V6_RUN_DIR=$(cat "$SCRIPT_DIR/v6_run_dir.txt")
fi
# Fallback: find newest run dir excluding all pre-v6 known runs
if [ -z "$V6_RUN_DIR" ]; then
    KNOWN_RUNS="run--cs2128ri\|run--ez1mqfie\|run--xnhklvch\|run--77l53ue1\|run--w1lwubg5\|run--xci958q7\|run--41t5d8xd\|run--1bj42atz\|run--jm8ljbms\|run--n1kp0qeb"
    V6_RUN_DIR=$(ls -td "$SCRIPT_DIR/wandb/run--"*/ | grep -v "$KNOWN_RUNS" | head -1)
fi
V6_FILES="${V6_RUN_DIR}files"
echo "V6 run dir: $V6_FILES"

echo ""
echo "=== PICMUS v5 score quality (GroupNorm+ELU baseline) ==="
$PYTHON -u score_quality_check.py \
    --run_dir wandb/run--n1kp0qeb/files \
    --data_root "$DATA_ROOT" \
    --t 0.01 \
    --n_batches 20

echo ""
echo "=== PICMUS v6 score quality (t-importance sampling, alpha=0.3) ==="
$PYTHON -u score_quality_check.py \
    --run_dir "$V6_FILES" \
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
