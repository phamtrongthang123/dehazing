#!/bin/bash
#SBATCH --job-name=train_pi_v6
#SBATCH --time=24:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/train_picmus_v6_%N_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

echo "Starting PICMUS tissue v6 training (t-importance sampling, alpha=0.3)..."

# Note: do NOT use --no_wandb; WANDB_MODE=disabled causes wandb to create
# a local run dir under wandb/run--XXXXX/ which the dynamic discovery scripts rely on.
$PYTHON -u train.py \
    -c "configs/training/score_picmus_tissue_v6.yaml" \
    --data_root "$DATA_ROOT"

# Find the new run dir (newest, excluding all known pre-v6 runs)
KNOWN_RUNS="run--cs2128ri\|run--ez1mqfie\|run--xnhklvch\|run--77l53ue1\|run--w1lwubg5\|run--xci958q7\|run--41t5d8xd\|run--1bj42atz\|run--jm8ljbms\|run--n1kp0qeb"
V6_RUN_DIR=$(ls -td "$SCRIPT_DIR/wandb/run--"*/ | grep -v "$KNOWN_RUNS" | head -1)
V6_FILES="${V6_RUN_DIR}files"
echo "V6 run dir: $V6_FILES"

# Save run dir to shared filesystem for downstream jobs
echo "$V6_RUN_DIR" > "$SCRIPT_DIR/v6_run_dir.txt"

# Copy config to wandb dir for later score quality checks
cp configs/training/score_picmus_tissue_v6.yaml "${V6_FILES}/config.yaml" 2>/dev/null || true

echo "Job finished at: $(date)"
