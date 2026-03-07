#!/bin/bash
#SBATCH --job-name=infer_pi_v7
#SBATCH --time=4:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/infer_picmus_v7_%N_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
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

# Read v7 run dir saved by slurm_train_picmus_v7.sh (shared filesystem)
if [ -f "$SCRIPT_DIR/v7_run_dir.txt" ]; then
    V7_RUN_DIR=$(cat "$SCRIPT_DIR/v7_run_dir.txt")
fi
# Fallback: find newest run dir excluding known pre-v7 IDs
if [ -z "$V7_RUN_DIR" ]; then
    KNOWN_RUNS="run--cs2128ri\|run--ez1mqfie\|run--xnhklvch\|run--77l53ue1\|run--w1lwubg5\|run--xci958q7\|run--41t5d8xd\|run--1bj42atz\|run--jm8ljbms\|run--n1kp0qeb"
    V7_RUN_DIR=$(ls -td "$SCRIPT_DIR/wandb/run--"*/ | grep -v "$KNOWN_RUNS" | head -1)
fi
V7_FILES="${V7_RUN_DIR}files"
echo "V7 run dir: $V7_FILES"

echo ""
echo "V7 latest checkpoints:"
ls "$V7_FILES/training_checkpoints"/ckpt-*.pt 2>/dev/null | sort -t'-' -k2 -n | tail -3

# Create inference config for v7 from v4 template
# Write to /scrfs (not /tmp) to avoid node-local filesystem issues
V7_CFG="$SCRIPT_DIR/configs/inference/paper/picmus_dehaze_v7_runtime.yaml"
sed "s|/scrfs/storage/tp030/home/dehazing/joint_diffusion/wandb/run--cs2128ri/files|${V7_FILES}|g" \
    configs/inference/paper/picmus_dehaze_v4.yaml > "$V7_CFG"

echo ""
echo "=== PICMUS dehazing inference: v7 score model (fine-tuned v5 + t-importance) ==="
$PYTHON -u inference.py -e paper/picmus_dehaze_v7_runtime -t denoise

echo ""
echo "Job finished at: $(date)"
