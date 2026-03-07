#!/bin/bash
#SBATCH --job-name=infer_pi_v6
#SBATCH --time=4:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/infer_picmus_v6_%N_%j.out
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

# Create inference config from v4 template, substituting v6 run path
# Write to /scrfs (not /tmp) to avoid node-local filesystem issues
V4_CFG="configs/inference/paper/picmus_dehaze_v4.yaml"
V6_CFG="$SCRIPT_DIR/configs/inference/paper/picmus_dehaze_v6_runtime.yaml"
sed "s|/scrfs/storage/tp030/home/dehazing/joint_diffusion/wandb/run--cs2128ri/files|${V6_FILES}|g" "$V4_CFG" > "$V6_CFG"
echo "Inference config: $V6_CFG"

echo "=== Running PICMUS dehazing inference with v6 score model (t-importance) ==="
$PYTHON -u inference.py -e paper/picmus_dehaze_v6_runtime -t denoise

echo "Job finished at: $(date)"
