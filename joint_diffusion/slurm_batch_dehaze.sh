#!/bin/bash
#SBATCH --job-name=batch_dehaze
#SBATCH --partition=agpu,vgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/batch_dehaze_%j.out
#SBATCH --exclude=c2110

INF_CONFIG=${1:-configs/inference/paper/picmus_dehaze_v4.yaml}

echo "=== Batch dehaze all PICMUS val frames ==="
echo "Config: $INF_CONFIG"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || source /home/tp030/.conda/etc/profile.d/conda.sh
conda activate dehazing 2>/dev/null || true
PYTHON=/home/tp030/.conda/envs/dehazing/bin/python
export WANDB_MODE=disabled

cd /scrfs/storage/tp030/home/dehazing/joint_diffusion

$PYTHON batch_dehaze_picmus.py \
    --inf_config "$INF_CONFIG" \
    --n_batch 4

echo "Job finished at: $(date)"
