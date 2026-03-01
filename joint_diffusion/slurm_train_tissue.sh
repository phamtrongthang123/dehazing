#!/bin/bash
#SBATCH --job-name=train_tissue
#SBATCH --time=3-00:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/tissue_%N_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --partition=agpu
#SBATCH --constraint=1a100

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Training: TISSUE model"

mkdir -p /scrfs/storage/tp030/home/dehazing/slurm_logs

ROOT_DIR="/scrfs/storage/tp030/home"
SCRIPT_DIR="$ROOT_DIR/dehazing/joint_diffusion"

apptainer exec --nv --writable-tmpfs \
  --bind /scrfs/storage/tp030/home:/scrfs/storage/tp030/home \
  --bind /home/tp030:/home/tp030 \
  "$HOME/qwen3vl-cu128.sif" \
  bash -c "
    set -euo pipefail
    source /home/tp030/.conda/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
    conda activate dehazing
    cd $SCRIPT_DIR

    echo '=== Environment ==='
    python --version
    python -c \"import torch; print(f'PyTorch: {torch.__version__}')\"
    python -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\"
    python -c \"import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')\"

    echo ''
    echo '=== Training TISSUE model ==='
    export WANDB_MODE=disabled
    python train.py \
        -c configs/training/score_zea_tissue.yaml \
        --data_root $ROOT_DIR/f2f_ldm/data
  "

echo "Job finished at: $(date)"
