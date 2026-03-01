#!/bin/bash
set -euo pipefail

# Usage:
#   ./train_run.sh tissue   # train tissue model
#   ./train_run.sh haze     # train haze model
#   ./train_run.sh          # defaults to tissue

MODEL="${1:-tissue}"

ROOT_DIR="/scrfs/storage/tp030/home"
SCRIPT_DIR="$ROOT_DIR/dehazing/joint_diffusion"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dehazing
cd "$SCRIPT_DIR"

echo "=== Environment ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Training ${MODEL^^} model ==="
export WANDB_MODE=disabled
python train.py \
    -c "configs/training/score_zea_${MODEL}.yaml" \
    --data_root "$ROOT_DIR/f2f_ldm/data"
