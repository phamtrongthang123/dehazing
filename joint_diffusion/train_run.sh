#!/bin/bash
set -euo pipefail

# Usage:
#   ./train_run.sh tissue          # train ZEA tissue model
#   ./train_run.sh haze            # train ZEA haze model
#   ./train_run.sh tissue picmus   # train PICMUS tissue model
#   ./train_run.sh haze picmus     # train PICMUS haze model

MODEL="${1:-tissue}"
DATASET="${2:-zea}"

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
echo "=== Training ${DATASET^^} ${MODEL^^} model ==="
export WANDB_MODE=disabled
python train.py \
    -c "configs/training/score_${DATASET}_${MODEL}.yaml" \
    --data_root "$ROOT_DIR/f2f_ldm/data"
