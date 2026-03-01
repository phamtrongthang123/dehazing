#!/bin/bash
set -euo pipefail

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
echo "=== Running sanity tests ==="
python test_sanity.py

echo ""
echo "=== Running training loop test (synthetic data) ==="
python test_training_loop.py

echo ""
echo "=== All tests passed ==="
