#!/bin/bash
#SBATCH --job-name=sweep_picmus
#SBATCH --time=8:00:00
#SBATCH --output=/scrfs/storage/tp030/home/dehazing/slurm_logs/sweep_picmus_%N_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=agpu
#SBATCH --constraint=1a100
#SBATCH --exclude=c2110,c2008

# Phase 4 hyperparameter sweep for PICMUS dehazing.
# Usage: sbatch slurm_sweep_phase4.sh <inf_config_path>
# Example: sbatch slurm_sweep_phase4.sh configs/inference/paper/picmus_dehaze_v4.yaml
# Or for v6 (using /tmp path):
#   sbatch slurm_sweep_phase4.sh /tmp/picmus_dehaze_v6_<job_id>.yaml

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

# Accept inference config path as first argument (default: v4 config)
INF_CONFIG="${1:-configs/inference/paper/picmus_dehaze_v4.yaml}"
echo "Using inference config: $INF_CONFIG"

# Back up any existing results.csv to avoid mixing with old model results
RESULTS_CSV="sweep_results/picmus/results.csv"
if [ -f "$RESULTS_CSV" ]; then
    BACKUP="sweep_results/picmus/results_backup_${SLURM_JOB_ID}.csv"
    mv "$RESULTS_CSV" "$BACKUP"
    echo "Backed up existing results.csv to $BACKUP"
fi

# Phase 1: Gamma sweep
echo ""
echo "=== Phase 1: gamma sweep ==="
$PYTHON -u sweep_picmus.py --phase 1 --num_img 3 --inf_config "$INF_CONFIG"

# Read best gamma from CSV
BEST_GAMMA=$($PYTHON -c "

import csv
with open('sweep_results/picmus/results.csv') as f:
    rows = [r for r in csv.DictReader(f) if r.get('phase')=='1']
rows.sort(key=lambda r: float(r['psnr']), reverse=True)
print(rows[0]['gamma'])
" 2>/dev/null || echo "0.1")
echo "Best gamma: $BEST_GAMMA"

# Phase 2: Lambda/kappa sweep
echo ""
echo "=== Phase 2: lambda/kappa sweep (gamma=$BEST_GAMMA) ==="
$PYTHON -u sweep_picmus.py --phase 2 --num_img 3 --inf_config "$INF_CONFIG" --best_gamma "$BEST_GAMMA"

# Read best lambda/kappa
BEST_LK=$($PYTHON -c "

import csv
with open('sweep_results/picmus/results.csv') as f:
    rows = [r for r in csv.DictReader(f) if r.get('phase')=='2']
rows.sort(key=lambda r: float(r['psnr']), reverse=True)
print(rows[0]['lambda_coeff'], rows[0]['kappa_coeff'])
" 2>/dev/null || echo "0.01 0.0")
BEST_LAMBDA=$(echo "$BEST_LK" | cut -d' ' -f1)
BEST_KAPPA=$(echo "$BEST_LK" | cut -d' ' -f2)
echo "Best lambda: $BEST_LAMBDA, kappa: $BEST_KAPPA"

# Phase 3: ccdf sweep
echo ""
echo "=== Phase 3: ccdf sweep (gamma=$BEST_GAMMA, lambda=$BEST_LAMBDA, kappa=$BEST_KAPPA) ==="
$PYTHON -u sweep_picmus.py --phase 3 --num_img 3 --inf_config "$INF_CONFIG" \
    --best_gamma "$BEST_GAMMA" --best_lambda "$BEST_LAMBDA" --best_kappa "$BEST_KAPPA"

echo ""
echo "=== Sweep complete. Results in sweep_results/picmus/results.csv ==="
cat sweep_results/picmus/results.csv 2>/dev/null

echo "Job finished at: $(date)"
