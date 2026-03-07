#!/bin/bash
# Run this script after v4 inference (job 179128) shows tissue structure.
# Usage: bash submit_phase4_sweep.sh
# Or with dependency: sbatch --dependency=afterok:179128 slurm_sweep_phase4.sh configs/inference/paper/picmus_dehaze_v4.yaml

cd /scrfs/storage/tp030/home/dehazing/joint_diffusion

echo "Submitting Phase 4 hyperparameter sweep for PICMUS v4 model..."
sbatch slurm_sweep_phase4.sh configs/inference/paper/picmus_dehaze_v4.yaml
echo "Done. Check squeue for job ID."
