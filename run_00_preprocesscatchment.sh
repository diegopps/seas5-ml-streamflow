#!/bin/bash
#SBATCH --job-name=FlCa-prep
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm_log/FlowCast-%j.out
#SBATCH --error=slurm_log/FlowCast-%j.err

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================================"

cd "${SLURM_SUBMIT_DIR}"   # repo root: the submit_*.sh scripts cd there before sbatch
source load_env.sh         # environment setup from config.toml [slurm] env_setup

python3 00_preprocess_catchment.py $CATCHMENT

echo ""
echo "============================================"
echo "End:    $(date)"
echo "Exit:   $?"
echo "============================================"
