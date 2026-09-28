#!/bin/bash
#SBATCH --job-name=FlCa-explore
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_log/explore_%x-%j.out
#SBATCH --error=slurm_log/explore_%x-%j.err

# =============================================================================
# run_01_exploration.sh
#
# Slurm job script for a single exploration run (one catchment, one model).
# Called by submit_01_exploration.sh — do not run directly.
#
# Environment variables set via --export by submit_01_exploration.sh:
#   CATCHMENT : catchment id (e.g. zeravshan)
#   MODEL     : model name (climatology | persistence | ridge | rf | xgb | lstm)
#
# Partition and QOS are chosen per-model by submit_01_exploration.sh; every
# model runs on partition=standard, and the LSTM asks for 16 CPUs.
# =============================================================================

set -eo pipefail

mkdir -p slurm_log

echo "============================================================"
echo "Exploration Job"
echo "  Model     : ${MODEL}"
echo "  Catchment : ${CATCHMENT}"
echo "  Node      : $(hostname)"
echo "  Started   : $(date)"
echo "============================================================"

cd "${SLURM_SUBMIT_DIR}"   # repo root: the submit_*.sh scripts cd there before sbatch
source load_env.sh         # environment setup from config.toml [slurm] env_setup

# The miniforge module sets MKL_NUM_THREADS=1, which pins torch to a single
# thread; use the cores Slurm allocated instead.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

START_TIME=$(date +%s)

python 01_exploration.py ${CATCHMENT} ${MODEL}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "  Runtime  : ${RUNTIME}s ($(( RUNTIME / 60 ))m $(( RUNTIME % 60 ))s)"
echo "============================================================"
