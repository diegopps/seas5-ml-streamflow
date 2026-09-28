#!/bin/bash
#SBATCH --job-name=FlCa-sens
#SBATCH --mem=128G
#SBATCH --output=slurm_log/sens_%x-%j.out
#SBATCH --error=slurm_log/sens_%x-%j.err

# =============================================================================
# run_03_sensitivity.sh
#
# Shared Slurm job body for the stage-03 climate-input sensitivity test.
# Called by submit_03_sensitivity.sh — do not run directly.
#
# Resource requests (partition, qos, cpus-per-task, gres, time) are set per
# model by the wrapper on the sbatch line, since the models need different
# core counts and wall times; they are deliberately not defaulted here.
#
# One job covers every fold of one (catchment, model), plus a final fit on all
# years that is saved to trained_models/. Each fold trains once and runs two
# inference passes, so a whole catchment costs about what a single stage-02
# outer fold does — per-fold jobs are unnecessary here.
#
# Environment variables set by the wrapper via --export:
#   CATCHMENT : catchment id (e.g. zeravshan)
#   MODEL     : model to test (ridge | rf | xgb | lstm)
#   SEED      : global seed, and the stage-02 search seed whose winners are read
# =============================================================================

set -eo pipefail

mkdir -p slurm_log

echo "============================================================"
echo "Stage-03 Climate-Input Sensitivity Job"
echo "  Catchment : ${CATCHMENT}"
echo "  Model     : ${MODEL}"
echo "  Seed      : ${SEED:-107}"
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

python 03_sensitivity_climate_input.py ${CATCHMENT} ${MODEL}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "  Runtime  : ${RUNTIME}s ($(( RUNTIME / 60 ))m $(( RUNTIME % 60 ))s)"
echo "============================================================"
