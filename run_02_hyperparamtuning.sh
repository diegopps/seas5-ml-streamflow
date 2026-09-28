#!/bin/bash
#SBATCH --job-name=FlCa-hptune
#SBATCH --mem=128G
#SBATCH --output=slurm_log/hptune_%x-%j.out
#SBATCH --error=slurm_log/hptune_%x-%j.err

# =============================================================================
# run_02_hyperparamtuning.sh
#
# Shared Slurm job body for the stage-02 nested hyperparameter search. Called
# by submit_02_hyperparamtuning_nested.sh — do not run directly.
#
# Resource requests (partition, qos, cpus-per-task, gres, time) are set per
# model by the wrapper on the sbatch line, since the models need different
# core counts and wall times; they are deliberately not defaulted here.
#
# Environment variables set by the wrapper via --export:
#   CATCHMENT : catchment id (e.g. zeravshan)
#   MODEL     : model to tune (lstm | rf | xgb)
#   OUTER_HY  : held-out hydro year for this outer fold
#   SEED      : global random seed
#   N_INNER   : number of inner folds
# =============================================================================

set -eo pipefail

mkdir -p slurm_log

echo "============================================================"
echo "Stage-02 Nested Hyperparameter Tuning Job"
echo "  Catchment : ${CATCHMENT}"
echo "  Model     : ${MODEL}"
echo "  Seed      : ${SEED:-107}"
echo "  OUTER_HY  : ${OUTER_HY:-<n/a>}"
echo "  N_INNER   : ${N_INNER:-<n/a>}"
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

python 02_hyperparamtuning_nested.py ${CATCHMENT} ${MODEL}

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "  Runtime  : ${RUNTIME}s ($(( RUNTIME / 60 ))m $(( RUNTIME % 60 ))s)"
echo "============================================================"
