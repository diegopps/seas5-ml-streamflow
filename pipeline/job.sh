#!/bin/bash
#SBATCH --job-name=s5ml
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# job.sh
#
# The one Slurm job body of the pipeline: sets up the environment and runs the
# command it is given. Every submit script (pipeline/*/submit*.sh) submits its
# jobs through it, with the resources, job name and log files on the sbatch
# line:
#
#     sbatch [resources] pipeline/job.sh python pipeline/1_models/run.py ...
#
# Submit from the repository root (the submit scripts cd there first): the
# job starts in SLURM_SUBMIT_DIR, and the command's paths are relative to it.
#
# Not for the SEAS5 download (pipeline/0_data/seas5_download/), which has its
# own scripts.
# =============================================================================

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"
source pipeline/load_env.sh   # environment setup from config.toml [slurm] env_setup

# The miniforge module sets MKL_NUM_THREADS=1, which pins torch (and numpy)
# to a single thread; use the cores Slurm allocated instead.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "============================================================"
echo "  Job      : ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "  Node     : $(hostname)  (${SLURM_CPUS_PER_TASK:-1} CPUs)"
echo "  Command  : $*"
echo "  Started  : $(date)"
echo "============================================================"

START_TIME=$(date +%s)

"$@"

RUNTIME=$(( $(date +%s) - START_TIME ))
echo ""
echo "============================================================"
echo "  Finished : $(date)"
echo "  Runtime  : ${RUNTIME}s ($(( RUNTIME / 60 ))m $(( RUNTIME % 60 ))s)"
echo "============================================================"
