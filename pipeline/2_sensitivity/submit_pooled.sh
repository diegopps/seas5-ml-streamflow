#!/bin/bash
# =============================================================================
# submit_pooled.sh — stage 2 (climate-input sensitivity test), pooled models
#
# Submits one Slurm job running pipeline/2_sensitivity/run_pooled.py: every
# pooled fold trains one model with its own tuned winner (stage 1,
# run_pooled.py tuned, must be complete) and scores every catchment under
# real and donor-year forcing; then the pool's final model (loyo: all-years
# fit; split: the fold model) is saved to data/trained_models/<design>/.
# Cost: ~one pooled fit per fold, plus one final fit.
#
# Resubmitting resumes: finished folds are checkpointed and skipped, so a job
# that hits its walltime can simply be submitted again.
#
# CPU only, QOS medium (7-day walltime), 16 CPUs.
#
# Usage:
#   bash pipeline/2_sensitivity/submit_pooled.sh [--model <m>] [--design <d>]
# =============================================================================

# Submit from the repo root so jobs start there (job.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm opens the log files before the job script runs

MODEL=pooled_lstm
DESIGN=loyo

# Slurm routing: "partition qos cpus mem time". Set mem and time from the
# timing variant's report (pipeline/1_models/submit_pooled.sh timing):
# about the default run's runtime plus one fit, at the winners' hidden sizes.
ROUTE="standard medium 16 TBD TBD"

while [[ "$1" == --* ]]; do
    case "$1" in
        --model)  MODEL="$2";  shift 2 ;;
        --design) DESIGN="$2"; shift 2 ;;
        *) echo "ERROR: unknown option $1"; exit 1 ;;
    esac
done
read -r PARTITION QOS CPUS MEM WALLTIME <<< "${ROUTE}"

TAG="${DESIGN}_pooled_sens_${MODEL}"
sbatch \
    --job-name=s5ml-${TAG} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --time=${WALLTIME} \
    --output=logs/${TAG}-%j.out \
    --error=logs/${TAG}-%j.err \
    pipeline/job.sh python pipeline/2_sensitivity/run_pooled.py "${MODEL}" --design "${DESIGN}"
echo "  Submitted pooled sensitivity: ${MODEL} (${DESIGN})  [${PARTITION}/${QOS}, ${CPUS} CPUs]"
