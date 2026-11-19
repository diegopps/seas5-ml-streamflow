#!/bin/bash
# =============================================================================
# submit_preprocess.sh
#
# Stage 0: submits one Slurm job per catchment running
# pipeline/0_data/preprocess.py, which writes the catchment's feature table
# and fold manifests to data/processed_data/.
#
# Usage:
#   bash pipeline/0_data/submit_preprocess.sh                  # all catchments
#   bash pipeline/0_data/submit_preprocess.sh zeravshan esil   # some
# =============================================================================

# Submit from the repo root so jobs start there (job.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm opens the log files before the job script runs

ALL_CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)

if [[ $# -gt 0 && "$1" != "all" ]]; then
    CATCHMENTS=("$@")
else
    CATCHMENTS=("${ALL_CATCHMENTS[@]}")
fi

for CATCHMENT in "${CATCHMENTS[@]}"; do
    sbatch \
        --job-name=s5ml-prep_${CATCHMENT} \
        --partition=standard \
        --qos=short \
        --cpus-per-task=8 \
        --mem=128G \
        --time=12:00:00 \
        --output=logs/prep_${CATCHMENT}-%j.out \
        --error=logs/prep_${CATCHMENT}-%j.err \
        pipeline/job.sh python pipeline/0_data/preprocess.py "${CATCHMENT}"
    echo "Submitted preprocessing: ${CATCHMENT}"
done
