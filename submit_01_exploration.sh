#!/bin/bash
# =============================================================================
# submit_01_exploration.sh
#
# Submits one Slurm job per (model, catchment) combination, all driven by the
# single runner 01_exploration.py. All models run on partition=standard; the
# LSTM asks for 16 CPUs. Outputs land under model_results/01_exploration/.
#
# The 'persistence' job produces both the persistence and damped_persistence
# result directories from a single run.
#
# Usage:
#   bash submit_01_exploration.sh                        # zeravshan (default)
#   bash submit_01_exploration.sh esil
#   bash submit_01_exploration.sh zeravshan esil ayat
#   bash submit_01_exploration.sh all                    # all defined catchments
# =============================================================================

# Submit from the repo root so jobs start there (run_*.sh cd to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")"

ALL_CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)


if [[ "$1" == "all" ]]; then
    CATCHMENTS=("${ALL_CATCHMENTS[@]}")
elif [[ $# -gt 0 ]]; then
    CATCHMENTS=("$@")
else
    CATCHMENTS=(zeravshan)
fi

# Format: "model  partition  qos  extra_sbatch_args"
MODELS=(
    "climatology  standard  short     "
    "persistence  standard  short     "
    "ridge        standard  short     "
    "rf           standard  short     "
    "xgb          standard  short     "
    "lstm         standard  short     --cpus-per-task=16"
)

for CATCHMENT in "${CATCHMENTS[@]}"; do
    echo "=== Submitting exploration jobs for: ${CATCHMENT} ==="

    for entry in "${MODELS[@]}"; do
        read -r MODEL PARTITION QOS EXTRA <<< "$entry"

        sbatch \
            --partition=${PARTITION} \
            --qos=${QOS} \
            --job-name=FlCa-explore_${MODEL}_${CATCHMENT} \
            --output=slurm_log/explore_${MODEL}_${CATCHMENT}-%j.out \
            --error=slurm_log/explore_${MODEL}_${CATCHMENT}-%j.err \
            --export=ALL,CATCHMENT=${CATCHMENT},MODEL=${MODEL} \
            ${EXTRA} \
            run_01_exploration.sh

        echo "  Submitted '${MODEL}'  [${PARTITION}/${QOS}]"
    done

    echo ""
done
