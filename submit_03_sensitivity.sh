#!/bin/bash
# =============================================================================
# submit_03_sensitivity.sh
#
# Climate-input sensitivity test. Submits one Slurm job per (catchment, model);
# each job covers every fold of that pair.
#
# Per fold the job trains one model and scores it twice — on real forcing and
# on a donor year's forcing — so the cost is roughly one training per fold.
# That puts a whole catchment in the same ballpark as a single stage-02 outer
# fold, which is why these are whole-catchment jobs rather than per-fold ones.
#
# rf, xgb, and lstm use one config per catchment: whichever config won the most
# outer folds in stage 02 (03_configranking.pdf's 'best'). Stage 02 should have
# COMPLETED for the pairs being submitted — a partially finished catchment will
# still run, but the winning config is then decided by a vote over however many
# folds happen to be done, and may change once the rest land. Ridge has no
# stage-02 search and runs on every fold regardless.
#
# Every fold runs for tuned models too, plus one final full-data fit at the end
# (saved to trained_models/), so a job is ~n_years + 1 trainings.
#
# Usage:
#   bash submit_03_sensitivity.sh                          # default catchments, all models
#   bash submit_03_sensitivity.sh all                      # all catchments, all models
#   bash submit_03_sensitivity.sh --model xgb all          # one model, all catchments
#   bash submit_03_sensitivity.sh --model lstm zeravshan   # one model, one catchment
# =============================================================================

# Submit from the repo root so jobs start there (run_*.sh cd to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")"

DEFAULT_CATCHMENTS=(zeravshan ayat kirchbichl)
ALL_CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)
ALL_MODELS=(ridge rf xgb lstm)

# Per-model Slurm routing: "partition qos cpus gres time"
# gres: "-" means no generic resource requested (CPU-only job)
# lstm runs on CPU rather than GPU: gpushort caps a user at 2 concurrent jobs.
# It uses the medium qos because a whole catchment can exceed short's 1-day
# cap: kirchbichl's winner (hidden 512) is ~24 h on 16 CPUs. Needs the thread
# exports in the run script (the miniforge module sets MKL_NUM_THREADS=1).
declare -A MODEL_ROUTE=(
    [lstm]="standard medium   16 -     48:00:00"
    [ridge]="standard short    16 -     12:00:00"
    [rf]="standard    short    16 -     12:00:00"
    [xgb]="standard   short    16 -     12:00:00"
)

MODELS=("${ALL_MODELS[@]}")
if [[ "$1" == "--model" ]]; then
    if [[ "$2" == "all" ]]; then
        MODELS=("${ALL_MODELS[@]}")
    else
        MODELS=("$2")
    fi
    shift 2
fi

if [[ "$1" == "all" ]]; then
    CATCHMENTS=("${ALL_CATCHMENTS[@]}")
elif [[ $# -gt 0 ]]; then
    CATCHMENTS=("$@")
else
    CATCHMENTS=("${DEFAULT_CATCHMENTS[@]}")
fi

SEED=107
TOTAL=0

for MODEL in "${MODELS[@]}"; do
    ROUTE="${MODEL_ROUTE[$MODEL]:-}"
    if [[ -z "$ROUTE" ]]; then
        echo "ERROR: no Slurm routing defined for model '${MODEL}'"
        echo "       (add it to MODEL_ROUTE above)"
        exit 1
    fi
    read -r PARTITION QOS CPUS GRES WALLTIME <<< "$ROUTE"

    GRES_ARG=()
    if [[ "$GRES" != "-" ]]; then
        GRES_ARG=(--gres="${GRES}")
    fi

    for CATCHMENT in "${CATCHMENTS[@]}"; do
        echo "=== Submitting SENSITIVITY job: ${CATCHMENT} / ${MODEL} ==="

        sbatch \
            --job-name=FlCa-sens_${CATCHMENT}_${MODEL} \
            --partition=${PARTITION} \
            --qos=${QOS} \
            --cpus-per-task=${CPUS} \
            "${GRES_ARG[@]}" \
            --time=${WALLTIME} \
            --output=slurm_log/sens_${CATCHMENT}_${MODEL}-%j.out \
            --error=slurm_log/sens_${CATCHMENT}_${MODEL}-%j.err \
            --export=ALL,CATCHMENT=${CATCHMENT},MODEL=${MODEL},SEED=${SEED} \
            run_03_sensitivity.sh
        TOTAL=$((TOTAL + 1))
        echo ""
    done
done

echo "=== Total SENSITIVITY jobs submitted: ${TOTAL} ==="
