#!/bin/bash
# =============================================================================
# submit_02_hyperparamtuning_nested.sh
#
# Nested-CV honest performance estimate. Submits the stage-02 hyperparameter
# search for one or more models across one or more catchments.
#
# One Slurm job = one OUTER fold (one held-out hydro year), per catchment, per
# model. Each job runs its config grid x N_INNER inner folds + a winner
# retrain; the untuned baseline is read from stage 01 (measured: lstm ~15-16 h on 16 CPUs for
# kirchbichl, the slowest catchment; rf ~2.5 h; xgb ~50 min). All models run
# on CPU. Per-fold jobs keep runtime independent of how many
# hydro years a catchment has, so no single catchment can blow through a
# job's wall-time budget.
#
# Ridge is not tuned here: RidgeCV already picks its alpha by generalised
# cross-validation on each fold's training data, so it is tuned honestly
# within stage 01 and stays a stage-01 reference model.
#
# The outer hydro years are read from the fold manifest each catchment's
# stage-00 run writes (data_processed/<catchment>_folds.txt), so no python
# runs on the login node and the list cannot go stale: stage 00 derives it by
# calling build_analysis_frame() on the parquet it just wrote, which is the
# same function the jobs themselves use. Each job still self-validates
# OUTER_HY against the live data and exits fast if a year is unusable.
#
# Usage:
#   bash submit_02_hyperparamtuning_nested.sh                          # default catchments, lstm
#   bash submit_02_hyperparamtuning_nested.sh kurshim
#   bash submit_02_hyperparamtuning_nested.sh all                      # all catchments, lstm
#   bash submit_02_hyperparamtuning_nested.sh --model rf all           # one model, all catchments
#   bash submit_02_hyperparamtuning_nested.sh --model all zeravshan    # all models, one catchment
#   bash submit_02_hyperparamtuning_nested.sh --model all all          # everything
# =============================================================================

# Submit from the repo root so jobs start there (run_*.sh cd to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")"

DATA_DIR=data_processed

DEFAULT_CATCHMENTS=(zeravshan ayat kirchbichl)
ALL_CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)
ALL_MODELS=(lstm rf xgb)

# Per-model Slurm routing: "partition qos cpus gres time"
# gres: "-" means no generic resource requested (CPU-only job)
# lstm runs on CPU rather than GPU: gpushort caps a user at 2 concurrent jobs,
# while CPU jobs run many at once. Needs the thread exports in the run script
# (the miniforge module sets MKL_NUM_THREADS=1, which pins torch to 1 thread).
declare -A MODEL_ROUTE=(
    [lstm]="standard short    16 -     24:00:00"
    [rf]="standard   short    16 -     08:00:00"
    [xgb]="standard  short    16 -     08:00:00"
)

MODELS=(lstm)
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
N_INNER=5

TOTAL=0

for MODEL in "${MODELS[@]}"; do
    ROUTE="${MODEL_ROUTE[$MODEL]:-}"
    if [[ -z "$ROUTE" ]]; then
        echo "ERROR: no Slurm routing defined for model '${MODEL}'"
        echo "       (add it to MODEL_ROUTE above and to flowcast_src/hptuning.py)"
        exit 1
    fi
    read -r PARTITION QOS CPUS GRES WALLTIME <<< "$ROUTE"

    GRES_ARG=()
    if [[ "$GRES" != "-" ]]; then
        GRES_ARG=(--gres="${GRES}")
    fi

    for CATCHMENT in "${CATCHMENTS[@]}"; do
        echo "=== Submitting NESTED jobs: ${CATCHMENT} / ${MODEL} ==="

        FOLD_FILE="${DATA_DIR}/${CATCHMENT}_folds.txt"
        if [[ ! -f "$FOLD_FILE" ]]; then
            echo "  WARNING: no fold manifest at ${FOLD_FILE} — skipping"
            echo "           (run stage 00 for ${CATCHMENT} first)"
            continue
        fi
        YEARS="$(<"$FOLD_FILE")"
        if [[ -z "${YEARS// }" ]]; then
            echo "  WARNING: fold manifest for ${CATCHMENT} is empty — skipping"
            continue
        fi
        echo "  Outer folds: $(echo $YEARS | wc -w) years"

        for HY in $YEARS; do
            sbatch \
                --job-name=FlCa-nested_${CATCHMENT}_${MODEL}_hy${HY} \
                --partition=${PARTITION} \
                --qos=${QOS} \
                --cpus-per-task=${CPUS} \
                "${GRES_ARG[@]}" \
                --time=${WALLTIME} \
                --output=slurm_log/nested_${CATCHMENT}_${MODEL}_hy${HY}-%j.out \
                --error=slurm_log/nested_${CATCHMENT}_${MODEL}_hy${HY}-%j.err \
                --export=ALL,CATCHMENT=${CATCHMENT},MODEL=${MODEL},OUTER_HY=${HY},SEED=${SEED},N_INNER=${N_INNER} \
                run_02_hyperparamtuning.sh
            TOTAL=$((TOTAL + 1))
        done
        echo ""
    done
done

echo "=== Total NESTED jobs submitted: ${TOTAL} ==="
