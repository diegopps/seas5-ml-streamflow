#!/bin/bash
# =============================================================================
# submit.sh — stage 1 (train and evaluate the models)
#
#   default  one job per (catchment, model), covering every fold with the
#            default configuration. Models: climatology, persistence (also
#            writes damped_persistence), ridge, rf, xgb, lstm.
#   tuned    one job per (catchment, model, fold): nested-CV search, winner
#            retrain and score for that fold (measured: lstm ~15-16 h on 16
#            CPUs for kirchbichl, the slowest catchment; rf ~2.5 h; xgb
#            ~50 min). Models: rf, xgb, lstm. Ridge is not tuned: RidgeCV
#            picks its alpha inside its default run. Per-fold jobs keep
#            runtime independent of how many hydro years a catchment has.
#
# The two variants are independent and can be submitted together. The fold
# ids of the tuned jobs come from the fold manifest each catchment's stage-0
# run writes (data/processed_data/<catchment>_folds_<design>.txt), so no
# Python runs on the login node and the list cannot go stale; each job still
# checks its fold against the live data.
#
# Every model runs on CPU (partition standard). The LSTM runs on CPU rather
# than GPU: gpushort caps a user at 2 concurrent jobs, while CPU jobs run
# many at once (pipeline/job.sh sets the torch thread count).
#
# Usage:
#   bash pipeline/1_models/submit.sh <variant> [--model <m>|all] [--design <d>] [catchments...|all]
#
#   bash pipeline/1_models/submit.sh default                       # all models, zeravshan
#   bash pipeline/1_models/submit.sh default all                   # all models, all catchments
#   bash pipeline/1_models/submit.sh default --model ridge esil    # one model, one catchment
#   bash pipeline/1_models/submit.sh tuned                         # lstm, zeravshan ayat kirchbichl
#   bash pipeline/1_models/submit.sh tuned --model all all         # every tuned job
# =============================================================================

# Submit from the repo root so jobs start there (job.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm opens the log files before the job script runs

ALL_CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)
DESIGN=loyo

VARIANT="$1"
shift
case "$VARIANT" in
    default)
        ALL_MODELS=(climatology persistence ridge rf xgb lstm)
        MODELS=("${ALL_MODELS[@]}")
        DEFAULT_CATCHMENTS=(zeravshan)
        # Per-model Slurm routing: "partition qos cpus mem time"
        declare -A ROUTE=(
            [climatology]="standard short  8 128G 12:00:00"
            [persistence]="standard short  8 128G 12:00:00"
            [ridge]="standard       short  8 128G 12:00:00"
            [rf]="standard          short  8 128G 12:00:00"
            [xgb]="standard         short  8 128G 12:00:00"
            [lstm]="standard        short 16 128G 12:00:00"
        )
        ;;
    tuned)
        ALL_MODELS=(lstm rf xgb)
        MODELS=(lstm)
        DEFAULT_CATCHMENTS=(zeravshan ayat kirchbichl)
        declare -A ROUTE=(
            [lstm]="standard short 16 128G 24:00:00"
            [rf]="standard   short 16 128G 08:00:00"
            [xgb]="standard  short 16 128G 08:00:00"
        )
        ;;
    *)
        echo "Usage: bash $0 default|tuned [--model <m>|all] [--design <d>] [catchments...|all]"
        exit 1
        ;;
esac

while [[ "$1" == --* ]]; do
    case "$1" in
        --model)
            if [[ "$2" == "all" ]]; then MODELS=("${ALL_MODELS[@]}"); else MODELS=("$2"); fi
            shift 2 ;;
        --design)
            DESIGN="$2"
            shift 2 ;;
        *)
            echo "ERROR: unknown option $1"; exit 1 ;;
    esac
done

if [[ "$1" == "all" ]]; then
    CATCHMENTS=("${ALL_CATCHMENTS[@]}")
elif [[ $# -gt 0 ]]; then
    CATCHMENTS=("$@")
else
    CATCHMENTS=("${DEFAULT_CATCHMENTS[@]}")
fi

submit() {   # submit <job-tag> <python args...>
    local tag="$1"; shift
    sbatch \
        --job-name=s5ml-${tag} \
        --partition=${PARTITION} \
        --qos=${QOS} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --time=${WALLTIME} \
        --output=logs/${tag}-%j.out \
        --error=logs/${tag}-%j.err \
        pipeline/job.sh python pipeline/1_models/run.py "$@" --design "${DESIGN}"
}

TOTAL=0
for MODEL in "${MODELS[@]}"; do
    if [[ -z "${ROUTE[$MODEL]:-}" ]]; then
        echo "ERROR: model '${MODEL}' has no ${VARIANT} run (one of: ${ALL_MODELS[*]})"
        exit 1
    fi
    read -r PARTITION QOS CPUS MEM WALLTIME <<< "${ROUTE[$MODEL]}"

    for CATCHMENT in "${CATCHMENTS[@]}"; do
        if [[ "$VARIANT" == "default" ]]; then
            submit "${DESIGN}_default_${CATCHMENT}_${MODEL}" \
                "${CATCHMENT}" "${MODEL}" default
            echo "  Submitted default: ${CATCHMENT} / ${MODEL}  [${PARTITION}/${QOS}, ${CPUS} CPUs]"
            TOTAL=$((TOTAL + 1))
            continue
        fi

        FOLD_FILE="data/processed_data/${CATCHMENT}_folds_${DESIGN}.txt"
        if [[ ! -f "$FOLD_FILE" ]]; then
            echo "  WARNING: no fold manifest at ${FOLD_FILE} — skipping ${CATCHMENT}"
            echo "           (run stage 0 for ${CATCHMENT} first)"
            continue
        fi
        FOLDS="$(<"$FOLD_FILE")"
        if [[ -z "${FOLDS// }" ]]; then
            echo "  WARNING: fold manifest for ${CATCHMENT} is empty — skipping"
            continue
        fi
        echo "=== Tuned: ${CATCHMENT} / ${MODEL}: $(echo $FOLDS | wc -w) folds ==="
        for FOLD in $FOLDS; do
            submit "${DESIGN}_tuned_${CATCHMENT}_${MODEL}_${FOLD}" \
                "${CATCHMENT}" "${MODEL}" tuned --fold "${FOLD}"
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo "=== Total ${VARIANT} jobs submitted: ${TOTAL} ==="
