#!/bin/bash
# =============================================================================
# submit.sh — stage 2 (climate-input sensitivity test)
#
# Submits one Slurm job per (catchment, model); each job covers every fold of
# that pair. Per fold the job trains one model and scores it twice — on real
# forcing and on a donor year's forcing — so the cost is roughly one training
# per fold, plus one final all-years fit (saved to
# data/trained_models/<design>/), i.e. ~n_years + 1 trainings per job.
#
# rf, xgb, and lstm use each fold's own tuned winner (stage 1, tuned
# variant), so the tuned run must be COMPLETE for the pairs submitted: a fold
# without its winner.csv stops the job. The final model uses the
# catchment-wide winner (most fold wins; 03_configranking.pdf's 'best').
# Ridge has no search and runs its default configuration.
#
# Usage:
#   bash pipeline/2_sensitivity/submit.sh                          # default catchments, all models
#   bash pipeline/2_sensitivity/submit.sh all                      # all catchments, all models
#   bash pipeline/2_sensitivity/submit.sh --model xgb all          # one model, all catchments
#   bash pipeline/2_sensitivity/submit.sh --model lstm zeravshan   # one model, one catchment
#   (add --design <d> for another evaluation design; default loyo)
# =============================================================================

# Submit from the repo root so jobs start there (job.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm opens the log files before the job script runs

DEFAULT_CATCHMENTS=(zeravshan ayat kirchbichl)
ALL_CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)
ALL_MODELS=(ridge rf xgb lstm)
DESIGN=loyo

# Per-model Slurm routing: "partition qos cpus mem time"
# lstm runs on CPU rather than GPU: gpushort caps a user at 2 concurrent jobs.
# It uses the medium qos because a whole catchment can exceed short's 1-day
# cap: kirchbichl's winner (hidden 512) is ~24 h on 16 CPUs.
declare -A ROUTE=(
    [lstm]="standard medium   16 128G 48:00:00"
    [ridge]="standard short    16 128G 12:00:00"
    [rf]="standard    short    16 128G 12:00:00"
    [xgb]="standard   short    16 128G 12:00:00"
)

MODELS=("${ALL_MODELS[@]}")
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

TOTAL=0
for MODEL in "${MODELS[@]}"; do
    if [[ -z "${ROUTE[$MODEL]:-}" ]]; then
        echo "ERROR: no Slurm routing defined for model '${MODEL}'"
        echo "       (add it to ROUTE above)"
        exit 1
    fi
    read -r PARTITION QOS CPUS MEM WALLTIME <<< "${ROUTE[$MODEL]}"

    for CATCHMENT in "${CATCHMENTS[@]}"; do
        TAG="${DESIGN}_sens_${CATCHMENT}_${MODEL}"
        sbatch \
            --job-name=s5ml-${TAG} \
            --partition=${PARTITION} \
            --qos=${QOS} \
            --cpus-per-task=${CPUS} \
            --mem=${MEM} \
            --time=${WALLTIME} \
            --output=logs/${TAG}-%j.out \
            --error=logs/${TAG}-%j.err \
            pipeline/job.sh python pipeline/2_sensitivity/run.py \
                "${CATCHMENT}" "${MODEL}" --design "${DESIGN}"
        echo "  Submitted sensitivity: ${CATCHMENT} / ${MODEL}  [${PARTITION}/${QOS}]"
        TOTAL=$((TOTAL + 1))
    done
done

echo "=== Total sensitivity jobs submitted: ${TOTAL} ==="
