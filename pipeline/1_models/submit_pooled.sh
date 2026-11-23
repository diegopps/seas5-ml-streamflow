#!/bin/bash
# =============================================================================
# submit_pooled.sh — stage 1, pooled models (pipeline/1_models/run_pooled.py)
#
#   timing   one job: times one pooled fold (full default fit + a short probe
#            per hidden size of the grid) and prints the projected runtimes
#            and suggested --time / --mem of the default and tuned jobs.
#            Writes no results.
#   default  one job: every pooled fold of the design, default configuration.
#   tuned    one job per pooled fold: nested-CV search, winner retrain and
#            per-catchment scores.
#
# The pooled model trains on every catchment at once, so its jobs need every
# catchment's stage-0 output. The tuned fold ids are the union of the
# catchments' fold manifests (data/processed_data/<catchment>_folds_<design>.txt;
# loyo: every hydro year of any catchment, split: hindcast), built here so no
# Python runs on the login node; each job checks its fold against the data.
#
# Resubmitting resumes: the default job continues from its fold checkpoints;
# tuned folds already finished (fold_<id>/COMPLETE in the pooled working
# folder) are skipped, and an unfinished one skips its completed inner runs.
#
# CPU only, QOS medium (7-day walltime), 16 CPUs (pipeline/job.sh sets the
# torch thread count).
#
# Usage:
#   bash pipeline/1_models/submit_pooled.sh <variant> [--model <m>] [--design <d>]
#
#   bash pipeline/1_models/submit_pooled.sh timing
#   bash pipeline/1_models/submit_pooled.sh default --design split
#   bash pipeline/1_models/submit_pooled.sh tuned
# =============================================================================

# Submit from the repo root so jobs start there (job.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm opens the log files before the job script runs

# Must match seas5ml.pooled.POOLED_CATCHMENTS.
POOL=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)
MODEL=pooled_lstm
DESIGN=loyo

# Slurm routing per variant (and design): "partition qos cpus mem time".
# default / tuned from the timing reports of 2026-11-20 (24 CPUs, which the
# cluster allocates whatever is requested; peak memory 16 GB):
#   loyo   default expected 28.5 h (upper bound 63.6 h at 50 epochs),
#          tuned per fold 72.5 h (163 h)
#   split  default 0.4 h (0.9 h), tuned 40.9 h (95.5 h)
# Every job resumes when resubmitted, so a walltime below the upper bound
# costs a resubmission, not the work done.
declare -A ROUTE=(
    [timing]="standard  medium 16 128G 2-00:00:00"
    [default_loyo]="standard  medium 16 32G 3-00:00:00"
    [tuned_loyo]="standard    medium 16 32G 4-13:00:00"
    [default_split]="standard medium 16 32G 0-02:00:00"
    [tuned_split]="standard   medium 16 32G 4-00:00:00"
)

VARIANT="$1"
shift
case "$VARIANT" in
    timing|default|tuned) ;;
    *) echo "Usage: bash $0 timing|default|tuned [--model <m>] [--design <d>]"
       exit 1 ;;
esac

while [[ "$1" == --* ]]; do
    case "$1" in
        --model)  MODEL="$2";  shift 2 ;;
        --design) DESIGN="$2"; shift 2 ;;
        *) echo "ERROR: unknown option $1"; exit 1 ;;
    esac
done
ROUTE_KEY=$([[ "$VARIANT" == "timing" ]] && echo timing || echo "${VARIANT}_${DESIGN}")
if [[ -z "${ROUTE[$ROUTE_KEY]:-}" ]]; then
    echo "ERROR: no Slurm route for ${VARIANT} under design ${DESIGN}"
    exit 1
fi
read -r PARTITION QOS CPUS MEM WALLTIME <<< "${ROUTE[$ROUTE_KEY]}"

submit() {   # submit <job-tag> <run_pooled.py args...>
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
        pipeline/job.sh python pipeline/1_models/run_pooled.py "${MODEL}" "$@" --design "${DESIGN}"
}

if [[ "$VARIANT" != "tuned" ]]; then
    submit "${DESIGN}_pooled_${VARIANT}_${MODEL}" "${VARIANT}"
    echo "  Submitted ${VARIANT}: ${MODEL} (${DESIGN})  [${PARTITION}/${QOS}, ${CPUS} CPUs]"
    exit 0
fi

# ── tuned: one job per pooled fold ──────────────────────────────────────────
FOLDS=()
for CATCHMENT in "${POOL[@]}"; do
    FOLD_FILE="data/processed_data/${CATCHMENT}_folds_${DESIGN}.txt"
    if [[ ! -f "$FOLD_FILE" ]]; then
        echo "ERROR: no fold manifest at ${FOLD_FILE} (run stage 0 for ${CATCHMENT} first);"
        echo "       the pooled model needs every catchment of the pool."
        exit 1
    fi
    # shellcheck disable=SC2207  # word-split the manifest on purpose
    FOLDS+=($(<"$FOLD_FILE"))
done
mapfile -t FOLDS < <(printf '%s\n' "${FOLDS[@]}" | sort -u)

WORK="data/results/${DESIGN}/tuned/_pooled/${MODEL}"   # seas5ml.paths.pooled_work_dir
TOTAL=0
SKIPPED=0
echo "=== Tuned: ${MODEL} (${DESIGN}): ${#FOLDS[@]} pooled folds ==="
for FOLD in "${FOLDS[@]}"; do
    if [[ -f "${WORK}/fold_${FOLD}/COMPLETE" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    submit "${DESIGN}_pooled_tuned_${MODEL}_${FOLD}" tuned --fold "${FOLD}"
    TOTAL=$((TOTAL + 1))
done
echo "=== Total tuned jobs submitted: ${TOTAL} (${SKIPPED} folds already complete) ==="
