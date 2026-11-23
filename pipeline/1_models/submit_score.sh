#!/bin/bash
# =============================================================================
# submit_score.sh — stage 1 scores at daily, weekly and monthly resolution
#
# One job per catchment: scores every model of the line-up (score.py) from
# the stage-1 predictions already on disk. Submit after the stage-1 runs of
# that line-up are complete (tuned: a tuned model with missing folds stops
# the job unless --partial). No training, CPU only.
#
#   default      every model's default run
#   tuned        the tuned models' folds + ridge and the baselines' default runs
#   sensitivity  stage 2's scenarios of the learned models (after stage 2):
#                each scenario scored, plus the paired deltas per lead month
#
# Usage:
#   bash pipeline/1_models/submit_score.sh <variant> [--design <d>] [--partial] [catchments...|all]
#   (--partial: tuned only)
#
#   bash pipeline/1_models/submit_score.sh default all
#   bash pipeline/1_models/submit_score.sh tuned --design split all
#
# Resources are a first guess (pandas, single-threaded; the largest member
# frames are Kirchbichl's): adjust after the first run.
# =============================================================================

# Submit from the repo root so jobs start there (job.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm opens the log files before the job script runs

ALL_CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)
DESIGN=loyo
EXTRA=()

PARTITION=standard
QOS=short
CPUS=4
MEM=64G
WALLTIME=04:00:00

VARIANT="$1"
shift
case "$VARIANT" in
    default|tuned|sensitivity) ;;
    *)
        echo "Usage: bash $0 default|tuned|sensitivity [--design <d>] [--partial] [catchments...|all]"
        exit 1 ;;
esac

while [[ "$1" == --* ]]; do
    case "$1" in
        --design)  DESIGN="$2"; shift 2 ;;
        --partial) EXTRA+=(--partial); shift ;;
        *) echo "ERROR: unknown option $1"; exit 1 ;;
    esac
done

if [[ "$1" == "all" || $# -eq 0 ]]; then
    CATCHMENTS=("${ALL_CATCHMENTS[@]}")
else
    CATCHMENTS=("$@")
fi

for CATCHMENT in "${CATCHMENTS[@]}"; do
    TAG="${DESIGN}_score_${VARIANT}_${CATCHMENT}"
    sbatch \
        --job-name=s5ml-${TAG} \
        --partition=${PARTITION} \
        --qos=${QOS} \
        --cpus-per-task=${CPUS} \
        --mem=${MEM} \
        --time=${WALLTIME} \
        --output=logs/${TAG}-%j.out \
        --error=logs/${TAG}-%j.err \
        pipeline/job.sh python pipeline/1_models/score.py \
            "${CATCHMENT}" "${VARIANT}" --design "${DESIGN}" "${EXTRA[@]}"
    echo "  Submitted ${VARIANT} scores: ${CATCHMENT}  [${DESIGN}]"
done
echo "=== Total score jobs submitted: ${#CATCHMENTS[@]} ==="
