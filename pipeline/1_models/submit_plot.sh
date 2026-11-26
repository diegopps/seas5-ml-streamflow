#!/bin/bash
# =============================================================================
# submit_plot.sh — stage 1 figures (pipeline/1_models/plot.py), one job
#
#   default  lead-month curves, per-year boxplots, start month x lead month
#            heatmaps, from the default score tables (run submit_score.sh
#            default first)
#   tuned    config ranking and tuning-vs-reference figures, from the tuned
#            folds
#
# Figures (and a .csv of what each plots) go to data/results/<design>/figures/.
#
# Usage:
#   bash pipeline/1_models/submit_plot.sh <variant> [--design <d>] [--by-regime] [catchments...]
#   (no catchments: every catchment with results)
#
#   bash pipeline/1_models/submit_plot.sh default
#   bash pipeline/1_models/submit_plot.sh default --design split
#
# Resources: the default figures of all 9 catchments render in about a minute
# (test renders 2026-11-26), CPU only.
# =============================================================================

# Submit from the repo root so jobs start there (job.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm opens the log files before the job script runs

DESIGN=loyo
EXTRA=()

PARTITION=standard
QOS=short
CPUS=4
MEM=16G
WALLTIME=01:00:00

VARIANT="$1"
shift
case "$VARIANT" in
    default|tuned) ;;
    *)
        echo "Usage: bash $0 default|tuned [--design <d>] [--by-regime] [catchments...]"
        exit 1 ;;
esac

while [[ "$1" == --* ]]; do
    case "$1" in
        --design)    DESIGN="$2"; shift 2 ;;
        --by-regime) EXTRA+=(--by-regime); shift ;;
        *) echo "ERROR: unknown option $1"; exit 1 ;;
    esac
done

TAG="${DESIGN}_plot_${VARIANT}"
sbatch \
    --job-name=s5ml-${TAG} \
    --partition=${PARTITION} \
    --qos=${QOS} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --time=${WALLTIME} \
    --output=logs/${TAG}-%j.out \
    --error=logs/${TAG}-%j.err \
    pipeline/job.sh python pipeline/1_models/plot.py "${VARIANT}" "$@" \
        --design "${DESIGN}" "${EXTRA[@]}"
echo "  Submitted ${VARIANT} figures  [${DESIGN}]${*:+ for $*}"
