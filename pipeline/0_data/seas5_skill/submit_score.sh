#!/bin/bash
# =============================================================================
# submit_score.sh — SEAS5 meteorological skill against ERA5 (TODO.md, R2)
#
# One job per catchment (score.py). Needs the catchment's stage 0 (features,
# forcing cells, fold manifests) and its ERA5 table (pipeline/0_data/era5/).
#
# Usage (any directory):
#   bash pipeline/0_data/seas5_skill/submit_score.sh [--design <d>] [catchments...|all]
#
# Resources: a first guess (Kirchbichl's sandbox run, see TODO.md R2 step 4).
# =============================================================================

set -eo pipefail
cd "$(dirname "$0")/../../.."       # repository root (job.sh starts in the submit dir)
mkdir -p logs

ALL_CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)
DESIGN=loyo
if [[ "$1" == "--design" ]]; then DESIGN="$2"; shift 2; fi
if [[ "$1" == "all" || $# -eq 0 ]]; then CATCHMENTS=("${ALL_CATCHMENTS[@]}"); else CATCHMENTS=("$@"); fi

for CATCHMENT in "${CATCHMENTS[@]}"; do
    TAG="${DESIGN}_seas5skill_${CATCHMENT}"
    sbatch --job-name=s5ml-${TAG} --partition=standard --qos=short \
        --cpus-per-task=2 --mem=32G --time=02:00:00 \
        --output=logs/${TAG}-%j.out --error=logs/${TAG}-%j.err \
        pipeline/job.sh python pipeline/0_data/seas5_skill/score.py "${CATCHMENT}" --design "${DESIGN}"
    echo "  Submitted SEAS5 skill: ${CATCHMENT}  [${DESIGN}]"
done
