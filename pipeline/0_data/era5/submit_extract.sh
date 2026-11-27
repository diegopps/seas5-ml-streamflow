#!/bin/bash
# =============================================================================
# submit_extract.sh
#
# ERA5 daily catchment series for the SEAS5 skill evaluation (TODO.md, R2):
# a Slurm array over years running extract_era5.py --array (one year, all
# catchments per task; ~2 min each), then one --combine job that starts when
# every task has succeeded and writes data/processed_data/<catchment>_era5.parquet.
#
# Usage (any directory):
#     bash pipeline/0_data/era5/submit_extract.sh                 # array + combine
#     bash pipeline/0_data/era5/submit_extract.sh --combine-only  # e.g. after the snow-depth download
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")/../../.."       # repository root (job.sh starts in the submit dir)
mkdir -p logs

COMMON=(--partition=standard --qos=short --cpus-per-task=1 --mem=8G)

if [[ "${1:-}" != "--combine-only" ]]; then
    # Read the array size in a subshell: the environment must not be loaded in
    # the shell that calls sbatch (sbatch exports it, job.sh loads it again).
    n_tasks=$(source pipeline/load_env.sh >/dev/null 2>&1 && python pipeline/0_data/era5/extract_era5.py --n-tasks) \
        || { echo "ERROR: could not run extract_era5.py --n-tasks" >&2; exit 1; }
    array_id=$(sbatch --parsable "${COMMON[@]}" --time=00:30:00 \
        --array=0-$((n_tasks - 1))%20 --job-name=s5ml-era5-extract \
        --output=logs/era5_extract-%A_%a.out --error=logs/era5_extract-%A_%a.err \
        pipeline/job.sh python pipeline/0_data/era5/extract_era5.py --array)
    echo "Submitted extraction array ${array_id} (${n_tasks} years)"
    DEP=(--dependency=afterok:${array_id})
else
    DEP=()
fi

combine_id=$(sbatch --parsable "${COMMON[@]}" --time=00:30:00 "${DEP[@]}" \
    --job-name=s5ml-era5-combine \
    --output=logs/era5_combine-%j.out --error=logs/era5_combine-%j.err \
    pipeline/job.sh python pipeline/0_data/era5/extract_era5.py --combine)
echo "Submitted combine ${combine_id}"
