#!/bin/bash
# =============================================================================
# submit_grib2netcdf.sh
#
# Computes the array size from seas5_settings.py (len(YEARS) x len(MONTHS),
# set in config.toml) and
# submits run_grib2netcdf.sh as a Slurm array job. The array size can't be
# an #SBATCH directive in the run script since it depends on config.toml and
# directives are parsed before any shell runs — so it's computed here and
# passed on the sbatch command line instead, which overrides any directive.
#
# Usage:
#   bash submit_grib2netcdf.sh
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")"
mkdir -p ../../../logs   # Slurm logs (paths in run_grib2netcdf.sh)

# Read the settings in a subshell: the environment (load_env.sh) must not be
# loaded in the shell that calls sbatch, since sbatch exports it and the job
# loads it again (loading it twice makes imports fail intermittently).
settings=$(source ../../load_env.sh >/dev/null 2>&1 && python3 -c \
    "import seas5_settings as s; print(len(s.YEARS)*len(s.MONTHS), s.REGION, s.YEARS[0] + '-' + s.YEARS[-1])") \
    || { echo "ERROR: could not read seas5_settings.py (environment or config.toml)" >&2; exit 1; }
read -r n_tasks region years <<< "$settings"

echo "Submitting $n_tasks tasks (array 0-$((n_tasks - 1))%30) for $region $years"
sbatch --array=0-$((n_tasks - 1))%30 run_grib2netcdf.sh
