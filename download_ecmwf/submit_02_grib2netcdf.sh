#!/bin/bash
# =============================================================================
# submit_02_grib2netcdf.sh
#
# Computes the array size from seas5_settings.py (len(YEARS) x len(MONTHS),
# set in config.toml) and
# submits run_02_grib2netcdf.sh as a Slurm array job. The array size can't be
# an #SBATCH directive in the run script since it depends on config.toml and
# directives are parsed before any shell runs — so it's computed here and
# passed on the sbatch command line instead, which overrides any directive.
#
# Usage:
#   bash submit_02_grib2netcdf.sh
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")"
source ../load_env.sh      # environment setup from config.toml [slurm] env_setup
mkdir -p sbatch_logs

n_tasks=$(python3 -c "import seas5_settings as s; print(len(s.YEARS)*len(s.MONTHS))")
region=$(python3 -c "import seas5_settings as s; print(s.REGION)")
years=$(python3 -c "import seas5_settings as s; print(s.YEARS[0] + '-' + s.YEARS[-1])")

echo "Submitting $n_tasks tasks (array 0-$((n_tasks - 1))%30) for $region $years"
sbatch --array=0-$((n_tasks - 1))%30 run_02_grib2netcdf.sh
