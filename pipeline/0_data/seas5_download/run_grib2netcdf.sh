#!/bin/bash
#SBATCH --job-name=s5ml-grib2netcdf
#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=../../../logs/grib2netcdf-%A_%a.out
#SBATCH --error=../../../logs/grib2netcdf-%A_%a.err

# =============================================================================
# run_grib2netcdf.sh
#
# Slurm array job script: converts one year-month of GRIB to NetCDF per task.
# Called by submit_grib2netcdf.sh — do not run directly (task-id 0..N-1
# indexes into the YEARS x MONTHS grid from seas5_settings.py, sized by the
# submitter).
# =============================================================================

cd "${SLURM_SUBMIT_DIR}"   # pipeline/0_data/seas5_download/: submit_grib2netcdf.sh cds there before sbatch
source ../../load_env.sh      # environment setup from config.toml [slurm] env_setup

python3 -u grib2netcdf.py --task-id "$SLURM_ARRAY_TASK_ID"
