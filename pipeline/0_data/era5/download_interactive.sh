#!/bin/bash
# =============================================================================
# download_interactive.sh
#
# Downloads ERA5 snow depth from the CDS (download_sd.py) for the SEAS5 skill
# evaluation (TODO.md, R2). The other ERA5 variables come from the PIK copy
# (extract_era5.py).
#
# NOT a Slurm job: compute nodes have no outbound network access, so the CDS
# API is only reachable from the login node. Run it AFTER the SEAS5 download
# (same CDS account and queue). Resumable: finished files are skipped, so just
# re-run it if it dies.
#
# Usage, inside screen / tmux so it survives a dropped connection:
#     bash pipeline/0_data/era5/download_interactive.sh
# Progress: logs/era5_sd_download.log
# Then: python pipeline/0_data/era5/extract_era5.py --combine  (adds sd;
#       or bash pipeline/0_data/era5/submit_extract.sh --combine-only)
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")/../../.."       # repository root
source pipeline/load_env.sh         # environment setup from config.toml [slurm] env_setup
python3 -u pipeline/0_data/era5/download_sd.py
