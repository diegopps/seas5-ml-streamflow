#!/bin/bash
# =============================================================================
# download_interactive.sh
#
# Downloads ECMWF SEAS5 GRIB files for the region/years set in config.toml
# ([seas5_download] section, at the repository root).
#
# NOT a Slurm job: compute nodes have no outbound network access, so the CDS
# API is only reachable from the login node. This must run interactively.
#
# Long-running (thousands of requests) but resumable — already-downloaded
# files are skipped, so just re-run it if it dies or you disconnect.
#
# Usage — run inside tmux/screen so it survives a dropped connection:
#     tmux new -s ecmwf
#     bash download_interactive.sh
#     # detach with Ctrl-b d, reattach later with: tmux attach -t ecmwf
#
# Or detached via nohup:
#     nohup bash download_interactive.sh > download_stdout.log 2>&1 &
#
# Progress: 01_download_ecmwf_parallel.py also appends to ecmwf_download.log
# =============================================================================

set -eo pipefail

cd "$(dirname "$0")"
source ../load_env.sh      # environment setup from config.toml [slurm] env_setup

region=$(python3 -c "import seas5_settings as s; print(s.REGION)")
years=$(python3 -c "import seas5_settings as s; print(s.YEARS[0] + '-' + s.YEARS[-1])")
area=$(python3 -c "import seas5_settings as s; print(s.AREA)")

echo "Downloading $region $years  area=$area"
python3 -u 01_download_ecmwf_parallel.py
