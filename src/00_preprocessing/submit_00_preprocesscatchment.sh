#!/bin/bash

# Submit from the repo root so jobs start there (run_*.sh cd to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm opens the log files before the job script runs

CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)


for catchment in "${CATCHMENTS[@]}"; do
    sbatch \
        --job-name=FlCa-prep_${catchment} \
        --output=logs/FlowCast_${catchment}-%j.out \
        --error=logs/FlowCast_${catchment}-%j.err \
        --export=ALL,CATCHMENT=${catchment} \
        src/00_preprocessing/run_00_preprocesscatchment.sh
    echo "Submitted job for: ${catchment}"
done

