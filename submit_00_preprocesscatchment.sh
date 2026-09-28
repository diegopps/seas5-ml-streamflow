#!/bin/bash

# Submit from the repo root so jobs start there (run_*.sh cd to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")"

CATCHMENTS=(ulkenboken zeravshan esil ayat kalkutan kurshim zhabay kirchbichl diepoldsau)


for catchment in "${CATCHMENTS[@]}"; do
    sbatch \
        --job-name=FlCa-prep_${catchment} \
        --output=slurm_log/FlowCast_${catchment}-%j.out \
        --error=slurm_log/FlowCast_${catchment}-%j.err \
        --export=ALL,CATCHMENT=${catchment} \
        run_00_preprocesscatchment.sh
    echo "Submitted job for: ${catchment}"
done

