#!/bin/bash
# =============================================================================
# submit_paper_figures.sh
#
# Submits one Slurm job (run_paper_figures.sh) that regenerates every
# code-produced figure of the paper under the file names latex/main.tex
# uses: data/results/figures/main/fig03-fig09.pdf and
# data/results/figures/appendix/figA01-figC03.pdf.
#
# Stages 01-03 must have completed for all nine catchments first. The job
# only plots existing results (a few minutes).
#
# Usage:
#   bash submit_paper_figures.sh
# =============================================================================

# Submit from the repo root so the job starts there (run_paper_figures.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/.."

mkdir -p logs

sbatch \
    --partition=standard \
    --qos=short \
    --cpus-per-task=4 \
    --mem=64G \
    --time=03:00:00 \
    src/run_paper_figures.sh
