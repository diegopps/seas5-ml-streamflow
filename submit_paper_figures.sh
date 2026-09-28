#!/bin/bash
# =============================================================================
# submit_paper_figures.sh
#
# Submits one Slurm job (run_paper_figures.sh) that regenerates every
# code-produced figure of the paper under the file names LaTeX_report/main.tex
# uses: model_results/00_plots/fig03-fig09.pdf and
# model_results/00_appendix/figA01-figC03.pdf.
#
# Stages 01-03 must have completed for all nine catchments first. The job
# only plots existing results (a few minutes).
#
# Usage:
#   bash submit_paper_figures.sh
# =============================================================================

# Submit from the repo root so jobs start there (run_*.sh cd to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")"

mkdir -p slurm_log

sbatch \
    --partition=standard \
    --qos=short \
    --cpus-per-task=4 \
    --mem=64G \
    --time=03:00:00 \
    run_paper_figures.sh
