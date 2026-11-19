#!/bin/bash
# =============================================================================
# submit.sh — stage 3 (paper figures)
#
# Submits one Slurm job running make_figures.sh, which regenerates every
# code-produced figure of the paper under data/results/paper_figures/
# (main/fig03-fig09.pdf, appendix/figA01-figC03.pdf).
#
# Stages 1 (default and tuned) and 2 must have completed for all nine
# catchments first. The job only plots existing results (a few minutes).
#
# Usage:
#   bash pipeline/3_paper_figures/submit.sh
# =============================================================================

# Submit from the repo root so the job starts there (job.sh cds to SLURM_SUBMIT_DIR).
cd "$(dirname "$0")/../.."
mkdir -p logs

sbatch \
    --job-name=s5ml-paperfigs \
    --partition=standard \
    --qos=short \
    --cpus-per-task=4 \
    --mem=64G \
    --time=03:00:00 \
    --output=logs/paperfigs-%j.out \
    --error=logs/paperfigs-%j.err \
    pipeline/job.sh bash pipeline/3_paper_figures/make_figures.sh
