#!/bin/bash
#SBATCH --job-name=FlCa-paperfigs
#SBATCH --output=slurm_log/paperfigs-%j.out
#SBATCH --error=slurm_log/paperfigs-%j.err

# =============================================================================
# run_paper_figures.sh
#
# Slurm job script. Called by submit_paper_figures.sh — do not run directly.
# Resource requests (partition, qos, cpus-per-task, mem, time) are set by the
# wrapper on the sbatch line.
#
# Regenerates every code-produced figure of the paper under the file names
# LaTeX_report/main.tex uses:
#
#   model_results/00_plots/fig03-fig09.pdf       main text (Zeravshan, Ayat,
#                                                Kirchbichl-Bichlwang)
#   model_results/00_appendix/figA01-figC03.pdf  Appendices A-C, one figure
#                                                per regime (mountain, steppe,
#                                                Europe)
#
# Requires the stage 01-03 results under model_results/ and the discharge
# CSVs set in config.toml. Fig. 1 (catchment map), Fig. 2 (workflow
# schematic) and the graphical abstract were made outside this code and are
# not produced here.
#
# The plot scripts write fixed file names into model_results/00_plots/, which
# the next run would overwrite, so each set of outputs is renamed right after
# the runs that made it.
# =============================================================================

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"   # repo root: submit_paper_figures.sh cds there before sbatch
source load_env.sh         # environment setup from config.toml [slurm] env_setup
set -u                     # only now: conda's activate scripts use unset variables

PLOTS=model_results/00_plots
APPENDIX=model_results/00_appendix
mkdir -p "${PLOTS}" "${APPENDIX}"

# Raw outputs of the three plot scripts, in paper order.
RAW=(01a_boxplotcomparison_short    # stage 1, lead days 0-30
     01b_boxplotcomparison_long     # stage 1, lead days 185-214
     02_leadtimeheatmap             # stage 1, ACC by init month x lead bin
     03_configranking               # stage 2, config ranking
     04a_tuningvsreference_short    # stage 2, lead days 0-30
     04b_tuningvsreference_long     # stage 2, lead days 185-214
     07_sensitivity)                # stage 3

plot_all() {
    python 01_plot_exploration.py "$@"
    python 02_plot_hyperparamtuning.py "$@"
    python 03_plot_sensitivity.py "$@"
}

# ── Main text: Figs. 3-9 ─────────────────────────────────────────────────
echo "=== Main text: zeravshan ayat kirchbichl ==="
plot_all zeravshan ayat kirchbichl
for i in "${!RAW[@]}"; do
    mv "${PLOTS}/${RAW[$i]}.pdf" "${PLOTS}/fig0$((i + 3)).pdf"
done

# ── Appendices A-C: one figure per regime (k = 1, 2, 3) ──────────────────
REGIMES=("zeravshan ulkenboken kurshim"      # k=1  mountain
         "kalkutan esil ayat zhabay"         # k=2  steppe
         "kirchbichl diepoldsau")            # k=3  Europe

for k in 1 2 3; do
    echo "=== Appendix, regime ${k}: ${REGIMES[$((k - 1))]} ==="
    # shellcheck disable=SC2086  # word-split the catchment list on purpose
    plot_all ${REGIMES[$((k - 1))]}
    mv "${PLOTS}/${RAW[0]}.pdf" "${APPENDIX}/figA0${k}.pdf"
    mv "${PLOTS}/${RAW[1]}.pdf" "${APPENDIX}/figA0$((k + 3)).pdf"
    mv "${PLOTS}/${RAW[2]}.pdf" "${APPENDIX}/figA0$((k + 6)).pdf"
    mv "${PLOTS}/${RAW[3]}.pdf" "${APPENDIX}/figB0${k}.pdf"
    mv "${PLOTS}/${RAW[4]}.pdf" "${APPENDIX}/figB0$((k + 3)).pdf"
    mv "${PLOTS}/${RAW[5]}.pdf" "${APPENDIX}/figB0$((k + 6)).pdf"
    mv "${PLOTS}/${RAW[6]}.pdf" "${APPENDIX}/figC0${k}.pdf"
done

# 02_plot_hyperparamtuning.py also writes the config ranking as a table.
rm -f "${PLOTS}/03_configranking.csv"

echo "Done: ${PLOTS}/fig03-fig09.pdf, ${APPENDIX}/figA01-figC03.pdf"
