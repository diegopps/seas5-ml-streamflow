#!/bin/bash
# =============================================================================
# make_figures.sh — stage 3 (paper figures)
#
# Run as a Slurm job by submit.sh (through pipeline/job.sh, which sets up the
# environment); run from the repository root.
#
# Regenerates every code-produced figure of the paper, from the loyo design:
#
#   data/results/paper_figures/main/fig03-fig09.pdf        main text (Zeravshan,
#                                                          Ayat, Kirchbichl-Bichlwang)
#   data/results/paper_figures/appendix/figA01-figC03.pdf  Appendices A-C, one
#                                                          figure per regime
#                                                          (mountain, steppe, Europe)
#
# Requires the stage 1 (default and tuned) and stage 2 results under
# data/results/loyo/ and the discharge CSVs set in config.toml. Fig. 1
# (catchment map), Fig. 2 (workflow schematic) and the graphical abstract
# were made outside this code and are not produced here.
#
# The plot scripts write fixed file names into data/results/loyo/figures/,
# which the next run would overwrite, so each set of outputs is moved into
# main/ or appendix/ right after the runs that made it.
# =============================================================================

set -euo pipefail

DESIGN=loyo
PLOTS=data/results/${DESIGN}/figures       # raw outputs of the plot scripts
MAIN=data/results/paper_figures/main
APPENDIX=data/results/paper_figures/appendix
mkdir -p "${PLOTS}" "${MAIN}" "${APPENDIX}"

# Raw outputs of the plot scripts, in paper order.
RAW=(01a_boxplotcomparison_short    # default run, lead days 0-30
     01b_boxplotcomparison_long     # default run, lead days 185-214
     02_leadtimeheatmap             # default run, ACC by init month x lead bin
     03_configranking               # tuned run, config ranking
     04a_tuningvsreference_short    # tuned run, lead days 0-30
     04b_tuningvsreference_long     # tuned run, lead days 185-214
     07_sensitivity)                # stage 2

plot_all() {
    python pipeline/1_models/plot.py default "$@" --design "${DESIGN}"
    python pipeline/1_models/plot.py tuned "$@" --design "${DESIGN}"
    python pipeline/2_sensitivity/plot.py "$@" --design "${DESIGN}"
}

# ── Main text: Figs. 3-9 ─────────────────────────────────────────────────
echo "=== Main text: zeravshan ayat kirchbichl ==="
plot_all zeravshan ayat kirchbichl
for i in "${!RAW[@]}"; do
    mv "${PLOTS}/${RAW[$i]}.pdf" "${MAIN}/fig0$((i + 3)).pdf"
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

# The tuned plots also write the config ranking as a table.
rm -f "${PLOTS}/03_configranking.csv"

echo "Done: ${MAIN}/fig03-fig09.pdf, ${APPENDIX}/figA01-figC03.pdf"
