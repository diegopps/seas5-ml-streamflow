#!/usr/bin/env python3
"""
03_plot_sensitivity.py – Stage-03 climate-input sensitivity plot.

Reads the per-fold sensitivity metrics written by
03_sensitivity_climate_input.py:

    model_results/03_sensitivity/results_<catchment>_<model>/
        sensitivity_fold_metrics.csv   one row per fold x scenario

Produces, under model_results/00_plots/, saved as .pdf:

    07_sensitivity   Does corrupting the meteorological forcing change the
                     models' skill? Per-fold paired difference between real
                     forcing and donor-year forcing, one row per catchment,
                     one column per metric (NSE, KGE, ACC, CRPS). Each dot is
                     one hydro year; above zero means the corruption hurt
                     skill.

Only the four learned models appear (ridge, rf, xgb, lstm). Climatology and
persistence never read the forcing, so corrupting it cannot move them and the
comparison would be vacuous.

Usage:
    python 03_plot_sensitivity.py                    # every catchment with results
    python 03_plot_sensitivity.py zeravshan kurshim  # specific catchments
"""

import os
import sys

from flowcast_src import plot_lib as pl
from flowcast_src import plot_sensitivity


def main():
    catchments = [a.lower() for a in sys.argv[1:]]

    os.makedirs(pl.PLOT_DIR, exist_ok=True)

    available = plot_sensitivity.find_catchments(plot_sensitivity.RESULTS_DIR)
    if not available:
        print("No stage-03 results found under "
              f"{os.path.normpath(plot_sensitivity.RESULTS_DIR)}.")
        print("Run 03_sensitivity_climate_input.py (via "
              "submit_03_sensitivity.sh) first.")
        sys.exit(1)

    shown = catchments if catchments else available
    print(f"Catchments: {', '.join(shown)}")
    print(f"Models    : {', '.join(plot_sensitivity.SENS_MODELS)}")

    print("\n=== Climate-input sensitivity ===")
    plot_sensitivity.run(catchments or None)

    print("\nDone.")


if __name__ == '__main__':
    main()
