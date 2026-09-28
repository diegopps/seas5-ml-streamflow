#!/usr/bin/env python3
"""
01_plot_exploration.py – Stage-01 exploration plots.

Produces figures for the model comparison (climatology, persistence,
damped persistence, ridge, RF, XGBoost, LSTM):

  model_results/00_plots/
      01a_boxplotcomparison_short.pdf  boxplots: NSE, KGE, ACC
                                               pooled over lead_day 0-30
      01b_boxplotcomparison_long.pdf   boxplots: NSE, KGE, ACC
                                               pooled over lead_day 185-214
      02_leadtimeheatmap.pdf           ACC by lead time x init month,
                                               one panel per (catchment, model)
                                               in a single trellis figure

Reads directly from each model's results directory — no intermediate
comparison CSV is required:
  model_results/01_exploration/results_<catchment>_<model>/
      fold_metrics.csv
      predictions_daily.parquet

Usage:
    python 01_plot_exploration.py                    # default: all available catchments
    python 01_plot_exploration.py zeravshan kurshim  # specific catchments
    python 01_plot_exploration.py --split            # heatmap grid split into
                                                     # per-regime-group figures
"""

import os
import sys

from flowcast_src import plot_lib as pl
from flowcast_src import plot_boxplots
from flowcast_src import plot_heatmap


def main():
    args = sys.argv[1:]

    # --split renders the heatmap grid as separate per-regime-group figures.
    split = "--split" in args
    args = [a for a in args if a != "--split"]

    catchments = pl.parse_catchment_args(args)
    if not catchments:
        sys.exit(1)

    os.makedirs(pl.PLOT_DIR, exist_ok=True)

    print("\n=== Boxplot comparison (short lead, 0-30 d) ===")
    plot_boxplots.run(catchments, 0, 30,
                      "LOGO-CV Aggregate Skill\n(Short) Lead time: 0-30 days",
                      "01a_boxplotcomparison_short")

    print("\n=== Boxplot comparison (long lead, 185-214 d) ===")
    plot_boxplots.run(catchments, 185, 214,
                      "LOGO-CV Aggregate Skill\n(Long) Lead time: 185-214 days",
                      "01b_boxplotcomparison_long")

    print("\n=== Lead-time x init-month heatmap grid ===")
    plot_heatmap.run(catchments, split=split)

    print("\nDone.")


if __name__ == '__main__':
    main()
