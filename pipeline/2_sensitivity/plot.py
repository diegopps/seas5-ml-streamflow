#!/usr/bin/env python3
"""
Stage 2 plot – climate-input sensitivity.

Reads the per-year sensitivity metrics written by
pipeline/2_sensitivity/run.py:

    data/results/<design>/sensitivity/<catchment>/<model>/
        sensitivity_fold_metrics.csv   one row per test year x scenario

Produces, under data/results/<design>/figures/, saved as .pdf:

    07_sensitivity   Does corrupting the meteorological forcing change the
                     models' skill? Per-year paired difference between real
                     forcing and donor-year forcing, one row per catchment,
                     one column per metric (NSE, KGE, ACC, CRPS). Each dot is
                     one hydro year; above zero means the corruption hurt
                     skill.

Only the four learned models appear (ridge, rf, xgb, lstm). Climatology and
persistence never read the forcing, so corrupting it cannot move them and the
comparison would be vacuous.

Usage:
    python pipeline/2_sensitivity/plot.py                    # every catchment with results
    python pipeline/2_sensitivity/plot.py zeravshan kurshim  # specific catchments
    (add --design <design> for another evaluation design; default loyo)
"""

import argparse
import os
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for seas5ml

from seas5ml import designs
from seas5ml.plots import common as pl
from seas5ml.plots import sensitivity as plot_sensitivity


def main():
    p = argparse.ArgumentParser(description="Stage 2 figure.")
    p.add_argument('catchments', nargs='*', type=str.lower)
    p.add_argument('--design', default=designs.DEFAULT_DESIGN,
                   choices=designs.DESIGNS)
    args = p.parse_args()
    pl.use_design(args.design)

    os.makedirs(pl.PLOT_DIR, exist_ok=True)
    results_dir = plot_sensitivity.default_results_dir()
    available = plot_sensitivity.find_catchments(results_dir)
    if not available:
        print(f"No stage-2 results found under {os.path.normpath(results_dir)}.")
        print("Run pipeline/2_sensitivity/run.py (via "
              "pipeline/2_sensitivity/submit.sh) first.")
        sys.exit(1)

    shown = args.catchments if args.catchments else available
    print(f"Catchments: {', '.join(shown)}")
    print(f"Models    : {', '.join(plot_sensitivity.SENS_MODELS)}")

    print("\n=== Climate-input sensitivity ===")
    plot_sensitivity.run(args.catchments or None)

    print("\nDone.")


if __name__ == '__main__':
    main()
