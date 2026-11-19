#!/usr/bin/env python3
"""
Stage 1 plots, per variant.

default  (data/results/<design>/default/<catchment>/<model>/)
    01a_boxplotcomparison_short   NSE, KGE, ACC, CRPSS per test year, lead
                                  days 0-30; one row per catchment
    01b_boxplotcomparison_long    same, lead days 185-214
    02_leadtimeheatmap            ACC by lead time x init month, one panel
                                  per (catchment, model)

tuned    (data/results/<design>/tuned/<catchment>/<model>/fold_<id>/,
          plus the default run, for the untuned and reference boxes)
    03_configranking              Is any candidate config a consistent
                                  winner? Mean rank per config across folds,
                                  one table per tuned model, one row per
                                  catchment. Written to 03_configranking.csv
                                  as well.
    04a_tuningvsreference_short   Does tuning close the gap to simple
                                  references, at lead days 0-30? Two
                                  reference boxes (climatology, damped
                                  persistence) beside a default/tuned pair per
                                  tuned model.
    04b_tuningvsreference_long    same, lead days 185-214

Figures go to data/results/<design>/figures/, as .pdf.

Usage:
    python pipeline/1_models/plot.py default                       # every catchment with results
    python pipeline/1_models/plot.py default zeravshan kurshim     # specific catchments
    python pipeline/1_models/plot.py default --by-regime           # heatmap grid split into
                                                                   # one figure per regime
    python pipeline/1_models/plot.py tuned [catchments ...]
    (add --design <design> for another evaluation design; default loyo)
"""

import argparse
import os
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for seas5ml

from seas5ml import designs
from seas5ml.plots import common as pl
from seas5ml.plots import boxplots, configrank, heatmap, tuningvsref


def plot_default(catchments, by_regime):
    catchments = pl.parse_catchment_args(catchments)
    if not catchments:
        sys.exit(1)
    os.makedirs(pl.PLOT_DIR, exist_ok=True)

    print("\n=== Boxplot comparison (short lead, 0-30 d) ===")
    boxplots.run(catchments, 0, 30,
                 f"{pl.design_label()} Aggregate Skill\n"
                 f"(Short) Lead time: 0-30 days",
                 "01a_boxplotcomparison_short")

    print("\n=== Boxplot comparison (long lead, 185-214 d) ===")
    boxplots.run(catchments, 185, 214,
                 f"{pl.design_label()} Aggregate Skill\n"
                 f"(Long) Lead time: 185-214 days",
                 "01b_boxplotcomparison_long")

    print("\n=== Lead-time x init-month heatmap grid ===")
    heatmap.run(catchments, by_regime=by_regime)


def plot_tuned(catchments):
    if not catchments:
        found = set()
        for model in pl.HPTUNE_MODELS:
            found.update(pl.find_hptune_catchments(model))
        catchments = pl.order_catchments(sorted(found))
    if not catchments:
        print(f"No tuned results found under "
              f"{pl.paths.kind_dir(pl.DESIGN, 'tuned')}.")
        sys.exit(1)
    os.makedirs(pl.PLOT_DIR, exist_ok=True)
    print(f"Catchments: {', '.join(catchments)}")
    print(f"Models    : {', '.join(pl.HPTUNE_MODELS)}")

    print("\n=== Config ranking ===")
    configrank.run(catchments)

    print("\n=== Tuning vs. reference (short lead, 0-30 d) ===")
    tuningvsref.run(catchments, 0, 30,
                    f"{pl.design_label()} Best parameter config vs. default\n"
                    "(Short) Lead time: 0-30 Days",
                    "04a_tuningvsreference_short")

    print("\n=== Tuning vs. reference (long lead, 185-214 d) ===")
    tuningvsref.run(catchments, 185, 214,
                    f"{pl.design_label()} Best parameter config vs. default\n"
                    "(Long) Lead time: 185-214 Days",
                    "04b_tuningvsreference_long")


def main():
    p = argparse.ArgumentParser(description="Stage 1 figures.")
    p.add_argument('variant', choices=('default', 'tuned'))
    p.add_argument('catchments', nargs='*', type=str.lower)
    p.add_argument('--by-regime', action='store_true',
                   help="default: one heatmap figure per regime group")
    p.add_argument('--design', default=designs.DEFAULT_DESIGN,
                   choices=designs.DESIGNS)
    args = p.parse_args()
    pl.use_design(args.design)

    if args.variant == 'default':
        plot_default(args.catchments, args.by_regime)
    else:
        plot_tuned(args.catchments)
    print("\nDone.")


if __name__ == '__main__':
    main()
