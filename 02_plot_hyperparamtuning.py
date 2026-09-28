#!/usr/bin/env python3
"""
02_plot_hyperparamtuning.py – Stage-02 hyperparameter search plots.

Reads the nested-CV results directly from each search's per-outer-fold
output — no intermediate comparison CSV is required:

    model_results/02_hyperparamtuning/results_<catchment>_<model>_nested_seed<SEED>/
        outer_<HY>/
            inner_cv_acc.csv          every config's inner-CV score
            winner.csv                the config selected for that fold
            outer_test_metrics.csv    winner + baseline outer-test scores
            predictions_daily.parquet winner's outer-test predictions

Produces, under model_results/00_plots/, saved as .pdf:

    03_configranking      Is any candidate config a consistent winner? Mean
                          rank per config across outer folds, one table per
                          tuned model, one row per catchment (no pooling
                          across catchments — each is tuned independently).
                          Written to 03_configranking.csv as well.
    04a_tuningvsreference_short  Does tuning close the gap to simple
                          references, at short lead (0-30 d)? Full fold-level
                          distributions (median + IQR): two reference boxes
                          (climatology, damped persistence) beside a paired
                          untuned/tuned box for each tuned model, one row per
                          catchment, one column per metric (NSE, KGE, ACC).
    04b_tuningvsreference_long   Same, at long lead (185-214 d).

Usage:
    python 02_plot_hyperparamtuning.py                    # every catchment with results
    python 02_plot_hyperparamtuning.py zeravshan kurshim  # specific catchments
    SEED=207 python 02_plot_hyperparamtuning.py           # a non-default search seed
"""

import os
import sys

from flowcast_src import plot_lib as pl
from flowcast_src import plot_configrank
from flowcast_src import plot_tuningvsref


def available_catchments():
    """Every catchment with stage-02 results for at least one model."""
    found = set()
    for model in pl.HPTUNE_MODELS:
        found.update(pl.find_hptune_catchments(model))
    return pl.order_catchments(sorted(found))


def main():
    args = [a.lower() for a in sys.argv[1:]]

    catchments = args if args else available_catchments()
    if not catchments:
        print("No stage-02 results found under "
              f"{pl.HPTUNE_DIR} (seed={pl.HPTUNE_SEED}).")
        sys.exit(1)

    os.makedirs(pl.PLOT_DIR, exist_ok=True)
    print(f"Catchments: {', '.join(catchments)}")
    print(f"Models    : {', '.join(pl.HPTUNE_MODELS)}  (seed={pl.HPTUNE_SEED})")

    print("\n=== Config ranking ===")
    plot_configrank.run(catchments)

    print("\n=== Tuning vs. reference (short lead, 0-30 d) ===")
    plot_tuningvsref.run(catchments, 0, 30,
                         "Best parameter config vs. default\n"
                         "(Short) Lead time: 0-30 Days",
                         "04a_tuningvsreference_short")

    print("\n=== Tuning vs. reference (long lead, 185-214 d) ===")
    plot_tuningvsref.run(catchments, 185, 214,
                         "Best parameter config vs. default\n"
                         "(Long) Lead time: 185-214 Days",
                         "04b_tuningvsreference_long")

    print("\nDone.")


if __name__ == '__main__':
    main()
