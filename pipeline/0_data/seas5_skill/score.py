#!/usr/bin/env python3
"""
SEAS5 meteorological forecast skill against ERA5, one catchment (TODO.md, R2).

Scores every forcing variable of the stage-0 feature table (the models' exact
input) against the catchment's ERA5 series at daily, weekly and monthly
resolution, on the streamflow scoring's lead-month grid, folds and designs
(seas5ml.metskill). Answers reviewer comment #2: at which leads, for which
variables, does SEAS5 beat climatology?

Usage:
    python pipeline/0_data/seas5_skill/score.py <catchment> [--design loyo|split]
        [--variables tas tp ...]

Input (seas5ml.paths):
    data/processed_data/<catchment>_features.parquet     stage 0 (SEAS5)
    data/processed_data/<catchment>_forcing_cells.csv    stage 0 weights
    data/processed_data/<catchment>_folds_loyo.txt       fold years
    data/processed_data/<catchment>_era5.parquet         extract_era5.py

Checks before scoring (stops otherwise): the ERA5 weights' SEAS5 cells and
weights (seas5ml.era5.seas5_weights) equal stage 0's forcing cells, so both
sides average the same area. A variable ERA5 has no value for (sd before the
snow-depth download) is skipped with a warning.

Output: data/results/<design>/seas5_skill/<catchment>/
    yearly.csv    per (variable, resolution, hydro year, lead month): ACC,
                  CRPS, CRPS of the climatological ensemble, CRPSS
    cells.csv     per (variable, resolution, start month, lead month): ACC
    bias.csv      per (variable, start month, lead month): raw SEAS5 - ERA5
    periods.parquet  every scored period (significance at figure time)
    stats.csv     periods kept / dropped per (variable, resolution)
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import numpy as np
import pandas as pd

from seas5ml import designs, era5, metskill, paths
from seas5ml.catchments import CATCHMENTS
from seas5ml.data import load_features


def parse_args():
    p = argparse.ArgumentParser(description="SEAS5 skill against ERA5 (R2).")
    p.add_argument("catchment", type=str.lower, choices=sorted(CATCHMENTS))
    p.add_argument("--design", default=designs.DEFAULT_DESIGN,
                   choices=designs.DESIGNS)
    p.add_argument("--variables", nargs="+", default=list(metskill.VARIABLES),
                   choices=metskill.VARIABLES)
    return p.parse_args()


def check_weights(catchment):
    """The ERA5 matching's SEAS5 cells and weights = stage 0's."""
    st0 = pd.read_csv(paths.forcing_cells_path(catchment))
    lat, lon, w = era5.seas5_weights(CATCHMENTS[catchment])
    mine = pd.DataFrame({"lat": np.repeat(lat, len(lon)),
                         "lon": np.tile(lon, len(lat)), "weight": w.ravel()})
    ok = (len(st0) == len(mine)
          and np.allclose(st0[["lat", "lon"]].to_numpy(), mine[["lat", "lon"]].to_numpy())
          and np.allclose(st0["weight"].to_numpy(), mine["weight"].to_numpy(), atol=1e-12))
    if not ok:
        raise ValueError(f"{catchment}: the ERA5 weights do not match stage 0's "
                         f"{paths.forcing_cells_path(catchment)} (rerun stage 0 "
                         f"or the ERA5 extraction)")
    print(f"  Weights = stage 0's ({int((st0.weight > 0).sum())} weighted cells)")


def fold_years(catchment):
    """The catchment's fold years: the loyo manifest of stage 0 (one fold per
    hydro year), i.e. the hydro years the streamflow models are scored on."""
    path = paths.fold_manifest_path("loyo", catchment)
    return [int(y) for y in path.read_text().split()]


def main():
    args = parse_args()
    c = args.catchment
    t0 = time.time()
    print("=" * 70)
    print(f"SEAS5 SKILL vs ERA5: {CATCHMENTS[c]['name']} ({c}) -- design {args.design}")
    print("=" * 70)
    check_weights(c)
    years = fold_years(c)
    print(f"  Fold years: {len(years)} ({years[0]}-{years[-1]})")
    cols = ["init_date", "date", "member", "lead_day", "init_month",
            "hydro_year"] + args.variables
    members = load_features(str(paths.features_path(c)))[cols]
    e5 = pd.read_parquet(paths.era5_path(c)).set_index("date")
    print(f"  SEAS5: {len(members):,} member rows, "
          f"{members.init_date.nunique()} starts | ERA5: {len(e5):,} days "
          f"({e5.index.min().date()} - {e5.index.max().date()})", flush=True)

    out = {k: [] for k in ("yearly", "cells", "bias", "periods", "stats")}
    for var in args.variables:
        tv = time.time()
        r = metskill.score_variable(members, e5[var], var, design=args.design,
                                    years=years)
        if r is None:
            print(f"  {var}: WARNING no ERA5 values -- skipped")
            continue
        for k in out:
            out[k].append(r[k])
        y = r["yearly"]
        med = y[y.resolution == "monthly"].groupby("lead_month")[["acc", "crpss"]].median()
        print(f"  {var:>6s} ({time.time() - tv:.0f}s): monthly median ACC by lead month "
              + " ".join(f"{v:+.2f}" for v in med["acc"])
              + " | CRPSS " + " ".join(f"{v:+.2f}" for v in med["crpss"]), flush=True)
    if not out["yearly"]:
        sys.exit("nothing scored")

    out_dir = paths.metskill_dir(args.design, c)
    os.makedirs(out_dir, exist_ok=True)
    for k in ("yearly", "cells", "bias", "stats"):
        pd.concat(out[k], ignore_index=True).to_csv(out_dir / f"{k}.csv", index=False)
    pd.concat(out["periods"], ignore_index=True).to_parquet(out_dir / "periods.parquet",
                                                            index=False)
    print(f"\n  Saved {out_dir}/  DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
