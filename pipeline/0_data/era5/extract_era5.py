#!/usr/bin/env python3
"""
ERA5 daily catchment series for the SEAS5 skill evaluation (TODO.md, R2).

Reads the PIK copy of ERA5 (config.toml [paths] era5_dir: hourly, global,
0.25 degree, <variable>/<variable>_YYYYMM.nc) and, for every catchment,
averages it over the same SEAS5 cells and weights as the catchment's forcing
and reduces it to days with the same definitions as the SEAS5 table
(seas5ml.era5). Snow depth is not in the PIK copy: it is downloaded from the
CDS (download_sd.py) and added by --combine.

Usage (from the repository root):
    python pipeline/0_data/era5/extract_era5.py --year 2000     # one year, all catchments
    python pipeline/0_data/era5/extract_era5.py --array         # year = FIRST_YEAR + $SLURM_ARRAY_TASK_ID
    python pipeline/0_data/era5/extract_era5.py --timing        # Jan 2000 only, timings, writes nothing
    python pipeline/0_data/era5/extract_era5.py --combine [catchment ...]
    python pipeline/0_data/era5/extract_era5.py --n-tasks       # number of years (array size)
Normally run through pipeline/0_data/era5/submit_extract.sh (array + combine).

Output:
    data/processed_data/era5_parts/<catchment>_<year>.parquet    per year (--year / --array)
    data/processed_data/<catchment>_era5.parquet                 --combine: one row per date,
        Oct 1980 - Dec 2025: date, tas, tasmax, tasmin, ssrd, tp, sd, sf
        (sd NaN, with a warning, until the snow-depth download is in)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for seas5ml

from seas5ml import era5                                         # noqa: E402
from seas5ml.catchments import CATCHMENTS                        # noqa: E402
from seas5ml.paths import ERA5_DIR, ERA5_SD_DIR, era5_part_path, era5_path   # noqa: E402

YEARS = list(range(era5.FIRST_YEAR, era5.LAST_YEAR + 1))
START_DATE = pd.Timestamp(f"{era5.FIRST_YEAR}-10-01")           # first day kept by --combine
END_DATE = pd.Timestamp(f"{era5.LAST_YEAR}-12-31")


def month_file(folder: str, year: int, month: int) -> Path:
    return ERA5_DIR / folder / f"{folder}_{year}{month:02d}.nc"


def read_windows(path: Path, var: str, windows: dict, first_hour_only=False):
    """Hourly values of `var` in each catchment's window. Returns (times,
    {catchment: (values (time, lat, lon), lat, lon)})."""
    with xr.open_dataset(path) as ds:
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        da = ds[var]
        if first_hour_only:
            da = da.isel({tname: slice(0, 1)})
        times = da[tname].values
        out = {}
        for c, ((lat_hi, lat_lo), (lon_lo, lon_hi)) in windows.items():
            sub = da.sel(latitude=slice(lat_hi, lat_lo), longitude=slice(lon_lo, lon_hi))
            out[c] = (sub.values.astype(np.float64), sub.latitude.values, sub.longitude.values)
    return times, out


def extract(year: int, months=range(1, 13), catchments=None, timing=False):
    """Daily series of every catchment for the given months of `year`."""
    catchments = catchments or list(CATCHMENTS)
    windows = {c: era5.era5_window(CATCHMENTS[c]) for c in catchments}
    months = list(months)
    nxt = (year + 1, 1) if months[-1] == 12 else (year, months[-1] + 1)
    series = {c: {} for c in catchments}
    weights, coords, dates = {}, {}, None

    for folder in dict.fromkeys(v[0] for v in era5.VARIABLES.values()):
        var = next(v[1] for v in era5.VARIABLES.values() if v[0] == folder)
        files = [(month_file(folder, year, m), False) for m in months] + [(month_file(folder, *nxt), True)]
        times, hourly = [], {c: [] for c in catchments}
        for path, first_only in files:
            t0 = time.time()
            t, data = read_windows(path, var, windows, first_hour_only=first_only)
            if first_only and pd.Timestamp(t[0]) != pd.Timestamp(f"{nxt[0]}-{nxt[1]:02d}-01"):
                raise ValueError(f"{path.name}: first step {t[0]} is not 00 UTC of the 1st")
            times.append(t)
            for c in catchments:
                vals, lat, lon = data[c]
                if c not in coords:
                    coords[c] = (lat, lon)
                elif not (np.array_equal(coords[c][0], lat) and np.array_equal(coords[c][1], lon)):
                    raise ValueError(f"{path.name}: {c} window grid differs from the first file's")
                hourly[c].append(vals)
            if timing:
                print(f"  {path.name}{' (first hour)' if first_only else ''}: {time.time() - t0:.1f} s", flush=True)
        times = np.concatenate(times)
        for c in catchments:
            if c not in weights:
                weights[c] = era5.era5_weights(CATCHMENTS[c], *coords[c])
            arr = np.concatenate(hourly[c])
            for out_var, (f, _, rule) in era5.VARIABLES.items():
                if f != folder:
                    continue
                d, fields = era5.daily_fields(arr, times, rule)
                if dates is None:
                    dates = d
                elif not d.equals(dates):
                    raise ValueError(f"{folder}: dates differ from the first variable's")
                series[c][out_var] = era5.weighted_series(fields, weights[c])

    frames = {c: pd.DataFrame({"date": dates, **series[c]}) for c in catchments}
    return frames, weights, coords


def run_year(year: int):
    t0 = time.time()
    frames, _, _ = extract(year)
    for c, df in frames.items():
        out = era5_part_path(c, year)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    print(f"{year}: {len(frames)} catchments, {len(next(iter(frames.values())))} days, "
          f"{time.time() - t0:.0f} s", flush=True)


def run_timing():
    t0 = time.time()
    frames, weights, coords = extract(2000, months=[1], timing=True)
    print(f"\nJan 2000, all {len(frames)} catchments: {time.time() - t0:.1f} s "
          f"(x 12 months x {len(YEARS)} years for the whole job)")
    for c, df in frames.items():
        lat, lon = coords[c]
        n_used = int((weights[c] > 0).sum())
        print(f"  {c:11s} window {len(lat)} x {len(lon)} ERA5 points, {n_used} weighted; "
              f"Jan 2000 mean tas {df.tas.mean() - 273.15:6.1f} C, tasmax {df.tasmax.mean() - 273.15:6.1f}, "
              f"tasmin {df.tasmin.mean() - 273.15:6.1f}, tp {df.tp.sum() * 1000:6.1f} mm, "
              f"sf {df.sf.sum() * 1000:6.1f} mm w.e., ssrd {df.ssrd.mean() / 1e6:5.1f} MJ m-2 d-1")


def add_snow_depth(c: str, df: pd.DataFrame) -> pd.DataFrame:
    """Snow depth from the CDS download: day D = 00 UTC of D+1 (end of day D)."""
    region = next(r for r, cs in era5.SD_REGIONS.items() if c in cs)
    files = sorted((ERA5_SD_DIR / region).glob(f"snow_depth_{region}_*.nc"))
    if not files:
        print(f"  WARNING {c}: no snow-depth files in {ERA5_SD_DIR / region}; sd left NaN")
        df["sd"] = np.nan
        return df
    (lat_hi, lat_lo), (lon_lo, lon_hi) = era5.era5_window(CATCHMENTS[c])
    vals, times, grid = [], [], None
    for f in files:
        with xr.open_dataset(f) as ds:
            tname = "valid_time" if "valid_time" in ds.dims else "time"
            sub = ds[era5.SD_VARIABLE].sel(latitude=slice(lat_hi, lat_lo), longitude=slice(lon_lo, lon_hi))
            g = (sub.latitude.values, sub.longitude.values)
            if grid is None:
                grid = g
            elif not (np.array_equal(grid[0], g[0]) and np.array_equal(grid[1], g[1])):
                raise ValueError(f"{f.name}: grid differs from the first snow-depth file's")
            vals.append(sub.values.astype(np.float64))
            times.append(sub[tname].values)
    times = pd.DatetimeIndex(np.concatenate(times))
    if (times.hour != 0).any() or times.duplicated().any():
        raise ValueError(f"{c}: snow-depth times must be 00 UTC, one per day")
    W = era5.era5_weights(CATCHMENTS[c], *grid)
    sd = pd.Series(era5.weighted_series(np.concatenate(vals), W), index=times - pd.Timedelta(days=1))
    df["sd"] = df["date"].map(sd)
    n_missing = int(df["sd"].isna().sum())
    if n_missing:
        print(f"  WARNING {c}: snow depth missing on {n_missing} days")
    return df


def run_combine(catchments):
    for c in catchments or list(CATCHMENTS):
        parts = [era5_part_path(c, y) for y in YEARS]
        missing = [p.name for p in parts if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{c}: missing yearly parts {missing}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)
        expected = pd.date_range(START_DATE, END_DATE, freq="D")
        if not pd.DatetimeIndex(df["date"]).equals(expected):
            raise ValueError(f"{c}: dates are not one per day {START_DATE.date()} .. {END_DATE.date()}")
        df = add_snow_depth(c, df)
        df = df[["date"] + era5.OUTPUT_COLUMNS]
        df.to_parquet(era5_path(c), index=False)
        print(f"{c}: {len(df)} days -> {era5_path(c)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--year", type=int)
    g.add_argument("--array", action="store_true", help="year = FIRST_YEAR + $SLURM_ARRAY_TASK_ID")
    g.add_argument("--timing", action="store_true")
    g.add_argument("--combine", nargs="*", metavar="CATCHMENT")
    g.add_argument("--n-tasks", action="store_true")
    a = ap.parse_args()
    if a.n_tasks:
        print(len(YEARS))
    elif a.timing:
        run_timing()
    elif a.combine is not None:
        run_combine(a.combine)
    else:
        year = a.year if a.year is not None else YEARS[int(os.environ["SLURM_ARRAY_TASK_ID"])]
        if year not in YEARS:
            sys.exit(f"year {year} outside {YEARS[0]}-{YEARS[-1]}")
        run_year(year)


if __name__ == "__main__":
    main()
