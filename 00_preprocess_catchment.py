#!/usr/bin/env python3
"""
00_preprocess_catchment.py – Preprocess SEAS5 data for all models.

This is the first step in the pipeline and must be run before any
exploration or training scripts.

Usage:
    python 00_preprocess_catchment.py <catchment_id>
    python 00_preprocess_catchment.py zeravshan
    python 00_preprocess_catchment.py ulkenboken

Reads catchment definitions from catchments.py.

Output: data_processed/<catchment_id>_features.parquet

Schema (one row per date × init_date × member):
    date, init_date, member, lead_day, source,
    hydro_year, init_month,
    a0,                                        ← observed log1p(Q) anomaly at init_date
    tas, tasmax, tasmin, ssrd, tp, sd, sf,   ← catchment-averaged met vars
    sin_doy, cos_doy,                         ← seasonal cycle
    Q                                         ← observed discharge (target, NaN for future dates)

Rows whose init_date has no observed discharge, or whose init_date falls in the
catchment's skip_years, are dropped before writing — every row therefore has a
finite, trustworthy a0 (see flowcast_src.climatology.compute_global_climatology
and the skip_years comment in catchments.py).

Units:
    tas, tasmax, tasmin: K (instantaneous / true daily extremes, mx2t24/mn2t24)
    sd:                  m of water equivalent (instantaneous snow depth)
    tp:                  m, daily total precipitation (day⁻¹)
    ssrd:                J m⁻², daily surface solar radiation (day⁻¹)
    sf:                  m water equivalent, daily snowfall (day⁻¹)

CDS archives tp/ssrd/sf as running totals accumulated from forecast step 0.
This script differences them along the lead-day axis (see ACCUM_VARS below)
so the feature table holds true per-day values, same as the other variables.
"""

import os
import sys
import glob
import re
import time
import numpy as np
import pandas as pd
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

from catchments import CATCHMENTS, SHARED
from flowcast_src.climatology import compute_global_climatology
from flowcast_src.data import build_analysis_frame
from flowcast_src.paths import PROCESSED_DIR

# ═══════════════════════════════════════════════════════════════════════
# VARIABLE MAPPING
# long name (as it appears as a NetCDF variable or folder) → short name
# ═══════════════════════════════════════════════════════════════════════

VAR_MAP = {
    'tas':    'tas',
    'tasmax': 'tasmax',
    'tasmin': 'tasmin',
    'ssrd':   'ssrd',
    'tp':     'tp',
    'sd':     'sd',
    'sf':     'sf',
}

# Final short names expected in output — read from catchments.py
OUTPUT_VARS = SHARED['met_variables']

# ═══════════════════════════════════════════════════════════════════════
# DAILY VALUES FROM RUNNING TOTALS
# SEAS5 archives these three as accumulations from forecast step 0: the
# stored value at lead h is the running total over [init, init + h + 1
# days]. Differencing along the lead axis recovers the per-day value.
# tas/tasmax/tasmin are instantaneous or true daily extremes and sd is an
# instantaneous state, so none of those are touched.
# ═══════════════════════════════════════════════════════════════════════

ACCUM_VARS = ('tp', 'ssrd', 'sf')
EXPECTED_STEPS = 215

_clip_stats = {'count': 0, 'max_magnitude': 0.0}


def deaccumulate(values):
    """Running total since initialisation -> per-day total."""
    daily = np.diff(values, prepend=0.0)
    negative = daily < 0
    if negative.any():
        _clip_stats['count'] += int(negative.sum())
        _clip_stats['max_magnitude'] = max(
            _clip_stats['max_magnitude'], float(-daily[negative].min())
        )
    # GRIB packing precision can make the running total dip by ~1e-9 between
    # steps; clip so no negative "rainfall" reaches the feature table.
    return np.clip(daily, 0.0, None)


# ═══════════════════════════════════════════════════════════════════════
# PARSE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <catchment_id>")
    print(f"Available: {', '.join(CATCHMENTS.keys())}")
    sys.exit(1)

CATCHMENT_ID = sys.argv[1].lower()
if CATCHMENT_ID not in CATCHMENTS:
    print(f"ERROR: Unknown catchment '{CATCHMENT_ID}'")
    print(f"Available: {', '.join(CATCHMENTS.keys())}")
    sys.exit(1)

CFG = CATCHMENTS[CATCHMENT_ID]
OUT_DIR = str(PROCESSED_DIR)

CATCH_LAT_MIN, CATCH_LAT_MAX = CFG['catch_lat']
CATCH_LON_MIN, CATCH_LON_MAX = CFG['catch_lon']
HINDCAST_DIR = CFG.get('hindcast_dir', SHARED['hindcast_dir'])
FORECAST_DIR = CFG.get('forecast_dir', SHARED['forecast_dir'])
SKIP_YEARS = CFG.get('skip_years', set())
CLIM_WINDOW     = SHARED.get('daily_clim_window', 31)
CLIM_START_YEAR = SHARED['clim_start_year']     # no .get() -- must be set


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def load_discharge():
    df = pd.read_csv(CFG['discharge_csv'])
    if df.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns (date, Q) in {CFG['discharge_csv']}, "
            f"found {df.shape[1]}: {list(df.columns)}"
        )
    df.columns = ['date', 'Q']
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.normalize()
    df = df.dropna(subset=['Q'])
    n_dupes = df['date'].duplicated().sum()
    if n_dupes:
        raise ValueError(
            f"{CFG['discharge_csv']} has {n_dupes} duplicate date(s) — "
            f"merge would silently fan out rows."
        )
    return df


def subset_catchment(ds):
    """Subset dataset to catchment bounding box, independent of axis direction."""
    lat_vals = ds.lat.values
    lon_vals = ds.lon.values

    lat_slice = (slice(CATCH_LAT_MAX, CATCH_LAT_MIN) if lat_vals[0] > lat_vals[-1]
                 else slice(CATCH_LAT_MIN, CATCH_LAT_MAX))
    lon_slice = (slice(CATCH_LON_MAX, CATCH_LON_MIN) if lon_vals[0] > lon_vals[-1]
                 else slice(CATCH_LON_MIN, CATCH_LON_MAX))

    ds_catch = ds.sel(lat=lat_slice, lon=lon_slice)

    if ds_catch.sizes.get('lat', 0) == 0 or ds_catch.sizes.get('lon', 0) == 0:
        raise ValueError(
            f"Catchment bbox lat={CATCH_LAT_MIN}-{CATCH_LAT_MAX}, "
            f"lon={CATCH_LON_MIN}-{CATCH_LON_MAX} selected an empty region. "
            f"File covers lat={lat_vals.min()}-{lat_vals.max()}, "
            f"lon={lon_vals.min()}-{lon_vals.max()}."
        )

    return ds_catch


def area_weighted_mean(da):
    """Compute latitude-weighted spatial mean."""
    weights = np.cos(np.deg2rad(da.lat))
    return da.weighted(weights).mean(dim=['lat', 'lon']).values


def check_units_match(hc_file, fc_file):
    """
    Verify hindcast and forecast files report the same NetCDF 'units'
    attribute for every variable in VAR_MAP. Raises ValueError on any
    mismatch or missing attribute — does not attempt to fix anything.
    """
    ds_hc = xr.open_dataset(hc_file)
    ds_fc = xr.open_dataset(fc_file)
    try:
        mismatches = []
        for var in VAR_MAP:
            if var not in ds_hc.data_vars or var not in ds_fc.data_vars:
                continue
            hc_units = ds_hc[var].attrs.get('units')
            fc_units = ds_fc[var].attrs.get('units')
            if hc_units != fc_units:
                mismatches.append(f"{var}: hindcast='{hc_units}' vs forecast='{fc_units}'")

        if mismatches:
            raise ValueError(
                "Unit mismatch between hindcast and forecast files:\n  "
                + "\n  ".join(mismatches)
                + f"\n  hindcast file: {hc_file}\n  forecast file: {fc_file}"
            )
    finally:
        ds_hc.close()
        ds_fc.close()


def extract_vars_from_ds(ds):
    """
    Extract all recognised variables from a dataset.
    Returns dict: short_name -> np.array of shape (time,)
    Raises KeyError if a required variable is missing.
    """
    ds_catch = subset_catchment(ds)
    result = {}

    for data_var in ds_catch.data_vars:
        if data_var in VAR_MAP:
            result[data_var] = area_weighted_mean(ds_catch[data_var])

    missing = [v for v in OUTPUT_VARS if v not in result]
    if missing:
        raise KeyError(f"Missing variables in file: {missing}")

    return result, pd.to_datetime(ds_catch.time.values)


# ═══════════════════════════════════════════════════════════════════════
# TRAJECTORY PROCESSING
# Both hindcast and forecast share the same filename convention:
#   forecast_YYYY_MM_daily_memberN.nc
# The only difference is which directory they live in, and the 'source'
# label assigned to the rows.
# ═══════════════════════════════════════════════════════════════════════

def process_trajectory(filepath, source):
    """
    Process one NetCDF file (hindcast or forecast).
    Expected filename: forecast_YYYY_MM_daily_memberN.nc
    source: 'hindcast' or 'forecast' — used to label the rows.
    """
    basename = os.path.basename(filepath)
    match = re.match(r'forecast_(\d{4})_(\d{2})_daily_member(\d+)\.nc', basename)
    if not match:
        return None

    year, month, member = int(match.group(1)), int(match.group(2)), int(match.group(3))
    init_date = pd.Timestamp(f'{year}-{month:02d}-01')

    ds = xr.open_dataset(filepath)
    try:
        vars_dict, times = extract_vars_from_ds(ds)
    finally:
        ds.close()

    if len(times) != EXPECTED_STEPS:
        raise ValueError(
            f"{basename}: expected {EXPECTED_STEPS} daily steps, got {len(times)}"
        )

    df = pd.DataFrame({
        'date':      times,
        'init_date': init_date,
        'member':    member,
        'lead_day':  np.arange(len(times)),
        'source':    source,
    })
    for short_name, values in vars_dict.items():
        if short_name in ACCUM_VARS:
            values = deaccumulate(values)
        df[short_name] = values

    return df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print(f"PREPROCESSING: {CFG['name']} ({CATCHMENT_ID})")
    print(f"  Catchment bbox: lat {CATCH_LAT_MIN}-{CATCH_LAT_MAX}, "
          f"lon {CATCH_LON_MIN}-{CATCH_LON_MAX}")
    print(f"  Output variables: {OUTPUT_VARS}")
    print("=" * 60)

    # ── 1. Discharge ─────────────────────────────────────────────────
    print("\n[1/5] Loading discharge data...")
    q_df = load_discharge()
    print(f"  Records: {len(q_df)}, "
          f"range: {q_df['date'].min().date()} to {q_df['date'].max().date()}")

    all_dfs = []

    # ── 2. Hindcast ──────────────────────────────────────────────────
    print("\n[2/5] Processing hindcast...")
    hc_files = sorted(glob.glob(os.path.join(HINDCAST_DIR, 'forecast_*.nc')))
    fc_files_check = sorted(glob.glob(os.path.join(FORECAST_DIR, 'forecast_*.nc')))

    if hc_files and fc_files_check:
        print("  Checking hindcast/forecast unit consistency...")
        check_units_match(hc_files[0], fc_files_check[0])
        print("  Units match.")

    total_hc = len(hc_files)
    print(f"  Found {total_hc:,} hindcast member files")
    errors = 0

    for i, fpath in enumerate(hc_files, 1):
        try:
            df = process_trajectory(fpath, source='hindcast')
            if df is not None:
                all_dfs.append(df)
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR: {os.path.basename(fpath)}: {e}", flush=True)

        if i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total_hc - i) / rate / 60
            print(f"  Hindcast: {i:,}/{total_hc:,} "
                  f"({i/total_hc*100:.0f}%) | "
                  f"errors: {errors} | "
                  f"elapsed: {elapsed/60:.1f}min | "
                  f"ETA: {eta:.1f}min", flush=True)

    print(f"  Hindcast complete: {len(all_dfs)} trajectories, {errors} errors")

    # ── 3. Forecast ──────────────────────────────────────────────────
    print(f"\n[3/5] Processing forecast...")
    fc_files = sorted(glob.glob(os.path.join(FORECAST_DIR, 'forecast_*.nc')))
    total_fc = len(fc_files)
    print(f"  Found {total_fc:,} forecast files")
    n_before = len(all_dfs)
    errors_fc = 0

    for i, fpath in enumerate(fc_files, 1):
        try:
            df = process_trajectory(fpath, source='forecast')
            if df is not None:
                all_dfs.append(df)
            else:
                errors_fc += 1
        except Exception as e:
            errors_fc += 1
            if errors_fc <= 5:
                print(f"  ERROR: {os.path.basename(fpath)}: {e}", flush=True)

        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Forecast: {i:,}/{total_fc:,} "
                  f"({i/total_fc*100:.0f}%) | "
                  f"errors: {errors_fc}", flush=True)

    print(f"  Forecast complete: {len(all_dfs) - n_before} trajectories, {errors_fc} errors")

    # ── 4. Concatenate & enrich ──────────────────────────────────────
    print(f"\n[4/5] Concatenating {len(all_dfs):,} trajectories...")
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"  Raw shape: {df_all.shape}")

    # Seasonal cycle encoding
    doy = df_all['date'].dt.dayofyear
    df_all['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    df_all['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)

    # Hydrological year (Oct 1 start)
    df_all['hydro_year'] = df_all['date'].dt.year
    df_all.loc[df_all['date'].dt.month >= 10, 'hydro_year'] += 1

    df_all['init_month'] = df_all['init_date'].dt.month

    # Merge observed discharge as target
    print("  Merging observed discharge (target Q)...")
    df_all = df_all.merge(q_df, on='date', how='left')

    n_with_q = df_all['Q'].notna().sum()
    n_total = len(df_all)
    print(f"  Rows with observed Q: {n_with_q:,} / {n_total:,} "
          f"({n_with_q/n_total*100:.1f}%)")

    # Antecedent discharge anomaly (a0): the observed log1p(Q) anomaly at each
    # row's init_date — the same quantity the persistence baselines carry
    # forward, now available to every model as an input feature.
    print("  Computing antecedent anomaly (a0)...")
    obs_daily_q = q_df.set_index('date')['Q'].sort_index()
    n_hit = df_all['init_date'].isin(obs_daily_q.index).sum()
    if n_hit == 0:
        raise ValueError("No init_date matched the discharge index — "
                         "check date normalisation in load_discharge().")

    doy_clim = compute_global_climatology(obs_daily_q, SKIP_YEARS, CLIM_WINDOW,
                                          start_year=CLIM_START_YEAR)
    print(f"  Global climatology: {CLIM_START_YEAR}+ minus skip_years "
          f"({len(SKIP_YEARS)} skipped), {CLIM_WINDOW}-day smoothing")

    init_unique = pd.DataFrame({'init_date': df_all['init_date'].unique()})
    init_doy = init_unique['init_date'].dt.dayofyear
    init_clim = init_doy.map(doy_clim)
    q_at_init = obs_daily_q.reindex(init_unique['init_date'])
    init_unique['a0'] = np.log1p(q_at_init.values) - init_clim.values
    df_all = df_all.merge(init_unique, on='init_date', how='left')

    # Drop rows whose initialisation is unobserved or falls in skip_years.
    # Both conditions are needed: hydro_year is derived from the VALID date,
    # not init_date, so a trajectory initialised inside a skip year can have
    # valid dates that fall outside it and would otherwise slip through the
    # downstream hydro_year-based skip_years filter while still carrying an
    # a0 read from the untrustworthy period.
    init_hy = (df_all['init_date'].dt.year
               + (df_all['init_date'].dt.month >= 10).astype(int))
    ok_init = df_all['a0'].notna() & ~init_hy.isin(SKIP_YEARS)

    n_before = len(df_all)
    df_all = df_all[ok_init].reset_index(drop=True)
    n_dropped = n_before - len(df_all)
    print(f"  Dropped {n_dropped:,} / {n_before:,} rows whose init_date is "
          f"unobserved or in skip_years ({n_dropped / n_before * 100:.1f}%)")

    # Reorder columns cleanly
    meta_cols  = ['date', 'init_date', 'member', 'lead_day', 'source',
                  'hydro_year', 'init_month', 'a0']
    feat_cols  = OUTPUT_VARS + ['sin_doy', 'cos_doy']
    target_col = ['Q']
    df_all = df_all[meta_cols + feat_cols + target_col]

    # ── 5. Save ──────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, f'{CATCHMENT_ID}_features.parquet')
    print(f"\n[5/5] Saving to {out_path}...")
    df_all.to_parquet(out_path, index=False, engine='pyarrow')

    # Fold manifest — the hydro years the later stages will actually see.
    # Derived by calling build_analysis_frame() on the file just written, so
    # the list cannot disagree with what those stages compute for themselves.
    # submit_02_hyperparamtuning_nested.sh reads this instead of carrying a
    # hardcoded per-catchment year list, which silently went stale whenever
    # skip_years or the discharge record changed.
    fold_path = os.path.join(OUT_DIR, f'{CATCHMENT_ID}_folds.txt')
    frame, fold_years = build_analysis_frame(out_path, obs_daily_q, SKIP_YEARS)
    del frame
    if not fold_years:
        raise ValueError(
            f"{CATCHMENT_ID}: build_analysis_frame() returned no hydro years — "
            f"the feature table has no rows with an observed target outside "
            f"skip_years ({sorted(SKIP_YEARS)}). Refusing to write an empty "
            f"fold manifest."
        )
    with open(fold_path, 'w') as fh:
        fh.write(' '.join(str(int(y)) for y in fold_years) + '\n')
    print(f"  Fold manifest: {fold_path}")
    print(f"    {len(fold_years)} hydro years "
          f"({fold_years[0]}-{fold_years[-1]})")

    file_size = os.path.getsize(out_path) / 1e6
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    print(f"  Output : {out_path} ({file_size:.1f} MB)")
    print(f"  Shape  : {df_all.shape}")
    print(f"  Columns: {list(df_all.columns)}")
    print(f"\n  Date range  : {df_all['date'].min().date()} to {df_all['date'].max().date()}")
    print(f"  Hydro years : {sorted(df_all['hydro_year'].unique())}")
    print(f"  Fold years  : {fold_years}")
    print(f"  Sources     : {df_all['source'].value_counts().to_dict()}")
    print(f"  Members     : hindcast {df_all[df_all.source=='hindcast']['member'].nunique()}, "
          f"forecast {df_all[df_all.source=='forecast']['member'].nunique()}")
    print(f"  Negative-increment clipping ({', '.join(ACCUM_VARS)}): "
          f"{_clip_stats['count']} negative increment(s), "
          f"max magnitude {_clip_stats['max_magnitude']:.3e}")


if __name__ == '__main__':
    main()
