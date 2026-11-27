#!/usr/bin/env python3
"""
Stage 0 – Preprocess SEAS5 data for all models.

This is the first step in the pipeline and must be run before any training
script.

Usage:
    python pipeline/0_data/preprocess.py <catchment_id>
    python pipeline/0_data/preprocess.py zeravshan

Reads catchment definitions from seas5ml/catchments.py.

Output:
    data/processed_data/<catchment_id>_features.parquet
    data/processed_data/<catchment_id>_folds_<design>.txt   (fold manifests,
        one per evaluation design in seas5ml.designs.DESIGNS: loyo, split;
        written only if the folds pass designs.check_sources)

Schema (one row per date × init_date × member):
    date, init_date, member, lead_day, source,
    hydro_year, init_month,
    logq0,                                     ← observed log1p(Q) at init_date
    tas, tasmax, tasmin, ssrd, tp, sd, sf,   ← catchment-mean met vars (see Forcing cells)
    sin_doy, cos_doy,                         ← seasonal cycle
    Q                                         ← observed discharge (target, NaN for future dates)

The antecedent anomaly a0 is NOT stored: it depends on the climatology, which
is fit per cross-validation fold on that fold's training years. Each stage
computes it at runtime as logq0 minus the fold's climatology on the init date
(seas5ml.features.set_a0). This script therefore needs no climatology.

Rows whose init_date has no observed discharge, or whose init_date falls in the
catchment's skip_years, are dropped before writing — every row therefore has a
finite, trustworthy logq0 (see the skip_years comment in
seas5ml/catchments.py).

Units:
    tas, tasmax, tasmin: K (instantaneous / true daily extremes, mx2t24/mn2t24)
    sd:                  m of water equivalent (instantaneous snow depth)
    tp:                  m, daily total precipitation (day⁻¹)
    ssrd:                J m⁻², daily surface solar radiation (day⁻¹)
    sf:                  m water equivalent, daily snowfall (day⁻¹)

CDS archives tp/ssrd/sf as running totals accumulated from forecast step 0.
This script differences them along the lead-day axis (see ACCUM_VARS below)
so the feature table holds true per-day values, same as the other variables.

Forcing cells:
    Each field is reduced to one series per catchment as the mean over the
    SEAS5 cells of the catchment's box (catch_lat / catch_lon), each cell
    weighted by its area of overlap with the catchment polygon
    (seas5ml.spatial.cell_weights; cells outside the catchment get weight 0).
    The cells and weights are written to
    data/processed_data/<catchment>_forcing_cells.csv.

Grid guard:
    Hindcast and operational files must be on the same grid, and every file
    must give exactly the box cells the weights were computed on; any
    mismatch stops the run (GridMismatchError). Both eras must be present.
    (The Central Asia operational files were once on a grid offset by half a
    degree, which silently widened their boxes; see TODO.md, SEAS5 FORCING.)

Dates:
    `date` is the calendar day (normalised; the NetCDF daily means are
    stamped 12:00 / 09:00, which once kept the discharge merge below from
    matching any row, so Q was all NaN; the later stages re-attach Q on load
    and never noticed). Every file's days must be init_date + lead day
    0 ... 214; otherwise the run stops (DateMismatchError).
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

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for seas5ml

from seas5ml import designs
from seas5ml.catchments import CATCHMENTS, SHARED
from seas5ml.data import build_analysis_frame, hydro_year_of
from seas5ml.paths import PROCESSED_DIR, features_path, fold_manifest_path, forcing_cells_path
from seas5ml.spatial import cell_weights, load_polygon

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

# Final short names expected in output — read from seas5ml/catchments.py
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
POLYGON_PATH = CFG['polygon']


class GridMismatchError(ValueError):
    """A SEAS5 file is not on the grid the forcing weights were computed on.
    Never skipped like an unreadable file: it stops the run."""


class DateMismatchError(ValueError):
    """A SEAS5 file's days are not init_date + lead day (0, 1, ... 214).
    Never skipped like an unreadable file: it stops the run."""


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


def check_grids_match(hc_file, fc_file):
    """
    Hindcast and operational files must share the whole grid (same lat / lon
    values). Raises GridMismatchError otherwise: a grid offset by half a
    degree makes the same box select other cells, so the two eras would be
    averaged over different areas.
    """
    with xr.open_dataset(hc_file) as hc, xr.open_dataset(fc_file) as fc:
        same = (np.array_equal(hc.lat.values, fc.lat.values)
                and np.array_equal(hc.lon.values, fc.lon.values))
        if not same:
            def desc(ds):
                return (f"{ds.sizes['lat']} x {ds.sizes['lon']}, lat "
                        f"{ds.lat.values[0]}..{ds.lat.values[-1]}, lon "
                        f"{ds.lon.values[0]}..{ds.lon.values[-1]}")
            raise GridMismatchError(
                f"hindcast and operational grids differ:\n"
                f"  hindcast    {desc(hc)}  ({hc_file})\n"
                f"  operational {desc(fc)}  ({fc_file})")


def forcing_weights(ref_file):
    """
    Weights of the box cells on the grid of `ref_file`: overlap of each cell
    with the catchment polygon, normalised (seas5ml.spatial.cell_weights).
    Returns the weights as a (lat, lon) DataArray on the box coordinates and
    a table of the cells (lat, lon, overlap_km2, weight).
    """
    with xr.open_dataset(ref_file) as ds:
        box = subset_catchment(ds)
        lat, lon = box.lat.values, box.lon.values
    polygon = load_polygon(POLYGON_PATH)
    w, overlap, covered = cell_weights(lat, lon, polygon)
    weights = xr.DataArray(w, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
    cells = pd.DataFrame({
        'lat': np.repeat(lat, len(lon)),
        'lon': np.tile(lon, len(lat)),
        'overlap_km2': overlap.ravel(),
        'weight': w.ravel(),
    })
    return weights, cells, covered


def weighted_mean(da, weights):
    """Mean over the box cells, each cell weighted by its overlap with the
    catchment polygon (weights from forcing_weights)."""
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


def extract_vars_from_ds(ds, weights):
    """
    Extract all recognised variables from a dataset as polygon-weighted
    catchment means.
    Returns dict: short_name -> np.array of shape (time,)
    Raises KeyError if a required variable is missing, GridMismatchError if
    the box cells are not the ones the weights were computed on.
    """
    ds_catch = subset_catchment(ds)
    if not (np.array_equal(ds_catch.lat.values, weights.lat.values)
            and np.array_equal(ds_catch.lon.values, weights.lon.values)):
        raise GridMismatchError(
            f"box cells lat {ds_catch.lat.values.tolist()} lon "
            f"{ds_catch.lon.values.tolist()} differ from the weighted cells lat "
            f"{weights.lat.values.tolist()} lon {weights.lon.values.tolist()}")
    result = {}

    for data_var in ds_catch.data_vars:
        if data_var in VAR_MAP:
            result[data_var] = weighted_mean(ds_catch[data_var], weights)

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

def process_trajectory(filepath, source, weights):
    """
    Process one NetCDF file (hindcast or forecast).
    Expected filename: forecast_YYYY_MM_daily_memberN.nc
    source: 'hindcast' or 'forecast' — used to label the rows.
    weights: forcing-cell weights (forcing_weights).
    """
    basename = os.path.basename(filepath)
    match = re.match(r'forecast_(\d{4})_(\d{2})_daily_member(\d+)\.nc', basename)
    if not match:
        return None

    year, month, member = int(match.group(1)), int(match.group(2)), int(match.group(3))
    init_date = pd.Timestamp(f'{year}-{month:02d}-01')

    ds = xr.open_dataset(filepath)
    try:
        vars_dict, times = extract_vars_from_ds(ds, weights)
    finally:
        ds.close()

    if len(times) != EXPECTED_STEPS:
        raise ValueError(
            f"{basename}: expected {EXPECTED_STEPS} daily steps, got {len(times)}"
        )

    # The CDO daymean time stamps carry a time of day (12:00 on lead day 0,
    # 09:00 after it): normalise to the calendar day, so `date` joins the
    # discharge (and ERA5) on the day. Each step must then be its lead day.
    days = pd.DatetimeIndex(times).normalize()
    expected = init_date + pd.to_timedelta(np.arange(len(days)), unit='D')
    if not days.equals(expected):
        bad = np.flatnonzero(days != expected)
        raise DateMismatchError(
            f"{basename}: day of lead day {bad[0]} is {days[bad[0]].date()}, "
            f"expected {expected[bad[0]].date()} ({len(bad)} steps off)")

    df = pd.DataFrame({
        'date':      days,
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

    # Both eras are required (the split design trains on one and tests on the
    # other); a missing folder would otherwise give a one-era table silently.
    for label, files, folder in (('hindcast', hc_files, HINDCAST_DIR),
                                 ('operational', fc_files_check, FORECAST_DIR)):
        if not files:
            raise FileNotFoundError(f"no {label} files (forecast_*.nc) in {folder}")

    print("  Checking hindcast/forecast unit consistency...")
    check_units_match(hc_files[0], fc_files_check[0])
    print("  Units match.")
    print("  Checking hindcast/forecast grid consistency...")
    check_grids_match(hc_files[0], fc_files_check[0])
    print("  Grids match.")

    # Forcing cells: overlap of each box cell with the catchment polygon,
    # computed once on the (shared) grid; every file is checked against it.
    weights, cells, covered = forcing_weights(hc_files[0])
    used = cells[cells['weight'] > 0]
    print(f"  Forcing cells: {len(used)} of {len(cells)} box cells overlap the "
          f"catchment polygon ({covered:.2%} of its area inside the box; "
          f"{POLYGON_PATH})")
    for row in used.sort_values('weight', ascending=False).itertuples():
        print(f"    lat {row.lat:6.2f}  lon {row.lon:7.2f}  weight {row.weight:.4f}  "
              f"({row.overlap_km2:,.0f} km²)")

    total_hc = len(hc_files)
    print(f"  Found {total_hc:,} hindcast member files")
    errors = 0

    for i, fpath in enumerate(hc_files, 1):
        try:
            df = process_trajectory(fpath, source='hindcast', weights=weights)
            if df is not None:
                all_dfs.append(df)
            else:
                errors += 1
        except (GridMismatchError, DateMismatchError) as e:
            raise type(e)(f"{os.path.basename(fpath)}: {e}") from e
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
            df = process_trajectory(fpath, source='forecast', weights=weights)
            if df is not None:
                all_dfs.append(df)
            else:
                errors_fc += 1
        except (GridMismatchError, DateMismatchError) as e:
            raise type(e)(f"{os.path.basename(fpath)}: {e}") from e
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

    # Hydrological year (Oct 1 start) of the verification date
    df_all['hydro_year'] = hydro_year_of(df_all['date'])

    df_all['init_month'] = df_all['init_date'].dt.month

    # Merge observed discharge as target
    print("  Merging observed discharge (target Q)...")
    df_all = df_all.merge(q_df, on='date', how='left')

    n_with_q = df_all['Q'].notna().sum()
    n_total = len(df_all)
    print(f"  Rows with observed Q: {n_with_q:,} / {n_total:,} "
          f"({n_with_q/n_total*100:.1f}%)")

    # Observed log1p(Q) at each row's init_date (logq0). The antecedent
    # anomaly a0 = logq0 - climatology(init doy) is formed at runtime against
    # the fold's own climatology (seas5ml.features.set_a0), so no climatology
    # is fit here.
    print("  Reading observed discharge at initialisation (logq0)...")
    obs_daily_q = q_df.set_index('date')['Q'].sort_index()
    n_hit = df_all['init_date'].isin(obs_daily_q.index).sum()
    if n_hit == 0:
        raise ValueError("No init_date matched the discharge index — "
                         "check date normalisation in load_discharge().")

    init_unique = pd.DataFrame({'init_date': df_all['init_date'].unique()})
    q_at_init = obs_daily_q.reindex(init_unique['init_date'])
    init_unique['logq0'] = np.log1p(q_at_init.values)
    df_all = df_all.merge(init_unique, on='init_date', how='left')

    # Drop rows whose initialisation is unobserved or falls in skip_years.
    # Both conditions are needed: hydro_year is derived from the VALID date,
    # not init_date, so a trajectory initialised inside a skip year can have
    # valid dates that fall outside it and would otherwise slip through the
    # downstream hydro_year-based skip_years filter while still carrying a
    # logq0 read from the untrustworthy period.
    init_hy = pd.Series(hydro_year_of(df_all['init_date']), index=df_all.index)
    ok_init = df_all['logq0'].notna() & ~init_hy.isin(SKIP_YEARS)

    n_before = len(df_all)
    df_all = df_all[ok_init].reset_index(drop=True)
    n_dropped = n_before - len(df_all)
    print(f"  Dropped {n_dropped:,} / {n_before:,} rows whose init_date is "
          f"unobserved or in skip_years ({n_dropped / n_before * 100:.1f}%)")

    # Reorder columns cleanly
    meta_cols  = ['date', 'init_date', 'member', 'lead_day', 'source',
                  'hydro_year', 'init_month', 'logq0']
    feat_cols  = OUTPUT_VARS + ['sin_doy', 'cos_doy']
    target_col = ['Q']
    df_all = df_all[meta_cols + feat_cols + target_col]

    # ── 5. Save ──────────────────────────────────────────────────────
    out_path = str(features_path(CATCHMENT_ID))
    print(f"\n[5/5] Saving to {out_path}...")
    df_all.to_parquet(out_path, index=False, engine='pyarrow')
    cells_path = forcing_cells_path(CATCHMENT_ID)
    cells.to_csv(cells_path, index=False)
    print(f"  Forcing cells: {cells_path}")

    # Fold manifests, one per evaluation design: the ids of the folds the
    # later stages will build (seas5ml.designs.folds), from the hydro years
    # they will actually see. Derived by calling build_analysis_frame() on
    # the file just written, so the list cannot disagree with what those
    # stages compute for themselves. The tuned submit script reads these
    # (one Slurm job per fold) instead of a hardcoded per-catchment list,
    # which silently went stale whenever skip_years or the discharge record
    # changed.
    #
    # Every design's folds are built and checked (designs.check_sources:
    # under split, training years hindcast-only and test years
    # operational-only) before any manifest is written. On a failed check
    # the catchment's manifests are removed: they described the feature
    # table this run has just replaced, and a stale split manifest would let
    # the tuned submit script start jobs on a split that does not hold.
    frame, fold_years = build_analysis_frame(out_path, obs_daily_q, SKIP_YEARS)
    if not fold_years:
        raise ValueError(
            f"{CATCHMENT_ID}: build_analysis_frame() returned no hydro years — "
            f"the feature table has no rows with an observed target outside "
            f"skip_years ({sorted(SKIP_YEARS)}). Refusing to write an empty "
            f"fold manifest."
        )
    print(f"  {len(fold_years)} hydro years ({fold_years[0]}-{fold_years[-1]})")
    design_folds = {d: designs.folds(d, fold_years) for d in designs.DESIGNS}
    try:
        for design, folds in design_folds.items():
            designs.check_sources(design, folds, frame)
    except ValueError:
        for design in designs.DESIGNS:
            fold_manifest_path(design, CATCHMENT_ID).unlink(missing_ok=True)
        print(f"  ERROR: fold check failed -- removed {CATCHMENT_ID}'s fold "
              f"manifests")
        raise
    del frame
    for design, folds in design_folds.items():
        fold_ids = [f.id for f in folds]
        fold_path = fold_manifest_path(design, CATCHMENT_ID)
        with open(fold_path, 'w') as fh:
            fh.write(' '.join(fold_ids) + '\n')
        print(f"  Fold manifest ({design}, {len(fold_ids)} folds): {fold_path}")
        if not fold_ids:
            print(f"  WARNING: design '{design}' has no fold for "
                  f"{CATCHMENT_ID} (empty manifest; its stage-1 tuned jobs "
                  f"will be skipped)")

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
