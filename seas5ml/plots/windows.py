"""
seas5ml.plots.windows – LEGACY plot-time scoring on fixed lead-day windows.

Per-year NSE / KGE / ACC / CRPS / CRPSS recomputed from the prediction files
over a lead-day window (e.g. 0-30 or 185-214 days), and the day-of-year
reference ACC on the same rows. This is how the figures scored before the
verification layer (seas5ml.scores, the score tables); the stage-1 default
figures no longer use it (plots.boxplots, plots.leadcurves and plots.heatmap
read the score tables).

Still used by, until they move to the score tables:
  - plots.tuningvsref (tuned stage)
  - plots.sensitivity (stage 2; window_crps)
  - data/temp-checks/check_agg.py (doy_reference_acc, as an independent
    reference for the doy_reference scores)
Moved here unchanged from plots.boxplots (2026-11-26).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from . import common as pl
from ..metrics import calc_nse, calc_kge, calc_acc, crps_from_pred_df

# Baselines are identical across members, so their CRPS is exactly MAE and is
# read from the ensemble-mean file. Everything else needs the member frame.
DETERMINISTIC_MODELS = {"climatology", "persistence", "damped_persistence"}
CRPSS_REFERENCE = "climatology"

# Day-of-year reference rows (dates, folds, Q_raw, Q_clim) are taken from the
# climatology baseline's predictions, so it is scored on exactly the rows the
# reference boxes are.
DOY_REF_SCAFFOLD = "climatology"


def fold_metrics_in_window(catchment, model, lo, hi, min_rows=30):
    """Per-fold NSE/KGE/ACC restricted to lead_day in [lo, hi]."""
    df = pl.load_predictions(catchment, model)
    if df is None:
        return None
    sub = (df[(df['lead_day'] >= lo) & (df['lead_day'] <= hi)]
             .dropna(subset=['Q_raw', 'Q_pred', 'Q_clim']))
    rows = []
    for hy, g in sub.groupby('hydro_year'):
        if len(g) < min_rows:
            continue
        rows.append({
            'hydro_year': hy,
            'nse': calc_nse(g['Q_raw'].values, g['Q_pred'].values),
            'kge': calc_kge(g['Q_raw'].values, g['Q_pred'].values),
            'acc': calc_acc(g['Q_raw'].values, g['Q_pred'].values,
                            g['Q_clim'].values),
        })
    return pd.DataFrame(rows) if rows else None


@lru_cache(maxsize=None)
def window_crps(catchment: str, model: str, lo: int, hi: int, min_rows=30):
    """
    Per-fold CRPS restricted to lead_day in [lo, hi], indexed by hydro_year.

    Deterministic baselines have no member frame; their members are identical
    so CRPS reduces exactly to MAE, taken from the ensemble-mean file. An
    ensemble model with no member frame returns None rather than falling back
    to MAE, which would not be its CRPS.
    """
    if model in DETERMINISTIC_MODELS:
        df = pl.load_predictions(catchment, model)
        if df is None:
            return None
        sub = (df[(df['lead_day'] >= lo) & (df['lead_day'] <= hi)]
                 .dropna(subset=['Q_raw', 'Q_pred']))
        rows = {}
        for hy, g in sub.groupby('hydro_year'):
            if len(g) < min_rows:
                continue
            rows[hy] = float(np.abs(g['Q_raw'] - g['Q_pred']).mean())
        return pd.Series(rows, dtype=float) if rows else None

    dfm = pl.load_member_predictions(catchment, model)
    if dfm is None:
        return None
    sub = (dfm[(dfm['lead_day'] >= lo) & (dfm['lead_day'] <= hi)]
             .dropna(subset=['Q_raw', 'Q_pred']))
    rows = {}
    for hy, g in sub.groupby('hydro_year'):
        # min_rows counts timesteps, not member-rows, to stay comparable with
        # the same guard applied to the ensemble-mean metrics.
        if g[['init_date', 'date']].drop_duplicates().shape[0] < min_rows:
            continue
        rows[hy] = crps_from_pred_df(g)
    return pd.Series(rows, dtype=float) if rows else None


@lru_cache(maxsize=None)
def window_crpss(catchment: str, lo: int, hi: int, min_rows=30) -> dict:
    """
    Per-fold CRPSS against climatology, as {model: Series indexed by
    hydro_year}. Folds where the reference CRPS is zero or non-finite are
    dropped, and the reference model itself is omitted (it is identically 0).
    """
    ref = window_crps(catchment, CRPSS_REFERENCE, lo, hi, min_rows)
    if ref is None:
        return {}
    ref = ref[np.isfinite(ref) & (ref > 0)]
    if len(ref) == 0:
        return {}

    out = {}
    for model in pl.MODEL_ORDER:
        if model == CRPSS_REFERENCE:
            continue
        crps = window_crps(catchment, model, lo, hi, min_rows)
        if crps is None:
            continue
        common = crps.index.intersection(ref.index)
        if len(common) == 0:
            continue
        out[model] = 1.0 - crps.loc[common] / ref.loc[common]
    return out


@lru_cache(maxsize=None)
def _climatology_period_obs(catchment: str) -> pd.Series:
    """Observed discharge over the years the climatology is fitted on:
    hydro years from clim_start_year onward, skip_years excluded."""
    # Imported here, not at module level: the plotting modules otherwise
    # receive their configuration from the calling driver.
    from ..catchments import CATCHMENTS, SHARED
    from ..climatology import eligible_hydro_years
    from ..data import load_obs_discharge, hydro_year_of
    cfg = CATCHMENTS[catchment]
    obs = load_obs_discharge(cfg['discharge_csv'])
    years = eligible_hydro_years(obs, cfg['skip_years'],
                                 start_year=SHARED['clim_start_year'])
    return obs[np.isin(hydro_year_of(obs.index), list(years))]


@lru_cache(maxsize=None)
def doy_reference_acc(catchment: str, lo: int, hi: int, min_rows=30):
    """
    Per-fold ACC of the day-of-year reference, restricted to lead_day in
    [lo, hi], indexed by hydro_year.

    For each fold the forecast on every date is the mean of log1p(Q) on that
    day of year over the fold's training hydro years, left unsmoothed, and
    back-transformed with expm1 so calc_acc sees it exactly as it sees every
    other forecast. Forecast and observed anomalies share one reference: the
    forecast is built from exactly the fold's training years (read from the
    scaffold run's folds.csv, common.load_folds), the same years the fold's
    climatology was fit on, and is scored against that fold climatology (the
    scaffold's Q_clim).
    """
    scaffold = pl.load_predictions(catchment, DOY_REF_SCAFFOLD)
    if scaffold is None:
        return None
    train_years = pl.load_folds(catchment, DOY_REF_SCAFFOLD)
    if train_years is None:
        print(f"  {catchment}: no folds.csv in the {DOY_REF_SCAFFOLD} default "
              f"run -- day-of-year reference skipped (rerun that run)")
        return None
    from ..data import hydro_year_of
    obs = _climatology_period_obs(catchment)
    obs_hy = hydro_year_of(obs.index)

    sub = (scaffold[(scaffold['lead_day'] >= lo) & (scaffold['lead_day'] <= hi)]
             .dropna(subset=['Q_raw', 'Q_clim']))
    rows = {}
    for hy, g in sub.groupby('hydro_year'):
        if len(g) < min_rows or int(hy) not in train_years:
            continue
        train = obs[np.isin(obs_hy, list(train_years[int(hy)]))]
        doy_mean = (np.log1p(train).groupby(train.index.dayofyear).mean()
                      .reindex(range(1, 367)).ffill().bfill())
        q_pred = np.expm1(g['date'].dt.dayofyear.map(doy_mean).values)
        rows[hy] = calc_acc(g['Q_raw'].values, q_pred, g['Q_clim'].values)
    return pd.Series(rows, dtype=float) if rows else None
