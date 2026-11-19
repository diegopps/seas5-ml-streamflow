"""
seas5ml.models.references – Deterministic reference baselines.

These baselines are identical across ensemble members, so rows are deduplicated
to one per (init_date, date) and no CRPS is computed.

Every fold uses its own climatology (and a0), fit on the fold's training
hydro years only. One call predicts one fold (a seas5ml.designs.Fold); the
stage-1 driver loops over the folds and scores each test year.

  climatology         predicts a zero anomaly at every lead day, i.e. the
                      fold's climatology (ACC = NaN by construction).
  persistence         carries the observed anomaly at initialisation forward
                      flat (phi = 1).
  damped_persistence  decays that anomaly geometrically, a_hat(h) = phi**h * a0,
                      with phi the lag-1 autocorrelation of the observed anomaly
                      record, estimated per fold on the fold's training hydro
                      years, against the fold's climatology (`fold_phi`).
"""

import numpy as np
import pandas as pd

from ..climatology import eligible_hydro_years
from ..data import GROUP_COL, hydro_year_of
from ..features import apply_fold_climatology

BASELINES = ('climatology', 'persistence', 'damped_persistence')

# Columns of a baseline's prediction frame (one row per init_date, date).
DET_COLS = ['init_date', 'date', 'lead_day', 'hydro_year',
            'Q_raw', 'Q_pred', 'Q_clim']


def dedup_unique(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (init_date, date, hydro_year) — members are identical."""
    return (df.drop_duplicates(subset=['init_date', 'date', 'hydro_year'])
              .reset_index(drop=True))


def estimate_phi(anomaly_series: pd.Series) -> float:
    """
    Lag-1 autocorrelation of a daily anomaly series, clipped to [0, 1].

    Only pairs (t, t-1) from contiguous daily observations are used; gaps in
    the date index are not paired across. Returns 0.0 if fewer than two valid
    contiguous pairs exist.
    """
    s = anomaly_series.dropna().sort_index()
    if len(s) < 2:
        return 0.0

    dates = s.index.to_series()
    is_contiguous = (dates.diff().dt.days == 1).values

    a_t = s.values[1:][is_contiguous[1:]]
    a_tm1 = s.values[:-1][is_contiguous[1:]]
    if len(a_t) < 2:
        return 0.0

    a_mean = np.mean(np.concatenate([a_t, a_tm1]))
    num = np.sum((a_t - a_mean) * (a_tm1 - a_mean))
    den = np.sum((a_tm1 - a_mean) ** 2)
    if den == 0:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def anomaly_record(obs_daily_q: pd.Series, doy_clim: dict,
                   skip_years: set, *, start_year: int,
                   train_years=None) -> pd.Series:
    """
    Observed log1p(Q) anomaly series against `doy_clim`, over the hydro years
    a climatology may use: hydro years from `start_year` onward, skip_years
    excluded (see seas5ml.climatology.eligible_hydro_years), and, if
    `train_years` is given, only those hydro years (same rule as
    seas5ml.climatology.compute_climatology). Days of excluded years are
    absent, not NaN, so the series has gaps there.
    """
    years = eligible_hydro_years(obs_daily_q, skip_years,
                                 start_year=start_year)
    if train_years is not None:
        years &= {int(y) for y in train_years}
    keep = obs_daily_q[np.isin(hydro_year_of(obs_daily_q.index), list(years))]
    doy = keep.index.dayofyear
    clim = pd.Series([doy_clim.get(d, np.nan) for d in doy], index=keep.index)
    return np.log1p(keep) - clim


def fold_phi(fit_clim, train_years) -> float:
    """
    Damped-persistence phi for one fold: lag-1 autocorrelation of the
    observed anomaly record over the fold's training hydro years, against the
    fold's climatology. `fit_clim` is the catchment's
    seas5ml.climatology.ClimatologyFitter, which also holds the observed record,
    skip_years and start year. Day pairs are never formed across the gap a
    held-out year leaves (see `estimate_phi`).
    """
    doy_clim = fit_clim(train_years)
    anom = anomaly_record(fit_clim.obs_daily_q, doy_clim, fit_clim.skip_years,
                          start_year=fit_clim.start_year,
                          train_years=train_years)
    return estimate_phi(anom)


def predict_baseline(model, df, fit_clim, fold):
    """
    One baseline's forecast of a fold's test years, against the fold's
    climatology (fit by `fit_clim`, a seas5ml.climatology.ClimatologyFitter,
    on fold.train_years).

      climatology         zero anomaly, i.e. the fold climatology itself
      persistence         a0 carried forward flat (phi = 1)
      damped_persistence  phi**lead_day * a0, phi estimated on the fold's
                          training years (`fold_phi`)

    Returns (pred, info): `pred` has one row per (init_date, date) with
    DET_COLS (valid rows only), or is None if the fold has no test rows;
    `info` holds the per-fold values to record ({'phi': ...} for the two
    persistence models).
    """
    if model not in BASELINES:
        raise ValueError(f"unknown baseline '{model}' (one of {BASELINES})")
    df_test = dedup_unique(df[df[GROUP_COL].isin(fold.test_years)])
    if len(df_test) == 0:
        return None, {}

    info = {}
    if model == 'climatology':
        doy_clim = fit_clim(fold.train_years)
        q_clim_log = df_test['date'].dt.dayofyear.map(doy_clim).values
        q_pred = np.expm1(q_clim_log)                 # zero anomaly -> clim
    else:
        # Only the test rows are used, so a0 is written on that slice alone.
        doy_clim = apply_fold_climatology(df_test, fit_clim, fold.train_years)
        phi = (1.0 if model == 'persistence'
               else fold_phi(fit_clim, fold.train_years))
        info['phi'] = phi
        q_clim_log = df_test['date'].dt.dayofyear.map(doy_clim).values
        lead_day = df_test['lead_day'].values.astype(float)
        pred_log_q = (phi ** lead_day) * df_test['a0'].values + q_clim_log
        q_pred = np.clip(np.expm1(pred_log_q), 0.0, None)

    pred = df_test.rename(columns={'Q': 'Q_raw'})
    pred['Q_pred'] = q_pred
    pred['Q_clim'] = np.expm1(q_clim_log)
    valid = pred['Q_raw'].notna() & pred['Q_pred'].notna()
    if not valid.any():
        return None, info
    return pred.loc[valid, DET_COLS].reset_index(drop=True), info

