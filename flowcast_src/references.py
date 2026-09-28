"""
flowcast_src.references – Deterministic reference baselines.

These baselines are identical across ensemble members, so rows are deduplicated
to one per (init_date, date) and no CRPS is computed.

  climatology         predicts a zero anomaly at every lead day (NSE = 0,
                      ACC = NaN by construction).
  persistence         carries the observed anomaly at initialisation forward
                      flat (phi = 1).
  damped_persistence  decays that anomaly geometrically, a_hat(h) = phi**h * a0,
                      with phi the lag-1 autocorrelation of the observed anomaly
                      record (estimated once per catchment).
"""

import time
import numpy as np
import pandas as pd

from .cv import make_folds
from .evaluate import score_frame


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
                   skip_years: set, *, start_year: int) -> pd.Series:
    """
    Observed log1p(Q) anomaly series (skip_years excluded), restricted to
    calendar years from `start_year` onward.
    """
    keep = obs_daily_q[(~obs_daily_q.index.year.isin(skip_years))
                       & (obs_daily_q.index.year >= start_year)]
    doy = keep.index.dayofyear
    clim = pd.Series([doy_clim.get(d, np.nan) for d in doy], index=keep.index)
    return np.log1p(keep) - clim


def run_climatology(df, unique_years, doy_clim, lead_bins):
    """LOGO-CV loop for the climatology baseline (zero-anomaly forecast)."""
    df_unique = dedup_unique(df)
    fold_metrics, all_pred_dfs = [], []

    for split in make_folds(unique_years):
        test_year = split['test_year']
        fold_t0 = time.time()

        df_test = df_unique[df_unique['hydro_year'] == test_year].copy()
        if len(df_test) == 0:
            continue

        df_test['doy'] = df_test['date'].dt.dayofyear
        q_clim_log = df_test['doy'].map(doy_clim).values
        df_test['Q_pred'] = np.expm1(q_clim_log)     # zero anomaly -> clim
        df_test['Q_clim'] = np.expm1(q_clim_log)
        df_test = df_test.rename(columns={'Q': 'Q_raw'})

        rec, ens = _score_deterministic(df_test, test_year, fold_t0)
        if rec is None:
            continue
        fold_metrics.append(rec)
        all_pred_dfs.append(ens)

    return _finalise_deterministic(fold_metrics, all_pred_dfs)


def run_persistence(df, unique_years, doy_clim, phi, model_name):
    """LOGO-CV loop for persistence (phi=1) or damped persistence (phi<1)."""
    df_unique = dedup_unique(df)
    fold_metrics, all_pred_dfs = [], []

    for split in make_folds(unique_years):
        test_year = split['test_year']
        fold_t0 = time.time()

        df_test = df_unique[df_unique['hydro_year'] == test_year].copy()
        if len(df_test) == 0:
            continue

        df_test['doy'] = df_test['date'].dt.dayofyear
        q_clim_log = df_test['doy'].map(doy_clim)

        a0 = df_test['a0'].values

        lead_day = df_test['lead_day'].values.astype(float)
        pred_log_q = (phi ** lead_day) * a0 + q_clim_log.values
        df_test['Q_pred'] = np.clip(np.expm1(pred_log_q), 0.0, None)
        df_test['Q_clim'] = np.expm1(q_clim_log.values)
        df_test = df_test.rename(columns={'Q': 'Q_raw'})

        rec, ens = _score_deterministic(df_test, test_year, fold_t0)
        if rec is None:
            continue
        fold_metrics.append(rec)
        all_pred_dfs.append(ens)

    return _finalise_deterministic(fold_metrics, all_pred_dfs)


def _score_deterministic(df_test, test_year, fold_t0):
    """Shared scoring for a deterministic baseline fold."""
    valid = df_test['Q_raw'].notna() & df_test['Q_pred'].notna()
    if valid.sum() == 0:
        return None, None

    ens = df_test.loc[valid, ['init_date', 'date', 'lead_day',
                              'Q_raw', 'Q_pred', 'Q_clim']].copy()
    ens['hydro_year'] = test_year
    scores = score_frame(ens)
    rec = {
        'hydro_year': test_year,
        'nse': scores['nse'], 'kge': scores['kge'], 'acc': scores['acc'],
        'n_test_rows': int(valid.sum()),
        'time_s': time.time() - fold_t0,
    }
    return rec, ens


def _finalise_deterministic(fold_metrics, all_pred_dfs):
    mdf = pd.DataFrame(fold_metrics)
    all_preds = (pd.concat(all_pred_dfs, ignore_index=True)
                 if all_pred_dfs else pd.DataFrame())
    return mdf, all_preds
