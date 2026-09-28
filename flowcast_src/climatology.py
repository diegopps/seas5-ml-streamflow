"""
flowcast_src.climatology – Smoothed day-of-year (DOY) discharge climatology.

The climatology is the seasonal reference against which every model's anomaly
target and ACC baseline are defined. A single global climatology is computed
once per catchment from the observed record from clim_start_year (1981) onward (excluding the
catchment's skip_years) and reused across every LOGO-CV fold and every model, so that the
anomaly definition is identical across the whole comparison.

The climatology is a smooth seasonal mean over that record, not a fit to
any individual fold's test data.
"""

import numpy as np
import pandas as pd


def compute_smoothed_doy_climatology(obs_daily_q: pd.Series,
                                     years: set,
                                     window: int = 31) -> dict:
    """
    Return dict doy (1-366) -> smoothed log1p(Q) climatology value.

    Parameters
    ----------
    obs_daily_q : pd.Series
        Observed daily discharge indexed by date.
    years : set
        Calendar years to include in the climatology.
    window : int
        Centred rolling-mean window (days), applied circularly across the
        year boundary.
    """
    sel = obs_daily_q[obs_daily_q.index.year.isin(years)]
    log_q = np.log1p(sel)
    doy = sel.index.dayofyear
    raw = log_q.groupby(doy).mean()
    half = window // 2
    vals = raw.reindex(range(1, 367), fill_value=np.nan).ffill().bfill()
    padded = pd.concat([vals.iloc[-half:], vals, vals.iloc[:half]])
    smoothed = padded.rolling(window, center=True, min_periods=1).mean()
    result = smoothed.iloc[half: half + 366]
    result.index = range(1, 367)
    return result.to_dict()


MIN_CLIM_YEARS = 10


def compute_global_climatology(obs_daily_q: pd.Series,
                               skip_years: set,
                               window: int = 31,
                               *,
                               start_year: int) -> dict:
    """
    Compute the single global DOY climatology used by every model and fold.

    Fit on observed years from `start_year` onward, excluding the catchment's
    skip_years (data-gap years). The same window is used for every model and
    every fold.

    `start_year` is keyword-only and has no default on purpose: every caller
    must pass SHARED['clim_start_year'] explicitly, so a call site that is
    missed raises TypeError instead of silently keeping the old full-record
    behaviour.
    """
    all_years = set(obs_daily_q.index.year.unique())
    keep_years = {y for y in all_years
                  if y >= start_year and y not in skip_years}

    if len(keep_years) < MIN_CLIM_YEARS:
        raise ValueError(
            f"climatology would be fit on only {len(keep_years)} year(s) "
            f"(start_year={start_year}, skip_years={sorted(skip_years)}, "
            f"record {min(all_years)}-{max(all_years)}). "
            f"Need at least {MIN_CLIM_YEARS}."
        )

    clim = compute_smoothed_doy_climatology(obs_daily_q, keep_years,
                                            window=window)

    # Downstream code relies on q_clim never being NaN -- build_flat_table's
    # dropna would otherwise silently delete rows, and the row set would stop
    # matching across models.
    if any(not np.isfinite(v) for v in clim.values()):
        raise ValueError(
            f"climatology contains non-finite values (start_year={start_year}, "
            f"{len(keep_years)} years) -- refusing to return it."
        )
    return clim
