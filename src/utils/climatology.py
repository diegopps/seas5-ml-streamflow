"""
utils.climatology – Smoothed day-of-year (DOY) discharge climatology.

The climatology is the seasonal reference against which every model's anomaly
target and ACC baseline are defined. It is the mean of log1p(Q) per day of
year, smoothed circularly over `window` days.

Years are always selected by HYDRO year (1 Oct - 30 Sep, see
utils.data.hydro_year_of), the same unit the cross-validation folds use.
Selecting by calendar year would leave Oct-Dec of an excluded hydro year in
the fit (and drop Oct-Dec of the following one).

Which hydro years may enter any climatology (`eligible_hydro_years`):
from SHARED['clim_start_year'] onward, the catchment's skip_years excluded.
`compute_climatology` fits on those, optionally restricted further to an
explicit set of training hydro years (`train_years`), e.g. one fold's.
"""

import numpy as np
import pandas as pd

from .data import hydro_year_of


def eligible_hydro_years(obs_daily_q: pd.Series, skip_years: set, *,
                         start_year: int) -> set:
    """
    Hydro years of the observed record that a climatology (or an anomaly
    record derived from it) may use: hydro years >= `start_year`, minus
    `skip_years` (also hydro years). Years without any observation are not
    returned.
    """
    hy = set(hydro_year_of(obs_daily_q.index).tolist())
    return {y for y in hy if y >= start_year and y not in skip_years}


def compute_smoothed_doy_climatology(obs_daily_q: pd.Series,
                                     hydro_years: set,
                                     window: int = 31) -> dict:
    """
    Return dict doy (1-366) -> smoothed log1p(Q) climatology value.

    Parameters
    ----------
    obs_daily_q : pd.Series
        Observed daily discharge indexed by date.
    hydro_years : set
        Hydro years to include in the climatology.
    window : int
        Centred rolling-mean window (days), applied circularly across the
        year boundary.
    """
    in_years = np.isin(hydro_year_of(obs_daily_q.index), list(hydro_years))
    sel = obs_daily_q[in_years]
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


def compute_climatology(obs_daily_q: pd.Series,
                        skip_years: set,
                        window: int = 31,
                        *,
                        start_year: int,
                        train_years=None) -> dict:
    """
    Compute the DOY climatology on an explicit set of hydro years.

    The fit uses the eligible hydro years (>= `start_year`, minus
    `skip_years`; see `eligible_hydro_years`). If `train_years` is given, the
    fit is further restricted to those hydro years, so a fold's climatology
    is built from its training years only; years in `train_years` that are
    not eligible are ignored. `train_years=None` uses every eligible year.

    `start_year` is keyword-only and has no default on purpose: every caller
    must pass SHARED['clim_start_year'] explicitly, so a call site that is
    missed raises TypeError instead of silently keeping the old full-record
    behaviour.
    """
    eligible = eligible_hydro_years(obs_daily_q, skip_years,
                                    start_year=start_year)
    keep_years = (eligible if train_years is None
                  else eligible & {int(y) for y in train_years})

    if len(keep_years) < MIN_CLIM_YEARS:
        all_hy = hydro_year_of(obs_daily_q.index)
        raise ValueError(
            f"climatology would be fit on only {len(keep_years)} hydro "
            f"year(s) (start_year={start_year}, skip_years="
            f"{sorted(skip_years)}, train_years="
            f"{'all' if train_years is None else len(set(train_years))}, "
            f"record HY{all_hy.min()}-HY{all_hy.max()}). "
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
            f"{len(keep_years)} hydro years) -- refusing to return it."
        )
    return clim
