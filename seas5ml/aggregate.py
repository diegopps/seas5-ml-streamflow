"""
seas5ml.aggregate – Daily predictions aggregated to weekly and monthly means.

The verification layer scores every run at three resolutions with the same
code: daily, weekly and monthly. This module turns a daily prediction frame
(per member, or ensemble mean / deterministic baseline) into the frame of one
resolution; seas5ml.scores scores it. Daily is the identity plus the lead-month
column, so all three share one grid.

Lead periods (forecasts start on the 1st of a month)
    daily    lead day + 1 (1-based)
    weekly   lead week k = lead days 7(k-1) ... 7k-1
    monthly  lead month m = the m-th calendar month from the start
             (lead month 1 = the start month)

`lead_month` is the common grid of every resolution: 1-7. 215 lead days always
cover 7 full calendar months (worst case Jul-Jan, exactly 215 days); what
follows is a partial 8th month and is dropped at every resolution, so daily
and monthly scores of a lead month cover the same days. A week belongs to the
lead month of its middle day (its 4th), so a lead month's weeks cover its
days up to the weeks straddling its edges.

Aggregation rules
    - Mean discharge over the period, in Q space (a volume): Q_raw, Q_pred
      and Q_clim are all averaged. The anomaly scored by
      seas5ml.metrics.calc_acc is then log1p(mean Q) - log1p(mean Q_clim):
      the aggregated flow against the fold climatology aggregated the same
      way, one reference for observation and forecast.
    - Complete periods only: every day of the period present (rows exist
      only where discharge was observed, so a partial period would be the
      mean of whichever days happen to be there). In a member frame, a
      period is kept only if every member has every day, so the members of a
      kept period form a complete ensemble and their mean equals the
      aggregated ensemble mean.
    - A period lies inside one hydro year. Under loyo, the days of one start
      that verify in different hydro years were predicted by different fold
      models, so a period crossing 1 October would mix two models. Calendar
      months never cross it; one lead week per start (Mar-Sep starts) does
      and is dropped. Same rule under every design.

Input: a frame with columns init_date, date, lead_day, hydro_year, Q_raw,
Q_pred, Q_clim (plus member for a per-member frame): predictions_daily.parquet,
predictions_members.parquet or a baseline's prediction frame. Each
(init_date, date[, member]) must appear once.

Output: one row per (init_date, period[, member]) with the same columns, plus
period, lead_month and n_days; `date` and `lead_day` are those of the period's
first day, so (init_date, date) still identifies a forecast timestep and
seas5ml.metrics.crps_from_pred_df works unchanged.
"""

import numpy as np
import pandas as pd

RESOLUTIONS = ('daily', 'weekly', 'monthly')
N_LEAD_MONTHS = 7       # lead months on the grid; the partial 8th is dropped
WEEK_DAYS = 7
WEEK_MID = 3            # offset of a week's middle day (its 4th)

VALUE_COLS = ['Q_raw', 'Q_pred', 'Q_clim']
REQUIRED_COLS = ['init_date', 'date', 'lead_day', 'hydro_year'] + VALUE_COLS


def check_resolution(resolution: str) -> str:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"unknown resolution '{resolution}' "
                         f"(one of {RESOLUTIONS})")
    return resolution


def lead_month_of(init_date, date) -> np.ndarray:
    """1-based calendar lead month of each `date` from its `init_date`
    (Series or array-likes of timestamps): 1 for the start month itself."""
    init = pd.DatetimeIndex(init_date)
    d = pd.DatetimeIndex(date)
    return np.asarray((d.year - init.year) * 12 + (d.month - init.month) + 1)


def _with_periods(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """`df` plus period, lead_month and n_expected (days in the period)."""
    out = df.copy()
    lead_day = out['lead_day'].to_numpy().astype(int)
    if resolution == 'daily':
        out['period'] = lead_day + 1
        out['lead_month'] = lead_month_of(out['init_date'], out['date'])
        out['n_expected'] = 1
    elif resolution == 'weekly':
        week = lead_day // WEEK_DAYS + 1
        mid = (out['init_date'].to_numpy().astype('datetime64[D]')
               + ((week - 1) * WEEK_DAYS + WEEK_MID).astype('timedelta64[D]'))
        out['period'] = week
        out['lead_month'] = lead_month_of(out['init_date'], mid)
        out['n_expected'] = WEEK_DAYS
    else:   # monthly
        out['period'] = lead_month_of(out['init_date'], out['date'])
        out['lead_month'] = out['period']
        out['n_expected'] = out['date'].dt.days_in_month.to_numpy()
    return out


def aggregate(pred: pd.DataFrame, resolution: str, *, return_stats=False):
    """
    Aggregate a daily prediction frame to `resolution` (see module
    docstring for the rules). Returns the aggregated frame, sorted by
    init_date, period[, member]; with `return_stats`, also a dict counting
    the dropped periods (per (init_date, period), members not counted
    separately): beyond_grid (lead month > N_LEAD_MONTHS), incomplete,
    crosses_hydro_year, and kept.
    """
    check_resolution(resolution)
    missing = [c for c in REQUIRED_COLS if c not in pred.columns]
    if missing:
        raise KeyError(f"prediction frame lacks column(s) {missing}")
    has_members = 'member' in pred.columns
    member_key = ['member'] if has_members else []

    df = pred.dropna(subset=VALUE_COLS)
    dup = df.duplicated(subset=['init_date', 'date'] + member_key)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (init_date, date"
                         f"{', member' if has_members else ''}) row(s): a "
                         f"day must be predicted once (frames concatenated "
                         f"from overlapping folds?)")
    df = _with_periods(df, resolution)

    period_key = ['init_date', 'period']
    n_all = df[period_key].drop_duplicates().shape[0]
    on_grid = df['lead_month'] <= N_LEAD_MONTHS
    df = df[on_grid]
    n_grid = df[period_key].drop_duplicates().shape[0]

    if resolution == 'daily':
        out = df.drop(columns='n_expected').assign(n_days=1)
        stats = {'beyond_grid': n_all - n_grid, 'incomplete': 0,
                 'crosses_hydro_year': 0, 'kept': n_grid}
    else:
        g = df.groupby(period_key + member_key, sort=False)
        agg = g.agg(date=('date', 'min'), lead_day=('lead_day', 'min'),
                    hydro_year=('hydro_year', 'first'),
                    n_hy=('hydro_year', 'nunique'),
                    lead_month=('lead_month', 'first'),
                    n_expected=('n_expected', 'first'),
                    n_days=('date', 'size'),
                    **{c: (c, 'mean') for c in VALUE_COLS}).reset_index()

        # Judged per (init_date, period): every day of every member present,
        # every member of the start present, one hydro year. A period
        # crossing 1 October counts as crosses_hydro_year whether or not it
        # is also incomplete.
        agg['complete'] = agg['n_days'] == agg['n_expected']
        per = agg.groupby(period_key).agg(complete=('complete', 'all'),
                                          n_hy=('n_hy', 'max'),
                                          n_members=('n_days', 'size'))
        if has_members:
            start_members = df.groupby('init_date')['member'].nunique()
            per['complete'] &= (per['n_members'].to_numpy()
                                == start_members.reindex(
                                    per.index.get_level_values('init_date')
                                ).to_numpy())
        one_hy = per['n_hy'] == 1
        keep = per.index[per['complete'] & one_hy]
        stats = {'beyond_grid': n_all - n_grid,
                 'incomplete': int((~per['complete'] & one_hy).sum()),
                 'crosses_hydro_year': int((~one_hy).sum()),
                 'kept': len(keep)}
        idx = pd.MultiIndex.from_frame(agg[period_key])
        out = agg[idx.isin(keep)].drop(columns=['n_hy', 'n_expected',
                                                'complete'])

    cols = (['init_date', 'date'] + member_key
            + ['lead_day', 'period', 'lead_month', 'hydro_year', 'n_days']
            + VALUE_COLS)
    out = (out[cols].sort_values(period_key + member_key)
              .reset_index(drop=True))
    return (out, stats) if return_stats else out
