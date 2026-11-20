"""
seas5ml.scores – Scores of one model's predictions at every resolution.

The verification layer's scoring step: one model's daily predictions
(seas5ml.aggregate's input) are aggregated to each resolution (daily, weekly,
monthly) and scored by the same code, on the common lead-month grid (1-7), so
the three resolutions are comparable at a glance.

Two score tables per resolution:

  yearly   one row per (hydro year, lead month): NSE, KGE and ACC of the
           ensemble mean over that lead month's periods verifying in that
           hydro year (every start month), the fair CRPS over the members,
           and CRPSS against climatology. What the per-year boxplots show,
           with a lead month as the window.
  cells    one row per (start month, lead month), all 84 cells: ACC pooled
           over every year's periods in the cell. What the heatmap shows. At
           monthly resolution a cell holds one value per year, so its ACC is
           pure year-to-year skill.

Rules
    - NSE, KGE in Q space; ACC on log1p anomalies against Q_clim
      (seas5ml.metrics.calc_acc), i.e. the aggregated flow against the fold
      climatology aggregated the same way.
    - CRPS: fair CRPS over the members (seas5ml.metrics.crps_from_pred_df);
      a deterministic model's CRPS is its absolute error.
    - CRPSS = 1 - CRPS / CRPS_clim, with CRPS_clim the absolute error of the
      climatology forecast. That forecast is Q_clim, carried by every model's
      frame (the fold climatology, fit on the fold's training years, the
      same one the climatology baseline predicts), so the reference is scored
      on exactly the model's own periods. The climatology model's CRPSS is 0.
    - Ensemble scores use the ensemble mean of the AGGREGATED members, so the
      ensemble-mean and member scores cover exactly the same periods.
    - Minimum counts (constants below), counted in start months so that
      the same starts and years are scored at every resolution. A start
      CONTRIBUTES to a (hydro year, lead month) if it covers at least half
      the lead month (MIN_CELL_COUNT: 15 days, 2 weeks, 1 month); the same
      rule makes a year count toward a cell. A yearly score uses only the
      periods of its contributing starts and needs MIN_YEAR_STARTS of them
      (of 12); a cell's ACC needs MIN_CELL_YEARS contributing years (per
      design: the split has only 3-8 test years).
      MIN_YEAR_STARTS = 3 keeps the partial years at the record's edges
      (Central Asia HY2025: last start Dec 2024; the Alpine last years,
      Oct-Dec only; the Alpine HY1981, from Jan 1981), whose scores cover
      only part of the seasonal cycle and, at monthly resolution, rest on as
      few as 3 values.
"""

import numpy as np
import pandas as pd

from .aggregate import RESOLUTIONS, N_LEAD_MONTHS, aggregate, check_resolution
from .designs import check_design
from .evaluate import score_frame
from .metrics import calc_acc, crps_from_pred_df

# Periods a start must have in a lead month to contribute to it (to a yearly
# score, or its year to a cell): half of a lead month's (daily ~30, weekly
# ~4, monthly 1).
MIN_CELL_COUNT = {'daily': 15, 'weekly': 2, 'monthly': 1}

# Contributing starts (of 12) for a (hydro year, lead month) to be scored,
# the same at every resolution.
MIN_YEAR_STARTS = 3

# Contributing years for a cell's ACC to be shown, per design (was
# plots.heatmap.MIN_YEARS_BY_DESIGN). With 6-8 years a split cell is
# descriptive only.
MIN_CELL_YEARS = {'loyo': 10, 'split': 6}

YEARLY_COLS = ['resolution', 'hydro_year', 'lead_month', 'n_starts', 'n',
               'nse', 'kge', 'acc', 'crps', 'crps_clim', 'crpss']
CELL_COLS = ['resolution', 'start_month', 'lead_month', 'n_years', 'n',
             'acc']


def ensemble_of(agg_members: pd.DataFrame) -> pd.DataFrame:
    """Ensemble mean of an aggregated member frame: one row per
    (init_date, period), Q_pred averaged over the members, every other
    column taken from the first member (identical across members)."""
    keys = ['init_date', 'period']
    first = (agg_members.drop_duplicates(subset=keys)
                        .drop(columns=['member', 'Q_pred']))
    q_pred = agg_members.groupby(keys, as_index=False)['Q_pred'].mean()
    return first.merge(q_pred, on=keys).reset_index(drop=True)


def _abs_error(obs, pred) -> float:
    return float(np.mean(np.abs(np.asarray(obs, float)
                                - np.asarray(pred, float))))


def yearly_scores(ens, members, resolution) -> pd.DataFrame:
    """
    The yearly table of one resolution (see module docstring). `ens` is the
    aggregated ensemble-mean (or deterministic) frame, `members` the
    aggregated member frame or None for a deterministic model. Only the
    periods of contributing starts are scored (`n_starts` of them, `n`
    periods).
    """
    check_resolution(resolution)
    keys = ['hydro_year', 'lead_month']
    by_year = (dict(tuple(members.groupby(keys))) if members is not None
               else None)
    rows = []
    for (hy, lm), g in ens.groupby(keys):
        per_start = g.groupby('init_date').size()
        starts = per_start.index[per_start >= MIN_CELL_COUNT[resolution]]
        if len(starts) < MIN_YEAR_STARTS:
            continue
        g = g[g['init_date'].isin(starts)]
        if by_year is not None:
            m = by_year[(hy, lm)]
            crps = crps_from_pred_df(m[m['init_date'].isin(starts)])
        else:
            crps = _abs_error(g['Q_raw'], g['Q_pred'])
        crps_clim = _abs_error(g['Q_raw'], g['Q_clim'])
        crpss = (1.0 - crps / crps_clim
                 if np.isfinite(crps_clim) and crps_clim > 0 else np.nan)
        rows.append({'resolution': resolution, 'hydro_year': int(hy),
                     'lead_month': int(lm), 'n_starts': len(starts),
                     'n': len(g), **score_frame(g),
                     'crps': crps, 'crps_clim': crps_clim, 'crpss': crpss})
    return pd.DataFrame(rows, columns=YEARLY_COLS)


def cell_scores(ens, resolution, design) -> pd.DataFrame:
    """
    The cells table of one resolution: every (start month, lead month)
    cell, ACC pooled over the cell's periods of every year that covers at
    least MIN_CELL_COUNT of them, NaN unless MIN_CELL_YEARS[design] years
    do. `n_years` and `n` count the contributing years and their periods.
    """
    check_resolution(resolution)
    min_years = MIN_CELL_YEARS[check_design(design)]
    d = ens.assign(start_month=ens['init_date'].dt.month)
    cells = {k: g for k, g in d.groupby(['start_month', 'lead_month'])}
    rows = []
    for sm in range(1, 13):
        for lm in range(1, N_LEAD_MONTHS + 1):
            row = {'resolution': resolution, 'start_month': sm,
                   'lead_month': lm, 'n_years': 0, 'n': 0, 'acc': np.nan}
            cell = cells.get((sm, lm))
            if cell is not None:
                counts = cell.groupby('hydro_year').size()
                years = counts.index[counts >= MIN_CELL_COUNT[resolution]]
                sub = cell[cell['hydro_year'].isin(years)]
                row.update(n_years=len(years), n=len(sub))
                if len(years) >= min_years:
                    row['acc'] = calc_acc(sub['Q_raw'].values,
                                          sub['Q_pred'].values,
                                          sub['Q_clim'].values)
            rows.append(row)
    return pd.DataFrame(rows, columns=CELL_COLS)


def score_model(pred: pd.DataFrame, *, design: str,
                resolutions=RESOLUTIONS) -> dict:
    """
    Aggregate and score one model's daily predictions at every resolution.

    `pred` is a per-member frame (column `member`: predictions_members
    .parquet) or a deterministic one (a baseline's predictions_daily
    .parquet). Returns {resolution: {'yearly': table, 'cells': table,
    'ens': aggregated ensemble-mean frame, 'stats': seas5ml.aggregate's
    dropped-period counts}}.
    """
    out = {}
    for res in resolutions:
        agg, stats = aggregate(pred, res, return_stats=True)
        if 'member' in agg.columns:
            members, ens = agg, ensemble_of(agg)
        else:
            members, ens = None, agg
        out[res] = {'yearly': yearly_scores(ens, members, res),
                    'cells': cell_scores(ens, res, design),
                    'ens': ens, 'stats': stats}
    return out
