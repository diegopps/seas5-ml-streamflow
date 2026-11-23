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

Stage 2 (sensitivity test) scores each scenario's predictions with the same
tables; `scenario_deltas` turns two yearly tables into the paired per-year
differences per lead month (m3).

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
      covers a lead month at one resolution if it has at least half the
      lead month's periods there (MIN_CELL_COUNT: 15 days, 2 weeks, 1
      month). It CONTRIBUTES to that lead month (to the yearly score of the
      lead month's hydro year, and its year to the (start month, lead
      month) cell) only if it covers it at ALL THREE resolutions
      (`common_contributors`): where discharge has gaps, a start can cover
      a lead month at daily (15 days) but not at monthly (the complete
      month), and would otherwise be scored at one resolution and not the
      other (Esil's winters; a few days elsewhere). A yearly score uses
      only the periods of its contributing starts and needs MIN_YEAR_STARTS
      of them (of 12); a cell's ACC needs MIN_CELL_YEARS contributing years
      (per design: the split has only 3-8 test years). The yearly rows,
      their n_starts and the cells' n_years are therefore identical at
      every resolution.
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

# Pooled cell ACC not reported (NaN; n_years and n kept) for these models at
# these resolutions. doy_reference forecasts the same day-of-year mean in
# every year of a fold, so a weekly or monthly cell holds (nearly) one
# forecast value per year: under split exactly one up to a leap-year shift,
# under loyo one up to leaving the test year out. Its ACC there measures that
# artefact, not skill. Its point is sub-monthly seasonal detail, shown at
# daily resolution only.
BLANK_CELLS = {'doy_reference': ('weekly', 'monthly')}

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


def contributing(ens, resolution) -> pd.MultiIndex:
    """The (init_date, lead_month) pairs of an aggregated ensemble frame in
    which the start covers the lead month at `resolution`: at least
    MIN_CELL_COUNT[resolution] periods."""
    check_resolution(resolution)
    per_start = ens.groupby(['init_date', 'lead_month']).size()
    return per_start.index[per_start >= MIN_CELL_COUNT[resolution]]


def common_contributors(ens_by_res: dict) -> pd.MultiIndex:
    """The (init_date, lead_month) pairs covered at every resolution of
    `ens_by_res` ({resolution: aggregated ensemble frame}): the starts that
    contribute to the scores (module docstring, minimum counts)."""
    out = None
    for res, ens in ens_by_res.items():
        c = contributing(ens, res)
        out = c if out is None else out.intersection(c)
    return out


def _restrict(frame, contrib):
    """Rows of `frame` whose (init_date, lead_month) is in `contrib`."""
    idx = pd.MultiIndex.from_arrays([frame['init_date'], frame['lead_month']])
    return frame[idx.isin(contrib)]


def yearly_scores(ens, members, resolution, contrib=None) -> pd.DataFrame:
    """
    The yearly table of one resolution (see module docstring). `ens` is the
    aggregated ensemble-mean (or deterministic) frame, `members` the
    aggregated member frame or None for a deterministic model. Only the
    periods of contributing starts are scored (`n_starts` of them, `n`
    periods): the (init_date, lead_month) pairs in `contrib`
    (`common_contributors`; default: this resolution's own, `contributing`).
    """
    check_resolution(resolution)
    if contrib is None:
        contrib = contributing(ens, resolution)
    ens = _restrict(ens, contrib)
    keys = ['hydro_year', 'lead_month']
    by_year = (dict(tuple(_restrict(members, contrib).groupby(keys)))
               if members is not None else None)
    rows = []
    for (hy, lm), g in ens.groupby(keys):
        starts = g['init_date'].unique()
        if len(starts) < MIN_YEAR_STARTS:
            continue
        if by_year is not None:
            crps = crps_from_pred_df(by_year[(hy, lm)])
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


def cell_scores(ens, resolution, design, contrib=None) -> pd.DataFrame:
    """
    The cells table of one resolution: every (start month, lead month)
    cell, ACC pooled over the cell's periods of every contributing year,
    NaN unless MIN_CELL_YEARS[design] years contribute. A year's start in
    the cell contributes if its (init_date, lead_month) is in `contrib`
    (`common_contributors`; default: this resolution's own, `contributing`);
    a cell holds one start per year. `n_years` and `n` count the
    contributing years and their periods.
    """
    check_resolution(resolution)
    min_years = MIN_CELL_YEARS[check_design(design)]
    if contrib is None:
        contrib = contributing(ens, resolution)
    d = _restrict(ens, contrib)
    d = d.assign(start_month=d['init_date'].dt.month)
    cells = {k: g for k, g in d.groupby(['start_month', 'lead_month'])}
    rows = []
    for sm in range(1, 13):
        for lm in range(1, N_LEAD_MONTHS + 1):
            row = {'resolution': resolution, 'start_month': sm,
                   'lead_month': lm, 'n_years': 0, 'n': 0, 'acc': np.nan}
            cell = cells.get((sm, lm))
            if cell is not None:
                n_years = cell['hydro_year'].nunique()
                row.update(n_years=n_years, n=len(cell))
                if n_years >= min_years:
                    row['acc'] = calc_acc(cell['Q_raw'].values,
                                          cell['Q_pred'].values,
                                          cell['Q_clim'].values)
            rows.append(row)
    return pd.DataFrame(rows, columns=CELL_COLS)


DELTA_KEYS = ['resolution', 'hydro_year', 'lead_month']
# Sign of each metric's paired difference so that positive = the corrupted
# forcing HURT skill (the sign of the stage-2 figure): normal - corrupted
# for higher-is-better metrics, corrupted - normal for CRPS.
DELTA_SIGN = {'nse': 1, 'kge': 1, 'acc': 1, 'crpss': 1, 'crps': -1}
DELTA_COLS = (['scenario'] + DELTA_KEYS + ['n_starts', 'n']
              + [f'd_{m}' for m in DELTA_SIGN])


def scenario_deltas(normal: pd.DataFrame, corrupted: pd.DataFrame,
                    scenario: str) -> pd.DataFrame:
    """
    Paired per-year differences between two `yearly_scores` tables of the
    stage-2 scenarios (`normal`, and the `corrupted` one named `scenario`),
    one row per (resolution, hydro year, lead month), d_<metric> signed by
    DELTA_SIGN (positive = corruption hurt). Both tables come from frames
    over the same paired rows (seas5ml.sensitivity), so they must hold the
    same keys with the same n_starts and n; raises otherwise. CRPS_clim is
    the same in both (same observations and Q_clim), so d_crpss =
    d_crps / CRPS_clim.
    """
    a = normal.set_index(DELTA_KEYS).sort_index()
    b = corrupted.set_index(DELTA_KEYS).sort_index()
    if not a.index.equals(b.index):
        only = a.index.symmetric_difference(b.index)
        raise ValueError(f"scenarios normal and {scenario} scored different "
                         f"(resolution, hydro year, lead month) rows: "
                         f"{len(only)} unpaired, e.g. {list(only[:3])}")
    for col in ('n_starts', 'n'):
        bad = (a[col] != b[col])
        if bad.any():
            raise ValueError(f"scenarios normal and {scenario} differ in "
                             f"{col} on {int(bad.sum())} rows, e.g. "
                             f"{list(a.index[bad][:3])}")
    out = a[['n_starts', 'n']].copy()
    for m, sign in DELTA_SIGN.items():
        out[f'd_{m}'] = sign * (a[m] - b[m])
    out = out.reset_index().assign(scenario=scenario)
    return out[DELTA_COLS]


def score_model(pred: pd.DataFrame, *, design: str, model: str = None,
                resolutions=RESOLUTIONS) -> dict:
    """
    Aggregate and score one model's daily predictions at every resolution.

    `pred` is a per-member frame (column `member`: predictions_members
    .parquet) or a deterministic one (a baseline's predictions_daily
    .parquet). Returns {resolution: {'yearly': table, 'cells': table,
    'ens': aggregated ensemble-mean frame, 'stats': seas5ml.aggregate's
    dropped-period counts, plus contrib_starts (the (start, lead month)
    pairs covered at this resolution) and contrib_dropped (of those, the
    ones not covered at every resolution, hence not scored)}}.

    The contributing starts are taken over ALL resolutions
    (`common_contributors`), whichever `resolutions` are returned, so a
    table does not depend on which others were asked for. `model` names the
    model scored: the cells' ACC is blanked where BLANK_CELLS says so.
    """
    for res in resolutions:
        check_resolution(res)
    parts = {}
    for res in RESOLUTIONS:
        agg, stats = aggregate(pred, res, return_stats=True)
        if 'member' in agg.columns:
            parts[res] = (agg, ensemble_of(agg), stats)
        else:
            parts[res] = (None, agg, stats)
    contrib = common_contributors({r: p[1] for r, p in parts.items()})
    out = {}
    for res in resolutions:
        members, ens, stats = parts[res]
        own = contributing(ens, res)
        stats = dict(stats, contrib_starts=len(own),
                     contrib_dropped=len(own) - len(own.intersection(contrib)))
        cells = cell_scores(ens, res, design, contrib)
        if res in BLANK_CELLS.get(model, ()):
            cells['acc'] = np.nan
        out[res] = {'yearly': yearly_scores(ens, members, res, contrib),
                    'cells': cells, 'ens': ens, 'stats': stats}
    return out
