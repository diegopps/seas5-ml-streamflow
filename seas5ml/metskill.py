"""
seas5ml.metskill – SEAS5 meteorological forecast skill against ERA5 (R2).

Does SEAS5 carry predictive information about each forcing variable, lead
month by lead month? The SEAS5 side is exactly the models' input (the stage-0
feature table: same starts, members, deaccumulation, polygon weights); the
reference is ERA5 averaged over the same SEAS5 cells with the same weights
and the same daily definitions (pipeline/0_data/era5/extract_era5.py). The
scores are laid out like the streamflow scores (seas5ml.scores): daily /
weekly / monthly on lead months 1-7, the same folds and designs, the same
contributing-start rules, so the two can be set side by side.

Anomalies
    Per fold, two climatologies on the index (start month, lead day), fit on
    the rows verifying in the fold's training hydro years: the SEAS5
    ensemble mean's (clim_S, which carries SEAS5's lead-dependent drift and
    bias) and ERA5's on the same starts and days (clim_E). Both smoothed
    along lead day over CLIM_WINDOW days, as the discharge climatology is
    along day of year: a centred running mean, local linear within half a
    window of the trajectory's ends (`local_linear`: a shrinking running
    mean there leaves the same slope offset in both climatologies, a shared
    signal that inflates the ACC). Each dataset is taken against its own climatology, i.e.
    the SEAS5 members are mean-corrected per (start month, lead day):
        Q_raw  = ERA5
        Q_pred = SEAS5 member - clim_S + clim_E
        Q_clim = clim_E
    which is the frame seas5ml.aggregate expects; periods are aggregated as
    for discharge (mean over the period, complete periods only, inside one
    hydro year, lead month <= 7). Verification days are NOT dropped for
    missing discharge: every stage-0 row is scored, on the catchment's fold
    years (the loyo fold manifest of stage 0).

Scores, per variable and resolution
    ACC     on additive anomalies (Q - Q_clim; the streamflow path uses
            log1p): ensemble mean of the aggregated members.
    CRPS    fair CRPS (seas5ml.metrics.fair_crps_per_step) of the corrected
            members, per period.
    CRPS_clim
            fair CRPS of the CLIMATOLOGICAL ENSEMBLE: the aggregated ERA5
            values of the same (start month, period) in every training year
            of the period's fold (loyo: every other year; split: HY <= 2016).
            Not the deterministic climatology of the streamflow scores:
            SEAS5 ensembles are well spread, and against a deterministic
            mean a reliable ensemble with no information already scores
            CRPSS ~ +0.29.
    CRPSS   1 - mean CRPS / mean CRPS_clim over the scored periods.
    bias    raw SEAS5 ensemble mean - ERA5 per (start month, lead month),
            on the daily rows (no correction; % for the positive variables).

Tables (module constants YEARLY_COLS, CELL_COLS, BIAS_COLS, PERIOD_COLS)
    yearly  per (variable, resolution, hydro year, lead month), as
            seas5ml.scores.yearly_scores: contributing starts only, at least
            MIN_YEAR_STARTS of them.
    cells   per (variable, resolution, start month, lead month), ACC pooled
            over years, NaN below MIN_CELL_YEARS[design] years.
    bias    per (variable, start month, lead month).
    periods every scored period (contributing starts): obs, clim, ensemble
            mean, CRPS, CRPS_clim; for significance tests at figure time.
"""

import numpy as np
import pandas as pd

from . import designs
from .aggregate import RESOLUTIONS, aggregate, lead_month_of, N_LEAD_MONTHS
from .metrics import fair_crps_per_step
from .scores import (MIN_CELL_YEARS, MIN_YEAR_STARTS, common_contributors,
                     contributing, ensemble_of)

VARIABLES = ("tas", "tasmax", "tasmin", "ssrd", "tp", "sf", "sd")
# Bias in % of ERA5 for these (positive quantities), in their units otherwise.
RELATIVE_BIAS = ("ssrd", "tp", "sf", "sd")
CLIM_WINDOW = 31        # days, centred, along lead day

KEYS = ["init_date", "lead_day"]
YEARLY_COLS = ["variable", "resolution", "hydro_year", "lead_month",
               "n_starts", "n", "acc", "crps", "crps_clim", "crpss"]
CELL_COLS = ["variable", "resolution", "start_month", "lead_month",
             "n_years", "n", "acc"]
BIAS_COLS = ["variable", "start_month", "lead_month", "n", "seas5", "era5",
             "bias", "bias_pct"]
PERIOD_COLS = ["variable", "resolution", "init_date", "date", "period",
               "lead_month", "hydro_year", "start_month", "n_days",
               "Q_raw", "Q_clim", "Q_pred", "crps", "crps_clim"]


# ═══════════════════════════════════════════════════════════════════════
# CLIMATOLOGIES AND THE PREDICTION FRAME
# ═══════════════════════════════════════════════════════════════════════

def local_linear(y: np.ndarray, window: int = CLIM_WINDOW) -> np.ndarray:
    """
    Local linear smoothing of an evenly spaced series: at each point, the
    least-squares line through the points within window // 2 of it
    (clipped at the ends), evaluated there. Where the window is complete it
    is symmetric and the value equals the centred running mean; at the ends
    it removes the slope bias a shrinking running mean has (one-sided, it
    would sit `slope x ~window/4` off: up to ~1 K on the spring / autumn
    temperature cycle, the same offset in SEAS5 and ERA5, which inflated
    the ACC of a no-skill forecast by ~+0.05-0.08 at lead months 1 and 7).
    """
    y = np.asarray(y, float)
    n, h = len(y), window // 2
    x = np.arange(n, dtype=float)
    c = lambda a: np.concatenate([[0.0], np.cumsum(a)])
    lo, hi = np.clip(x - h, 0, n).astype(int), np.clip(x + h + 1, 0, n).astype(int)
    s0, s1, s2 = (c(np.ones(n))[hi] - c(np.ones(n))[lo], c(x)[hi] - c(x)[lo],
                  c(x * x)[hi] - c(x * x)[lo])
    t0, t1 = c(y)[hi] - c(y)[lo], c(x * y)[hi] - c(x * y)[lo]
    den = s0 * s2 - s1 ** 2
    slope = np.where(den > 0, (s0 * t1 - s1 * t0) / np.where(den > 0, den, 1), 0.0)
    return t0 / s0 + slope * (x - s1 / s0)


def smooth_along_lead(clim: pd.Series, window: int = CLIM_WINDOW) -> pd.Series:
    """`local_linear` smoothing of a (init_month, lead_day) series along
    lead day, per start month (every lead day 0 ... 214 must be present)."""
    out = []
    for _, s in clim.sort_index().groupby(level="init_month"):
        lead = s.index.get_level_values("lead_day").to_numpy()
        if not np.array_equal(lead, np.arange(len(s))):
            raise ValueError("climatology lacks lead days: cannot smooth")
        out.append(pd.Series(local_linear(s.to_numpy(), window), index=s.index))
    return pd.concat(out)


def fold_climatologies(ens: pd.DataFrame, train_years) -> pd.DataFrame:
    """
    clim_S (SEAS5 ensemble mean) and clim_E (ERA5) per (init_month,
    lead_day), means over the rows of `ens` verifying in `train_years`,
    smoothed along lead day. `ens`: one row per (init_date, lead_day) with
    init_month, hydro_year, seas5 (ensemble mean), era5. Rows with a missing
    value on either side are left out of both climatologies.
    """
    tr = ens[ens["hydro_year"].isin(train_years)].dropna(subset=["seas5", "era5"])
    raw = tr.groupby(["init_month", "lead_day"])[["seas5", "era5"]].mean()
    return pd.DataFrame({"clim_S": smooth_along_lead(raw["seas5"]),
                         "clim_E": smooth_along_lead(raw["era5"])})


def prediction_frame(members: pd.DataFrame, ens: pd.DataFrame, var: str,
                     design_folds) -> pd.DataFrame:
    """
    The member frame scored for `var`: every row of `members` verifying in a
    fold's test years, with that fold's climatologies (module docstring:
    Q_raw = ERA5, Q_pred = member - clim_S + clim_E, Q_clim = clim_E).
    `members`: init_date, date, member, lead_day, init_month, hydro_year,
    `var`, era5. Each row is in exactly one fold's test years, so the folds'
    frames do not overlap.
    """
    parts = []
    for fold in design_folds:
        test = members[members["hydro_year"].isin(fold.test_years)]
        if test.empty:
            continue
        clim = fold_climatologies(ens, fold.train_years)
        idx = pd.MultiIndex.from_arrays([test["init_month"], test["lead_day"]])
        c = clim.reindex(idx)
        parts.append(pd.DataFrame({
            "init_date": test["init_date"].to_numpy(),
            "date": test["date"].to_numpy(),
            "member": test["member"].to_numpy(),
            "lead_day": test["lead_day"].to_numpy(),
            "hydro_year": test["hydro_year"].to_numpy(),
            "Q_raw": test["era5"].to_numpy(),
            "Q_pred": (test[var].to_numpy() - c["clim_S"].to_numpy()
                       + c["clim_E"].to_numpy()),
            "Q_clim": c["clim_E"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# SCORES
# ═══════════════════════════════════════════════════════════════════════

def acc_additive(obs, pred, clim) -> float:
    """ACC on additive anomalies (obs - clim, pred - clim); NaN if either
    anomaly series has zero variance or fewer than 3 values."""
    a = np.asarray(obs, float) - np.asarray(clim, float)
    b = np.asarray(pred, float) - np.asarray(clim, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def crps_per_period(agg_members: pd.DataFrame) -> pd.Series:
    """Fair CRPS of each (init_date, period) of an aggregated member frame,
    each over its own members (25 hindcast / 51 operational: grouped by
    member count, as seas5ml.metrics.crps_from_pred_df)."""
    keys = ["init_date", "period"]
    n_members = agg_members.groupby(keys)["member"].transform("nunique")
    out = []
    for _, sub in agg_members.groupby(n_members.to_numpy()):
        piv = sub.pivot_table(index=keys, columns="member", values="Q_pred",
                              aggfunc="mean")
        obs = (sub.drop_duplicates(subset=keys).set_index(keys)["Q_raw"]
                  .reindex(piv.index).to_numpy())
        out.append(pd.Series(fair_crps_per_step(obs, piv.to_numpy()),
                             index=piv.index))
    return pd.concat(out).sort_index()


def climatological_crps(periods: pd.DataFrame, reference: pd.DataFrame,
                        design_folds) -> np.ndarray:
    """
    Fair CRPS of the climatological ensemble for each row of `periods`
    (start_month, period, hydro_year, Q_raw): its members are the values of
    `reference` (start_month, period, hydro_year, value: the aggregated ERA5
    of every start) with the same (start month, period) verifying in the
    training years of the fold that scores the row's hydro year.
    """
    train_of = {hy: frozenset(f.train_years)
                for f in design_folds for hy in f.test_years}
    out = np.full(len(periods), np.nan)
    ref_groups = {k: g for k, g in reference.groupby(["start_month", "period"])}
    pos = np.arange(len(periods))
    for key, g in periods.assign(_pos=pos).groupby(["start_month", "period"]):
        ref = ref_groups.get(key)
        if ref is None:
            continue
        ry = ref["hydro_year"].to_numpy()
        rv = ref["value"].to_numpy(float)
        o = g["Q_raw"].to_numpy(float)
        mask = np.array([[y in train_of[hy] for y in ry]
                         for hy in g["hydro_year"].to_numpy()], dtype=float)
        m = mask.sum(1)
        term1 = (mask * np.abs(rv[None, :] - o[:, None])).sum(1) / m
        pair = np.abs(rv[:, None] - rv[None, :])
        spread = np.einsum("ij,jk,ik->i", mask, pair, mask) / (2 * m * (m - 1))
        val = term1 - spread
        val[m < 2] = np.nan
        out[g["_pos"].to_numpy()] = val
    return out


def _restrict(frame, contrib):
    idx = pd.MultiIndex.from_arrays([frame["init_date"], frame["lead_month"]])
    return frame[idx.isin(contrib)]


def yearly_table(per, var, res) -> pd.DataFrame:
    rows = []
    for (hy, lm), g in per.groupby(["hydro_year", "lead_month"]):
        n_starts = g["init_date"].nunique()
        if n_starts < MIN_YEAR_STARTS:
            continue
        crps, crps_clim = g["crps"].mean(), g["crps_clim"].mean()
        rows.append({"variable": var, "resolution": res,
                     "hydro_year": int(hy), "lead_month": int(lm),
                     "n_starts": n_starts, "n": len(g),
                     "acc": acc_additive(g["Q_raw"], g["Q_pred"], g["Q_clim"]),
                     "crps": crps, "crps_clim": crps_clim,
                     "crpss": (1 - crps / crps_clim
                               if np.isfinite(crps_clim) and crps_clim > 0
                               else np.nan)})
    return pd.DataFrame(rows, columns=YEARLY_COLS)


def cell_table(per, var, res, design) -> pd.DataFrame:
    min_years = MIN_CELL_YEARS[designs.check_design(design)]
    cells = {k: g for k, g in per.groupby(["start_month", "lead_month"])}
    rows = []
    for sm in range(1, 13):
        for lm in range(1, N_LEAD_MONTHS + 1):
            row = {"variable": var, "resolution": res, "start_month": sm,
                   "lead_month": lm, "n_years": 0, "n": 0, "acc": np.nan}
            g = cells.get((sm, lm))
            if g is not None:
                n_years = g["hydro_year"].nunique()
                row.update(n_years=n_years, n=len(g))
                if n_years >= min_years:
                    row["acc"] = acc_additive(g["Q_raw"], g["Q_pred"],
                                              g["Q_clim"])
            rows.append(row)
    return pd.DataFrame(rows, columns=CELL_COLS)


def bias_table(ens: pd.DataFrame, var: str, scored_years) -> pd.DataFrame:
    """Raw SEAS5 ensemble mean vs ERA5 per (start month, lead month), on the
    daily rows verifying in `scored_years` (every fold's test years)."""
    d = ens[ens["hydro_year"].isin(scored_years)].dropna(subset=["seas5", "era5"])
    d = d.assign(lead_month=lead_month_of(d["init_date"], d["date"]))
    d = d[d["lead_month"] <= N_LEAD_MONTHS]
    g = (d.groupby(["init_month", "lead_month"])
          .agg(n=("seas5", "size"), seas5=("seas5", "mean"),
               era5=("era5", "mean")).reset_index()
          .rename(columns={"init_month": "start_month"}))
    g["bias"] = g["seas5"] - g["era5"]
    g["bias_pct"] = (100 * (g["seas5"] / g["era5"] - 1)
                     if var in RELATIVE_BIAS else np.nan)
    return g.assign(variable=var)[BIAS_COLS]


# ═══════════════════════════════════════════════════════════════════════
# ONE VARIABLE
# ═══════════════════════════════════════════════════════════════════════

def score_variable(members: pd.DataFrame, era5: pd.Series, var: str, *,
                   design: str, years, resolutions=RESOLUTIONS) -> dict:
    """
    Score SEAS5's `var` against ERA5 (module docstring).

    `members`: the stage-0 rows (init_date, date normalised; member,
    lead_day, init_month, hydro_year, `var`); `era5`: the ERA5 series of
    `var` indexed by date; `years`: the catchment's fold years. Returns
    {'yearly', 'cells', 'bias', 'periods', 'stats'} (tables over every
    resolution), or None if ERA5 has no value of `var` on these days.
    """
    m = members[members["hydro_year"].isin(years)][
        ["init_date", "date", "member", "lead_day", "init_month",
         "hydro_year", var]].copy()
    m["era5"] = m["date"].map(era5).to_numpy()
    if m["era5"].isna().all():
        return None
    ens = (m.groupby(KEYS, as_index=False)
            .agg(date=("date", "first"), init_month=("init_month", "first"),
                 hydro_year=("hydro_year", "first"), seas5=(var, "mean"),
                 era5=("era5", "first")))
    design_folds = designs.folds(design, years)
    pred = prediction_frame(m, ens, var, design_folds)

    # The climatological ensemble's members: ERA5 of every start, aggregated
    # like the forecasts (the Q_pred / Q_clim columns are placeholders).
    ref_daily = ens.rename(columns={"era5": "Q_raw"}).assign(
        Q_pred=lambda d: d["Q_raw"], Q_clim=lambda d: d["Q_raw"])

    parts = {}
    for res in RESOLUTIONS:
        agg, stats = aggregate(pred, res, return_stats=True)
        parts[res] = (agg, ensemble_of(agg), stats)
    contrib = common_contributors({r: p[1] for r, p in parts.items()})

    yearly, cells, periods, stats_rows = [], [], [], []
    for res in resolutions:
        agg, ens_agg, stats = parts[res]
        own = contributing(ens_agg, res)
        per = _restrict(ens_agg, contrib).copy()
        per["start_month"] = per["init_date"].dt.month
        crps = crps_per_period(_restrict(agg, contrib))
        per["crps"] = crps.reindex(
            pd.MultiIndex.from_frame(per[["init_date", "period"]])).to_numpy()
        ref = aggregate(ref_daily, res)
        ref = pd.DataFrame({"start_month": ref["init_date"].dt.month,
                            "period": ref["period"],
                            "hydro_year": ref["hydro_year"],
                            "value": ref["Q_raw"]})
        per["crps_clim"] = climatological_crps(per, ref, design_folds)
        yearly.append(yearly_table(per, var, res))
        cells.append(cell_table(per, var, res, design))
        periods.append(per.assign(variable=var, resolution=res)[PERIOD_COLS])
        stats_rows.append({"variable": var, "resolution": res, **stats,
                           "contrib_starts": len(own),
                           "contrib_dropped": len(own) - len(own.intersection(contrib)),
                           "n_periods_scored": len(per)})
    scored_years = sorted({y for f in design_folds for y in f.test_years})
    return {"yearly": pd.concat(yearly, ignore_index=True),
            "cells": pd.concat(cells, ignore_index=True),
            "bias": bias_table(ens, var, scored_years),
            "periods": pd.concat(periods, ignore_index=True),
            "stats": pd.DataFrame(stats_rows)}
