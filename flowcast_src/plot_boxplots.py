"""
flowcast_src.plot_boxplots – Model comparison boxplots (NSE, KGE, ACC, CRPSS).

Produces two merged figures across catchments, one per lead-time window:

  01a_boxplotcomparison_short.pdf — lead_day 0-30
  01b_boxplotcomparison_long.pdf  — lead_day 185-214

Both windows are 30 days wide on purpose (equal sample size per fold) so box
widths are comparable between the two figures. Per-fold NSE/KGE/ACC are
recomputed directly from predictions_daily.parquet, filtered to the window's
lead_day range — fold_metrics.csv pools all 215 lead days into one score and
has no lead breakdown, so it cannot be used as the data source here.

CRPS appears as a skill score against climatology, not raw. Two reasons: raw
CRPS carries the units of discharge, so it cannot share one hardcoded axis
across catchments of different size; and it is lower-is-better, which would
invert the reading of the panel beside it. CRPSS = 1 - CRPS/CRPS_clim is
dimensionless, capped at 1, and higher-is-better like the other three.

Per-fold CRPS within a window needs the per-member predictions, read from
predictions_members.parquet. Deterministic baselines write no such file, but
their members are identical, so their CRPS is exactly MAE and is taken from
the ensemble-mean file instead. Climatology is the reference and would be
identically zero, so it is left out of the CRPSS panel.

The ACC column also carries the day-of-year reference as a dashed horizontal
line at its median across folds (see doy_reference_acc). It is a forecast
that knows nothing but the date, resolving the seasonal cycle more finely than
the 31-day climatology, so ACC above that line rather than above zero is skill
beyond the seasonal cycle. It is scored with the same calc_acc, against the
same stored climatology, on the same rows as the reference boxes.

Statistical notes:
  - Boxplots (median + IQR) are used instead of mean +/- SE bars, since
    fold-level metrics are often skewed.
  - Y-axis limits are hardcoded to (-0.5, 1.0) for every panel in both
    figures, rather than fit to the data. Per-fold NSE is unbounded below on
    ephemeral rivers (it normalises by that year's own flow variance, which
    can be tiny in a low-flow year), so a handful of folds routinely fall far
    below any reasonable axis. Folds below the floor are counted and marked
    with a small annotation at the bottom of their box instead of being
    allowed to blow out the shared axis.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from . import plot_lib as pl
from .metrics import calc_nse, calc_kge, calc_acc, crps_from_pred_df

# Built at print size (pl.FIG_WIDTH_IN wide). Seven boxes per panel leave no
# room for model names under them at this width, so boxes are identified by
# colour and the shared legend only.
BAR_FIG_HEIGHT_ROW = 1.7    # inches per catchment row
BAR_LEGEND_HEIGHT  = 0.75   # inches reserved below the grid for the legend
BOX_WIDTH          = 0.7
BOX_ALPHA          = 0.85
ANNOT_OFFSET_FRAC  = 0.025

Y_LIM = (-2.0, 1.0)

METRICS = [
    {"key": "nse", "title": "NSE"},
    {"key": "kge", "title": "KGE"},
    {"key": "acc", "title": "ACC (per year)"},
    {"key": "crpss", "title": "CRPSS"},   # reference (climatology) stated in the caption
]

# Baselines are identical across members, so their CRPS is exactly MAE and is
# read from the ensemble-mean file. Everything else needs the member frame.
DETERMINISTIC_MODELS = {"climatology", "persistence", "damped_persistence"}
CRPSS_REFERENCE = "climatology"

# Day-of-year reference line in the ACC column. Its rows (dates, folds, Q_raw,
# Q_clim) are taken from the climatology baseline's predictions, so it is
# scored on exactly the rows the reference boxes are.
DOY_REF_SCAFFOLD = "climatology"
DOY_REF_COLOR    = "#8B1E3F"
DOY_REF_LABEL    = "Day-of-year reference (median)"


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
    calendar years from clim_start_year onward, skip_years excluded."""
    # Imported here, not at module level: flowcast_src otherwise receives its
    # configuration from the calling driver. catchments.py sits at the
    # repository root, which is on sys.path when 01_plot_exploration.py runs.
    from catchments import CATCHMENTS, SHARED
    from .data import load_obs_discharge
    cfg = CATCHMENTS[catchment]
    obs = load_obs_discharge(cfg['discharge_csv'])
    keep = ((obs.index.year >= SHARED['clim_start_year'])
            & ~obs.index.year.isin(cfg['skip_years']))
    return obs[keep]


@lru_cache(maxsize=None)
def doy_reference_acc(catchment: str, lo: int, hi: int, min_rows=30):
    """
    Per-fold ACC of the day-of-year reference, restricted to lead_day in
    [lo, hi], indexed by hydro_year.

    For each fold the forecast on every date is the mean of log1p(Q) on that
    day of year over the fold's training hydro years, left unsmoothed, and
    back-transformed with expm1 so calc_acc sees it exactly as it sees every
    other forecast. Because the climatology is smoothed over 31 days, it
    flattens short features of the seasonal cycle such as a sharp freshet;
    this forecast resolves them, and so earns a positive ACC without any
    information about the year being forecast. It is the analytic limit of a
    model trained on day-of-year features alone.
    """
    scaffold = pl.load_predictions(catchment, DOY_REF_SCAFFOLD)
    if scaffold is None:
        return None
    obs = _climatology_period_obs(catchment)
    obs_hy = obs.index.year + (obs.index.month >= 10).astype(int)

    sub = (scaffold[(scaffold['lead_day'] >= lo) & (scaffold['lead_day'] <= hi)]
             .dropna(subset=['Q_raw', 'Q_clim']))
    rows = {}
    for hy, g in sub.groupby('hydro_year'):
        if len(g) < min_rows:
            continue
        train = obs[obs_hy != hy]
        doy_mean = (np.log1p(train).groupby(train.index.dayofyear).mean()
                      .reindex(range(1, 367)).ffill().bfill())
        q_pred = np.expm1(g['date'].dt.dayofyear.map(doy_mean).values)
        rows[hy] = calc_acc(g['Q_raw'].values, q_pred, g['Q_clim'].values)
    return pd.Series(rows, dtype=float) if rows else None


def load_window_fold_metrics(catchment: str, lo: int, hi: int) -> dict:
    """fold_metrics_in_window for every model in MODEL_ORDER available for
    this catchment, with per-fold CRPSS joined on. Missing models are simply
    absent from the dict; a model without CRPSS simply lacks that column."""
    crpss = window_crpss(catchment, lo, hi)
    out = {}
    for model in pl.MODEL_ORDER:
        df = fold_metrics_in_window(catchment, model, lo, hi)
        if df is None or len(df) == 0:
            continue
        s = crpss.get(model)
        if s is not None and len(s) > 0:
            df = df.merge(
                s.rename('crpss').rename_axis('hydro_year').reset_index(),
                on='hydro_year', how='left')
        out[model] = df
    return out


def _sig_vs_reference(catchments: list, fold_data: dict) -> dict:
    """
    (catchment, metric, model) -> +1 / -1 / 0 for every learned model against
    damped persistence, paired over hydro years; FDR-adjusted per metric
    across the whole figure (see plot_lib.paired_tests_vs_reference).
    """
    out = {}
    for key in pl.SIG_METRICS:
        pairs = {}
        for catchment in catchments:
            models_here = fold_data[catchment]
            ref = models_here.get(pl.SIG_REFERENCE)
            if ref is None or key not in ref.columns:
                continue
            ref_s = ref.set_index('hydro_year')[key].dropna()
            for m, df_m in models_here.items():
                if m in DETERMINISTIC_MODELS or key not in df_m.columns:
                    continue
                pairs[(catchment, key, m)] = (
                    df_m.set_index('hydro_year')[key].dropna(), ref_s)
        out.update(pl.paired_tests_vs_reference(pairs))
    return out


def draw_below_floor_counts(ax, xs, data):
    """
    Values below the hardcoded Y_LIM floor are clipped off-axis; mark each
    affected box with its off-axis count instead of letting a single bad fold
    (e.g. a low-flow year on an ephemeral river) blow out the shared axis.
    """
    y_annot = Y_LIM[0] + (Y_LIM[1] - Y_LIM[0]) * ANNOT_OFFSET_FRAC
    for x, vals in zip(xs, data):
        n_below = int(np.sum(np.asarray(vals) < Y_LIM[0]))
        if n_below > 0:
            ax.plot(x, y_annot, marker='v', markersize=2.5,
                    color='#666666', zorder=4, clip_on=False)
            ax.text(x, y_annot + (Y_LIM[1] - Y_LIM[0]) * 0.04,
                    str(n_below), ha='center', va='bottom',
                    fontsize=pl.FONT_ANNOT - 1, color='#666666', zorder=4)


def plot_bar_chart(catchments: list, fold_data: dict, title: str, out_name: str,
                   doy_ref: dict = None):
    """doy_ref maps catchment -> median day-of-year reference ACC, drawn as a
    dashed line in the ACC column; catchments without an entry get no line."""
    pl.apply_plot_style()
    doy_ref = doy_ref or {}
    sig = _sig_vs_reference(catchments, fold_data)

    n_rows = len(catchments)
    n_cols = len(METRICS)
    fig_h = BAR_FIG_HEIGHT_ROW * n_rows + BAR_LEGEND_HEIGHT
    fig, axes = plt.subplots(n_rows, n_cols, sharey=True,
                             figsize=(pl.FIG_WIDTH_IN, fig_h), squeeze=False)

    for row, catchment in enumerate(catchments):
        models_here = fold_data[catchment]
        letter  = pl.catchment_letter(row)
        display = pl.catchment_display(catchment)

        present_models = [m for m in pl.MODEL_ORDER if m in models_here]

        for col, metric in enumerate(METRICS):
            ax  = axes[row, col]
            key = metric["key"]

            box_data  = []
            box_years = {}
            for m in present_models:
                df_m = models_here[m]
                if key not in df_m.columns:
                    continue
                vals = df_m[key].dropna()
                if len(vals) == 0:
                    continue
                box_data.append(vals.values)
                box_years[m] = (vals.values,
                                df_m.loc[vals.index, 'hydro_year'].values)

            x_all = np.arange(len(present_models))
            plotted_models = [m for m in present_models if m in box_years]
            x_plotted = [present_models.index(m) for m in plotted_models]
            data_plotted = [box_years[m][0] for m in plotted_models]

            if data_plotted:
                bp = ax.boxplot(
                    data_plotted, positions=x_plotted, widths=BOX_WIDTH,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color='#222222', linewidth=0.9),
                    flierprops=dict(marker='o', markersize=1.5,
                                    markerfacecolor='#888888',
                                    markeredgecolor='none', alpha=0.5),
                    whiskerprops=dict(linewidth=0.6, color='#333333'),
                    capprops=dict(linewidth=0.6, color='#333333'),
                    boxprops=dict(linewidth=0.6),
                    zorder=3,
                )
                for patch, m in zip(bp['boxes'], plotted_models):
                    patch.set_facecolor(pl.MODEL_COLORS[m])
                    patch.set_alpha(BOX_ALPHA)
                    patch.set_edgecolor('#333333')

            ax.set_ylim(*Y_LIM)
            ax.axhline(0.0, color='#444444', lw=0.6, alpha=0.6, zorder=2)
            if key == 'acc' and catchment in doy_ref:
                ax.axhline(doy_ref[catchment], color=DOY_REF_COLOR, lw=0.9,
                           ls='--', zorder=2.5)

            draw_below_floor_counts(ax, x_plotted, data_plotted)

            for x, m in zip(x_plotted, plotted_models):
                pl.draw_sig_marker(ax, x, sig.get((catchment, key, m)))

            if row == 0:
                ax.set_title(metric["title"], fontweight='bold', pad=7)

            if col == 0:
                # Letter prefixed into the row label rather than floated above
                # the panel, where it would collide with the metric titles.
                ax.set_ylabel(f"{letter})  {display}", fontweight='bold',
                             fontsize=pl.FONT_AXIS_LABEL, labelpad=3)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False)

            ax.set_xticks(x_all)
            ax.set_xticklabels([])
            ax.set_xlim(-0.6, len(present_models) - 0.4)
            pl.style_axes(ax)
            ax.tick_params(axis='x', length=0)

    legend_handles = [
        mpatches.Patch(color=pl.MODEL_COLORS[m], alpha=BOX_ALPHA, label=pl.MODEL_LABELS[m])
        for m in pl.MODEL_ORDER
        if any(m in fold_data[c] for c in catchments)
    ]
    if any(c in doy_ref for c in catchments):
        legend_handles.append(mlines.Line2D([], [], color=DOY_REF_COLOR, lw=0.9,
                                            ls='--', label=DOY_REF_LABEL))
    legend_handles += pl.sig_legend_handles()
    fig.legend(handles=legend_handles, loc='lower center',
              ncol=3, fontsize=pl.FONT_LEGEND,
              frameon=False, bbox_to_anchor=(0.5, 0.0),
              handlelength=1.8, handleheight=0.9, columnspacing=1.2,
              labelspacing=0.3)

    top = 1.0
    if pl.SHOW_SUPTITLE:
        fig.suptitle(title, fontsize=pl.FONT_SUPTITLE, fontweight='bold')
        top = 1.0 - 0.35 / fig_h
    plt.tight_layout(rect=[0, BAR_LEGEND_HEIGHT / fig_h, 1, top],
                     w_pad=0.6, h_pad=0.8)

    # After tight_layout: axes positions are only final once layout has run,
    # so the separator must be placed here to land in the row gutter.
    pl.draw_regime_separator(fig, axes, catchments)

    pl.savefig(fig, f"{pl.PLOT_DIR}/{out_name}")
    plt.close()


def run(catchments: list, lo: int, hi: int, title: str, out_name: str):
    """Build the boxplot comparison figure for the given catchments, using
    only lead_day in [lo, hi]."""
    catchments = pl.order_catchments(catchments)
    fold_data = {}
    doy_ref = {}
    for catchment in catchments:
        models_here = load_window_fold_metrics(catchment, lo, hi)
        if models_here:
            fold_data[catchment] = models_here
        acc_doy = doy_reference_acc(catchment, lo, hi)
        if acc_doy is not None and acc_doy.notna().any():
            doy_ref[catchment] = float(acc_doy.median())
            print(f"  {catchment}: day-of-year reference median ACC "
                  f"{doy_ref[catchment]:+.3f} ({acc_doy.notna().sum()} folds)")

    box_catchments = [c for c in catchments if c in fold_data]
    if box_catchments:
        plot_bar_chart(box_catchments, fold_data, title, out_name,
                       doy_ref=doy_ref)
    else:
        print(f"  Skipping boxplot figure {out_name} (no predictions found)")
