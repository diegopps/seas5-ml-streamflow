"""
seas5ml.plots.tuningvsref – Tuning gain against the reference gap.

Produces two merged figures across catchments, one per lead-time window
(short-lead 0-30 d, long-lead 185-214 d — same windows as the default run's
boxplots):

  04a_tuningvsreference_short.pdf — lead_day 0-30
  04b_tuningvsreference_long.pdf  — lead_day 185-214

Every panel places two grey/blue reference boxes (climatology, damped
persistence) beside a pair of boxes per tuned model (rf, xgb, lstm,
pooled_lstm): the
untuned default-hyperparameter distribution (hatched) and the nested-CV
winner's distribution (solid).

Data sources, and why they differ from 04's old whole-window figure:
  Both reference boxes and the untuned/tuned pair need per-year NSE/KGE/ACC
  recomputed from daily predictions filtered to the window's lead_day range —
  fold_metrics.csv only holds whole-window (0-215 d) scores, so it cannot
  support a short/long split.

  - Tuned winner: predictions_daily.parquet under each tuned fold
    (tuned/<catchment>/<model>/fold_<id>/, the winner's test-year
    predictions) is windowed directly, exactly like the default run's
    boxplots.
  - Untuned: the default run of the same model (default configuration,
    trained on the same fold years and early-stopping split as the tuned
    winner), read from its predictions and windowed like every other box.
  - Reference models (climatology, damped_persistence): the default run
    writes both whole-window fold_metrics.csv AND predictions_daily.parquet,
    so these can be windowed the same way its own boxplots already do.

  Every box except the tuned one is therefore read from the default run's
  results (default/<catchment>/<model>/predictions_daily.parquet) via the
  same fold_metrics_in_window() its boxplots use, and the tuned box from the
  tuned run's. Both are windowed identically; nothing here is pooled against
  something windowed, which would be a misleading comparison.

Why paired boxplots rather than a point-estimate comparison:
  Fold-to-fold variation in every metric here is large (NSE and KGE routinely
  span 1-2 units across hydro years; ACC swings from strongly negative to
  strongly positive). A plot of mean or median values alone would either
  hide that variation or, worse, make a hyperparameter-tuning shift that is
  tiny relative to the real spread look decisive. Showing full distributions
  side by side lets the untuned/tuned overlap speak for itself: where the two
  boxes for a model are indistinguishable, that is the finding, not a
  limitation of the plot.

Shared y-limits:
  Every panel uses the default boxplots' hardcoded boxplots.Y_LIM (-2, 1), so
  this figure reads directly against them. A handful of near-zero-variance
  dry-year folds can send NSE/KGE to -100; those fall below the floor and are
  counted with the same off-axis marker the default boxplots draw, rather
  than being allowed to blow out the shared axis.

The ACC column carries the same day-of-year reference line as the default
boxplots (plots.windows.doy_reference_acc): median across years, scored with the
same calc_acc on the default run's climatology rows, so both figures show the
identical value.

CRPS appears as CRPSS against climatology, for the reasons the default
boxplots give: raw CRPS carries discharge units and is lower-is-better, so it
would neither share an axis across catchments nor read in the same direction
as the panels beside it. The per-year CRPS behind it is recomputed per window
from predictions_members.parquet — the tuned run's own file for the tuned
box, the default run's for the untuned and reference boxes, matching where
each of those box kinds already takes its other metrics from.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from . import common as pl
from ..metrics import calc_nse, calc_kge, calc_acc
from ..metrics import crps_from_pred_df
# Legacy window scoring (plots.windows) until this figure moves to the
# score tables at the tuned stage; drawing helpers from the boxplots.
from .windows import (fold_metrics_in_window, window_crps, window_crpss,
                      CRPSS_REFERENCE, doy_reference_acc)
from .boxplots import (DOY_REF_COLOR, DOY_REF_LABEL, Y_LIM,
                       draw_below_floor_counts)

TUNED_MODELS = ["rf", "xgb", "lstm", "pooled_lstm"]
REF_MODELS = ["climatology", "damped_persistence"]

METRICS = [
    {"key": "nse", "title": "NSE"},
    {"key": "kge", "title": "KGE"},
    {"key": "acc", "title": "ACC (per year)"},
    {"key": "crpss", "title": "CRPSS"},   # reference (climatology) stated in the caption
]

# Built at print size (pl.FIG_WIDTH_IN wide), matching the default run's boxplots.
# Eight boxes per panel leave no room for model names under them at this
# width, so boxes are identified by colour, hatch and the shared legend only.
FIG_HEIGHT_ROW = 1.7    # inches per catchment row
LEGEND_HEIGHT = 0.8     # inches reserved below the grid for the legend
BOX_WIDTH = 0.7
BOX_ALPHA_TUNED = 0.85
BOX_ALPHA_UNTUNED = 0.55
HATCH_UNTUNED = "////"
HATCH_LW = 0.4
GAP_BEFORE_MODELS = 0.5   # extra x-gap between reference boxes and model boxes


def _tuned_crpss(catchment, model, lo, hi, min_rows=30):
    """
    Nested-CV winner's per-fold CRPSS, windowed to lead_day in [lo, hi].

    The winner's CRPS comes from the tuned run's own per-member predictions; the
    climatology reference comes from the default run, which is where the reference
    models are scored for every other box in this figure.
    """
    dfm = pl.load_hptune_member_predictions(catchment, model)
    if dfm is None:
        return None
    ref = window_crps(catchment, CRPSS_REFERENCE, lo, hi, min_rows)
    if ref is None:
        return None
    ref = ref[np.isfinite(ref) & (ref > 0)]
    if len(ref) == 0:
        return None

    sub = (dfm[(dfm['lead_day'] >= lo) & (dfm['lead_day'] <= hi)]
             .dropna(subset=['Q_raw', 'Q_pred']))
    vals = {}
    for hy, g in sub.groupby('hydro_year'):
        if hy not in ref.index:
            continue
        if g[['init_date', 'date']].drop_duplicates().shape[0] < min_rows:
            continue
        vals[hy] = 1.0 - crps_from_pred_df(g) / ref.loc[hy]
    vals = pd.Series(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals if len(vals) else None


def _tuned_values(catchment, model, metric, lo, hi):
    """Nested-CV winner's fold-level values for one model/metric, windowed
    to lead_day in [lo, hi], from the tuned run's own predictions. A Series
    indexed by hydro_year, so it can be paired against a reference."""
    if metric == 'crpss':
        return _tuned_crpss(catchment, model, lo, hi)
    df = pl.load_hptune_predictions(catchment, model)
    if df is None:
        return None
    sub = (df[(df['lead_day'] >= lo) & (df['lead_day'] <= hi)]
             .dropna(subset=['Q_raw', 'Q_pred', 'Q_clim']))
    if len(sub) == 0:
        return None
    calc = {'nse': calc_nse, 'kge': calc_kge}.get(metric)
    vals = {}
    for hy, g in sub.groupby('hydro_year'):
        if len(g) < 30:
            continue
        vals[hy] = (calc_acc(g['Q_raw'].values, g['Q_pred'].values, g['Q_clim'].values)
                    if metric == 'acc' else calc(g['Q_raw'].values, g['Q_pred'].values))
    vals = pd.Series(vals, dtype=float).dropna()
    return vals if len(vals) else None


def _windowed_values(catchment, model, metric, lo, hi):
    """Fold-level values for a default-run model (used both as reference boxes
    and as the untuned-baseline stand-in), windowed to lead_day in [lo, hi].
    A Series indexed by hydro_year."""
    if metric == 'crpss':
        # CRPSS is not part of fold_metrics_in_window's per-model frame: it
        # needs the catchment's climatology reference, so it is computed for
        # every model at once.
        s = window_crpss(catchment, lo, hi).get(model)
        if s is None:
            return None
        v = s.astype(float).dropna()
        return v if len(v) else None
    d = fold_metrics_in_window(catchment, model, lo, hi)
    if d is None or metric not in d.columns:
        return None
    v = d.set_index('hydro_year')[metric].astype(float).dropna()
    return v if len(v) else None


def _sig_vs_reference(catchments, lo, hi) -> dict:
    """(catchment, metric, model) -> +1 / -1 / 0 for each TUNED winner against
    damped persistence, paired over hydro years; FDR-adjusted per metric
    across the whole figure. Untuned boxes are not tested: they are the
    default-run models, already tested in the default boxplots."""
    out = {}
    for key in pl.SIG_METRICS:
        pairs = {}
        for catchment in catchments:
            ref = _windowed_values(catchment, pl.SIG_REFERENCE, key, lo, hi)
            if ref is None:
                continue
            for model in TUNED_MODELS:
                v = _tuned_values(catchment, model, key, lo, hi)
                if v is not None:
                    pairs[(catchment, key, model)] = (v, ref)
        out.update(pl.paired_tests_vs_reference(pairs))
    return out


def _build_positions():
    """x-position and (kind, model) for every possible box in a panel:
    reference boxes, a gap, then each tuned model's untuned/tuned pair."""
    xs, kinds = [], []
    x = 0.0
    for ref in REF_MODELS:
        xs.append(x); kinds.append(("ref", ref)); x += 1.0
    x += GAP_BEFORE_MODELS
    for model in TUNED_MODELS:
        xs.append(x); kinds.append(("untuned", model)); x += 0.85
        xs.append(x); kinds.append(("tuned", model)); x += 1.0
    return xs, kinds


def _panel_boxes(catchment, metric_key, xs, kinds, lo, hi):
    """Collect (data, position, color, hatch) for every box present in one
    (catchment, metric) panel, windowed to lead_day in [lo, hi]."""
    data, pos, colors, hatches = [], [], [], []
    for x, (kind, model) in zip(xs, kinds):
        if kind == "ref":
            v = _windowed_values(catchment, model, metric_key, lo, hi)
            if v is None:
                continue
            data.append(v.values); pos.append(x)
            colors.append(pl.MODEL_COLORS[model]); hatches.append(None)
        elif kind == "untuned":
            v = _windowed_values(catchment, model, metric_key, lo, hi)
            if v is None:
                continue
            data.append(v.values); pos.append(x)
            colors.append(pl.MODEL_COLORS[model]); hatches.append(HATCH_UNTUNED)
        else:
            v = _tuned_values(catchment, model, metric_key, lo, hi)
            if v is None:
                continue
            data.append(v.values); pos.append(x)
            colors.append(pl.MODEL_COLORS[model]); hatches.append(None)
    return data, pos, colors, hatches


def plot_tuning_vs_reference(catchments: list, lo: int, hi: int,
                             title: str, out_name: str):
    pl.apply_plot_style()

    catchments = pl.order_catchments(catchments)
    xs, kinds = _build_positions()

    doy_ref = {}
    for catchment in catchments:
        acc_doy = doy_reference_acc(catchment, lo, hi)
        if acc_doy is not None and acc_doy.notna().any():
            doy_ref[catchment] = float(acc_doy.median())
    sig = _sig_vs_reference(catchments, lo, hi)

    n_rows = len(catchments)
    fig_h = FIG_HEIGHT_ROW * n_rows + LEGEND_HEIGHT
    # Hatch stroke width is a global rcParam, read at draw time; scope it to
    # this figure so it does not leak into later figures in the same run.
    with plt.rc_context({"hatch.linewidth": HATCH_LW}):
        fig, axes = plt.subplots(n_rows, len(METRICS), sharey=True,
                                 figsize=(pl.FIG_WIDTH_IN, fig_h), squeeze=False)
        _draw_panels(fig, axes, catchments, xs, kinds, lo, hi, doy_ref, sig)
        _draw_legend(fig, doy_ref)

        top = 1.0
        if pl.SHOW_SUPTITLE:
            fig.suptitle(title, fontsize=pl.FONT_SUPTITLE, fontweight="bold")
            top = 1.0 - 0.35 / fig_h
        plt.tight_layout(rect=[0, LEGEND_HEIGHT / fig_h, 1, top],
                         w_pad=0.6, h_pad=0.8)

        # After tight_layout: axes positions are only final once layout has
        # run, so the separator must be placed here to land in the row gutter.
        pl.draw_regime_separator(fig, axes, catchments)

        pl.savefig(fig, f"{pl.PLOT_DIR}/{out_name}")
    plt.close()


def _draw_panels(fig, axes, catchments, xs, kinds, lo, hi, doy_ref, sig):
    """Fill every (catchment, metric) panel of the grid."""
    for r, catchment in enumerate(catchments):
        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            key = metric["key"]
            data, pos, colors, hatches = _panel_boxes(catchment, key, xs, kinds, lo, hi)

            if not data:
                ax.axis("off")
                continue

            bp = ax.boxplot(
                data, positions=pos, widths=BOX_WIDTH, patch_artist=True,
                showfliers=True,
                medianprops=dict(color="#222222", linewidth=0.9),
                flierprops=dict(marker="o", markersize=1.5,
                                markerfacecolor="#888888",
                                markeredgecolor="none", alpha=0.5),
                whiskerprops=dict(linewidth=0.6, color="#333333"),
                capprops=dict(linewidth=0.6, color="#333333"),
                boxprops=dict(linewidth=0.6),
                zorder=3)
            for patch, col, hatch in zip(bp["boxes"], colors, hatches):
                patch.set_facecolor(col)
                patch.set_alpha(BOX_ALPHA_UNTUNED if hatch else BOX_ALPHA_TUNED)
                patch.set_edgecolor("#333333")
                if hatch:
                    patch.set_hatch(hatch)

            ax.set_ylim(*Y_LIM)
            ax.axhline(0.0, color="#444444", lw=0.6, alpha=0.6, zorder=2)
            if key == "acc" and catchment in doy_ref:
                ax.axhline(doy_ref[catchment], color=DOY_REF_COLOR, lw=0.9,
                           ls="--", zorder=2.5)
            draw_below_floor_counts(ax, pos, data)

            for x, (kind, model) in zip(xs, kinds):
                if kind == "tuned" and x in pos:
                    pl.draw_sig_marker(ax, x, sig.get((catchment, key, model)))

            if r == 0:
                ax.set_title(metric["title"], fontweight="bold", pad=7)
            if c == 0:
                # Letter prefixed into the row label rather than floated above
                # the panel, where it would collide with the metric titles.
                ax.set_ylabel(f"{pl.catchment_letter(r)})  "
                              f"{pl.catchment_display(catchment)}",
                              fontweight="bold", fontsize=pl.FONT_AXIS_LABEL,
                              labelpad=3)
            else:
                ax.tick_params(axis="y", labelleft=False)

            # Every box slot is ticked, not only the drawn ones, so the gap
            # left by a missing box (e.g. climatology has no ACC) stays put.
            ax.set_xticks(xs)
            ax.set_xticklabels([])
            ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
            pl.style_axes(ax)
            ax.tick_params(axis="x", length=0)


def _draw_legend(fig, doy_ref):
    """
    One legend below the grid, in three columns: models, box style / reference
    line, significance symbols. Columns are padded with blank entries so each
    group keeps a column of its own (the legend fills column by column).
    """
    def blank():
        return mpatches.Patch(visible=False, label="")

    model_handles = [
        mpatches.Patch(facecolor=pl.MODEL_COLORS[m], alpha=BOX_ALPHA_TUNED,
                      edgecolor="#333333", lw=0.5, label=pl.MODEL_LABELS[m])
        for m in REF_MODELS + TUNED_MODELS]
    style_handles = [
        mpatches.Patch(facecolor="#999999", alpha=BOX_ALPHA_UNTUNED,
                      hatch=HATCH_UNTUNED, edgecolor="#333333", lw=0.5,
                      label="Untuned (Stage 1 default)"),
        mpatches.Patch(facecolor="#999999", alpha=BOX_ALPHA_TUNED,
                      edgecolor="#333333", lw=0.5,
                      label="Tuned (nested-CV winner)"),
    ]
    if doy_ref:
        style_handles.append(mlines.Line2D([], [], color=DOY_REF_COLOR, lw=0.9,
                                           ls="--", label=DOY_REF_LABEL))
    sig_handles = pl.sig_legend_handles()

    n = max(len(model_handles), len(style_handles), len(sig_handles))
    handles = []
    for group in (model_handles, style_handles, sig_handles):
        handles += group + [blank() for _ in range(n - len(group))]

    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=pl.FONT_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, handleheight=0.9, columnspacing=1.2,
               labelspacing=0.3)


def run(catchments: list, lo: int, hi: int, title: str, out_name: str):
    """Build the tuning-vs-reference figure for the given catchments, using
    only lead_day in [lo, hi]."""
    available = [c for c in catchments
                 if any(pl.load_tuned_metrics(c, m) is not None
                       for m in TUNED_MODELS)]
    if not available:
        print("  No tuned results found — skipping tuning-vs-reference figure")
        return
    plot_tuning_vs_reference(available, lo, hi, title, out_name)
