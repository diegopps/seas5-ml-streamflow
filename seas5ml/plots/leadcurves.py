"""
seas5ml.plots.leadcurves – Skill by lead month, relative to damped persistence.

The core figure of the verification layer: how a model's skill over damped
persistence evolves with lead time, at daily, weekly and monthly resolution
side by side. One figure per metric:

  01_leadcurves_acc.pdf    ACC  (anomaly correlation, per year)
  01_leadcurves_crpss.pdf  CRPSS (against deterministic climatology, per year)

Layout: rows = catchments (regime order), columns = resolution (daily /
weekly / monthly), one shared y axis. x = lead month 1-7 (lead month 1 is the
calendar month the forecast starts in). Data: the score tables
(plots.common.load_yearly), which score every model on the same years and
starts at every resolution, so the three columns are directly comparable.

What is plotted, per (catchment, resolution, lead month, model):
  - the median over hydro years of the per-year difference
    score(model) - score(damped persistence), paired by year
    (plots.common.paired_with_reference). Damped persistence is the
    strongest cheap reference, so it is the zero line;
  - the interquartile range of those yearly differences, as a thin vertical
    bar, dodged per model so the bars of one lead month do not overlap.
Lines:
  - learned models: solid, with markers; the marker is filled where the
    difference from damped persistence is significant, open where not;
  - persistence: solid, no markers (a reference, not tested);
  - day-of-year reference: dashed, daily column only
    (plots.common.DOY_REF_RESOLUTIONS);
  - climatology: grey, CRPSS figure only (its CRPSS is 0 by definition, so
    its curve is minus damped persistence's CRPSS: where damped persistence
    beats the no-skill floor). Its ACC is undefined.

Significance: the boxplots' test (plots.common.paired_bootstrap_diff, the
paired bootstrap of the MEAN yearly difference, resampling years), for every
learned model at every (catchment, resolution, lead month) point of the
figure, Benjamini-Hochberg over all of them together (one figure = one
metric). The figure plots the median and tests the mean: stated in the
caption. A point needs at least MIN_YEARS paired years to be drawn or tested.

Every plotted value is also written to <figure>.csv beside the figure
(median, quartiles, n years, mean difference, p, q, significant).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from . import common as pl

REFERENCE = pl.SIG_REFERENCE                 # damped_persistence
TESTED = ["ridge", "rf", "xgb", "lstm", "pooled_lstm"]
MIN_YEARS = 3                                # = paired_bootstrap_diff's floor

METRICS = {
    "acc":   {"label": "ACC", "lines": ["persistence", pl.DOY_REFERENCE] + TESTED},
    "crpss": {"label": "CRPSS", "lines": ["climatology", "persistence",
                                          pl.DOY_REFERENCE] + TESTED},
}

ROW_H       = 1.25    # inches per catchment row
LEGEND_H    = 0.65    # inches below the grid for the legend
DODGE       = 0.42    # total x spread of one lead month's bars (lead months 1 apart)
MARKER_SIZE = 2.6
LINE_W      = 0.9
BAR_W       = 0.6
Y_CLIP      = (-1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def curve_table(catchments: list, metric: str, variant: str) -> pd.DataFrame:
    """
    One row per (catchment, resolution, model, lead month) plotted:
    n_years, median, q25, q75 of the paired yearly differences, mean
    difference, and (tested models) p, q, sig (+1 / -1 / 0).
    """
    rows = []
    for c in catchments:
        ref = pl.load_yearly(c, REFERENCE, variant)
        if ref is None:
            print(f"  {c}: no {REFERENCE} scores -- skipped")
            continue
        for m in METRICS[metric]["lines"]:
            y = pl.load_yearly(c, m, variant)
            if y is None:
                continue
            pairs = pl.paired_with_reference(y, ref, metric)
            for (res, lm), g in pairs.groupby(["resolution", "lead_month"]):
                if m == pl.DOY_REFERENCE and res not in pl.DOY_REF_RESOLUTIONS:
                    continue
                if len(g) < MIN_YEARS:
                    continue
                d = g["diff"].to_numpy()
                row = {"catchment": c, "resolution": res, "model": m,
                       "lead_month": int(lm), "n_years": len(g),
                       "median": float(np.median(d)),
                       "q25": float(np.percentile(d, 25)),
                       "q75": float(np.percentile(d, 75)),
                       "mean": float(d.mean()), "p": np.nan}
                if m in TESTED:
                    yrs = g["hydro_year"].to_numpy()
                    row["p"] = pl.paired_bootstrap_diff(
                        g["model"].to_numpy(), yrs,
                        g["ref"].to_numpy(), yrs)["p"]
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    tested = df["p"].notna()
    df["q"] = np.nan
    df.loc[tested, "q"] = pl.bh_adjust(df.loc[tested, "p"].to_numpy())
    df["sig"] = np.where(tested & (df["q"] < pl.ALPHA_LEVEL),
                         np.sign(df["mean"]), 0).astype(int)
    df.loc[~tested, "sig"] = 0
    return df


# ═══════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════

def _style(model):
    """(colour, linestyle, has markers) of one line."""
    if model == pl.DOY_REFERENCE:
        return pl.DOY_REF_COLOR, "--", False
    return pl.MODEL_COLORS[model], "-", model in TESTED


def _ylim(df):
    lo = min(df["q25"].min(), df["median"].min(), 0.0)
    hi = max(df["q75"].max(), df["median"].max(), 0.0)
    pad = 0.06 * (hi - lo)
    return max(Y_CLIP[0], lo - pad), min(Y_CLIP[1], hi + pad)


def plot_curves(catchments: list, df: pd.DataFrame, metric: str,
                out_name: str):
    pl.apply_plot_style()
    lines = [m for m in METRICS[metric]["lines"] if m in set(df["model"])]
    offsets = dict(zip(lines, np.linspace(-DODGE / 2, DODGE / 2, len(lines))))
    n_rows, n_cols = len(catchments), len(pl.RESOLUTIONS)
    fig_h = ROW_H * n_rows + LEGEND_H
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True,
                             figsize=(pl.FIG_WIDTH_IN, fig_h), squeeze=False)
    ylim = _ylim(df)

    for r, c in enumerate(catchments):
        for k, res in enumerate(pl.RESOLUTIONS):
            ax = axes[r, k]
            ax.axhline(0.0, color=pl.MODEL_COLORS[REFERENCE], lw=0.8, zorder=2)
            sub = df[(df["catchment"] == c) & (df["resolution"] == res)]
            for m in lines:
                g = sub[sub["model"] == m].sort_values("lead_month")
                if g.empty:
                    continue
                color, ls, markers = _style(m)
                x = g["lead_month"].to_numpy() + offsets[m]
                ax.vlines(x, g["q25"], g["q75"], color=color, lw=BAR_W,
                          alpha=0.55, zorder=3)
                ax.plot(x, g["median"], color=color, ls=ls, lw=LINE_W,
                        zorder=4)
                if markers:
                    sig = g["sig"].to_numpy() != 0
                    ax.plot(x[sig], g["median"].to_numpy()[sig], "o",
                            ms=MARKER_SIZE, color=color, mec="#222222",
                            mew=0.4, zorder=5)
                    ax.plot(x[~sig], g["median"].to_numpy()[~sig], "o",
                            ms=MARKER_SIZE, mfc="white", mec=color,
                            mew=0.7, zorder=5)
            ax.set_ylim(*ylim)
            ax.set_xlim(0.5, 7.5)
            ax.set_xticks(pl.LEAD_MONTHS)
            pl.style_axes(ax)
            if r == 0:
                ax.set_title(pl.RESOLUTION_LABELS[res], fontweight="bold",
                             pad=4)
            if k == 0:
                ax.set_ylabel(pl.catchment_row_label(r, c), fontweight="bold",
                              fontsize=pl.FONT_AXIS_LABEL, labelpad=3,
                              linespacing=0.95)
            if r == n_rows - 1:
                ax.set_xlabel("Lead month")

    label = METRICS[metric]["label"]
    handles = [mlines.Line2D([], [], color=pl.MODEL_COLORS[REFERENCE], lw=0.8,
                             label=f"{pl.MODEL_LABELS[REFERENCE]} (= 0)")]
    for m in lines:
        color, ls, markers = _style(m)
        handles.append(mlines.Line2D([], [], color=color, ls=ls, lw=LINE_W,
                                     marker="o" if markers else None,
                                     ms=MARKER_SIZE, mec="#222222", mew=0.4,
                                     label=pl.MODEL_LABELS[m]))
    handles += [
        mlines.Line2D([], [], ls="none", marker="o", ms=MARKER_SIZE,
                      color="#777777", mec="#222222", mew=0.4,
                      label="Significant"),
        mlines.Line2D([], [], ls="none", marker="o", ms=MARKER_SIZE,
                      mfc="white", mec="#777777", mew=0.7,
                      label="Not significant"),
        mlines.Line2D([], [], color="#777777", lw=BAR_W, alpha=0.55,
                      marker="|", ms=6, ls="none",
                      label="IQR over years"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=pl.FONT_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.4,
               handletextpad=0.4, columnspacing=0.8, labelspacing=0.3)
    fig.supylabel(f"{label} − {label} of damped persistence (median over "
                  f"years)", fontsize=pl.FONT_AXIS_LABEL, x=0.005)

    top = 1.0
    if pl.SHOW_SUPTITLE:
        fig.suptitle(f"{pl.design_label()} {label} by lead month, relative to "
                     f"damped persistence", fontsize=pl.FONT_SUPTITLE,
                     fontweight="bold")
        top = 1.0 - 0.35 / fig_h
    plt.tight_layout(rect=[0.02, LEGEND_H / fig_h, 1, top], w_pad=0.5,
                     h_pad=0.6)
    pl.draw_regime_separator(fig, axes, catchments)
    pl.savefig(fig, f"{pl.PLOT_DIR}/{out_name}")
    plt.close()


def run(catchments: list, metric: str, variant: str = "default",
        out_name: str = None):
    """Lead-month curves of one metric ('acc' or 'crpss') for the given
    catchments; writes <out_name>.pdf and <out_name>.csv."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    out_name = out_name or f"01_leadcurves_{metric}"
    catchments = [c for c in pl.order_catchments(catchments)
                  if pl.load_yearly(c, REFERENCE, variant) is not None]
    if not catchments:
        print(f"  Skipping {out_name}: no {variant} scores found")
        return
    df = curve_table(catchments, metric, variant)
    if df.empty:
        print(f"  Skipping {out_name}: nothing to plot")
        return
    df.to_csv(f"{pl.PLOT_DIR}/{out_name}.csv", index=False)
    n_sig = int((df["sig"] != 0).sum())
    n_test = int(df["p"].notna().sum())
    print(f"  {out_name}: {len(catchments)} catchments, {len(df)} points, "
          f"{n_sig}/{n_test} tested points significant "
          f"(+{int((df['sig'] > 0).sum())} / -{int((df['sig'] < 0).sum())})")
    plot_curves(catchments, df, metric, out_name)
