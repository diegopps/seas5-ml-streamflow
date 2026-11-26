"""
seas5ml.plots.boxplots – Per-year model comparison (NSE, KGE, ACC, CRPSS).

One figure, at one (resolution, lead month); the stage-1 default figure is
daily resolution, lead month 1 (the first calendar month of each forecast):

  02_boxplots_daily_lm1.pdf  (+ .csv of the box statistics)

How skill changes with lead time is the lead-month curves' job
(plots.leadcurves); this figure shows the spread over years and the
absolute level of every score at the shortest lead.

Rows = catchments, columns = metrics, one box per model (colour = model, see
the shared legend); a box is the model's per-year scores over the test years.
Data: the score tables (plots.common.load_yearly), the same years and starts
for every model. CRPSS is against deterministic climatology, so climatology
itself (identically 0) has no CRPSS box, and its ACC is undefined.

  - ACC column: the day-of-year reference's median ACC as a dashed line
    (daily resolution only, plots.common.DOY_REF_RESOLUTIONS). ACC above it is
    skill beyond the seasonal cycle.
  - Significance: every learned model against damped persistence, paired over
    years (plots.common.paired_tests_vs_reference: paired bootstrap of the
    mean difference, Benjamini-Hochberg per metric over the figure), as a
    symbol above the box.
  - The y axis is fixed to Y_LIM for every panel: per-year NSE is unbounded
    below on ephemeral rivers, so years below the floor are counted and
    marked at the bottom of their box instead.
  - Split design (3-8 test years): every year is also drawn as a dot over
    its box, since a box hides how few points it has.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from . import common as pl

# Built at print size (pl.FIG_WIDTH_IN wide). Eight boxes per panel leave no
# room for model names under them at this width, so boxes are identified by
# colour and the shared legend only.
BAR_FIG_HEIGHT_ROW = 1.7    # inches per catchment row
BAR_LEGEND_HEIGHT  = 0.75   # inches reserved below the grid for the legend
BOX_WIDTH          = 0.7
BOX_ALPHA          = 0.85
ANNOT_OFFSET_FRAC  = 0.025
DOT_SIZE           = 1.6
DOT_JITTER         = 0.18

Y_LIM = (-2.0, 1.0)

METRICS = [
    {"key": "nse", "title": "NSE"},
    {"key": "kge", "title": "KGE"},
    {"key": "acc", "title": "ACC (per year)"},
    {"key": "crpss", "title": "CRPSS"},   # reference (climatology) stated in the caption
]

# Not tested against damped persistence: the baselines themselves.
DETERMINISTIC_MODELS = {"climatology", "persistence", "damped_persistence"}
CRPSS_REFERENCE = "climatology"

DOY_REF_COLOR = pl.DOY_REF_COLOR
DOY_REF_LABEL = "Day-of-year reference (median)"


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def load_year_scores(catchment: str, variant: str, resolution: str,
                     lead_month: int) -> dict:
    """{model: per-year scores (hydro_year, nse, kge, acc, crpss)} at one
    (resolution, lead month), for every model of MODEL_ORDER with scores.
    Climatology's CRPSS (identically 0) is left out."""
    out = {}
    for m in pl.MODEL_ORDER:
        y = pl.load_yearly(catchment, m, variant, resolution)
        if y is None:
            continue
        y = y[y["lead_month"] == lead_month]
        if y.empty:
            continue
        y = y[["hydro_year"] + [k["key"] for k in METRICS]].copy()
        if m == CRPSS_REFERENCE:
            y["crpss"] = np.nan
        out[m] = y.reset_index(drop=True)
    return out


def doy_reference_median(catchment: str, variant: str, resolution: str,
                         lead_month: int):
    """Median per-year ACC of the day-of-year reference, or None (not shown
    at this resolution, or no scores)."""
    if resolution not in pl.DOY_REF_RESOLUTIONS:
        return None
    y = pl.load_yearly(catchment, pl.DOY_REFERENCE, variant, resolution)
    if y is None:
        return None
    acc = y.loc[y["lead_month"] == lead_month, "acc"].dropna()
    return float(acc.median()) if len(acc) else None


def _sig_vs_reference(catchments: list, fold_data: dict) -> dict:
    """
    (catchment, metric, model) -> +1 / -1 / 0 for every learned model against
    damped persistence, paired over hydro years; FDR-adjusted per metric
    across the whole figure (see plots.common.paired_tests_vs_reference).
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


def box_table(catchments: list, fold_data: dict, sig: dict) -> pd.DataFrame:
    """The box statistics as drawn: one row per (catchment, metric, model)."""
    rows = []
    for c in catchments:
        for m, df_m in fold_data[c].items():
            for k in METRICS:
                v = df_m[k["key"]].dropna()
                if v.empty:
                    continue
                rows.append({"catchment": c, "metric": k["key"], "model": m,
                             "n_years": len(v), "median": v.median(),
                             "q25": v.quantile(0.25), "q75": v.quantile(0.75),
                             "min": v.min(), "max": v.max(),
                             "n_below_floor": int((v < Y_LIM[0]).sum()),
                             "sig_vs_damped_persistence":
                                 sig.get((c, k["key"], m), np.nan)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════

def draw_below_floor_counts(ax, xs, data):
    """
    Values below the hardcoded Y_LIM floor are clipped off-axis; mark each
    affected box with its off-axis count instead of letting a single bad year
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


def draw_year_dots(ax, xs, data, seed=0):
    """Every year as a small dot over its box (horizontal jitter, fixed
    seed), within the axis range."""
    rng = np.random.default_rng(seed)
    for x, vals in zip(xs, data):
        v = np.asarray(vals)
        v = v[v >= Y_LIM[0]]
        ax.scatter(x + rng.uniform(-DOT_JITTER, DOT_JITTER, len(v)), v,
                   s=DOT_SIZE, color='#222222', linewidths=0, zorder=4.5)


def plot_bar_chart(catchments: list, fold_data: dict, title: str, out_name: str,
                   doy_ref: dict = None, show_years: bool = False):
    """doy_ref maps catchment -> median day-of-year reference ACC, drawn as a
    dashed line in the ACC column; catchments without an entry get no line.
    show_years draws every year as a dot over its box."""
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
        present_models = [m for m in pl.MODEL_ORDER if m in models_here]

        for col, metric in enumerate(METRICS):
            ax  = axes[row, col]
            key = metric["key"]

            box_years = {}
            for m in present_models:
                vals = models_here[m][key].dropna()
                if len(vals):
                    box_years[m] = vals.values

            x_all = np.arange(len(present_models))
            plotted_models = [m for m in present_models if m in box_years]
            x_plotted = [present_models.index(m) for m in plotted_models]
            data_plotted = [box_years[m] for m in plotted_models]

            if data_plotted:
                bp = ax.boxplot(
                    data_plotted, positions=x_plotted, widths=BOX_WIDTH,
                    patch_artist=True, showfliers=not show_years,
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
                if show_years:
                    draw_year_dots(ax, x_plotted, data_plotted, seed=row * 10 + col)

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
                ax.set_ylabel(pl.catchment_row_label(row, catchment,
                                                     max_chars=24),
                              fontweight='bold', fontsize=pl.FONT_AXIS_LABEL,
                              labelpad=3, linespacing=0.95)
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
    if show_years:
        legend_handles.append(mlines.Line2D([], [], ls='none', marker='o',
                                            markersize=2, color='#222222',
                                            label='One test year'))
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
    return sig


def run(catchments: list, variant: str = "default", resolution: str = "daily",
        lead_month: int = 1, out_name: str = None, show_years: bool = None):
    """
    Build the per-year boxplot figure for the given catchments at one
    (resolution, lead month). show_years defaults to True under the split
    design (few test years), False otherwise. Writes <out_name>.pdf and
    <out_name>.csv (the box statistics).
    """
    out_name = out_name or f"02_boxplots_{resolution}_lm{lead_month}"
    if show_years is None:
        show_years = pl.DESIGN == "split"
    catchments = pl.order_catchments(catchments)
    fold_data, doy_ref = {}, {}
    for catchment in catchments:
        models_here = load_year_scores(catchment, variant, resolution,
                                       lead_month)
        if models_here:
            fold_data[catchment] = models_here
        med = doy_reference_median(catchment, variant, resolution, lead_month)
        if med is not None:
            doy_ref[catchment] = med
            print(f"  {catchment}: day-of-year reference median ACC {med:+.3f}")

    box_catchments = [c for c in catchments if c in fold_data]
    if not box_catchments:
        print(f"  Skipping boxplot figure {out_name} (no {variant} scores found)")
        return
    title = (f"{pl.design_label()} per-year skill, "
             f"{pl.RESOLUTION_LABELS[resolution].lower()}, lead month "
             f"{lead_month}")
    sig = plot_bar_chart(box_catchments, fold_data, title, out_name,
                         doy_ref=doy_ref, show_years=show_years)
    box_table(box_catchments, fold_data, sig).to_csv(
        f"{pl.PLOT_DIR}/{out_name}.csv", index=False)
