"""
flowcast_src.plot_heatmap – Lead-time x initialisation-month ACC heatmap grid.

Binning skill by lead time alone mixes initialisation months — e.g. an
October initialisation at 200 days lead verifies in the spring melt peak,
while an April initialisation at 200 days verifies in autumn low flow.
Disaggregating ACC on a (init-month x lead-bin) grid removes that confound
and shows where in the seasonal cycle a model's anomaly skill actually sits.

Every (catchment, model) combination is shown as one mini-heatmap in a single
trellis figure: rows = catchment, columns = model, each panel init-month (12)
x lead-bin (7), coloured by ACC on a shared diverging scale centred at zero.
This packs the full skill landscape into one image.

Output:
    00_plots/02_leadtimeheatmap.pdf                     (all catchments)
    00_plots/02_leadtimeheatmap_mountain.pdf  \\  when split rendering
    00_plots/02_leadtimeheatmap_steppe.pdf    /   is requested

Cell handling:
  - Each cell's ACC is computed across the years contributing to that
    (init-month, lead-bin) combination.
  - Cells with fewer than MIN_YEARS contributing years are blanked (grey),
    to avoid showing striking-but-hollow ACC values off a handful of years.
  - Diverging colormap centred at 0 (ACC has a meaningful no-skill zero),
    symmetric shared scale across every panel.
"""

import os
import calendar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

from . import plot_lib as pl
from .metrics import calc_acc

# climatology is excluded — its ACC is undefined by construction (zero
# predicted anomaly).
HEATMAP_MODEL_ORDER = [m for m in pl.MODEL_ORDER if m != "climatology"]

# Minimum number of contributing YEARS for a cell's ACC to be shown.
MIN_YEARS = 10

# Minimum daily samples within a (year, cell) for that year to count
# toward the cell (mirrors the >10-day guard used in the lead-time plot).
MIN_DAYS_PER_YEAR_CELL = 10

CMAP     = "RdBu_r"
ACC_VMAX = 1.0   # full ACC range: pooled ACC often exceeds 0.6, which clipped

# Built at print size (pl.FIG_WIDTH_IN wide); only the height scales with the
# number of catchment rows.
PANEL_H      = 1.15   # inches per catchment row
EXTRA_H      = 0.8    # inches for column titles, lead-bin ticks and axis label
FONT_PANEL_TITLE = pl.FONT_TITLE - 1
FONT_ROW_LABEL   = pl.FONT_AXIS_LABEL
FONT_CELL_TICK   = 5.5

# Short column titles: full model names do not fit over a ~0.75 in panel.
MODEL_LABELS_SHORT = {**pl.MODEL_LABELS_BAR, "persistence": "Pers.",
                      "damped_persistence": "D. pers."}
# One-letter month ticks; twelve full abbreviations overlap at this height.
MONTH_TICKS = [calendar.month_abbr[m][0] for m in range(1, 13)]


def compute_heatmap(df):
    """
    Build the (init_month x lead_bin) ACC grid for one (catchment, model).

    For each cell, collect years with >= MIN_DAYS_PER_YEAR_CELL daily samples;
    if at least MIN_YEARS such years exist, compute ACC pooled across those
    years' daily anomaly pairs, otherwise leave the cell NaN (blanked).

    Returns a 12 x len(LEAD_BINS) array, rows = init months 1..12.
    """
    n_bins = len(pl.LEAD_BINS)
    acc_grid = np.full((12, n_bins), np.nan)

    if df is None:
        return acc_grid

    d = df.dropna(subset=['Q_raw', 'Q_pred', 'Q_clim']).copy()
    if 'init_date' not in d.columns:
        return acc_grid
    d['init_month'] = d['init_date'].dt.month

    for im in range(1, 13):
        d_im = d[d['init_month'] == im]
        if len(d_im) == 0:
            continue
        for bi, (lo, hi, _centre, _label) in enumerate(pl.LEAD_BINS):
            cell = d_im[(d_im['lead_day'] >= lo) & (d_im['lead_day'] <= hi)]
            if len(cell) == 0:
                continue
            contributing_years = [y for y, g in cell.groupby('hydro_year')
                                  if len(g) >= MIN_DAYS_PER_YEAR_CELL]
            if len(contributing_years) >= MIN_YEARS:
                sub = cell[cell['hydro_year'].isin(contributing_years)]
                acc_grid[im - 1, bi] = calc_acc(
                    sub['Q_raw'].values, sub['Q_pred'].values, sub['Q_clim'].values)
    return acc_grid


def _plot_grid(catchments: list, out_name: str, title_suffix: str = "",
              models: list = None, pred_loader=None, model_labels: dict = None):
    """
    Render one trellis figure (rows = catchments, cols = models).

    models: column set; defaults to every stage-01 model except climatology.
    pred_loader: (catchment, model) -> predictions dataframe; defaults to the
        stage-01 loader. Passed in by the stage-02 winner variant to read
        each catchment's tuned-winner predictions instead.
    model_labels: column-title override; defaults to MODEL_LABELS_SHORT.
    """
    pl.apply_plot_style()

    catchments = pl.order_catchments(catchments)
    models = HEATMAP_MODEL_ORDER if models is None else models
    pred_loader = pl.load_predictions if pred_loader is None else pred_loader
    model_labels = MODEL_LABELS_SHORT if model_labels is None else model_labels
    n_rows, n_cols = len(catchments), len(models)

    norm = TwoSlopeNorm(vmin=-ACC_VMAX, vcenter=0.0, vmax=ACC_VMAX)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(color='#e8e8e8')

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(pl.FIG_WIDTH_IN, PANEL_H * n_rows + EXTRA_H),
        squeeze=False, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.03, hspace=0.04)

    n_bins = len(pl.LEAD_BINS)
    month_labels = MONTH_TICKS
    lead_labels  = [b[3] for b in pl.LEAD_BINS]

    for r, catchment in enumerate(catchments):
        for c, model in enumerate(models):
            ax = axes[r][c]
            df = pred_loader(catchment, model)
            acc_grid = compute_heatmap(df)

            masked = np.ma.masked_invalid(acc_grid)
            ax.imshow(masked, aspect='auto', cmap=cmap, norm=norm,
                      origin='upper')

            if r == 0:
                ax.set_title(model_labels.get(model, model),
                             fontsize=FONT_PANEL_TITLE, fontweight='bold',
                             pad=3)

            if c == 0:
                ax.set_yticks(range(12))
                ax.set_yticklabels(month_labels, fontsize=FONT_CELL_TICK)
                # Letter prefixed into the row label rather than floated above
                # the panel, where it would collide with the model titles.
                ax.set_ylabel(f"{pl.catchment_letter(r)})  "
                              f"{pl.catchment_display(catchment)}",
                              fontsize=FONT_ROW_LABEL, fontweight='bold',
                              labelpad=2)
            else:
                ax.set_yticks([])

            if r == n_rows - 1:
                ax.set_xticks(range(n_bins))
                ax.set_xticklabels(lead_labels, rotation=90,
                                   fontsize=FONT_CELL_TICK)
            else:
                ax.set_xticks([])

            ax.tick_params(length=0, pad=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color('#bbbbbb')

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.6,
                        pad=0.01, aspect=25)
    cbar.set_label("Pooled ACC", fontsize=pl.FONT_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=pl.FONT_TICK, length=2, width=0.5)
    cbar.outline.set_linewidth(0.4)

    fig.supxlabel("Forecast lead time (days)", fontsize=pl.FONT_AXIS_LABEL)
    fig.supylabel("Initialisation month", fontsize=pl.FONT_AXIS_LABEL)

    if pl.SHOW_SUPTITLE:
        title = "LOGO-CV ACC by Lead Time × Init Month"
        if title_suffix:
            title += f" - {title_suffix}"
        fig.suptitle(title, fontsize=pl.FONT_SUPTITLE, fontweight='bold')

    # Separators between regime-group catchment blocks (only apply when a
    # single grid mixes multiple regimes -- the split renders keep them in
    # separate figures already). Placed last: this figure uses a constrained
    # layout, whose axes positions are only final once it has been drawn.
    pl.draw_regime_separator(fig, axes, catchments)

    pl.savefig(fig, f"{pl.PLOT_DIR}/{out_name}")
    plt.close()


def run(catchments: list, split: bool = False,
       out_prefix: str = "02_leadtimeheatmap", title_suffix: str = ""):
    """
    Build the compressed heatmap grid for the given catchments.

    split=False (default): one figure with all catchments.
    split=True: one figure per regime group, named after the group.

    out_prefix / title_suffix let a caller redirect the output filenames and
    annotate the title without touching the grid logic itself. Defaults
    reproduce stage 01's current behaviour exactly.
    """
    os.makedirs(pl.PLOT_DIR, exist_ok=True)

    def _suffix(group_label):
        return f"{title_suffix} - {group_label}" if title_suffix else group_label

    if not split:
        _plot_grid(catchments, out_prefix, title_suffix)
        return

    # Groups come back in REGIME_GROUPS order with `other` last, so pair them
    # with their names rather than unpacking positionally — a group added to
    # or reordered in plot_lib.REGIME_GROUPS would otherwise silently write
    # itself into the previous group's filename.
    group_names = [name for name, _ in pl.REGIME_GROUPS] + ["other"]
    for name, group in zip(group_names, pl.split_catchment_groups(catchments)):
        if group:
            _plot_grid(group, f"{out_prefix}_{name}",
                       _suffix(f"{name} catchments"))
