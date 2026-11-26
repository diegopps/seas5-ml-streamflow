"""
seas5ml.plots.heatmap – ACC by start month x lead month, at one resolution.

Binning skill by lead alone mixes start months: an October start at lead
month 7 verifies in the spring melt, an April start at lead month 7 in
autumn low flow. A (start month x lead month) grid removes that confound and
shows where in the seasonal cycle a model's anomaly skill sits.

One figure per resolution (stage-1 default: all three):

  03_heatmap_daily.pdf    the daily anomalies, pooled over years
  03_heatmap_weekly.pdf
  03_heatmap_monthly.pdf  monthly means: pure year-to-year skill
  (+ .csv of every plotted cell; with by_regime, one figure per regime group)

Trellis: rows = catchments, columns = models, each panel 12 start months x
7 lead months, coloured by the pooled ACC of the cell (all years' anomaly
pairs of that start month and lead month together) on one diverging scale
centred at 0. Data: the score tables' cells.csv (plots.common.load_cells),
which already leave a cell blank (grey) where it has too few years
(seas5ml.scores.MIN_CELL_YEARS, per design) and blank the day-of-year
reference beyond daily (seas5ml.scores.BLANK_CELLS). Climatology has no
column (its ACC is undefined); the day-of-year reference has one at daily
resolution only (plots.common.DOY_REF_RESOLUTIONS).
"""

import os
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

from . import common as pl

# Column order: the references first, then the learned models.
HEATMAP_MODEL_ORDER = ["persistence", "damped_persistence", pl.DOY_REFERENCE,
                       "ridge", "rf", "xgb", "lstm", "pooled_lstm"]

CMAP     = "RdBu_r"
ACC_VMAX = 1.0   # full ACC range: pooled ACC often exceeds 0.6

# Built at print size (pl.FIG_WIDTH_IN wide); only the height scales with the
# number of catchment rows.
PANEL_H      = 1.15   # inches per catchment row
EXTRA_H      = 0.8    # inches for column titles, lead-month ticks and axis label
FONT_PANEL_TITLE = pl.FONT_TITLE - 1
FONT_ROW_LABEL   = pl.FONT_AXIS_LABEL
FONT_CELL_TICK   = 5.5

# Short column titles: full model names do not fit over a ~0.75 in panel.
MODEL_LABELS_SHORT = {**pl.MODEL_LABELS_BAR, "persistence": "Pers.",
                      "damped_persistence": "D. pers."}
# One-letter month ticks; twelve full abbreviations overlap at this height.
MONTH_TICKS = [calendar.month_abbr[m][0] for m in range(1, 13)]


def cell_grid(cells) -> np.ndarray:
    """12 x 7 array of cell ACC (rows = start months 1-12, columns = lead
    months 1-7) from one model's cells at one resolution; NaN = blank."""
    grid = np.full((12, len(pl.LEAD_MONTHS)), np.nan)
    if cells is None or cells.empty:
        return grid
    for r in cells.itertuples():
        if 1 <= r.lead_month <= len(pl.LEAD_MONTHS):
            grid[int(r.start_month) - 1, int(r.lead_month) - 1] = r.acc
    return grid


def columns_for(catchments: list, variant: str, resolution: str) -> list:
    """The heatmap columns: models of HEATMAP_MODEL_ORDER with scores for any
    of the catchments; the day-of-year reference at daily only."""
    cols = []
    for m in HEATMAP_MODEL_ORDER:
        if m == pl.DOY_REFERENCE and resolution not in pl.DOY_REF_RESOLUTIONS:
            continue
        if any(pl.load_cells(c, m, variant) is not None for c in catchments):
            cols.append(m)
    return cols


def _plot_grid(catchments: list, models: list, variant: str, resolution: str,
               out_name: str, title_suffix: str = ""):
    """One trellis figure (rows = catchments, columns = models); writes
    <out_name>.pdf and <out_name>.csv (every plotted cell)."""
    pl.apply_plot_style()
    catchments = pl.order_catchments(catchments)
    n_rows, n_cols = len(catchments), len(models)

    norm = TwoSlopeNorm(vmin=-ACC_VMAX, vcenter=0.0, vmax=ACC_VMAX)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(color='#e8e8e8')

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(pl.FIG_WIDTH_IN, PANEL_H * n_rows + EXTRA_H),
        squeeze=False, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.03, hspace=0.04)

    rows = []
    for r, catchment in enumerate(catchments):
        for c, model in enumerate(models):
            ax = axes[r][c]
            cells = pl.load_cells(catchment, model, variant, resolution)
            grid = cell_grid(cells)
            if cells is not None:
                rows.append(cells.assign(catchment=catchment, model=model))
            ax.imshow(np.ma.masked_invalid(grid), aspect='auto', cmap=cmap,
                      norm=norm, origin='upper')

            if r == 0:
                ax.set_title(MODEL_LABELS_SHORT.get(model, model),
                             fontsize=FONT_PANEL_TITLE, fontweight='bold',
                             pad=3)
            if c == 0:
                ax.set_yticks(range(12))
                ax.set_yticklabels(MONTH_TICKS, fontsize=FONT_CELL_TICK)
                # Letter prefixed into the row label rather than floated above
                # the panel, where it would collide with the model titles.
                ax.set_ylabel(pl.catchment_row_label(r, catchment),
                              fontsize=FONT_ROW_LABEL, fontweight='bold',
                              labelpad=2, linespacing=0.95)
            else:
                ax.set_yticks([])
            if r == n_rows - 1:
                ax.set_xticks(range(len(pl.LEAD_MONTHS)))
                ax.set_xticklabels([str(m) for m in pl.LEAD_MONTHS],
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
    cbar.set_label(f"Pooled ACC ({pl.RESOLUTION_LABELS[resolution].lower()})",
                   fontsize=pl.FONT_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=pl.FONT_TICK, length=2, width=0.5)
    cbar.outline.set_linewidth(0.4)

    fig.supxlabel("Lead month", fontsize=pl.FONT_AXIS_LABEL)
    fig.supylabel("Start month", fontsize=pl.FONT_AXIS_LABEL)

    if pl.SHOW_SUPTITLE:
        title = (f"{pl.design_label()} ACC by start month × lead month "
                 f"({pl.RESOLUTION_LABELS[resolution].lower()})")
        if title_suffix:
            title += f" - {title_suffix}"
        fig.suptitle(title, fontsize=pl.FONT_SUPTITLE, fontweight='bold')

    # Separators between regime-group catchment blocks. Placed last: this
    # figure uses a constrained layout, whose axes positions are only final
    # once it has been drawn.
    pl.draw_regime_separator(fig, axes, catchments)

    pl.savefig(fig, f"{pl.PLOT_DIR}/{out_name}")
    plt.close()
    if rows:
        pd.concat(rows, ignore_index=True)[
            ["catchment", "model", "resolution", "start_month", "lead_month",
             "n_years", "n", "acc"]].to_csv(f"{pl.PLOT_DIR}/{out_name}.csv",
                                            index=False)


def run(catchments: list, variant: str = "default", resolution: str = "daily",
        by_regime: bool = False, out_prefix: str = None,
        title_suffix: str = ""):
    """
    Build the heatmap grid of one resolution for the given catchments.

    by_regime=False (default): one figure with all catchments.
    by_regime=True: one figure per regime group, named after the group.
    """
    os.makedirs(pl.PLOT_DIR, exist_ok=True)
    out_prefix = out_prefix or f"03_heatmap_{resolution}"
    catchments = [c for c in pl.order_catchments(catchments)
                  if pl.load_cells(c, pl.SIG_REFERENCE, variant) is not None]
    if not catchments:
        print(f"  Skipping {out_prefix}: no {variant} scores found")
        return
    models = columns_for(catchments, variant, resolution)

    def _suffix(group_label):
        return f"{title_suffix} - {group_label}" if title_suffix else group_label

    if not by_regime:
        _plot_grid(catchments, models, variant, resolution, out_prefix,
                   title_suffix)
        return

    # Groups come back in REGIME_GROUPS order with `other` last, so pair them
    # with their names rather than unpacking positionally.
    group_names = [name for name, _ in pl.REGIME_GROUPS] + ["other"]
    for name, group in zip(group_names, pl.split_catchment_groups(catchments)):
        if group:
            _plot_grid(group, models, variant, resolution,
                       f"{out_prefix}_{name}", _suffix(f"{name} catchments"))
