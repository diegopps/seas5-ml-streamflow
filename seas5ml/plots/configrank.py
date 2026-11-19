"""
seas5ml.plots.configrank – Is any hyperparameter config a real winner?

Produces one figure from the tuned runs of stage 1:

  03_configranking.pdf (+ .csv) — for each tuned model, a table of
      every candidate config's mean rank across the outer folds, one row per
      catchment. Under the split design (one outer fold) the ranks are
      taken across the inner folds of the single search instead (see
      RANK_UNIT); everything below reads the same with "inner" for "outer".

Reading the figure:
  - A cell is a config's MEAN RANK across that catchment's outer folds, where
    within each outer fold the 15 configs are ranked 1 (best inner-CV ACC) to
    15 (worst). Under the null "all configs are equally good" every config
    sits at rank 8.0, so the colour scale diverges around 8.0: saturated blue
    = consistently better than chance, saturated red = consistently worse,
    pale = indistinguishable from chance.
  - A row of uniformly pale cells means the search found no config that is
    reliably better — whichever config happens to win the most folds is
    winning on noise.
  - Mean rank is deliberately used instead of mean ACC. Outer folds share
    almost all of their training data, so fold-to-fold scores are not
    independent and formal significance tests on win counts would overstate
    confidence. A rank average needs no independence assumption and is
    directly comparable across catchments, whose absolute ACC levels differ
    by a lot.

Summary columns:
  - `best`     the config that won the most outer folds outright.
  - `wins`     how often it won, out of that row's completed folds. Compare
               against chance (folds / number of configs, written to the CSV
               as `chance_wins`) — a config barely above chance is not a
               winner in any useful sense, and its mean rank will say so.
  - `ACC range` the spread between the best and worst config's mean inner-CV
               ACC. This is the magnitude check the ranks cannot give: if the
               whole config grid spans a few thousandths of ACC, the ranking
               is precise but the choice does not matter.

Every value is printed as a number as well as encoded in colour, and the same
table is written to CSV alongside the figure.

Configs are model-specific — RF's R01 shares nothing with XGB's R01 — so each
model gets its own table and the tables are never compared cell to cell.

Rows are catchments and nothing is pooled across them: each catchment is
tuned independently and a config is only ever deployed per catchment, so a
combined ranking would describe a choice nobody makes, and would mask a
config that wins one catchment outright while ranking badly in the rest.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

from . import common as pl

# Rank under the null that every config is equally good, for n configs:
# the mean of 1..n. Set per model from the config count actually found.
CMAP = "RdBu_r"          # low rank (good) = blue, matching the default run's ACC heatmap

# What the configs are ranked across, per design.
#   outer  loyo: within each outer fold by its mean inner-CV ACC; mean rank
#          across the outer folds; `best` = the config winning the most outer
#          folds, `wins` = how many.
#   inner  split: one outer fold, so "across outer folds" would be a single
#          ranking. Ranked within each inner fold of that search by that inner
#          fold's ACC instead; mean rank across the inner folds; `best` = the
#          split's winner (winner.csv, best mean inner-CV ACC), `wins` = in
#          how many inner folds it ranked first.
# Both ask the same question -- is any config consistently better? -- at the
# level each design has several folds.
RANK_UNIT = {"loyo": "outer", "split": "inner"}
UNIT_LABEL = {"outer": "outer folds", "inner": "inner folds"}

# Built at print size (pl.FIG_WIDTH_IN wide); the heatmap takes whatever
# width the row labels and summary block leave, and only the height scales
# with the number of catchment rows and models.
ROW_H = 0.2            # inches per catchment row
TABLE_EXTRA_H = 0.42   # inches per table for its title and config tick labels
CBAR_H = 0.5           # inches for the colour bar and its label
SUMMARY_FRAC = 0.32    # summary block width, as a fraction of the heatmap's

FONT_CELL = 5.5
FONT_ROW_LABEL = pl.FONT_AXIS_LABEL - 0.5
FONT_COL_LABEL = pl.FONT_ANNOT
FONT_SUMMARY = pl.FONT_ANNOT


def config_stats(catchment: str, model: str):
    """
    Per-config ranking statistics for one (catchment, model) search.

    The configs are ranked (1 = best) within each ranking unit of the
    design (RANK_UNIT): each outer fold by its mean inner-CV ACC (loyo), or
    each inner fold of the one outer fold by its ACC (split). Returns None
    if this search has no completed folds.
    """
    inner = pl.load_inner_cv(catchment, model)
    if inner is None:
        return None
    unit = RANK_UNIT[pl.DESIGN]

    # One ACC per (ranking unit, config).
    keys = ['fold'] if unit == 'outer' else ['fold', 'inner_fold']
    per_unit = inner.groupby(keys + ['config'])['acc'].mean().unstack('config')
    labels = sorted(per_unit.columns)
    per_unit = per_unit.reindex(columns=labels)

    # Rank within each unit; NaN scores stay NaN rather than ranking last.
    ranks = per_unit.rank(axis=1, ascending=False, na_option='keep')

    winners = pl.load_winners(catchment, model)
    have_winners = winners is not None and 'winner_config' in winners.columns
    if unit == 'outer':
        wins = (winners['winner_config'].value_counts() if have_winners
                else pd.Series(dtype=float))
        wins = wins.reindex(labels).fillna(0).astype(int)
        best = wins.idxmax() if wins.sum() > 0 else None
        best_wins = int(wins.max()) if best is not None else 0
    else:
        best = str(winners['winner_config'].iloc[0]) if have_winners else None
        firsts = per_unit.dropna(how='all').idxmax(axis=1).value_counts()
        best_wins = int(firsts.get(best, 0)) if best is not None else 0

    return {
        'labels': labels,
        'ranks': ranks,
        'mean_acc': per_unit.mean(),
        'best': best,
        'best_wins': best_wins,
        'n_folds': int(len(per_unit)),
    }


def build_model_table(model: str, catchments: list):
    """
    Assemble the mean-rank table for one model.

    Rows are catchments; there is deliberately no pooled row. Each catchment
    is tuned independently and a config is only ever deployed per catchment,
    so a "best across all catchments" config would describe a choice nobody
    makes — and would hide the case where a config wins one catchment
    outright while ranking badly in the rest.

    Returns (rank_df, summary_df), or (None, None) if no catchment has data.
    """
    rank_rows, summary_rows = {}, {}
    labels = None

    for catchment in catchments:
        st = config_stats(catchment, model)
        if st is None:
            continue
        labels = st['labels']

        rank_rows[catchment] = st['ranks'].mean()

        summary_rows[catchment] = {
            'best': st['best'] if st['best'] is not None else "—",
            'wins': st['best_wins'],
            'n_folds': st['n_folds'],
            'acc_range': float(st['mean_acc'].max() - st['mean_acc'].min()),
        }

    if not rank_rows:
        return None, None

    rank_df = pd.DataFrame(rank_rows).T.reindex(columns=labels)
    summary_df = pd.DataFrame(summary_rows).T.reindex(rank_df.index)
    return rank_df, summary_df


def _draw_model_panel(ax_heat, ax_sum, model, rank_df, summary_df, norm, cmap):
    """Render one model's table into a heatmap axis plus a text axis."""
    values = rank_df.values.astype(float)
    n_rows, n_cols = values.shape

    ax_heat.imshow(np.ma.masked_invalid(values), aspect='auto',
                   cmap=cmap, norm=norm, origin='upper')

    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(rank_df.columns, fontsize=FONT_COL_LABEL)
    # Row letters go into the tick label here: rows live inside one axis, so
    # there is no per-row panel to float a letter above. Two-station names
    # (e.g. Kirchbichl-Bichlwang) break after the hyphen: the longest label
    # sets the heatmap's width, and a row is tall enough for two lines.
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(
        [f"{pl.catchment_letter(i)})  "
         + pl.catchment_display(c).replace("-", "-\n", 1)
         for i, c in enumerate(rank_df.index)],
        fontsize=FONT_ROW_LABEL, linespacing=0.95)
    ax_heat.tick_params(length=0, pad=1.5)

    # Value in every cell, so the table never encodes by colour alone.
    for r in range(n_rows):
        for c in range(n_cols):
            v = values[r, c]
            if not np.isfinite(v):
                continue
            # White ink only where the fill is dark enough to need it.
            strong = abs(v - norm.vcenter) / (norm.vmax - norm.vcenter) > 0.55
            ax_heat.text(c, r, f"{v:.1f}", ha='center', va='center',
                         fontsize=FONT_CELL,
                         color='white' if strong else '#1a1a19')

    for spine in ax_heat.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color('#bbbbbb')

    ax_heat.set_title(pl.MODEL_LABELS.get(model, model.upper()),
                      fontsize=pl.FONT_TITLE, fontweight='bold', loc='left',
                      pad=3)

    # Separator between regime-group catchment rows.
    for brk in pl.regime_breaks(list(rank_df.index)):
        ax_heat.axhline(brk - 0.5, color='#1a1a19', lw=0.6)

    # ── Summary text block ────────────────────────────────────────────
    ax_sum.set_xlim(0, 1)
    ax_sum.set_ylim(n_rows - 0.5, -0.5)     # match heatmap row coordinates
    ax_sum.axis('off')

    # Headers bottom-aligned, so the two-line one grows upward and all three
    # sit on a common baseline.
    cols = [(0.17, 'best'), (0.5, 'wins'), (0.83, 'ACC\nrange')]
    for x, header in cols:
        ax_sum.text(x, -0.6, header, fontsize=FONT_COL_LABEL,
                    fontweight='bold', ha='center', va='bottom',
                    color='#52514e')

    # Uniform styling: whether a config is a real winner is read off its mean
    # rank in the heatmap, not asserted by emphasising the summary text.
    for r, idx in enumerate(rank_df.index):
        s = summary_df.loc[idx]
        rng = s['acc_range']
        for (x, _), txt in zip(cols, (str(s['best']),
                                      f"{s['wins']}/{s['n_folds']}",
                                      "—" if not np.isfinite(rng) else f"{rng:.3f}")):
            ax_sum.text(x, r, txt, fontsize=FONT_SUMMARY,
                        ha='center', va='center', color='#52514e')

    for brk in pl.regime_breaks(list(rank_df.index)):
        ax_sum.axhline(brk - 0.5, color='#1a1a19', lw=0.6)


def plot_config_ranks(catchments: list):
    """Build the config-ranking table figure across all tuned models."""
    pl.apply_plot_style()

    tables = {}
    for model in pl.HPTUNE_MODELS:
        requested = set(catchments)
        available = [c for c in pl.find_hptune_catchments(model)
                     if c in requested]
        rank_df, summary_df = build_model_table(model, available)
        if rank_df is not None:
            tables[model] = (rank_df, summary_df)

    if not tables:
        print("  No tuned results found — skipping config-ranking figure")
        return

    n_configs = max(len(t[0].columns) for t in tables.values())
    # Null expectation: the mean of ranks 1..n_configs.
    chance_rank = (n_configs + 1) / 2
    norm = TwoSlopeNorm(vmin=1.0, vcenter=chance_rank, vmax=float(n_configs))
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(color='#e8e8e8')

    heights = [len(t[0]) for t in tables.values()]
    fig_h = sum(h * ROW_H for h in heights) + TABLE_EXTRA_H * len(tables) + CBAR_H

    fig = plt.figure(figsize=(pl.FIG_WIDTH_IN, fig_h), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.02, hspace=0.06)
    gs = fig.add_gridspec(
        len(tables), 2,
        width_ratios=[1.0, SUMMARY_FRAC], height_ratios=heights)

    for i, (model, (rank_df, summary_df)) in enumerate(tables.items()):
        _draw_model_panel(fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]),
                          model, rank_df, summary_df, norm, cmap)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=fig.axes, location='bottom',
                        shrink=0.45, aspect=40, pad=0.02)
    # Direction is written into the label itself; separate end-labels would
    # collide with the tick numbers.
    cbar.set_label(
        f"← better        Mean rank across {UNIT_LABEL[RANK_UNIT[pl.DESIGN]]} "
        f"        worse →",
        fontsize=pl.FONT_AXIS_LABEL)
    # Integer ranks, with the chance rank marked.
    cbar.set_ticks(sorted({1, round(chance_rank, 1), n_configs}
                          | set(range(4, n_configs, 4))))
    cbar.ax.tick_params(labelsize=pl.FONT_TICK, length=2, width=0.5)
    cbar.outline.set_linewidth(0.4)

    if pl.SHOW_SUPTITLE:
        fig.suptitle("Hyperparameter Tuning\nMean rank of parameter configs",
                     fontsize=pl.FONT_SUPTITLE, fontweight='bold')

    pl.savefig(fig, f"{pl.PLOT_DIR}/03_configranking")
    plt.close()

    # CSV twin of the figure — exact numbers for the write-up.
    out_rows = []
    for model, (rank_df, summary_df) in tables.items():
        for idx in rank_df.index:
            row = {'model': model, 'catchment': idx}
            row.update({c: round(float(rank_df.loc[idx, c]), 2)
                        for c in rank_df.columns})
            row.update({
                'best_config': summary_df.loc[idx, 'best'],
                'best_wins': summary_df.loc[idx, 'wins'],
                'n_folds': summary_df.loc[idx, 'n_folds'],
                'chance_wins': round(
                    summary_df.loc[idx, 'n_folds'] / len(rank_df.columns), 2),
                'acc_range': round(float(summary_df.loc[idx, 'acc_range']), 4),
            })
            out_rows.append(row)
    csv_path = f"{pl.PLOT_DIR}/03_configranking.csv"
    pd.DataFrame(out_rows).to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")


def run(catchments: list):
    """Build the config-ranking figure for the given catchments."""
    os.makedirs(pl.PLOT_DIR, exist_ok=True)
    plot_config_ranks(catchments)
