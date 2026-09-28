"""
flowcast_src.plot_sensitivity – Does corrupting the climate input change anything?

Produces one merged figure:

  07_sensitivity.pdf — per-fold paired skill change when the
      meteorological forcing is replaced by a donor year's at inference time.
      Rows are catchments, columns are NSE / KGE / ACC / CRPS, and each panel
      holds one strip per model.

Why paired differences rather than raw scores:
  Fold-to-fold hydrology varies far more than the effect of corrupting the
  forcing, so the raw distributions of normal and mixed overlap almost
  completely even when every single fold moved the same way. The signal is
  paired — the same hydro year under real forcing versus donor-year forcing —
  so the per-fold difference is what gets plotted, with zero marked. A strip
  sitting on zero means the model is indifferent to the corruption.

Sign convention, identical in every panel:
  Above zero = corruption HURT skill = the model was genuinely using the
  forcing.
    NSE / KGE / ACC (higher is better):  delta = normal - corrupted
    CRPS            (lower  is better):  delta = corrupted - normal
  CRPS is flipped so "worse under corruption" still reads upward, like the
  other three.

Reading the figure:
  A cloud pinned to zero is the headline negative result: that model scores
  the same whether it is shown the real weather or another year's weather — so
  its skill never came from the forcing. It came from the seasonal cycle,
  which the calendar encodings supply and which the corruption does not touch.

Why a strip plot and not boxplots:
  A box would hide the case where half the folds improved and half degraded
  around a median of zero — which draws identically to genuine indifference
  but means something quite different. Every fold is one hydro year and there
  are only ~30 of them, so the individual dots are worth showing; a median
  rule supplies the point estimate a box would have given.

Encoding:
  Colour is the model, matching every other figure in the project. Marker
  shape carries the corruption scenario. Only one scenario is currently
  plotted, so shape is redundant with position today, but the mapping is kept
  so that adding a second scenario needs no re-encoding.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from . import plot_lib as pl
from . import paths
from .plot_boxplots import window_crps, CRPSS_REFERENCE

# Default location of stage-03 output; overridable so the same renderer can be
# pointed at an alternative results tree.
RESULTS_DIR = str(paths.RESULTS_ROOT / "03_sensitivity")

# climatology and persistence are absent by design: they never read the met
# forcing, so corrupting it is a no-op for them and the test is vacuous.
SENS_MODELS = ["ridge", "rf", "xgb", "lstm"]

CORRUPT_SCENARIOS = ["mixed"]
SCENARIO_LABELS = {"mixed": "False Year"}
# Shape, not colour, separates the scenarios — colour is already spoken for by
# the model, and shape survives greyscale.
SCENARIO_MARKER = {"mixed": "o"}

METRICS = [
    {"key": "nse",  "title": "NSE",  "lower_is_better": False},
    {"key": "kge",  "title": "KGE",  "lower_is_better": False},
    {"key": "acc",  "title": "ACC (per year)",  "lower_is_better": False},
    # reference (climatology) stated in the caption
    {"key": "crpss", "title": "CRPSS", "lower_is_better": False},
]

# Stage 03 scores the whole forecast horizon, so the CRPSS reference spans it.
MAX_LEAD_DAY = 214

# Built at print size (pl.FIG_WIDTH_IN wide), like the stage-01/02 boxplots.
# Unlike those, panels keep their own y-limits (see _panel_ylim): the deltas
# span +-0.01 in one panel and +-1.5 in another, so a shared axis would pin
# most strips flat onto zero.
FIG_HEIGHT_ROW   = 1.7    # inches per catchment row
LEGEND_HEIGHT    = 0.4    # inches reserved below the grid for the legend
PAIR_GAP         = 0.80   # x-offset between a model's two scenario strips
MODEL_GAP        = 1.55   # x-offset between models
DOT_SIZE         = 3.5    # marker area, pt^2
DOT_ALPHA        = 0.55
JITTER           = 0.22
JITTER_SEED      = 7
MEDIAN_HALF_W    = 0.32   # half-width of the median rule over each strip
MEDIAN_LW        = 1.2
ZERO_LW          = 0.6
LEGEND_MARKERSIZE = 4

# Central share of the pooled fold deltas a panel's y-axis must cover. The
# remainder is drawn as an edge count, not discarded (see _panel_ylim).
AXIS_PERCENTILE  = 96.0


def fold_metrics_path(catchment: str, model: str, results_dir: str) -> str:
    return os.path.join(results_dir, f"results_{catchment}_{model}",
                        "sensitivity_fold_metrics.csv")


def load_fold_metrics(catchment: str, model: str, results_dir: str):
    """
    Per-fold, per-scenario metrics for one (catchment, model), or None.

    Adds a `crpss` column derived from the stored whole-window CRPS against
    stage 01's climatology, so the CRPS panel can be read on the same
    dimensionless, higher-is-better scale as the other three. Stage 03 scores
    every lead day, so the reference is taken over the full horizon too.
    """
    path = fold_metrics_path(catchment, model, results_dir)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    if 'crps' in df.columns:
        ref = window_crps(catchment, CRPSS_REFERENCE, 0, MAX_LEAD_DAY)
        if ref is not None:
            ref = ref[np.isfinite(ref) & (ref > 0)]
            denom = df['hydro_year'].map(ref)
            df['crpss'] = 1.0 - df['crps'] / denom
    return df


def find_catchments(results_dir: str) -> list:
    """Catchments with stage-03 results for at least one model."""
    found = set()
    for model in SENS_MODELS:
        for d in glob.glob(os.path.join(results_dir, f"results_*_{model}")):
            name = os.path.basename(d)
            found.add(name[len("results_"):-len(f"_{model}")])
    return pl.order_catchments(sorted(found))


def paired_deltas(df, metric: str, scenario: str, lower_is_better: bool):
    """
    Per-fold change in one metric between real and corrupted forcing.

    Pivoted on hydro_year so the same year's normal and corrupted scores are
    what get subtracted; a year missing either scenario is dropped rather than
    misaligned. Positive = corruption hurt skill, for every metric.
    """
    if df is None or metric not in df.columns:
        return np.array([])

    pivot = df.pivot_table(index='hydro_year', columns='scenario',
                           values=metric, aggfunc='mean')
    if 'normal' not in pivot.columns or scenario not in pivot.columns:
        return np.array([])

    pair = pivot[['normal', scenario]].dropna()
    if len(pair) == 0:
        return np.array([])

    if lower_is_better:
        return pair[scenario].values - pair['normal'].values
    return pair['normal'].values - pair[scenario].values


def _build_positions():
    """x position and (model, scenario) for every strip in a panel.

    Scenarios within a model sit PAIR_GAP apart; the step after a model's last
    scenario is the wider MODEL_GAP, so models stay visually separated however
    many scenarios there are (including one).
    """
    xs, keys = [], []
    x = 0.0
    last = len(CORRUPT_SCENARIOS) - 1
    for model in SENS_MODELS:
        for i, scenario in enumerate(CORRUPT_SCENARIOS):
            xs.append(x)
            keys.append((model, scenario))
            x += MODEL_GAP if i == last else PAIR_GAP
    return xs, keys


def _model_tick_positions(xs):
    """Centre of each model's scenario pair, for a single per-model x-tick."""
    n = len(CORRUPT_SCENARIOS)
    return [np.mean(xs[i * n:(i + 1) * n]) for i in range(len(SENS_MODELS))]


def _panel_ylim(data):
    """
    Panel limits from the pooled central AXIS_PERCENTILE of the fold deltas,
    always containing zero.

    NSE and KGE deltas are mathematically unbounded, and on the flashier
    steppe catchments a few near-zero-variance hydro years send a single
    model's deltas into the tens. An IQR-based bound does not help there,
    because it is that model's own box that is enormous — letting it set the
    axis flattens every other box onto the zero line and the panel stops
    saying anything. Clipping the axis to the bulk of the distribution keeps
    the comparison legible; the folds that fall outside are counted and
    annotated at the panel edge (see `_annotate_offscale`) rather than
    silently dropped.

    Each limit snaps to the nearest actual fold on the inner side of its tail
    rather than interpolating. With ~110 folds pooled, a 2 % tail is ~2 folds,
    and when a third outlier sits far out (Ayat NSE: -7.9, -5.7, -3.5, then
    -0.8) interpolation lands the limit inside that empty gap and stretches
    the axis to -3.

    Zero is the reference the entire figure is read against, so it is always
    inside the limits even if every delta sits to one side.
    """
    pooled = np.concatenate(data)
    lo_p = (100.0 - AXIS_PERCENTILE) / 2.0
    lo = np.nanpercentile(pooled, lo_p, method='higher')
    hi = np.nanpercentile(pooled, 100.0 - lo_p, method='lower')

    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return -1.0, 1.0
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def _annotate_offscale(ax, data, pos, lo, hi):
    """Mark folds outside the panel limits with a count at the panel edge, so
    clipped folds stay visible as information rather than disappearing."""
    for d, x in zip(data, pos):
        n_hi = int(np.sum(d > hi))
        n_lo = int(np.sum(d < lo))
        if n_hi:
            ax.annotate(f"▲{n_hi}", xy=(x, hi), xycoords=('data', 'data'),
                        xytext=(0, -1), textcoords='offset points',
                        ha='center', va='top', fontsize=pl.FONT_ANNOT - 1,
                        color='#c0392b', zorder=6, annotation_clip=False)
        if n_lo:
            ax.annotate(f"▼{n_lo}", xy=(x, lo), xycoords=('data', 'data'),
                        xytext=(0, 1), textcoords='offset points',
                        ha='center', va='bottom', fontsize=pl.FONT_ANNOT - 1,
                        color='#c0392b', zorder=6, annotation_clip=False)


def plot_sensitivity(catchments: list, results_dir: str, out_name: str):
    pl.apply_plot_style()

    catchments = pl.order_catchments(catchments)
    xs, keys = _build_positions()
    tick_pos = _model_tick_positions(xs)
    rng = np.random.default_rng(JITTER_SEED)

    n_rows = len(catchments)
    fig_h = FIG_HEIGHT_ROW * n_rows + LEGEND_HEIGHT
    fig, axes = plt.subplots(n_rows, len(METRICS),
                             figsize=(pl.FIG_WIDTH_IN, fig_h), squeeze=False)

    for r, catchment in enumerate(catchments):
        per_model = {m: load_fold_metrics(catchment, m, results_dir)
                     for m in SENS_MODELS}

        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            key, lower = metric["key"], metric["lower_is_better"]

            data, pos, colors, markers = [], [], [], []
            for x, (model, scenario) in zip(xs, keys):
                d = paired_deltas(per_model[model], key, scenario, lower)
                if len(d) == 0:
                    continue
                data.append(d)
                pos.append(x)
                colors.append(pl.MODEL_COLORS[model])
                markers.append(SCENARIO_MARKER[scenario])

            if not data:
                ax.axis("off")
                continue

            # One dot per hydro year, jittered so overlapping folds stay
            # countable, plus a median rule as the point estimate.
            for d, x, col, marker in zip(data, pos, colors, markers):
                ax.scatter(x + rng.uniform(-JITTER, JITTER, size=len(d)), d,
                           s=DOT_SIZE, marker=marker, color=col,
                           alpha=DOT_ALPHA, edgecolors="none", zorder=4)
                # Near-black rather than the model's hue: it has to stay
                # legible on top of a dense cloud of that same hue, and the
                # colour association is already carried by the dots beneath.
                med = float(np.median(d))
                ax.plot([x - MEDIAN_HALF_W, x + MEDIAN_HALF_W], [med, med],
                        color="#1a1a19", lw=MEDIAN_LW, solid_capstyle="round",
                        zorder=6)

            # The "no effect" reference the whole figure is read against.
            ax.axhline(0.0, color="#222222", lw=ZERO_LW, zorder=5)

            lo, hi = _panel_ylim(data)
            ax.set_ylim(lo, hi)
            _annotate_offscale(ax, data, pos, lo, hi)

            # Four strips leave room for model names under them at print
            # width, so they are kept alongside the legend's colours.
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([pl.MODEL_LABELS_BAR[m] for m in SENS_MODELS],
                               fontsize=pl.FONT_TICK - 1)
            ax.set_xlim(min(xs) - 0.9, max(xs) + 0.9)

            if r == 0:
                ax.set_title(metric["title"], fontweight="bold", pad=4)
            if c == 0:
                # Letter prefixed into the row label rather than floated above
                # the panel, where it would collide with the metric titles.
                ax.set_ylabel(f"{pl.catchment_letter(r)})  "
                              f"{pl.catchment_display(catchment)}",
                              fontweight="bold", fontsize=pl.FONT_AXIS_LABEL,
                              labelpad=3)
            pl.style_axes(ax)
            ax.tick_params(axis="x", length=0)
            # Per-panel y-limits, so every panel keeps its own tick labels;
            # the smaller size keeps them clear of the neighbouring panel.
            ax.tick_params(axis="y", labelsize=pl.FONT_TICK - 0.5)

    model_handles = [
        mpatches.Patch(facecolor=pl.MODEL_COLORS[m], alpha=0.85,
                       edgecolor="#333333", lw=0.5, label=pl.MODEL_LABELS[m])
        for m in SENS_MODELS
    ]
    style_handles = [
        Line2D([0], [0], marker=SCENARIO_MARKER[s], color="none",
               markerfacecolor="#777777", markeredgecolor="none",
               markersize=LEGEND_MARKERSIZE, label=lbl)
        for s, lbl in (
            ("mixed", f"{SCENARIO_LABELS['mixed']} (donor-year forcing)"),)
    ] + [
        Line2D([0], [0], color="#1a1a19", lw=MEDIAN_LW, label="Median"),
    ]
    # One legend, filled column by column: models on the left two columns,
    # scenario and median rule in the third.
    fig.legend(handles=model_handles + style_handles, loc="lower center",
               ncol=3, fontsize=pl.FONT_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, handleheight=0.9, columnspacing=1.2,
               labelspacing=0.3)

    top = 1.0
    if pl.SHOW_SUPTITLE:
        fig.suptitle("Climate Input Sensitivity — per-fold skill change",
                     fontsize=pl.FONT_SUPTITLE, fontweight="bold")
        top = 1.0 - 0.35 / fig_h
    plt.tight_layout(rect=[0, LEGEND_HEIGHT / fig_h, 1, top],
                     w_pad=0.6, h_pad=0.8)

    # After tight_layout: axes positions are only final once layout has run,
    # so the separator must be placed here to land in the row gutter.
    pl.draw_regime_separator(fig, axes, catchments)

    pl.savefig(fig, f"{pl.PLOT_DIR}/{out_name}")
    plt.close()


def run(catchments: list = None, results_dir: str = None,
        out_name: str = "07_sensitivity"):
    """Build the sensitivity figure for the given catchments."""
    results_dir = RESULTS_DIR if results_dir is None else results_dir
    os.makedirs(pl.PLOT_DIR, exist_ok=True)

    available = find_catchments(results_dir)
    if catchments:
        wanted = set(catchments)
        available = [c for c in available if c in wanted]
    if not available:
        print(f"  No stage-03 results found under {results_dir} — "
              f"skipping sensitivity figure")
        return
    plot_sensitivity(available, results_dir, out_name)
