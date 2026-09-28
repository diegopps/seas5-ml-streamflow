"""
flowcast_src.plot_lib – Shared configuration, loaders, and statistics helpers
for the figures of every stage.
"""

import os
import glob
import numpy as np
import pandas as pd

from . import paths

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

RESULTS_DIR = str(paths.RESULTS_ROOT / "01_exploration")
PLOT_DIR    = str(paths.PLOT_DIR)

# Stage-02 nested hyperparameter search. Results are one directory per
# (catchment, model), holding one subdirectory per outer fold.
HPTUNE_DIR  = str(paths.RESULTS_ROOT / "02_hyperparamtuning")
HPTUNE_SEED = int(os.environ.get("SEED", 107))
HPTUNE_MODELS = ["lstm", "rf", "xgb"]

# ═══════════════════════════════════════════════════════════════════════
# CATCHMENTS
# ═══════════════════════════════════════════════════════════════════════

CATCHMENT_DISPLAY = {
    "zeravshan":       "Zeravshan",
    "kurshim":         "Kurshim",
    "ulkenboken":      "Ulken Boken",
    "kalkutan":        "Kalkutan",
    "kirchbichl":      "Kirchbichl-Bichlwang",
    "diepoldsau":      "Diepoldsau-Rietbruecke",
}


# Catchment regime groups — mountain (snowmelt-driven, higher relief) and
# steppe (flat, flashy/ephemeral). Used for optional split rendering and
# to order catchments so related regimes sit together in grid figures.
MOUNTAIN_CATCHMENTS = ["zeravshan", "ulkenboken", "kurshim"]
STEPPE_CATCHMENTS   = ["kalkutan", "esil", "ayat", "zhabay"]
EUROPE_CATCHMENTS   = ["kirchbichl", "diepoldsau"]

# Ordered (name, catchments) regime groups, in the order they should appear
# in grid figures. Adding a new regime group only requires updating this
# list — ordering, splitting, and separator drawing all derive from it.
REGIME_GROUPS = [
    ("mountain",    MOUNTAIN_CATCHMENTS),
    ("steppe",      STEPPE_CATCHMENTS),
    ("europe",      EUROPE_CATCHMENTS)
]
CATCHMENT_ORDER = [c for _, group in REGIME_GROUPS for c in group]


def catchment_display(name: str) -> str:
    return CATCHMENT_DISPLAY.get(name, name.replace('_', ' ').title())


def catchment_letter(i: int) -> str:
    return chr(ord('A') + i)


def order_catchments(catchments: list) -> list:
    """Order given catchments by regime group, in `REGIME_GROUPS` order
    (mountain, then steppe, then europe), with any unrecognised
    catchments appended alphabetically."""
    given = set(catchments)
    ordered = [c for c in CATCHMENT_ORDER if c in given]
    extras = sorted(given - set(CATCHMENT_ORDER))
    return ordered + extras


def split_catchment_groups(catchments: list):
    """Partition the given catchments into one list per regime group in
    `REGIME_GROUPS` (mountain, steppe, europe), plus a trailing
    `other` list, each in canonical order."""
    given = set(catchments)
    groups = [[c for c in group if c in given] for _, group in REGIME_GROUPS]
    other  = sorted(given - set(CATCHMENT_ORDER))
    return (*groups, other)


def regime_breaks(catchments: list):
    """
    Row indices of every regime-group boundary in an `order_catchments`-
    ordered list (e.g. mountain/steppe, steppe/europe), skipping
    boundaries where either side is empty. A grid figure with one row per
    catchment draws a separator immediately after each index — e.g.
    `ax.axhline(brk - 0.5)` for each `brk` in `regime_breaks(catchments)`.
    """
    ordered = order_catchments(catchments)
    breaks = []
    cursor = 0
    for _, group in REGIME_GROUPS:
        n = sum(1 for c in ordered if c in group)
        cursor += n
        if n > 0 and cursor < len(ordered):
            breaks.append(cursor)
    return breaks


def draw_regime_separator(fig, axes, catchments: list):
    """
    Draw a separator line across a row-per-catchment grid at every regime-
    group boundary (e.g. mountain/steppe, steppe/europe).

    Must be called AFTER the figure's layout is final (after tight_layout, or
    after a draw for constrained layouts): axes positions shift during layout,
    so a line placed earlier ends up sitting on top of the panels instead of
    in the gutter between them.

    Each line is centred in the gap between the last row of one group and the
    first row of the next, and inset horizontally so it does not run through
    the row labels on the left.
    """
    breaks = regime_breaks(catchments)
    if not breaks:
        return

    import matplotlib.pyplot as plt

    fig.canvas.draw()
    x0 = min(axes[r, 0].get_position().x0 for r in range(axes.shape[0]))
    x1 = max(axes[r, -1].get_position().x1 for r in range(axes.shape[0]))

    for brk in breaks:
        y_above = axes[brk - 1, 0].get_position().y0   # bottom of last row above
        y_below = axes[brk, 0].get_position().y1       # top of first row below
        y_line = (y_above + y_below) / 2
        fig.add_artist(plt.Line2D([x0, x1], [y_line, y_line],
                                  color='#999999', lw=0.6,
                                  transform=fig.transFigure, zorder=0))


def find_catchments() -> list:
    """Discover catchments from results directories (climatology or lstm)."""
    names = set()
    for d in glob.glob(os.path.join(RESULTS_DIR, "results_*_climatology")):
        name = os.path.basename(d).replace("results_", "").replace("_climatology", "")
        names.add(name)
    for d in glob.glob(os.path.join(RESULTS_DIR, "results_*_lstm")):
        name = os.path.basename(d).replace("results_", "").replace("_lstm", "")
        names.add(name)
    return sorted(names)


def parse_catchment_args(args: list) -> list:
    """Shared CLI convention: no args -> discover all, else literal list."""
    if not args:
        catchments = find_catchments()
        if not catchments:
            print("No results found.")
            print(f"Expected in: {RESULTS_DIR}")
        else:
            print(f"Using all catchments: {', '.join(catchments)}")
    else:
        catchments = [a.lower() for a in args]
    return catchments

# ═══════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════

# Full model order, used by the boxplot and lead-time figures.
MODEL_ORDER = ["climatology", "persistence", "damped_persistence",
              "ridge", "rf", "xgb", "lstm"]

# Heatmap ACC is undefined for climatology by construction (zero predicted
# anomaly) — the heatmap figure filters it out of MODEL_ORDER itself.

MODEL_LABELS = {
    "climatology":        "Climatology",
    "persistence":        "Persistence",
    "damped_persistence": "Damped persistence",
    "ridge":              "Ridge",
    "rf":                 "Random Forest",
    "xgb":                "XGBoost",
    "lstm":               "LSTM",
}
MODEL_LABELS_BAR = {
    "climatology":        "Clim",
    "persistence":        "Pers",
    "damped_persistence": "DPers",
    "ridge":              "Ridge",
    "rf":                 "RF",
    "xgb":                "XGB",
    "lstm":               "LSTM",
}
# Colorblind-friendly categorical palette (validated adjacent-pair order).
# climatology stays neutral grey (it is the no-skill floor, not a model to
# distinguish). persistence / damped_persistence share the blue hue family
# (two shades) since they are variants of the same cheap baseline; ridge,
# rf, xgb, lstm each get one of the remaining maximally-distinct hues.
MODEL_COLORS = {
    "climatology":        "#adb5bd",  # neutral grey — no-skill floor
    "persistence":        "#2a78d6",  # blue
    "damped_persistence": "#184f95",  # dark blue (same hue family)
    "ridge":              "#eb6834",  # orange
    "rf":                 "#1baf7a",  # aqua
    "xgb":                "#e87ba4",  # magenta
    "lstm":               "#008300",  # green
}

# ═══════════════════════════════════════════════════════════════════════
# LEAD-TIME BINS
# (lo, hi inclusive in days, centre for x-position, label) — the canonical
# definition used by the boxplot/lead-time/heatmap figures. The heatmap
# figure ignores `centre` (it uses categorical columns, not a numeric axis).
# ═══════════════════════════════════════════════════════════════════════

LEAD_BINS = [
    (0,   14,   7,   "0-14"),
    (15,  30,   22,  "15-30"),
    (31,  60,   45,  "31-60"),
    (61,  90,   75,  "61-90"),
    (91,  120,  105, "91-120"),
    (121, 150,  135, "121-150"),
    (151, 215,  183, "151-215"),
]

# ═══════════════════════════════════════════════════════════════════════
# SHARED MATPLOTLIB STYLE
# ═══════════════════════════════════════════════════════════════════════

# Print sizes (pt): figures are built at their final 140 mm width, so these
# are the sizes that appear on the page. 6 pt is the smallest legible size.
FONT_TITLE      = 8
FONT_AXIS_LABEL = 7.5
FONT_TICK       = 6.5
FONT_LEGEND     = 7
FONT_ANNOT      = 6
FONT_SUPTITLE   = 9

# Journal figures carry their title in the LaTeX caption, not in the image.
SHOW_SUPTITLE = False

GRID_ALPHA = 0.25
GRID_LW    = 0.6

DPI     = 300
FORMATS = ("pdf",)   # journal submission requires vector PDF

# Every figure is built at the journal's 1.5-column width (Elsevier: 140 mm)
# so fonts print at their nominal point size instead of being shrunk.
FIG_WIDTH_MM = 140
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4


def savefig(fig, out_path_no_ext: str):
    """
    Save a figure in every format listed in FORMATS.

    The tight-bbox padding is capped so the saved page never exceeds
    FIG_WIDTH_IN: a constrained-layout figure already spans the full width,
    and the default 0.1 in each side would push it to ~144 mm.
    """
    fig.canvas.draw()
    content_w = fig.get_tightbbox(fig.canvas.get_renderer()).width
    pad = max(0.0, min(0.1, (FIG_WIDTH_IN - content_w) / 2))
    for fmt in FORMATS:
        path = f"{out_path_no_ext}.{fmt}"
        fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=pad,
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {path}")


def apply_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size':       FONT_TICK,
        'axes.titlesize':  FONT_TITLE,
        'axes.labelsize':  FONT_AXIS_LABEL,
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'legend.fontsize': FONT_LEGEND,
        'font.family':     'sans-serif',
        'pdf.fonttype':    42,   # embed TrueType, not Type 3 (journal requirement)
        'axes.linewidth':  0.6,
        'lines.linewidth': 1.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })


def style_axes(ax):
    """Common axis cosmetics shared by all figures."""
    ax.grid(True, alpha=GRID_ALPHA, lw=GRID_LW, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.6)
    ax.tick_params(axis='both', length=2, pad=1.5)

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════

def load_fold_metrics(catchment: str, model: str):
    """Load fold_metrics.csv for one (catchment, model) pair, or None."""
    path = os.path.join(RESULTS_DIR, f"results_{catchment}_{model}",
                        "fold_metrics.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_predictions(catchment: str, model: str):
    """
    Load predictions_daily.parquet for one (catchment, model) pair, or None.
    Parses both `date` and `init_date`, and derives `hydro_year` if absent.
    """
    path = os.path.join(RESULTS_DIR, f"results_{catchment}_{model}",
                        "predictions_daily.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    if 'init_date' in df.columns:
        df['init_date'] = pd.to_datetime(df['init_date'])
    if 'hydro_year' not in df.columns:
        df['hydro_year'] = df['date'].dt.year
        df.loc[df['date'].dt.month >= 10, 'hydro_year'] += 1
    return df


def load_member_predictions(catchment: str, model: str):
    """
    Load predictions_members.parquet for one (catchment, model) pair, or None.

    Only ensemble models write this file, and only runs from the stage-01
    version that persists it — callers must treat None as "no CRPS available"
    rather than as an error.
    """
    path = os.path.join(RESULTS_DIR, f"results_{catchment}_{model}",
                        "predictions_members.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    if 'init_date' in df.columns:
        df['init_date'] = pd.to_datetime(df['init_date'])
    if 'hydro_year' not in df.columns:
        df['hydro_year'] = df['date'].dt.year
        df.loc[df['date'].dt.month >= 10, 'hydro_year'] += 1
    return df

# ═══════════════════════════════════════════════════════════════════════
# STAGE-02 LOADERS — nested hyperparameter search
# ═══════════════════════════════════════════════════════════════════════

def hptune_root(catchment: str, model: str, seed: int = None) -> str:
    """Results directory for one (catchment, model) nested search."""
    seed = HPTUNE_SEED if seed is None else seed
    return os.path.join(
        HPTUNE_DIR, f"results_{catchment}_{model}_nested_seed{seed}")


def _load_outer_files(catchment: str, model: str, filename: str,
                      seed: int = None):
    """
    Concatenate one per-outer-fold CSV across every outer fold of a
    (catchment, model) search, tagging each row with its `outer_hy`.

    Folds that have not written this file yet (still running, or failed) are
    skipped, so a partially complete search still loads. Returns None if no
    fold has the file.
    """
    root = hptune_root(catchment, model, seed)
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "outer_*", filename))):
        outer_hy = os.path.basename(os.path.dirname(path)).replace("outer_", "")
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        df["outer_hy"] = int(outer_hy)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_inner_cv(catchment: str, model: str, seed: int = None):
    """
    Inner-CV scores for every config, pooled across outer folds.

    Columns: config, inner_fold, acc, nse, kge, ..., outer_hy — one row per
    (outer fold, config, inner fold). This is the only place all candidate
    configs are scored; the outer-test files hold just the winner and the
    baseline.
    """
    return _load_outer_files(catchment, model, "inner_cv_acc.csv", seed)


def load_winners(catchment: str, model: str, seed: int = None):
    """Selected config per outer fold, one row per fold."""
    return _load_outer_files(catchment, model, "winner.csv", seed)


def load_outer_metrics(catchment: str, model: str, seed: int = None):
    """
    Outer-test scores, two rows per outer fold: the selected winner and the
    fixed baseline, both scored on the same held-out year.
    """
    return _load_outer_files(catchment, model, "outer_test_metrics.csv", seed)


def _load_hptune_frames(catchment: str, model: str, seed, filename: str):
    """Concatenate `filename` across every outer fold of one nested search."""
    root = hptune_root(catchment, model, seed)
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "outer_*", filename))):
        df = pd.read_parquet(path)
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    if 'init_date' in df.columns:
        df['init_date'] = pd.to_datetime(df['init_date'])
    if 'hydro_year' not in df.columns:
        df['hydro_year'] = df['date'].dt.year
        df.loc[df['date'].dt.month >= 10, 'hydro_year'] += 1
    return df


def load_hptune_predictions(catchment: str, model: str, seed: int = None):
    """
    Concatenate predictions_daily.parquet across every outer fold of a
    (catchment, model) nested search — the tuned winner's held-out-year
    predictions, ensemble-mean, one row per (date, lead_day).

    Returns None if no outer fold has written this file yet.
    """
    return _load_hptune_frames(catchment, model, seed,
                               "predictions_daily.parquet")


def load_hptune_member_predictions(catchment: str, model: str,
                                   seed: int = None):
    """
    Same, but the tuned winner's per-member predictions — needed for any CRPS
    scored over a subset of lead days. Returns None for searches run before
    stage 02 began writing this file.
    """
    return _load_hptune_frames(catchment, model, seed,
                               "predictions_members.parquet")


def find_hptune_catchments(model: str, seed: int = None) -> list:
    """Catchments with at least one completed outer fold for this model."""
    seed = HPTUNE_SEED if seed is None else seed
    pattern = os.path.join(HPTUNE_DIR,
                           f"results_*_{model}_nested_seed{seed}")
    found = []
    for d in glob.glob(pattern):
        name = os.path.basename(d)
        catchment = name[len("results_"):-len(f"_{model}_nested_seed{seed}")]
        if glob.glob(os.path.join(d, "outer_*", "inner_cv_acc.csv")):
            found.append(catchment)
    return order_catchments(found)


# ═══════════════════════════════════════════════════════════════════════
# STATISTICS — paired bootstrap (boxplot brackets)
# ═══════════════════════════════════════════════════════════════════════

N_BOOT      = 5000
BOOT_SEED   = 42
ALPHA_LEVEL = 0.05   # two-sided 95% CI


def paired_bootstrap_diff(metric_a, years_a, metric_b, years_b,
                          n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA_LEVEL):
    """
    Paired bootstrap CI on mean(metric_a) - mean(metric_b), resampling
    hydrological years WITH REPLACEMENT and applying the SAME resampled
    year set to both models (paired), rather than bootstrapping each
    model's mean independently. Both models are evaluated on the same
    years, so "hard" and "easy" years contribute correlated noise that
    cancels in the difference.

    Only years present in BOTH metric_a/years_a and metric_b/years_b are
    used (intersection) — if the two models were scored on different year
    sets, this restricts to the overlap and reports how many years that was.

    Returns a dict with obs_diff, ci_lo, ci_hi, n_years, significant (True
    if the CI excludes zero) and p, the two-sided bootstrap p-value (twice
    the smaller tail share of resampled differences beyond zero, floored at
    1/n_boot). All NaN / significant=False / p=1 if fewer than 3 common
    years are available.
    """
    s_a = pd.Series(np.asarray(metric_a, dtype=float), index=years_a)
    s_b = pd.Series(np.asarray(metric_b, dtype=float), index=years_b)
    s_a = s_a[~s_a.isna()]
    s_b = s_b[~s_b.isna()]
    common_years = np.array(sorted(set(s_a.index) & set(s_b.index)))

    if len(common_years) < 3:
        return dict(obs_diff=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    n_years=len(common_years), significant=False, p=1.0)

    a = s_a.loc[common_years].values
    b = s_b.loc[common_years].values
    n = len(common_years)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    obs_diff = float(a.mean() - b.mean())
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci_lo, ci_hi = np.percentile(boot_diffs, [lo_pct, hi_pct])
    significant = not (ci_lo <= 0 <= ci_hi)
    p = 2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean())

    return dict(obs_diff=obs_diff, ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                n_years=n, significant=significant,
                p=float(min(1.0, max(p, 1 / n_boot))))


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (q-values), same order as p."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.empty(n)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1.0)


# Significance markers on the per-year boxplots: every learned model is tested
# against damped persistence, the strongest reference, with the paired
# bootstrap above, and the p-values of one figure's metric column (all its
# catchments x models) are FDR-adjusted together.
SIG_REFERENCE = "damped_persistence"
SIG_METRICS   = ("acc", "crpss")
SIG_COLOR     = "#222222"


def paired_tests_vs_reference(pairs: dict, fdr=ALPHA_LEVEL) -> dict:
    """
    pairs maps any key -> (model Series, reference Series), each indexed by
    hydro_year. Returns key -> +1 (model significantly better), -1 (reference
    significantly better) or 0, after Benjamini-Hochberg over all pairs.
    """
    keys = list(pairs)
    res = [paired_bootstrap_diff(a.values, a.index, b.values, b.index)
           for a, b in (pairs[k] for k in keys)]
    q = bh_adjust([r['p'] for r in res])
    return {k: (int(np.sign(r['obs_diff'])) if qk < fdr else 0)
            for k, r, qk in zip(keys, res, q)}


def sig_legend_handles() -> list:
    """Legend entries for the two significance triangles."""
    import matplotlib.lines as mlines
    kw = dict(color=SIG_COLOR, markeredgewidth=0, linestyle='none', markersize=4)
    return [mlines.Line2D([], [], marker='^', label='Better than d. persistence', **kw),
            mlines.Line2D([], [], marker='v', label='Worse than d. persistence', **kw),
            mlines.Line2D([], [], marker='_', label='No significant difference',
                          **{**kw, 'markeredgewidth': 1.0})]


def draw_sig_marker(ax, x: float, sign):
    """Symbol just above the panel over box x: up triangle if the model beat
    the reference, down triangle if the reference beat it, a dash if the
    difference is not significant. None (box not tested) draws nothing."""
    if sign is None:
        return
    marker = {1: '^', -1: 'v', 0: '_'}[sign]
    ax.plot(x, 1.02, marker=marker, markersize=3.2, color=SIG_COLOR,
            markeredgewidth=1.0 if sign == 0 else 0, linestyle='none',
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
