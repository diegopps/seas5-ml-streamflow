"""
seas5ml.tuning – Nested-CV hyperparameter search for one outer fold.

The outer fold is a seas5ml.designs.Fold. Its training years are partitioned
into N_INNER inner folds; each candidate configuration (seas5ml.models
.configs.GRIDS) is trained on each inner fold's inner-training years -- with
its own early-stopping split nested inside -- and scored on the inner fold's
held-out years. The configuration with the best mean inner-CV ACC is the
fold's winner. The caller then retrains the winner on the full fold (its
tr_yrs / val_yrs, the same split the default configuration trains on) and
scores it on the fold's test years: the leakage-free score, since no test
year contributes to the selection.

Correctness invariants:
  - An inner holdout is used only to score a configuration for selection; it
    is never in that inner run's training or early-stopping years.
  - The fold's test years are never touched during inner CV.
  - Every run fits its climatology (and a0) and draws its training
    trajectories from its own training years only (seas5ml.models.fit).
"""

import time

import numpy as np
import pandas as pd

from . import models
from .designs import SEED, VAL_FRAC
from .evaluate import score_members

N_INNER = 5

# Offsets of the val-split seed coordinates (see `val_seed_for`). They mark
# coordinates outside the (inner fold, config) space -- the winner retrain
# and the old baseline retrain -- and are kept so that the inner-CV seeds stay
# those of earlier versions of the search.
INNER_IDX_RETRAIN = -1
CONFIG_IDX_BASELINE = -2


def make_val_split(train_years, *, val_seed):
    """Whole-year train/val split for early stopping (VAL_FRAC of the years,
    at least 1). Returns (tr_yrs, val_yrs) as sets."""
    train_years = list(train_years)
    n_val = max(1, int(len(train_years) * VAL_FRAC))
    rng = np.random.default_rng(seed=val_seed)
    val_idx = set(rng.choice(len(train_years), size=n_val,
                             replace=False).tolist())
    val_yrs = {train_years[i] for i in val_idx}
    tr_yrs = {train_years[i] for i in range(len(train_years))
              if i not in val_idx}
    return tr_yrs, val_yrs


def val_seed_for(outer_idx, inner_idx, config_idx, *, seed=SEED):
    """
    Reproducible, collision-free seed for an inner run's val split, keyed by
    the full coordinate (outer fold, inner fold, config), so distinct
    coordinates never share a seed regardless of iteration order. SeedSequence
    rejects negative entries, so the coordinates are shifted by the sentinel
    offsets above.
    """
    ss = np.random.SeedSequence([seed, outer_idx,
                                 inner_idx - INNER_IDX_RETRAIN,
                                 config_idx - CONFIG_IDX_BASELINE])
    return int(ss.generate_state(1)[0])


def inner_partition(outer_train_years, *, n_inner, seed):
    """Partition the outer-training years into `n_inner` roughly-equal disjoint
    groups (the inner folds). Reproducible: shuffles the sorted year list with
    a fixed SeedSequence, then round-robins into n_inner buckets so sizes
    differ by at most 1. Returns a list of sets (empty buckets dropped)."""
    years = sorted(outer_train_years)
    ss = np.random.SeedSequence([seed, 987654321])   # fixed salt
    rng = np.random.default_rng(ss)
    order = rng.permutation(len(years))
    buckets = [set() for _ in range(n_inner)]
    for rank, idx in enumerate(order):
        buckets[rank % n_inner].add(years[idx])
    return [b for b in buckets if b]


def inner_cv(model, grid, df, *, fit_clim, fold, n_inner=None,
             seed=SEED, log=print):
    """
    Score every configuration of `grid` on every inner fold of `fold`
    (`n_inner` inner folds; N_INNER if None).

    Returns one row per (config, inner fold): config, inner_fold, acc, nse,
    kge, crps, best_epoch, time_s. ACC (and the other scores) pool the inner
    holdout's years. Runs that produce no model or no predictions are left
    out.
    """
    n_inner = N_INNER if n_inner is None else n_inner
    inner_folds = inner_partition(fold.train_years, n_inner=n_inner,
                                  seed=seed)
    log(f"  {len(grid)} configs x {len(inner_folds)} inner folds = "
        f"{len(grid) * len(inner_folds)} trainings; inner-fold sizes "
        f"{[len(f) for f in inner_folds]}")

    rows = []
    for cfg_idx, config in enumerate(grid):
        for inner_idx, holdout in enumerate(inner_folds):
            t0 = time.time()
            inner_train = [y for y in fold.train_years if y not in holdout]
            tr_yrs, val_yrs = make_val_split(
                inner_train,
                val_seed=val_seed_for(fold.index, inner_idx, cfg_idx,
                                      seed=seed))
            fitted = models.fit(model, config, df, fit_clim=fit_clim,
                                tr_yrs=tr_yrs, val_yrs=val_yrs)
            if fitted is None:
                continue
            scores = score_members(models.predict(fitted, df,
                                                  score_years=holdout))
            if scores is None:
                continue
            rows.append({
                'config': config['label'], 'inner_fold': inner_idx,
                'acc': scores['acc'], 'nse': scores['nse'],
                'kge': scores['kge'], 'crps': scores['crps'],
                'best_epoch': fitted['info'].get('best_epoch', np.nan),
                'time_s': round(time.time() - t0, 1),
            })
            log(f"    {config['label']}  inner {inner_idx + 1}/"
                f"{len(inner_folds)}  ACC={scores['acc']:+.3f}  "
                f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def select_winner(inner_df):
    """
    The configuration with the best mean inner-CV ACC, as (label, mean ACC),
    or (None, nan) if every configuration's ACC is NaN. Ties go to the
    alphabetically first label.
    """
    mean_acc = inner_df.groupby('config')['acc'].mean().dropna()
    if mean_acc.empty:
        return None, np.nan
    label = mean_acc.idxmax()
    return label, float(mean_acc[label])


def catchment_winner(winner_frames):
    """
    The configuration that won the most folds, from the folds' winner.csv
    frames; ties go to the alphabetically first label (as in the
    config-ranking figure). Returns None if no fold has a winner.
    """
    labels = [str(w['winner_config'].iloc[0]) for w in winner_frames
              if w is not None and len(w)]
    if not labels:
        return None
    return pd.Series(labels).value_counts().sort_index().idxmax()
