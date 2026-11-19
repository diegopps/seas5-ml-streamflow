"""
seas5ml.designs – Evaluation designs: how hydro years are split into folds.

A design turns a catchment's hydro years into a list of `Fold`s. Every stage
loops over these folds and never builds a split itself, so a design is the one
place that decides which years a model may learn from and which it is scored
on.

A fold carries:
    train_years  every year the fold may learn from: its climatology, a0 and
                 damped-persistence phi, the model fit, and early stopping
    test_years   the years it is scored on
    tr_yrs, val_yrs
                 train_years split for early stopping (VAL_FRAC of the years
                 in val_yrs). Models that do not early-stop train on both.

Metrics are written per scored hydro year, not per fold, so a fold with
several test years gives several rows.

Designs
    loyo   leave-one-hydro-year-out: one fold per hydro year, testing on that
           year and training on all others.

Every split is seeded, so every model and every stage sees identical folds.
"""

from dataclasses import dataclass

import numpy as np

SEED = 107          # global seed: fold splits, model initialisation
VAL_FRAC = 0.15     # share of a fold's training years held for early stopping

DESIGNS = ("loyo",)
DEFAULT_DESIGN = "loyo"


@dataclass(frozen=True)
class Fold:
    index: int               # position in the design's fold list
    id: str                  # stable name: output folders, Slurm jobs
    train_years: tuple       # sorted
    test_years: tuple        # sorted
    tr_yrs: frozenset        # train_years minus val_yrs
    val_yrs: frozenset       # early-stopping years


def _val_split(train_years, *, seed):
    """Whole-year early-stopping split: VAL_FRAC of `train_years` (at least
    one), drawn with `seed`. Returns (tr_yrs, val_yrs) as frozensets."""
    train_years = list(train_years)
    n_val = max(1, int(len(train_years) * VAL_FRAC))
    rng = np.random.default_rng(seed=seed)
    val_idx = set(rng.choice(len(train_years), size=n_val,
                             replace=False).tolist())
    val_yrs = frozenset(train_years[i] for i in val_idx)
    tr_yrs = frozenset(y for i, y in enumerate(train_years)
                       if i not in val_idx)
    return tr_yrs, val_yrs


def _loyo(years, seed):
    folds = []
    for i, test_year in enumerate(years):
        train_years = [y for y in years if y != test_year]
        if not train_years:
            continue
        # Per-fold seed (fold index + global seed), so the partition is
        # reproducible and identical across models.
        tr_yrs, val_yrs = _val_split(train_years, seed=i + seed)
        folds.append(Fold(index=i, id=str(test_year),
                          train_years=tuple(train_years),
                          test_years=(test_year,),
                          tr_yrs=tr_yrs, val_yrs=val_yrs))
    return folds


_BUILDERS = {"loyo": _loyo}


def check_design(design: str) -> str:
    if design not in _BUILDERS:
        raise ValueError(f"unknown design '{design}' (one of {DESIGNS})")
    return design


def folds(design: str, years, *, seed: int = SEED) -> list:
    """Every fold of `design` over a catchment's sorted hydro `years`."""
    years = [int(y) for y in sorted(years)]
    return _BUILDERS[check_design(design)](years, seed)


def get_fold(design: str, years, fold_id) -> Fold:
    """The fold named `fold_id`; raises if `design` has no such fold."""
    fold_id = str(fold_id)
    for fold in folds(design, years):
        if fold.id == fold_id:
            return fold
    raise ValueError(f"design '{design}' has no fold '{fold_id}' for hydro "
                     f"years {list(years)}")


def donor_pool(design: str, fold: Fold, test_year: int, years) -> list:
    """
    Hydro years the sensitivity test may draw a donor year's forcing from
    when scoring `test_year` of `fold`. Under loyo: every other hydro year.
    """
    check_design(design)
    return [int(y) for y in years if y != test_year]
