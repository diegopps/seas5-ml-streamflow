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
    split  train on hindcasts, test on operational forecasts: one fold
           (id SPLIT_FOLD_ID), training on the hydro years up to
           SPLIT_TRAIN_LAST and testing on those from SPLIT_TEST_FIRST.
           SEAS5 hindcasts (25 members) cover inits up to Dec 2016 and the
           operational forecasts (51 members) start in Jan 2017, so every row
           verifying in a training year comes from a hindcast and every row
           verifying in a test year from an operational forecast. The years
           in between (hydro year 2017, the only one mixing both) are a
           buffer, neither trained on nor scored. Stage 0 checks both sides
           against the data's `source` column.

The split runs through exactly the same code as loyo: a fold is used the same
way whatever its design (climatology, a0, phi and model fit on its training
years; the same early-stopping rule; scored on its test years). Only the
donor pool of the sensitivity test differs (see `donor_pool`).

Every split is seeded, so every model and every stage sees identical folds.
"""

from dataclasses import dataclass

import numpy as np

SEED = 107          # global seed: fold splits, model initialisation
VAL_FRAC = 0.15     # share of a fold's training years held for early stopping

DESIGNS = ("loyo", "split")
DEFAULT_DESIGN = "loyo"

# split design: last hindcast-only and first operational-only hydro year.
SPLIT_TRAIN_LAST = 2016
SPLIT_TEST_FIRST = 2018
SPLIT_FOLD_ID = "hindcast"


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


def _split(years, seed):
    train_years = [y for y in years if y <= SPLIT_TRAIN_LAST]
    test_years = [y for y in years if y >= SPLIT_TEST_FIRST]
    if not train_years or not test_years:
        # No fold: stage 0 writes an empty manifest and warns.
        return []
    # One fold, index 0: same seed rule as loyo (fold index + global seed).
    tr_yrs, val_yrs = _val_split(train_years, seed=0 + seed)
    return [Fold(index=0, id=SPLIT_FOLD_ID,
                 train_years=tuple(train_years),
                 test_years=tuple(test_years),
                 tr_yrs=tr_yrs, val_yrs=val_yrs)]


_BUILDERS = {"loyo": _loyo, "split": _split}


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
    when scoring `test_year` of `fold`.

      loyo   every other hydro year.
      split  the fold's other test years: operational forecasts like the
             test year's own, so the donor forcing has the same 51 members
             (a 25-member hindcast donor would be wrapped onto the 51 test
             members, giving the mixed ensemble only 25 distinct forcings
             and shrinking its spread) and comes from the same era.
    """
    if check_design(design) == "split":
        return [int(y) for y in fold.test_years if y != test_year]
    return [int(y) for y in years if y != test_year]
