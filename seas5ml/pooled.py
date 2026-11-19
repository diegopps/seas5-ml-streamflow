"""
seas5ml.pooled – The catchment pool of the pooled models, and its folds.

A pooled model (pooled_lstm) is trained on every catchment at once and scored
per catchment, like any other model. Its unit of work is therefore one fold
across the whole pool, not one (catchment, model) pair.

Folds are built by the evaluation design (seas5ml.designs) over the UNION of
the catchments' hydro years, then restricted to each catchment's own years
(designs.restrict):

  loyo   one fold per hydro year of the union. When Y is the test year it is
         removed from every catchment at once (neighbours share that year's
         weather and anomalies; Kirchbichl and Diepoldsau have overlapping
         SEAS5 boxes). Restricted to a catchment, the fold has exactly the
         training and test years of that catchment's own loyo fold for Y, so
         its climatology, a0, Q_clim and scored rows are those of the
         single-catchment models. A catchment without year Y still trains in
         fold Y; it is not scored there.
  split  one fold: training years <= SPLIT_TRAIN_LAST, test years >=
         SPLIT_TEST_FIRST, for every catchment.

Early-stopping years are drawn once per pooled fold, from the union of the
training years (the design's own rule), and shared by every catchment, so no
year is early-stopping data for one catchment while it is training data for
a neighbour with nearly the same forcing. This is the one deliberate
difference from the single-catchment folds, whose early-stopping years are
drawn per catchment.

Each catchment keeps its own climatology fitter: every catchment's
climatology and a0 are fit on its own training years, as in the
single-catchment models.
"""

from dataclasses import dataclass

import pandas as pd

from . import designs
from .climatology import ClimatologyFitter
from .data import load_catchment

# The catchments of the pool, in a fixed order: a catchment's position is its
# index in the pooled model's one-hot catchment ID, so this order must not
# change between training and prediction. Keep in sync with POOL in
# pipeline/1_models/submit_pooled.sh.
POOLED_CATCHMENTS = ("ulkenboken", "zeravshan", "esil", "ayat", "kalkutan",
                     "kurshim", "zhabay", "kirchbichl", "diepoldsau")


@dataclass
class PoolMember:
    """One catchment of the pool: what seas5ml.data.load_catchment returns."""
    catchment: str
    df: pd.DataFrame
    years: list
    fit_clim: ClimatologyFitter


def catchment_index(catchment: str) -> int:
    """Position of `catchment` in POOLED_CATCHMENTS (its one-hot index)."""
    return POOLED_CATCHMENTS.index(catchment)


def load_pool(catchments=POOLED_CATCHMENTS) -> dict:
    """{catchment: PoolMember} for every catchment of the pool, in pool
    order."""
    pool = {}
    for c in catchments:
        print(f"\n  [{c}]")
        df, years, fit_clim = load_catchment(c)
        pool[c] = PoolMember(c, df, [int(y) for y in years], fit_clim)
    return pool


def pool_years(pool: dict) -> list:
    """Sorted union of the pool's hydro years."""
    return sorted({y for m in pool.values() for y in m.years})


def pooled_folds(design: str, pool: dict) -> list:
    """
    Every pooled fold of `design`, as (fold, {catchment: restricted fold}):
    `fold` is built over the union of the pool's years, each restriction
    over one catchment's (see module docstring).
    """
    return [(fold, {c: designs.restrict(fold, m.years)
                    for c, m in pool.items()})
            for fold in designs.folds(design, pool_years(pool))]


def get_pooled_fold(design: str, pool: dict, fold_id):
    """The pooled fold named `fold_id`, as in `pooled_folds`; raises if the
    design has no such fold."""
    fold = designs.get_fold(design, pool_years(pool), fold_id)
    return fold, {c: designs.restrict(fold, m.years) for c, m in pool.items()}
