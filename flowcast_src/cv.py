"""
flowcast_src.cv – Leave-One-Hydro-Year-Out cross-validation folds.

Each fold holds out one complete hydro year as the test set and trains on all
remaining years. A fixed fraction of the training years is reserved as a
validation set (used by models that early-stop; ignored by the rest). Fold
splits are seeded so every model sees identical folds.
"""

import numpy as np

SEED = 107
VAL_FRAC = 0.15


def make_folds(unique_years: list,
               val_frac: float = VAL_FRAC,
               seed: int = SEED):
    """
    Yield one split per hydro year.

    For each test year, the remaining years are partitioned into training and
    validation subsets. The validation subset is drawn with a per-fold seed
    (fold_index + seed) so the partition is reproducible and identical across
    models.

    Yields
    ------
    dict with keys:
        fold_i    : int         zero-based fold index
        test_year : int         held-out hydro year
        train_years : list[int] all non-test years (train + val)
        tr_yrs    : set[int]     training years (excludes val)
        val_yrs   : set[int]     validation years
    """
    for fold_i, test_year in enumerate(unique_years):
        train_years = [y for y in unique_years if y != test_year]
        if not train_years:
            continue

        n_val = max(1, int(len(train_years) * val_frac))
        rng = np.random.default_rng(seed=fold_i + seed)
        val_idx = set(rng.choice(len(train_years), size=n_val,
                                 replace=False).tolist())
        val_yrs = {train_years[i] for i in val_idx}
        tr_yrs = {train_years[i] for i in range(len(train_years))
                  if i not in val_idx}

        yield {
            'fold_i': fold_i,
            'test_year': test_year,
            'train_years': train_years,
            'tr_yrs': tr_yrs,
            'val_yrs': val_yrs,
        }
