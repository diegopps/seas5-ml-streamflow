"""
seas5ml.features – Feature definitions and flat-table assembly.

The dynamic predictors are the seven catchment-averaged meteorological
variables, the two seasonal-cycle encodings, and the antecedent discharge
anomaly at initialisation (a0). For the tabular models these are flattened,
together with lead_day, into one row per forecast timestep per ensemble
member.

a0 is not stored in the feature parquet, because it depends on the
climatology, which belongs to a fold. The parquet stores logq0 (observed
log1p(Q) on the init date, written by stage 0, pipeline/0_data/preprocess.py), and
`set_a0` forms a0 = logq0 - climatology(day of year of init_date) against
the climatology the caller passes in.
"""

import numpy as np
import pandas as pd

from .data import GROUP_COL

# Per-timestep predictors: 7 met vars + seasonal cycle encodings + the
# antecedent log1p(Q) anomaly at initialisation (baked in at preprocessing).
DYN_VARS = ['tas', 'tasmax', 'tasmin', 'ssrd', 'tp', 'sd', 'sf',
            'sin_doy', 'cos_doy', 'a0']

# Tabular feature set adds lead_day to the dynamic predictors.
TABULAR_FEATURES = DYN_VARS + ['lead_day']


def set_a0(df: pd.DataFrame, doy_clim: dict) -> pd.DataFrame:
    """
    Write the antecedent anomaly column `a0` = logq0 - doy_clim[doy(init_date)]
    into `df`, IN PLACE (any existing a0 is overwritten), and return `df`.

    `doy_clim` must be the climatology of the fold being trained or scored,
    so a0 and the target share one anomaly definition. In place on purpose:
    the analysis frame is large and every fold rewrites the column before
    use, so no stale a0 from another fold survives a call.
    """
    if 'logq0' not in df.columns:
        raise KeyError(
            "feature frame has no 'logq0' column -- it was written by an older "
            "preprocessing script (which stored a0). Rerun stage 0 (pipeline/0_data/preprocess.py).")
    init_clim = df['init_date'].dt.dayofyear.map(doy_clim)
    a0 = df['logq0'] - init_clim
    if not np.isfinite(a0.values).all():
        raise ValueError(f"{int((~np.isfinite(a0.values)).sum())} non-finite "
                         f"a0 value(s): logq0 or the climatology is missing "
                         f"for some init dates.")
    df['a0'] = a0.values
    return df


def apply_fold_climatology(df: pd.DataFrame, fit_clim, train_years) -> dict:
    """
    Prepare one fold: fit its climatology on `train_years` (hydro years) with
    `fit_clim` (a seas5ml.climatology.ClimatologyFitter), write a0 against that
    climatology into `df` in place (see `set_a0`), and return the climatology.

    Call it at the top of every fold, before any feature table or trajectory
    is built from `df`, and pass the returned climatology to everything the
    fold builds (target, back-transform, Q_clim). `train_years` is every year
    the fold may learn from, early-stopping years included; never the years
    the fold scores.
    """
    doy_clim = fit_clim(train_years)
    set_a0(df, doy_clim)
    return doy_clim


def build_flat_table(df: pd.DataFrame, doy_clim: dict) -> pd.DataFrame:
    """
    Flatten the feature frame to one row per forecast timestep per member.

    Adds the log1p(Q) anomaly target relative to the DOY climatology, and
    keeps only rows with a finite target and finite features.
    """
    df = df.copy()
    df['doy'] = df['date'].dt.dayofyear
    df['q_clim'] = df['doy'].map(doy_clim)          # log1p space
    df['log_q'] = np.log1p(df['Q'].values)
    df['target'] = df['log_q'] - df['q_clim']       # anomaly

    keep_cols = TABULAR_FEATURES + ['target', 'q_clim', 'Q',
                                    'date', 'init_date', 'member', GROUP_COL]
    df = df[keep_cols].dropna(subset=['target'] + TABULAR_FEATURES)
    return df.reset_index(drop=True)
