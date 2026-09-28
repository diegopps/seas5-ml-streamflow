"""
flowcast_src.features – Feature definitions and flat-table assembly.

The dynamic predictors are the seven catchment-averaged meteorological
variables, the two seasonal-cycle encodings, and the antecedent discharge
anomaly at initialisation (a0, baked into the parquet by
00_preprocess_catchment.py). For the tabular models these are flattened,
together with lead_day, into one row per forecast timestep per ensemble
member.
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
