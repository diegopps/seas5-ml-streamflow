"""
seas5ml.evaluate – Scoring helpers, shared by every stage.

Given per-member (or already-deterministic) daily predictions, these helpers
collapse the ensemble to its mean and score NSE/KGE/ACC (and the fair CRPS
over the members). Predictions are expected in a long frame with columns:
init_date, date, lead_day, hydro_year, Q_raw, Q_pred, Q_clim (and member, for
ensemble models).
"""

import pandas as pd

from .metrics import calc_nse, calc_kge, calc_acc, crps_from_pred_df


def ensemble_mean(pred_df: pd.DataFrame, test_year=None) -> pd.DataFrame:
    """
    Collapse per-member predictions to the ensemble mean per (init_date, date).

    Averages Q_pred across members; all other retained columns are identical
    within an (init_date, date) group and taken from the first row.
    """
    q_pred_mean = pred_df.groupby(['init_date', 'date'])['Q_pred'].mean()
    first_rows = pred_df.drop_duplicates(subset=['init_date', 'date'])

    keep = ['init_date', 'date', 'Q_raw', 'Q_clim', 'lead_day']
    if 'hydro_year' in first_rows.columns and test_year is None:
        keep.append('hydro_year')

    ens = first_rows[keep].copy()
    ens = ens.merge(q_pred_mean, on=['init_date', 'date']).reset_index(drop=True)
    if test_year is not None:
        ens['hydro_year'] = test_year
    return ens


def score_frame(ens: pd.DataFrame) -> dict:
    """Score NSE, KGE, ACC on an ensemble-mean frame (valid rows only)."""
    valid = ens['Q_raw'].notna() & ens['Q_pred'].notna()
    return {
        'nse': calc_nse(ens.loc[valid, 'Q_raw'].values,
                        ens.loc[valid, 'Q_pred'].values),
        'kge': calc_kge(ens.loc[valid, 'Q_raw'].values,
                        ens.loc[valid, 'Q_pred'].values),
        'acc': calc_acc(ens.loc[valid, 'Q_raw'].values,
                        ens.loc[valid, 'Q_pred'].values,
                        ens.loc[valid, 'Q_clim'].values),
    }


def score_members(pred_df: pd.DataFrame, test_year=None):
    """
    NSE/KGE/ACC on the ensemble mean, plus the fair CRPS over the members, of
    a per-member prediction frame. With several hydro years in `pred_df` the
    scores pool them (as the inner CV of the tuning does); pass one year's
    rows for a per-year score. Returns None for an empty frame.
    """
    if pred_df is None or len(pred_df) == 0:
        return None
    scores = score_frame(ensemble_mean(pred_df, test_year=test_year))
    scores['crps'] = crps_from_pred_df(pred_df)
    return scores

