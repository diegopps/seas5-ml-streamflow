"""
flowcast_src.metrics – Evaluation metrics, shared by every stage.

NSE, KGE and CRPS are scored in real discharge (Q) space; ACC is scored on
log1p anomalies, matching the space the models are trained in (see calc_acc).
NSE, KGE, and ACC operate on the ensemble-mean forecast; CRPS operates on the
per-member ensemble spread.
"""

import numpy as np
import pandas as pd


def calc_nse(obs, pred):
    """Nash-Sutcliffe Efficiency. NaN if observations have zero variance."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    return np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


def calc_kge(obs, pred):
    """Kling-Gupta Efficiency. NaN if either series has zero variance."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if np.std(obs) == 0 or np.std(pred) == 0:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def calc_acc(obs, pred, clim):
    """
    Anomaly Correlation Coefficient, computed on log1p anomalies.

    `obs`, `pred` and `clim` are all passed in real Q space (non-negative) and
    transformed here, so call sites are unchanged.

    The anomaly is taken in log1p space because that is the space the models
    are trained in: the target is log1p(Q) minus the log1p-space DOY
    climatology. Differencing in real Q space instead compares against
    expm1(mean(log1p Q)), a geometric-type mean that sits below the true DOY
    mean by an amount that grows with the skew of the discharge record (at
    Ayat, skew 15.6, it recovers only 41% of mean discharge against 95-96% at
    Zeravshan and Kirchbichl). That shortfall leaves a seasonal signal in the
    residual, so any forecast merely tracking the seasonal cycle correlates
    with it: scored in real space a day-of-year-only forecast reached ACC 0.44
    at Ayat against 0.04 for persistence, and in log1p space the ordering
    reverses to 0.02 and 0.44.

    Note this corrects the ACC reference only. Predictions are still
    back-transformed as expm1(anomaly + log-climatology), which carries the
    same low bias into NSE, KGE and CRPS; and the smoothed climatology cannot
    represent a sharp freshet, which leaves a residual skew-linked inflation
    in ACC at flashy catchments. Both are documented, not corrected.

    NaN if either anomaly series has zero variance (e.g. a predicted anomaly
    that is identically zero, as for the climatology reference forecast).
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    clim = np.asarray(clim, dtype=float)
    obs_anom = np.log1p(obs) - np.log1p(clim)
    pred_anom = np.log1p(pred) - np.log1p(clim)
    if np.std(obs_anom) == 0 or np.std(pred_anom) == 0:
        return np.nan
    return float(np.corrcoef(obs_anom, pred_anom)[0, 1])


def calc_crps(obs: np.ndarray, member_preds: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score for an ensemble forecast, via the
    pairwise-difference estimator:

        CRPS = mean_i |x_i - y|  -  0.5 * mean_ij |x_i - x_j|

    Parameters
    ----------
    obs : array of shape (T,)
        Observed values, one per timestep.
    member_preds : array of shape (T, N)
        Per-member predictions for the same T timesteps, N ensemble members.
        Row t must correspond to obs[t].

    Returns
    -------
    Mean CRPS across all valid (non-NaN) timesteps. NaN if no valid
    timesteps are present.
    """
    valid = np.isfinite(obs) & np.all(np.isfinite(member_preds), axis=1)
    if not valid.any():
        return np.nan

    o = obs[valid]                      # (T_valid,)
    x = member_preds[valid]             # (T_valid, N)

    term1 = np.mean(np.abs(x - o[:, None]), axis=1)                      # (T_valid,)
    term2 = np.mean(np.abs(x[:, :, None] - x[:, None, :]), axis=(1, 2))  # (T_valid,)

    crps_per_t = term1 - 0.5 * term2
    return float(np.mean(crps_per_t))


def crps_from_pred_df(pred_df: pd.DataFrame) -> float:
    """
    Pivot per-member predictions to a (timestep x member) matrix and score
    CRPS in real Q space against the aligned observations.

    Expects columns: init_date, date, member, Q_pred, Q_raw.
    """
    pivot = pred_df.pivot_table(
        index=['init_date', 'date'], columns='member',
        values='Q_pred', aggfunc='mean'  # aggfunc shouldn't matter if unique
    )
    obs_aligned = (pred_df.drop_duplicates(subset=['init_date', 'date'])
                          .set_index(['init_date', 'date'])['Q_raw']
                          .reindex(pivot.index))

    return calc_crps(obs_aligned.values, pivot.values)
