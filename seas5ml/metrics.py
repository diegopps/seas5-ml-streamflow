"""
seas5ml.metrics – Evaluation metrics, shared by every stage.

NSE, KGE and CRPS are scored in real discharge (Q) space; ACC is scored on
log1p anomalies, matching the space the models are trained in (see calc_acc).
NSE, KGE, and ACC operate on the ensemble-mean forecast; CRPS operates on the
per-member ensemble and is the fair (ensemble-size adjusted) CRPS of
Ferro (2014), so 25- and 51-member forecasts are scored comparably.
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


def fair_crps_per_step(obs: np.ndarray, member_preds: np.ndarray) -> np.ndarray:
    """
    Fair (ensemble-size adjusted) CRPS of each timestep's ensemble
    (Ferro 2014, QJRMS 140:1917-1923, doi:10.1002/qj.2270; Ferro et al. 2008,
    Meteorol. Appl. 15:19-24, doi:10.1002/met.45):

        fCRPS = (1/M) sum_i |x_i - y|  -  1/(2 M (M-1)) sum_{i,j} |x_i - x_j|

    The usual estimator divides the spread term by 2 M^2 instead; its
    expectation depends on the ensemble size M, so it penalises the
    25-member hindcasts more than the 51-member operational forecasts for
    the same underlying forecast distribution. The fair version is an
    unbiased estimate of the CRPS of the distribution the members are drawn
    from, so scores are comparable across ensemble sizes. It treats the
    members as a random (exchangeable, independent) sample.

    The pairwise sum uses the sorted-member identity
    sum_{i,j} |x_i - x_j| = 2 sum_k (2k - M - 1) x_(k)   (k = 1..M),
    O(M log M) per timestep instead of O(M^2) memory.

    Parameters
    ----------
    obs : array of shape (T,)
    member_preds : array of shape (T, M), M >= 2, all finite
        Row t must correspond to obs[t].

    Returns
    -------
    array of shape (T,)
    """
    x = np.asarray(member_preds, dtype=float)
    o = np.asarray(obs, dtype=float)
    M = x.shape[1]
    if M < 2:
        raise ValueError("fair CRPS needs at least 2 ensemble members; a "
                         "deterministic forecast's CRPS is its absolute error.")
    term1 = np.mean(np.abs(x - o[:, None]), axis=1)
    k = np.arange(1, M + 1)
    pair_sum = 2.0 * np.sort(x, axis=1) @ (2 * k - M - 1)
    return term1 - pair_sum / (2.0 * M * (M - 1))


def calc_crps(obs: np.ndarray, member_preds: np.ndarray) -> float:
    """
    Mean fair CRPS (see `fair_crps_per_step`) over the valid timesteps of a
    rectangular (T, M) ensemble. Timesteps with a non-finite observation or
    member are skipped. NaN if none is valid.
    """
    obs = np.asarray(obs, dtype=float)
    member_preds = np.asarray(member_preds, dtype=float)
    valid = np.isfinite(obs) & np.all(np.isfinite(member_preds), axis=1)
    if not valid.any():
        return np.nan
    return float(np.mean(fair_crps_per_step(obs[valid], member_preds[valid])))


def crps_from_pred_df(pred_df: pd.DataFrame) -> float:
    """
    Mean fair CRPS, in real Q space, over every (init_date, date) timestep
    of a per-member prediction frame.

    Each timestep is scored over its OWN members. Ensemble size differs by
    initialisation (25 hindcast, 51 operational) and hydro year 2017 holds
    both, so timesteps are grouped by member count and each group pivoted
    separately; a single 51-column pivot would leave the 25-member
    timesteps with empty columns and silently drop them.

    Expects columns: init_date, date, member, Q_pred, Q_raw.
    """
    df = pred_df.dropna(subset=['Q_pred', 'Q_raw'])
    if len(df) == 0:
        return np.nan
    keys = ['init_date', 'date']
    n_members = df.groupby(keys)['member'].transform('nunique')

    per_step = []
    for _, sub in df.groupby(n_members.values):
        pivot = sub.pivot_table(index=keys, columns='member', values='Q_pred',
                                aggfunc='mean')   # one value per cell
        obs = (sub.drop_duplicates(subset=keys).set_index(keys)['Q_raw']
                  .reindex(pivot.index).values)
        per_step.append(fair_crps_per_step(obs, pivot.values))
    return float(np.mean(np.concatenate(per_step)))
