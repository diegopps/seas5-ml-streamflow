"""
flowcast_src.tabular – Flat-table regression models (Ridge, Random Forest,
XGBoost) under LOGO-CV.

All three share one training loop: per fold, build the flat train/test tables,
standardise features on the training rows, fit the estimator on the anomaly
target, back-transform predictions to discharge, and score the ensemble mean.
The models differ only in the estimator constructed per fold and, for Ridge,
a per-fold hyperparameter (the selected alpha) recorded in the metrics.
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .cv import make_folds
from .features import TABULAR_FEATURES, build_flat_table
from .metrics import crps_from_pred_df
from .evaluate import ensemble_mean, score_frame


def run_tabular_model(model_name, df, unique_years, doy_clim,
                      estimator_factory, lead_bins,
                      per_fold_extra=None):
    """
    Run LOGO-CV for one flat-table regression model.

    Parameters
    ----------
    model_name : str
        Short label ('ridge' | 'rf' | 'xgb').
    df : pd.DataFrame
        Analysis frame (features + observed Q, skip_years removed).
    unique_years : list[int]
        Hydro years present in df.
    doy_clim : dict
        Global DOY climatology (log1p space).
    estimator_factory : callable(X_tr, y_tr) -> (fitted_estimator, fold_info)
        Builds and fits the estimator for a fold. `fold_info` is a dict of
        extra per-fold values to record (e.g. {'best_alpha': ...}); may be {}.
    lead_bins : list of (lo, hi, label)
    per_fold_extra : unused hook kept for symmetry.

    Returns
    -------
    mdf : pd.DataFrame          per-fold metrics
    all_preds : pd.DataFrame    ensemble-mean predictions across folds
    all_members : pd.DataFrame  per-member predictions across folds (for CRPS)
    """
    fold_metrics = []
    all_pred_dfs = []
    all_member_dfs = []

    print(f"  {'Fold':>4s}  {'HY':>6s}  {'NSE':>8s}  {'KGE':>8s}  "
          f"{'ACC':>8s}  {'CRPS':>8s}  {'Time':>6s}")
    print(f"  {'-' * 60}")

    for split in make_folds(unique_years):
        fold_i = split['fold_i']
        test_year = split['test_year']
        tr_yrs = set(split['train_years'])   # tabular models use all non-test years
        fold_t0 = time.time()

        df_tr = df[df['hydro_year'].isin(tr_yrs)]
        df_test = df[df['hydro_year'] == test_year]
        if len(df_test) == 0:
            continue

        flat_tr = build_flat_table(df_tr, doy_clim)
        flat_test = build_flat_table(df_test, doy_clim)

        X_tr = flat_tr[TABULAR_FEATURES].values.astype(np.float32)
        y_tr = flat_tr['target'].values.astype(np.float32)
        X_te = flat_test[TABULAR_FEATURES].values.astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        model, fold_info = estimator_factory(X_tr, y_tr)

        pred_anom = model.predict(X_te)
        q_clim_log = flat_test['q_clim'].values
        pred_q = np.clip(np.expm1(pred_anom + q_clim_log), 0.0, None)

        flat_test = flat_test.copy()
        flat_test['Q_pred'] = pred_q
        flat_test['Q_clim'] = np.expm1(q_clim_log)
        flat_test_scored = flat_test.rename(columns={'Q': 'Q_raw'})

        ens = ensemble_mean(flat_test_scored, test_year=test_year)
        scores = score_frame(ens)
        f_crps = crps_from_pred_df(flat_test_scored)

        valid = ens['Q_raw'].notna() & ens['Q_pred'].notna()
        rec = {
            'hydro_year': test_year,
            'nse': scores['nse'], 'kge': scores['kge'],
            'acc': scores['acc'], 'crps': f_crps,
            'n_train_rows': len(flat_tr),
            'n_test_rows': int(valid.sum()),
            'time_s': time.time() - fold_t0,
        }
        rec.update(fold_info)
        fold_metrics.append(rec)

        all_pred_dfs.append(ens[valid])
        all_member_dfs.append(flat_test_scored)

        print(f"  {fold_i+1:4d}  {test_year:6d}  {scores['nse']:+8.3f}  "
              f"{scores['kge']:+8.3f}  {scores['acc']:+8.3f}  {f_crps:8.3f}  "
              f"{rec['time_s']:5.0f}s", flush=True)

    mdf = pd.DataFrame(fold_metrics)
    all_preds = pd.concat(all_pred_dfs, ignore_index=True)
    all_members = pd.concat(all_member_dfs, ignore_index=True)
    return mdf, all_preds, all_members
