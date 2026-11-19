"""
seas5ml.models.tabular – Flat-table regression models (Ridge, Random
Forest, XGBoost).

Each forecast timestep of each ensemble member is one row (the features of
seas5ml.features.TABULAR_FEATURES); the target is the log1p(Q) anomaly
against the training run's climatology. Features are standardised with a
scaler fit on the training rows.

Training rule: a fit uses only rows that verify in its training years AND
belong to a trajectory initialised in one (seas5ml.data.initialised_in), so
no held-out year's discharge reaches training, not even as the a0 of a
trajectory crossing 1 October.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

from ..data import GROUP_COL, initialised_in
from ..designs import SEED
from ..features import (TABULAR_FEATURES, build_flat_table, set_a0,
                        apply_fold_climatology)

RF_N_JOBS = -1
XGB_N_JOBS = -1

# Prediction frame contract shared by every learned model.
PRED_COLS = ['date', 'init_date', 'member', 'lead_day', 'hydro_year',
             'Q_raw', 'Q_pred', 'Q_clim']


def build_estimator(model, config):
    """Construct the (unfitted) estimator for one tabular configuration
    (see seas5ml.models.configs)."""
    if model == 'ridge':
        return RidgeCV(alphas=config['alphas'], fit_intercept=True)
    if model == 'rf':
        return RandomForestRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            max_features=config['max_features'],
            min_samples_leaf=config['min_samples_leaf'],
            n_jobs=RF_N_JOBS,
            random_state=SEED,
        )
    if model == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['lr'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            min_child_weight=config['min_child_weight'],
            n_jobs=XGB_N_JOBS,
            random_state=SEED,
            verbosity=0,
        )
    raise ValueError(f"No tabular estimator defined for model '{model}'")


def fit_tabular(model, config, df, *, fit_clim, fit_yrs):
    """
    Fit one tabular model on the hydro years `fit_yrs`, against the
    climatology of those years (`fit_clim`, a
    seas5ml.climatology.ClimatologyFitter; a0 is written into `df` in
    place). Returns the fitted-model dict (see seas5ml.models), or None if
    the training table is empty.
    """
    fit_yrs = set(fit_yrs)
    doy_clim = apply_fold_climatology(df, fit_clim, fit_yrs)
    df_tr = df[df[GROUP_COL].isin(fit_yrs) & initialised_in(df, fit_yrs)]
    flat_tr = build_flat_table(df_tr, doy_clim)
    if len(flat_tr) == 0:
        return None

    X_tr = flat_tr[TABULAR_FEATURES].values.astype(np.float32)
    y_tr = flat_tr['target'].values.astype(np.float32)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)

    estimator = build_estimator(model, config)
    estimator.fit(X_tr, y_tr)

    info = {'n_train': len(flat_tr)}
    if model == 'ridge':
        info['best_alpha'] = float(estimator.alpha_)
    return {'kind': 'tabular', 'model': model, 'config': config,
            'estimator': estimator, 'scaler': scaler, 'doy_clim': doy_clim,
            'info': info}


def predict_tabular(fitted, df_score):
    """
    One inference pass over `df_score` (not modified), against the model's
    own climatology: a0, the back-transform and Q_clim all use
    `fitted['doy_clim']`. Returns a per-member prediction frame (PRED_COLS),
    or None if `df_score` has no usable rows.
    """
    doy_clim = fitted['doy_clim']
    df_score = set_a0(df_score.copy(), doy_clim)
    flat = build_flat_table(df_score, doy_clim)
    if len(flat) == 0:
        return None

    X = fitted['scaler'].transform(
        flat[TABULAR_FEATURES].values.astype(np.float32))
    pred_anom = fitted['estimator'].predict(X)

    q_clim_log = flat['q_clim'].values
    scored = flat.copy()
    scored['Q_pred'] = np.clip(np.expm1(pred_anom + q_clim_log), 0.0, None)
    scored['Q_clim'] = np.expm1(q_clim_log)
    return scored.rename(columns={'Q': 'Q_raw'})[PRED_COLS]
