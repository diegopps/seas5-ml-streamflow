"""
flowcast_src.estimators – Per-fold estimator factories for the tabular models.

Each factory builds and fits one scikit-learn / XGBoost estimator on the
standardised training features and anomaly target, returning the fitted model
and a dict of any per-fold values worth recording in the metrics.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

from .cv import SEED

# ── Ridge ────────────────────────────────────────────────────────────────
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# ── Random Forest ────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_MAX_FEATURES = 1.0
RF_N_JOBS = -1

# ── XGBoost ──────────────────────────────────────────────────────────────
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.3
XGB_SUBSAMPLE = 1.0
XGB_COLSAMPLE_BYTREE = 1.0
XGB_N_JOBS = -1


def ridge_factory(X_tr, y_tr):
    model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True)
    model.fit(X_tr, y_tr)
    return model, {'best_alpha': float(model.alpha_)}


def rf_factory(X_tr, y_tr):
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        n_jobs=RF_N_JOBS,
        random_state=SEED,
    )
    model.fit(X_tr, y_tr)
    return model, {}


def xgb_factory(X_tr, y_tr):
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        n_jobs=XGB_N_JOBS,
        random_state=SEED,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model, {}


TABULAR_FACTORIES = {
    'ridge': ridge_factory,
    'rf': rf_factory,
    'xgb': xgb_factory,
}
