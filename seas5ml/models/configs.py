"""
seas5ml.models.configs – Hyperparameter configurations of the learned models.

DEFAULTS    the hand-set configuration of every learned model: the `default`
            variant of stage 1, and the reference the tuned variant is
            compared against.
GRIDS       the candidate configurations of the nested-CV search (`tuned`
            variant). Only models whose hyperparameters are otherwise set by
            hand are tuned: rf, xgb and lstm. Ridge is not: RidgeCV selects
            its regularisation strength by generalised cross-validation on
            each fold's training data, so its default run is already tuned
            honestly and an external search adds nothing.

Every configuration is a dict with a 'label'; the remaining keys are read by
seas5ml.models.tabular.build_estimator and seas5ml.models.lstm.fit_lstm.
"""

LEARNED_MODELS = ("ridge", "rf", "xgb", "lstm")   # trained per catchment

# Trained on the whole catchment pool (seas5ml.pooled), scored per catchment;
# same default configuration and grid as the single-catchment LSTM.
POOLED_MODELS = ("pooled_lstm",)

DEFAULTS = {
    # alpha is chosen from `alphas` by RidgeCV (generalised cross-validation).
    'ridge': {'label': 'default',
              'alphas': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    'rf':    {'label': 'default', 'n_estimators': 100, 'max_depth': None,
              'max_features': 1.0, 'min_samples_leaf': 1},
    'xgb':   {'label': 'default', 'n_estimators': 100, 'max_depth': 6,
              'lr': 0.3, 'subsample': 1.0, 'colsample_bytree': 1.0,
              'min_child_weight': 1},
    'lstm':  {'label': 'default', 'hidden': 128, 'lr': 3e-4, 'wd': 1e-4,
              'batch': 512},
}
DEFAULTS['pooled_lstm'] = dict(DEFAULTS['lstm'])

# ── LSTM ─────────────────────────────────────────────────────────────────
# Random-search grid (R01-R15), a frozen log-uniform sample over
# (hidden, lr, weight_decay); batch size fixed across the grid.
LSTM_GRID = [
    {'label': 'R01', 'hidden':  64, 'lr': 3.723866e-04, 'wd': 1e-04, 'batch': 512},
    {'label': 'R02', 'hidden': 128, 'lr': 8.077867e-04, 'wd': 1e-04, 'batch': 512},
    {'label': 'R03', 'hidden':  64, 'lr': 1.859148e-03, 'wd': 0.0,   'batch': 512},
    {'label': 'R04', 'hidden': 256, 'lr': 1.053642e-03, 'wd': 1e-04, 'batch': 512},
    {'label': 'R05', 'hidden': 256, 'lr': 3.854478e-04, 'wd': 0.0,   'batch': 512},
    {'label': 'R06', 'hidden': 256, 'lr': 1.606013e-03, 'wd': 1e-05, 'batch': 512},
    {'label': 'R07', 'hidden': 512, 'lr': 1.176082e-03, 'wd': 1e-05, 'batch': 512},
    {'label': 'R08', 'hidden': 256, 'lr': 1.975352e-04, 'wd': 1e-05, 'batch': 512},
    {'label': 'R09', 'hidden':  64, 'lr': 1.210677e-04, 'wd': 1e-05, 'batch': 512},
    {'label': 'R10', 'hidden': 512, 'lr': 6.634594e-04, 'wd': 1e-04, 'batch': 512},
    {'label': 'R11', 'hidden':  64, 'lr': 2.892337e-04, 'wd': 1e-04, 'batch': 512},
    {'label': 'R12', 'hidden':  64, 'lr': 1.452036e-03, 'wd': 1e-04, 'batch': 512},
    {'label': 'R13', 'hidden': 256, 'lr': 1.791558e-04, 'wd': 1e-04, 'batch': 512},
    {'label': 'R14', 'hidden': 128, 'lr': 1.140224e-04, 'wd': 1e-05, 'batch': 512},
    {'label': 'R15', 'hidden': 256, 'lr': 7.738685e-04, 'wd': 0.0,   'batch': 512},
]

# ── Random Forest ────────────────────────────────────────────────────────
# n_estimators is capped at 400: the flat table holds one row per forecast
# timestep per ensemble member, so deep unbounded forests over many trees get
# expensive fast across configs x inner folds x outer folds.
RF_GRID = [
    {'label': 'R01', 'n_estimators': 200, 'max_depth': None, 'max_features': 1.0,  'min_samples_leaf': 1},
    {'label': 'R02', 'n_estimators': 200, 'max_depth': 10,   'max_features': 0.6,  'min_samples_leaf': 5},
    {'label': 'R03', 'n_estimators': 400, 'max_depth': 20,   'max_features': 0.3,  'min_samples_leaf': 1},
    {'label': 'R04', 'n_estimators': 100, 'max_depth': None, 'max_features': 0.6,  'min_samples_leaf': 10},
    {'label': 'R05', 'n_estimators': 400, 'max_depth': None, 'max_features': 0.3,  'min_samples_leaf': 5},
    {'label': 'R06', 'n_estimators': 200, 'max_depth': 20,   'max_features': 1.0,  'min_samples_leaf': 2},
    {'label': 'R07', 'n_estimators': 300, 'max_depth': 15,   'max_features': 0.6,  'min_samples_leaf': 1},
    {'label': 'R08', 'n_estimators': 100, 'max_depth': 10,   'max_features': 0.3,  'min_samples_leaf': 2},
    {'label': 'R09', 'n_estimators': 400, 'max_depth': 10,   'max_features': 1.0,  'min_samples_leaf': 10},
    {'label': 'R10', 'n_estimators': 300, 'max_depth': None, 'max_features': 1.0,  'min_samples_leaf': 5},
    {'label': 'R11', 'n_estimators': 200, 'max_depth': 30,   'max_features': 0.6,  'min_samples_leaf': 2},
    {'label': 'R12', 'n_estimators': 100, 'max_depth': 20,   'max_features': 0.6,  'min_samples_leaf': 1},
    {'label': 'R13', 'n_estimators': 300, 'max_depth': 30,   'max_features': 0.3,  'min_samples_leaf': 10},
    {'label': 'R14', 'n_estimators': 400, 'max_depth': 15,   'max_features': 0.3,  'min_samples_leaf': 2},
    {'label': 'R15', 'n_estimators': 300, 'max_depth': 10,   'max_features': 1.0,  'min_samples_leaf': 1},
]

# ── XGBoost ──────────────────────────────────────────────────────────────
XGB_GRID = [
    {'label': 'R01', 'n_estimators': 300, 'max_depth': 3, 'lr': 0.05,  'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1},
    {'label': 'R02', 'n_estimators': 800, 'max_depth': 6, 'lr': 0.01,  'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 5},
    {'label': 'R03', 'n_estimators': 300, 'max_depth': 9, 'lr': 0.10,  'subsample': 1.0, 'colsample_bytree': 0.8, 'min_child_weight': 1},
    {'label': 'R04', 'n_estimators': 500, 'max_depth': 6, 'lr': 0.05,  'subsample': 0.6, 'colsample_bytree': 1.0, 'min_child_weight': 5},
    {'label': 'R05', 'n_estimators': 800, 'max_depth': 3, 'lr': 0.05,  'subsample': 0.8, 'colsample_bytree': 0.6, 'min_child_weight': 3},
    {'label': 'R06', 'n_estimators': 300, 'max_depth': 6, 'lr': 0.10,  'subsample': 0.6, 'colsample_bytree': 0.8, 'min_child_weight': 10},
    {'label': 'R07', 'n_estimators': 500, 'max_depth': 9, 'lr': 0.01,  'subsample': 0.8, 'colsample_bytree': 1.0, 'min_child_weight': 1},
    {'label': 'R08', 'n_estimators': 800, 'max_depth': 9, 'lr': 0.05,  'subsample': 1.0, 'colsample_bytree': 0.6, 'min_child_weight': 5},
    {'label': 'R09', 'n_estimators': 300, 'max_depth': 3, 'lr': 0.01,  'subsample': 1.0, 'colsample_bytree': 1.0, 'min_child_weight': 3},
    {'label': 'R10', 'n_estimators': 500, 'max_depth': 3, 'lr': 0.10,  'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 1},
    {'label': 'R11', 'n_estimators': 800, 'max_depth': 6, 'lr': 0.10,  'subsample': 0.8, 'colsample_bytree': 1.0, 'min_child_weight': 10},
    {'label': 'R12', 'n_estimators': 500, 'max_depth': 6, 'lr': 0.03,  'subsample': 1.0, 'colsample_bytree': 0.8, 'min_child_weight': 3},
    {'label': 'R13', 'n_estimators': 300, 'max_depth': 9, 'lr': 0.03,  'subsample': 0.8, 'colsample_bytree': 0.6, 'min_child_weight': 5},
    {'label': 'R14', 'n_estimators': 800, 'max_depth': 12, 'lr': 0.01, 'subsample': 0.6, 'colsample_bytree': 0.8, 'min_child_weight': 10},
    {'label': 'R15', 'n_estimators': 500, 'max_depth': 12, 'lr': 0.05, 'subsample': 1.0, 'colsample_bytree': 1.0, 'min_child_weight': 1},
]

GRIDS = {
    'lstm': LSTM_GRID,
    'rf': RF_GRID,
    'xgb': XGB_GRID,
    'pooled_lstm': LSTM_GRID,
}
# Tuned per catchment by stage 1's run.py; the pooled models are tuned by
# run_pooled.py.
TUNED_MODELS = ('lstm', 'rf', 'xgb')


def grid_config(model: str, label: str) -> dict:
    """The configuration named `label` in `model`'s grid; raises if absent."""
    by_label = {c['label']: c for c in GRIDS[model]}
    if label not in by_label:
        raise ValueError(f"config '{label}' is not in the {model} grid")
    return by_label[label]
