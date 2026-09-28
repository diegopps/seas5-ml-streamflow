"""
flowcast_src.hptuning – Nested-CV hyperparameter search.

One outer fold = one held-out hydro year. The outer-training years are
partitioned into inner folds; each candidate config is trained on each inner
fold's inner-training years (with its own early-stopping split nested inside)
and scored on the inner fold's held-out year(s). The config with the best mean
inner-CV ACC is the winner for that outer fold, retrained on the full
outer-training set and scored on the outer-test year — the leakage-free score,
since the outer-test year never contributes to config selection.

The untuned reference the winner is compared against is stage 01's own result
for the same catchment, model and year — not a retrain. Stage 01 fits these
same default configs on the same split, so its per-fold row already is the
untuned baseline, and reusing it keeps every consumer (the metrics file and
the tuning figure alike) on one identical set of numbers. BASELINE_CONFIG
below therefore documents which config that reference corresponds to; it is
recorded in the outputs rather than trained here. Both stages use the same
single global climatology, so ACC values are comparable across stages, not
just within this one.

The families tuned here are the ones whose hyperparameters are otherwise fixed
by hand: `lstm`, `rf`, and `xgb`. Ridge is not among them — RidgeCV selects its
regularisation strength by generalised cross-validation on each fold's training
data, so it is already tuned honestly inside stage 01 and an external search
adds nothing.

Every model family is keyed by name in `CONFIGS`, `BASELINE_CONFIG`, and
`TRAIN_AND_SCORE`, so the nested loop itself is model-agnostic: adding a family
means adding a config grid, a baseline, and a train/score function here.

Each family's train/score function shares one signature — (df, obs_daily_q,
doy_clim, hp, tr_yrs, val_yrs, score_yrs, ckpt_path) -> (metrics, pred_df) —
even where an argument does not apply. The LSTM early-stops on `val_yrs` and
checkpoints to `ckpt_path`; the tabular models do neither, so they fold
`val_yrs` back into training and ignore `ckpt_path`.

`doy_clim` is the single global climatology for the catchment (see
`flowcast_src.climatology.compute_global_climatology`), computed once and
passed in by the caller — exactly as stage 01 does — rather than refit per
fold, so every model, every config, the winner, and the baseline all share one
anomaly definition and are directly comparable.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from .cv import SEED, VAL_FRAC
from .evaluate import ensemble_mean, score_frame
from .features import TABULAR_FEATURES, build_flat_table
from .metrics import crps_from_pred_df
from . import estimators as est
from . import lstm as lstm_mod

GROUP_COL = 'hydro_year'

# Columns every family's prediction frame carries, so downstream consumers see
# one contract regardless of which model produced the predictions.
PRED_COLS = ['date', 'init_date', 'member', 'lead_day', 'hydro_year',
             'Q_raw', 'Q_pred', 'Q_clim']

# Random-search grid (R01-R15), a frozen log-uniform sample over
# (hidden, lr, weight_decay); batch size fixed across the grid.
LSTM_CONFIGS = [
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

# The stage-01 exploration LSTM's fixed defaults — the untuned reference every
# outer fold's winner is compared against.
LSTM_BASELINE_CONFIG = {
    'label': 'baseline', 'hidden': lstm_mod.HIDDEN_SIZE, 'lr': lstm_mod.LR,
    'wd': lstm_mod.WEIGHT_DECAY, 'batch': lstm_mod.BATCH_SIZE,
}

# ── Random Forest ────────────────────────────────────────────────────────
# n_estimators is capped at 400: the flat table holds one row per forecast
# timestep per ensemble member, so deep unbounded forests over many trees get
# expensive fast across configs x inner folds x outer folds.
RF_CONFIGS = [
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

RF_BASELINE_CONFIG = {
    'label': 'baseline',
    'n_estimators': est.RF_N_ESTIMATORS,
    'max_depth': est.RF_MAX_DEPTH,
    'max_features': est.RF_MAX_FEATURES,
    'min_samples_leaf': 1,
}

# ── XGBoost ──────────────────────────────────────────────────────────────
XGB_CONFIGS = [
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

XGB_BASELINE_CONFIG = {
    'label': 'baseline',
    'n_estimators': est.XGB_N_ESTIMATORS,
    'max_depth': est.XGB_MAX_DEPTH,
    'lr': est.XGB_LEARNING_RATE,
    'subsample': est.XGB_SUBSAMPLE,
    'colsample_bytree': est.XGB_COLSAMPLE_BYTREE,
    'min_child_weight': 1,
}


# Only model families whose hyperparameters are fixed by hand are tuned here.
# Ridge is deliberately absent: RidgeCV already selects its regularisation
# strength by generalised cross-validation on each fold's training data, so it
# is tuned honestly within stage 01 and needs no external search. It stays a
# stage-01 reference model, alongside climatology and persistence.
CONFIGS = {
    'lstm': LSTM_CONFIGS,
    'rf': RF_CONFIGS,
    'xgb': XGB_CONFIGS,
}
BASELINE_CONFIG = {
    'lstm': LSTM_BASELINE_CONFIG,
    'rf': RF_BASELINE_CONFIG,
    'xgb': XGB_BASELINE_CONFIG,
}


def make_val_split(train_years, *, val_seed):
    """Whole-year train/val split for early stopping (VAL_FRAC of the years,
    at least 1). `train_years` is a list; returns (tr_yrs, val_yrs) as sets."""
    train_years = list(train_years)
    n_val = max(1, int(len(train_years) * VAL_FRAC))
    rng = np.random.default_rng(seed=val_seed)
    val_idx = set(rng.choice(len(train_years), size=n_val,
                             replace=False).tolist())
    val_yrs = {train_years[i] for i in val_idx}
    tr_yrs = {train_years[i] for i in range(len(train_years))
              if i not in val_idx}
    return tr_yrs, val_yrs


# Sentinels for the two retrains, which sit outside the (inner fold, config)
# coordinate space that inner CV walks.
INNER_IDX_RETRAIN = -1
CONFIG_IDX_BASELINE = -2


def val_seed_for(outer_idx, inner_idx, config_idx, *, seed=SEED):
    """
    Reproducible, collision-free seed for a val-split, keyed by the full
    coordinate (outer fold, inner fold, config), so distinct coordinates never
    share a seed regardless of iteration order. `seed` is the run's global
    seed, so a run under a different global seed also draws different splits.

    inner_idx=INNER_IDX_RETRAIN marks a retrain (no inner level).
    config_idx=CONFIG_IDX_BASELINE marked the baseline retrain, which stage 02
    no longer performs; the sentinel is kept so that seeds drawn for the other
    coordinates are unchanged from earlier runs. SeedSequence
    rejects negative entries, so both coordinates are shifted up into
    non-negative territory before being hashed.
    """
    ss = np.random.SeedSequence([seed, outer_idx,
                                 inner_idx - INNER_IDX_RETRAIN,
                                 config_idx - CONFIG_IDX_BASELINE])
    return int(ss.generate_state(1)[0])


def inner_partition(outer_train_years, *, n_inner, seed):
    """Partition the outer-training years into `n_inner` roughly-equal disjoint
    groups (the inner folds). Reproducible: shuffles the sorted year list with
    a fixed SeedSequence, then round-robins into n_inner buckets so sizes
    differ by at most 1. Returns a list of sets (empty buckets dropped)."""
    years = sorted(outer_train_years)
    ss = np.random.SeedSequence([seed, 987654321])   # fixed salt
    rng = np.random.default_rng(ss)
    order = rng.permutation(len(years))
    buckets = [set() for _ in range(n_inner)]
    for rank, idx in enumerate(order):
        buckets[rank % n_inner].add(years[idx])
    return [b for b in buckets if b]


def train_and_score_lstm(df, obs_daily_q, *, doy_clim, hp, tr_yrs, val_yrs,
                         score_yrs, ckpt_path):
    """
    Train the LSTM for hyperparameter config `hp` on `tr_yrs` (with `val_yrs`
    reserved for early stopping) and score it on `score_yrs` (one or more
    held-out hydro years), against the shared global `doy_clim`. Returns
    (metrics_dict, pred_df), or (None, None) if `score_yrs` has no data.
    """
    if not df[GROUP_COL].isin(score_yrs).any():
        return None, None

    # Whole trajectories from the full frame; each call scores only its own
    # years (see lstm.build_trajectories).
    tr_trajs, dyn_sc = lstm_mod.build_trajectories(df, doy_clim, None, True,
                                                   score_years=tr_yrs)
    val_trajs, _ = lstm_mod.build_trajectories(df, doy_clim, dyn_sc, False,
                                               score_years=val_yrs)
    score_trajs, _ = lstm_mod.build_trajectories(df, doy_clim, dyn_sc, False,
                                                 score_years=score_yrs)

    from torch.utils.data import DataLoader
    tr_loader = DataLoader(lstm_mod.TrajectoryDataset(tr_trajs),
                           batch_size=hp['batch'], shuffle=True,
                           collate_fn=lstm_mod.collate_fn,
                           num_workers=0, pin_memory=True)
    val_loader = DataLoader(lstm_mod.TrajectoryDataset(val_trajs),
                            batch_size=hp['batch'], shuffle=False,
                            collate_fn=lstm_mod.collate_fn,
                            num_workers=0, pin_memory=True)

    model = lstm_mod.DischargeLSTM(len(lstm_mod.DYN_VARS), hp['hidden'],
                                   lstm_mod.N_LAYERS, lstm_mod.DROPOUT
                                   ).to(lstm_mod.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'],
                                 weight_decay=hp['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lstm_mod.LR_FACTOR,
        patience=lstm_mod.LR_PATIENCE)
    amp_scaler = torch.amp.GradScaler('cuda', enabled=lstm_mod.USE_AMP)

    best_val_loss = np.inf
    best_epoch = 0
    patience_count = 0

    for epoch in range(1, lstm_mod.MAX_EPOCHS + 1):
        lstm_mod.run_epoch(model, tr_loader, optimizer, amp_scaler, True)
        val_loss = lstm_mod.run_epoch(model, val_loader, optimizer,
                                      amp_scaler, False)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= lstm_mod.PATIENCE:
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=lstm_mod.DEVICE,
                                     weights_only=True))
    pred_df = lstm_mod.predict_trajectories(model, score_trajs)

    ens = ensemble_mean(pred_df)
    scores = score_frame(ens)
    metrics = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'nse': scores['nse'], 'kge': scores['kge'], 'acc': scores['acc'],
        'crps': crps_from_pred_df(pred_df),
        'n_train_traj': len(tr_trajs),
        'n_score_traj': len(score_trajs),
    }
    return metrics, pred_df


def build_tabular_estimator(model, hp):
    """Construct (unfitted) the estimator for one tabular config."""
    if model == 'rf':
        return RandomForestRegressor(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            max_features=hp['max_features'],
            min_samples_leaf=hp['min_samples_leaf'],
            n_jobs=est.RF_N_JOBS,
            random_state=SEED,
        )
    if model == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=hp['lr'],
            subsample=hp['subsample'],
            colsample_bytree=hp['colsample_bytree'],
            min_child_weight=hp['min_child_weight'],
            n_jobs=est.XGB_N_JOBS,
            random_state=SEED,
            verbosity=0,
        )
    raise ValueError(f"No tabular estimator defined for model '{model}'")


def train_and_score_tabular(model):
    """
    Build the train/score function for one tabular family.

    The returned function matches the shared train/score signature. Tabular
    models do not early-stop, so `val_yrs` is folded back into the training
    years and `ckpt_path` is unused.
    """
    def _run(df, obs_daily_q, *, doy_clim, hp, tr_yrs, val_yrs, score_yrs,
            ckpt_path=None):
        fit_yrs = set(tr_yrs) | set(val_yrs)
        df_tr = df[df[GROUP_COL].isin(fit_yrs)]
        df_score = df[df[GROUP_COL].isin(score_yrs)]
        if len(df_score) == 0:
            return None, None

        flat_tr = build_flat_table(df_tr, doy_clim)
        flat_score = build_flat_table(df_score, doy_clim)
        if len(flat_tr) == 0 or len(flat_score) == 0:
            return None, None

        X_tr = flat_tr[TABULAR_FEATURES].values.astype(np.float32)
        y_tr = flat_tr['target'].values.astype(np.float32)
        X_sc = flat_score[TABULAR_FEATURES].values.astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_sc = scaler.transform(X_sc)

        estimator = build_tabular_estimator(model, hp)
        estimator.fit(X_tr, y_tr)

        pred_anom = estimator.predict(X_sc)
        q_clim_log = flat_score['q_clim'].values
        pred_q = np.clip(np.expm1(pred_anom + q_clim_log), 0.0, None)

        scored = flat_score.copy()
        scored['Q_pred'] = pred_q
        scored['Q_clim'] = np.expm1(q_clim_log)
        scored = scored.rename(columns={'Q': 'Q_raw'})[PRED_COLS]

        ens = ensemble_mean(scored)
        scores = score_frame(ens)
        metrics = {
            'best_epoch': np.nan,       # not applicable: no iterative training
            'best_val_loss': np.nan,
            'nse': scores['nse'], 'kge': scores['kge'], 'acc': scores['acc'],
            'crps': crps_from_pred_df(scored),
            'n_train_traj': len(flat_tr),
            'n_score_traj': len(flat_score),
        }
        return metrics, scored

    return _run


TRAIN_AND_SCORE = {
    'lstm': train_and_score_lstm,
    'rf': train_and_score_tabular('rf'),
    'xgb': train_and_score_tabular('xgb'),
}
