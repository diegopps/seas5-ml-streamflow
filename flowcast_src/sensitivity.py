"""
flowcast_src.sensitivity – Climate-input sensitivity test.

Asks whether the learned models actually use the meteorological forcing at
inference time, or whether they are really tracking the seasonal cycle and
would score the same on forcing that carries no information about the target
year. A model whose skill barely moves when its forcing is replaced by another
year's was never using that forcing.

Two inference scenarios per fold, both scored on the same held-out year:

  normal  real forcing for the target year — the reference score.
  mixed   the seven meteorological variables are replaced wholesale by a
          randomly chosen donor hydro year's, matched on the exact structural
          analogue (init month, lead day, ensemble member slot) so the swapped
          series keeps the real forecast's seasonal position, lead-time
          structure, and ensemble spread. Only the year the weather came from
          changes.

Donor-year swapping rather than added noise
-------------------------------------------
The corruption is deliberately a real year's weather rather than synthetic
noise. Noise-corrupted forcing pushes the inputs out of the distribution the
model was trained on, so a skill drop has two possible causes that cannot be
told apart: the model was genuinely using the forcing, or it was merely thrown
off by values it had never seen. A donor year is in-distribution by
construction — real weather, real ensemble spread, real lead-time structure —
so a skill drop can only mean the model was using *this* year's weather
specifically. That is the question, and swapping years asks it cleanly.

The calendar encodings (sin_doy, cos_doy) are never corrupted. They carry no
weather information — they are the model's legitimate access to "what time of
year is it", and corrupting them would test something else entirely (whether
the model knows the season), which is not the question. The antecedent anomaly
(a0) is likewise never corrupted: it is the observed log1p(Q) anomaly at
initialisation, not meteorological forcing. The corruption writes only
MET_VARS, so a0 passes through untouched, and the test measures use of the
forcing rather than use of the initial condition.

Train once, infer twice
-----------------------
Each fold trains exactly one model and then runs two inference passes over it.
Retraining per scenario would confound the effect of corrupting the input with
ordinary training stochasticity — for the LSTM especially, two runs of the same
config on the same data differ enough to swamp the effect being measured.
Scenario deltas are therefore differences in inference input alone, with model
weights held fixed.

Which hyperparameters
---------------------
rf, xgb, and lstm each use ONE config per catchment: whichever config won the
most outer folds in stage 02's nested search (the 'best' column of
03_configranking.pdf/.csv), applied to every fold here. This tests the single
config that would actually be deployed for the catchment, rather than a
different per-fold winner. Ridge has no stage-02 search (RidgeCV selects its
own regularisation by internal generalised cross-validation) and uses its
stage-01 definition.

Paired-row guarantee
--------------------
The two scenarios are compared per fold as paired differences, so they must be
scored over identical rows. A donor year can be missing an (init month, lead
day, member) combination that the target year has — rare, but real at the
record's edges. Rows that fail to find a donor match are dropped from both
scenarios, not just from `mixed`, so every comparison stays paired.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor

from .cv import SEED
from .evaluate import ensemble_mean, score_frame
from .features import TABULAR_FEATURES, build_flat_table
from .metrics import crps_from_pred_df
from . import estimators as est
from . import lstm as lstm_mod

GROUP_COL = 'hydro_year'

# The seven meteorological drivers — the only columns the corruption touches.
# `corrupt_mixed` writes these by explicit loop, so every other dynamic
# predictor (sin_doy, cos_doy, a0) passes through untouched by construction.
MET_VARS = ['tas', 'tasmax', 'tasmin', 'ssrd', 'tp', 'sd', 'sf']

SCENARIOS = ('normal', 'mixed')

# Fixed salt keeping the donor draw reproducible from the fold seed alone.
DONOR_SALT = 811

# Prediction frame contract, matching stage 01/02.
PRED_COLS = ['date', 'init_date', 'member', 'lead_day', 'hydro_year',
             'Q_raw', 'Q_pred', 'Q_clim']


# ═══════════════════════════════════════════════════════════════════════
# INPUT CORRUPTION
# ═══════════════════════════════════════════════════════════════════════

def pick_donor_year(unique_years, test_year, *, fold_seed):
    """Choose the donor hydro year for the `mixed` scenario, reproducibly."""
    candidates = [y for y in unique_years if y != test_year]
    if not candidates:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([fold_seed, DONOR_SALT]))
    return int(rng.choice(candidates))


def corrupt_mixed(df_test, df_full, donor_year):
    """
    Replace MET_VARS with the donor hydro year's values.

    Matching is on (init month, lead day, member slot), which is the donor's
    structurally equivalent forecast: same point in the seasonal cycle, same
    forecast horizon, same position in the ensemble. Verified unique on this
    dataset — (hydro_year, init_month, lead_day, member) has no duplicates.

    Member counts differ by era (25 in the hindcast years, 51 in the forecast
    years), so the target's members are mapped onto the donor's by position,
    wrapping if the donor has fewer. This keeps genuine member-to-member
    variation in the swapped forcing rather than collapsing every member onto
    one averaged series, which would silently shrink ensemble spread and, via
    the pairwise-spread term, flatter CRPS for reasons unrelated to skill.

    Returns the corrupted frame; rows with no donor match carry NaN met values
    and are filtered by the caller (see `paired_row_mask`).
    """
    donor = df_full[df_full[GROUP_COL] == donor_year]
    if len(donor) == 0:
        return None

    test_members = np.sort(df_test['member'].unique())
    donor_members = np.sort(donor['member'].unique())
    slot = {int(m): int(donor_members[i % len(donor_members)])
            for i, m in enumerate(test_members)}

    donor = donor.copy()
    donor['_im'] = donor['init_date'].dt.month
    lut = donor.set_index(['_im', 'lead_day', 'member'])[MET_VARS]

    out = df_test.copy()
    out['_im'] = out['init_date'].dt.month
    out['_dm'] = out['member'].map(slot)

    idx = pd.MultiIndex.from_arrays([out['_im'], out['lead_day'], out['_dm']])
    swapped = lut.reindex(idx)
    for v in MET_VARS:
        out[v] = swapped[v].values

    return out.drop(columns=['_im', '_dm'])


def paired_row_mask(df_mixed):
    """Rows whose donor swap produced complete met values. Applied to both
    scenarios so they are scored over identical rows (see module docstring)."""
    return df_mixed[MET_VARS].notna().all(axis=1).values


def build_scenarios(df_test, df_full, unique_years, test_year, *, fold_seed):
    """
    Build the two scenario frames for one fold, restricted to the rows both
    scenarios can supply.

    `df_test` holds every trajectory that reaches `test_year`, whole, so the
    LSTM can read each from initialisation (see lstm.build_trajectories);
    rows from neighbouring hydro years are input context and never scored.
    The donor swap covers the context rows too, so a corrupted trajectory is
    corrupted from lead 0. Only test-year rows are subject to the paired-row
    filter: dropping a context row would open a gap inside a sequence. A
    context row without a donor match keeps NaN forcing, which the LSTM's
    standardisation maps to the training mean.

    Returns (frames, donor_year, n_rows, n_dropped) where `frames` maps
    scenario name to its frame and the counts refer to test-year rows, or
    (None, ...) if no donor year is available.
    """
    donor_year = pick_donor_year(unique_years, test_year, fold_seed=fold_seed)
    if donor_year is None:
        return None, None, 0, 0

    df_mixed_full = corrupt_mixed(df_test, df_full, donor_year)
    if df_mixed_full is None:
        return None, donor_year, 0, 0

    in_test = (df_mixed_full[GROUP_COL] == test_year).values
    matched = paired_row_mask(df_mixed_full)
    keep = matched | ~in_test
    n_dropped = int((in_test & ~matched).sum())

    df_base = df_test.loc[keep].reset_index(drop=True)
    df_mixed = df_mixed_full.loc[keep].reset_index(drop=True)

    frames = {'normal': df_base, 'mixed': df_mixed}
    return frames, donor_year, int((in_test & matched).sum()), n_dropped


# ═══════════════════════════════════════════════════════════════════════
# FIT ONCE / PREDICT MANY
# ═══════════════════════════════════════════════════════════════════════
#
# Stage 01 and stage 02 both train and score in a single call, which is the
# right shape when each trained model is scored once. This stage scores one
# trained model twice, so training and inference are separated here.

def build_estimator(model, config):
    """
    Construct the (unfitted) tabular estimator for one model.

    `config` is a stage-02 winner config for rf/xgb. Ridge takes none: RidgeCV
    picks its own alpha by internal generalised cross-validation on the
    training fold, which is why it is absent from the stage-02 search.
    """
    if model == 'ridge':
        return RidgeCV(alphas=est.RIDGE_ALPHAS, fit_intercept=True)
    if model == 'rf':
        return RandomForestRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            max_features=config['max_features'],
            min_samples_leaf=config['min_samples_leaf'],
            n_jobs=est.RF_N_JOBS,
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
            n_jobs=est.XGB_N_JOBS,
            random_state=SEED,
            verbosity=0,
        )
    raise ValueError(f"No tabular estimator defined for model '{model}'")


def fit_tabular(model, df, *, doy_clim, config, fit_yrs):
    """Fit one tabular model on `fit_yrs`. Returns a dict holding everything
    inference needs, or None if the training fold is empty."""
    df_tr = df[df[GROUP_COL].isin(fit_yrs)]
    flat_tr = build_flat_table(df_tr, doy_clim)
    if len(flat_tr) == 0:
        return None

    X_tr = flat_tr[TABULAR_FEATURES].values.astype(np.float32)
    y_tr = flat_tr['target'].values.astype(np.float32)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)

    estimator = build_estimator(model, config)
    estimator.fit(X_tr, y_tr)

    return {'kind': 'tabular', 'estimator': estimator, 'scaler': scaler,
            'n_train_rows': len(flat_tr)}


def predict_tabular(fitted, df_score, *, doy_clim):
    """Run one tabular inference pass. Returns a per-member prediction frame."""
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


def fit_lstm(df, *, doy_clim, config, tr_yrs, val_yrs, ckpt_path):
    """
    Train the LSTM for one config, early-stopping on `val_yrs`.

    Assembled from the same `flowcast_src.lstm` primitives stage 02 uses, but
    returning the trained model rather than a score, so the caller can run
    several inference passes over identical weights.
    """
    from torch.utils.data import DataLoader

    if not (df[GROUP_COL].isin(tr_yrs).any() and df[GROUP_COL].isin(val_yrs).any()):
        return None

    # Whole trajectories from the full frame; each call scores only its own
    # years (see lstm.build_trajectories).
    tr_trajs, dyn_sc = lstm_mod.build_trajectories(df, doy_clim, None, True,
                                                   score_years=tr_yrs)
    val_trajs, _ = lstm_mod.build_trajectories(df, doy_clim, dyn_sc, False,
                                               score_years=val_yrs)

    tr_loader = DataLoader(lstm_mod.TrajectoryDataset(tr_trajs),
                           batch_size=config['batch'], shuffle=True,
                           collate_fn=lstm_mod.collate_fn,
                           num_workers=0, pin_memory=True)
    val_loader = DataLoader(lstm_mod.TrajectoryDataset(val_trajs),
                            batch_size=config['batch'], shuffle=False,
                            collate_fn=lstm_mod.collate_fn,
                            num_workers=0, pin_memory=True)

    model = lstm_mod.DischargeLSTM(len(lstm_mod.DYN_VARS), config['hidden'],
                                   lstm_mod.N_LAYERS, lstm_mod.DROPOUT
                                   ).to(lstm_mod.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=config['wd'])
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
    # `arch` records what rebuilding the module needs, so a saved state dict
    # can be reloaded without re-deriving the shape from the config.
    arch = {'n_dyn': len(lstm_mod.DYN_VARS), 'hidden': config['hidden'],
            'n_layers': lstm_mod.N_LAYERS, 'dropout': lstm_mod.DROPOUT}
    return {'kind': 'lstm', 'model': model, 'dyn_scaler': dyn_sc, 'arch': arch,
            'best_epoch': best_epoch, 'best_val_loss': float(best_val_loss),
            'n_train_rows': len(tr_trajs)}


def predict_lstm(fitted, df_score, *, doy_clim, score_years):
    """Run one LSTM inference pass. Returns a per-member prediction frame,
    for rows in `score_years` only; other rows of `df_score` are read as
    trajectory context.

    The trajectory builder standardises the (corrupted) features with the
    training fold's scaler, exactly as at training time — corrupted inputs are
    deliberately NOT re-standardised to their own statistics, since a model
    fed out-of-distribution values is precisely what this test measures."""
    trajs, _ = lstm_mod.build_trajectories(df_score, doy_clim,
                                           fitted['dyn_scaler'], False,
                                           score_years=score_years)
    if not trajs:
        return None
    return lstm_mod.predict_trajectories(fitted['model'], trajs)


def predict(fitted, df_score, *, doy_clim, score_years):
    """Dispatch one inference pass to the right backend. Only rows in
    `score_years` are predicted; tabular models need no context rows."""
    if fitted['kind'] == 'lstm':
        return predict_lstm(fitted, df_score, doy_clim=doy_clim,
                            score_years=score_years)
    df_score = df_score[df_score[GROUP_COL].isin(score_years)]
    return predict_tabular(fitted, df_score, doy_clim=doy_clim)


def save_fitted(fitted, path, *, meta=None):
    """
    Persist a fitted model so it can be reloaded without retraining.

    Everything inference needs travels with the weights: the tabular models
    carry their fitted StandardScaler, the LSTM its dynamic-feature scaler and
    the architecture arguments needed to rebuild the module before loading the
    state dict. Without those, saved weights are not enough to reproduce a
    prediction — the scalers are fitted objects, not derivable from the config.

    `meta` is free-form provenance (catchment, config label, training years)
    stored alongside, so a saved model can always be traced back to the run
    that produced it.
    """
    import joblib

    payload = {'kind': fitted['kind'], 'meta': meta or {}}

    if fitted['kind'] == 'lstm':
        payload['state_dict'] = {k: v.cpu() for k, v
                                 in fitted['model'].state_dict().items()}
        payload['arch'] = fitted['arch']
        payload['dyn_scaler'] = fitted['dyn_scaler']
        payload['best_epoch'] = fitted.get('best_epoch')
        torch.save(payload, path)
    else:
        payload['estimator'] = fitted['estimator']
        payload['scaler'] = fitted['scaler']
        joblib.dump(payload, path)

    return path


def load_fitted(path):
    """
    Reload a model saved by `save_fitted` into the same dict shape `predict`
    expects, so a reloaded model is used exactly like a freshly fitted one.
    """
    import joblib

    if str(path).endswith('.pt'):
        payload = torch.load(path, map_location=lstm_mod.DEVICE,
                             weights_only=False)
        arch = payload['arch']
        model = lstm_mod.DischargeLSTM(arch['n_dyn'], arch['hidden'],
                                       arch['n_layers'], arch['dropout']
                                       ).to(lstm_mod.DEVICE)
        model.load_state_dict(payload['state_dict'])
        model.eval()
        return {'kind': 'lstm', 'model': model,
                'dyn_scaler': payload['dyn_scaler'],
                'best_epoch': payload.get('best_epoch'),
                'meta': payload.get('meta', {})}

    payload = joblib.load(path)
    return {'kind': 'tabular', 'estimator': payload['estimator'],
            'scaler': payload['scaler'], 'meta': payload.get('meta', {})}


def score_predictions(pred_df, test_year):
    """NSE/KGE/ACC on the ensemble mean, plus CRPS over the members."""
    if pred_df is None or len(pred_df) == 0:
        return None
    ens = ensemble_mean(pred_df, test_year=test_year)
    scores = score_frame(ens)
    scores['crps'] = crps_from_pred_df(pred_df)
    return scores
