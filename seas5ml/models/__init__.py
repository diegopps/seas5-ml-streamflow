"""
seas5ml.models – One fit / predict interface for every learned model.

Every stage trains and predicts through these four functions, so the training
rules (fold climatology and a0, training only on trajectories initialised in
the training years, the early-stopping split) live in one place.

    fit(model, config, df, fit_clim=, tr_yrs=, val_yrs=)  -> fitted | None
    predict(fitted, df, score_years=)                     -> member frame | None
    save_fitted(fitted, path, meta=) / load_fitted(path)

`model` is one of configs.LEARNED_MODELS; `config` comes from
configs.DEFAULTS or configs.GRIDS. The LSTM trains on `tr_yrs` and
early-stops on `val_yrs`; the tabular models do not early-stop and train on
both. Either way the run's climatology (and a0) is fit on tr_yrs | val_yrs,
never on a year it is scored on.

A fitted model is a dict carrying everything prediction needs, the training
climatology included (`doy_clim`), plus `info` (per-fit values worth
recording: n_train, and best_alpha for ridge or best_epoch / best_val_loss for
the LSTM).

The deterministic baselines (climatology, persistence, damped persistence)
are not fitted models; see seas5ml.models.references.
"""

from ..data import GROUP_COL
from . import configs, lstm, tabular


def fit(model, config, df, *, fit_clim, tr_yrs, val_yrs):
    """Train `model` with `config` (see module docstring)."""
    if model == 'lstm':
        return lstm.fit_lstm(config, df, fit_clim=fit_clim,
                             tr_yrs=tr_yrs, val_yrs=val_yrs)
    if model in configs.LEARNED_MODELS:
        return tabular.fit_tabular(model, config, df, fit_clim=fit_clim,
                                   fit_yrs=set(tr_yrs) | set(val_yrs))
    raise ValueError(f"unknown learned model '{model}'")


def predict(fitted, df, *, score_years):
    """
    Per-member predictions for the rows of `df` verifying in `score_years`,
    against the model's own climatology. `df` is not modified. The LSTM reads
    every trajectory reaching those years whole, from lead 0; rows outside
    `score_years` are input context only and are not returned.
    """
    score_years = {int(y) for y in score_years}
    in_scored = df[GROUP_COL].isin(score_years)
    if fitted['kind'] == 'lstm':
        inits = df.loc[in_scored, 'init_date'].unique()
        return lstm.predict_lstm(fitted, df[df['init_date'].isin(inits)],
                                 score_years=score_years)
    return tabular.predict_tabular(fitted, df[in_scored])


def save_fitted(fitted, path, *, meta=None):
    """
    Persist a fitted model so it can be reloaded without retraining.

    Everything inference needs travels with the weights: the tabular models
    carry their fitted StandardScaler, the LSTM its dynamic-feature scaler and
    the architecture arguments needed to rebuild the module before loading the
    state dict. Both also carry the DOY climatology they were trained against
    (a0, back-transform and Q_clim at prediction). `meta` is free-form
    provenance (catchment, config, training years) stored alongside.
    LSTM models are written with torch.save (use a .pt path), tabular models
    with joblib.
    """
    import joblib
    import torch

    payload = {'kind': fitted['kind'], 'model': fitted['model'],
               'config': fitted['config'], 'doy_clim': fitted['doy_clim'],
               'info': fitted['info'], 'meta': meta or {}}
    if fitted['kind'] == 'lstm':
        payload['state_dict'] = {k: v.cpu() for k, v
                                 in fitted['net'].state_dict().items()}
        payload['arch'] = fitted['arch']
        payload['dyn_scaler'] = fitted['dyn_scaler']
        torch.save(payload, path)
    else:
        payload['estimator'] = fitted['estimator']
        payload['scaler'] = fitted['scaler']
        joblib.dump(payload, path)
    return path


def load_fitted(path):
    """Reload a model saved by `save_fitted`, in the dict shape `predict`
    expects."""
    import joblib
    import torch

    if str(path).endswith('.pt'):
        payload = torch.load(path, map_location=lstm.DEVICE,
                             weights_only=False)
        arch = payload['arch']
        net = lstm.DischargeLSTM(arch['n_dyn'], arch['hidden'],
                                 arch['n_layers'], arch['dropout']
                                 ).to(lstm.DEVICE)
        net.load_state_dict(payload['state_dict'])
        net.eval()
        payload['net'] = net
        del payload['state_dict']
        return payload
    return joblib.load(path)
