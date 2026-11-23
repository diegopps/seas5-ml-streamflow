"""
seas5ml.models.lstm – LSTM (zero hidden-state initialisation).

Each forecast trajectory (one ensemble member of one initialisation) is
processed as a sequence of dynamic features from a zero hidden state; there are
no static features. Trajectories are always read whole, from lead 0, even
where they cross into a hydro year outside the run's scored years (see
build_trajectories). Training and early-stopping sets use only trajectories
initialised in the run's training years (`init_years`; see
seas5ml.data.initialised_in for why). The network predicts a log1p(Q)
anomaly relative to the DOY climatology, which is back-transformed to
discharge for scoring. All ensemble members are treated as independent
training sequences and averaged at evaluation.

The tunable hyperparameters (hidden size, learning rate, weight decay, batch
size) come from a configuration (seas5ml.models.configs); the constants
below are fixed for every configuration.

The pooled LSTM (pooled_lstm; end of this module) is the same network and
training loop trained on every catchment of the pool at once, with a one-hot
catchment ID, one shared input scaler and the NSE* loss.
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..data import hydro_year_of
from ..designs import SEED
from ..features import DYN_VARS, set_a0, apply_fold_climatology

# ── Architecture ─────────────────────────────────────────────────────────
N_LAYERS = 2
DROPOUT = 0.2

# ── Training ─────────────────────────────────────────────────────────────
MAX_EPOCHS = 50
PATIENCE = 20
LR_PATIENCE = 5
LR_FACTOR = 0.5

LOSS_SPACE = 'anomaly'   # 'real' or 'anomaly'

GROUP_COL = 'hydro_year'


# ═══════════════════════════════════════════════════════════════════════
# DEVICE / AMP
# ═══════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# AMP_DTYPE is always defined so the CPU path never references an unset name;
# AMP itself is only enabled on CUDA.
AMP_DTYPE = torch.float16
USE_AMP = False
if DEVICE.type == 'cuda':
    USE_AMP = True
    AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                 else torch.float16)


# ═══════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════

def nse_loss_real(pred, tgt, clim, valid):
    """NSE loss in real Q space (back-transformed from log1p-anomaly)."""
    pred_q = torch.expm1(pred[valid] + clim[valid]).clamp(min=0.0)
    obs_q = torch.expm1(tgt[valid] + clim[valid]).clamp(min=0.0)
    ss_res = ((pred_q - obs_q) ** 2).sum()
    ss_tot = ((obs_q - obs_q.mean()) ** 2).sum()
    eps = 1e-6 * ss_tot.detach() + 1e-8
    return ss_res / (ss_tot + eps)


def nse_loss_anomaly(pred, tgt, clim, valid):
    """NSE-form loss directly in log1p-anomaly space (clim unused)."""
    p = pred[valid]
    o = tgt[valid]
    ss_res = ((p - o) ** 2).sum()
    ss_tot = ((o - o.mean()) ** 2).sum()
    eps = 1e-6 * ss_tot.detach() + 1e-8
    return ss_res / (ss_tot + eps)


_LOSS_FN = nse_loss_real if LOSS_SPACE == 'real' else nse_loss_anomaly


# Pooled model: NSE* (Kratzert et al. 2019, HESS 23, 5089-5110, eq. 2): each
# squared error is divided by (s_c + eps)^2, s_c the standard deviation of
# catchment c's target over its training years, and averaged over all
# samples. Without it the steppe catchments, whose anomalies are much larger,
# would dominate the pooled loss. eps as in the paper.
NSE_STAR_EPS = 0.1


def nse_star_loss(pred, tgt, weight, valid):
    """NSE* loss in log1p-anomaly space; `weight` holds 1 / (s_c + eps)^2 at
    every step of a catchment-c trajectory."""
    se = (pred[valid] - tgt[valid]) ** 2
    return (weight[valid] * se).mean()


# ═══════════════════════════════════════════════════════════════════════
# TRAJECTORIES / DATASET
# ═══════════════════════════════════════════════════════════════════════

def build_trajectories(df, doy_clim, dyn_scaler, is_train, score_years=None,
                       init_years=None):
    """
    Convert the flat feature frame into a list of trajectory dicts, one per
    (init_date, member). Dynamic features are standardised using the training
    scaler (fit here when is_train, otherwise applied).

    score_years: the hydro years this call trains or predicts on. Records
    belong to the hydro year of their VERIFICATION date, so a trajectory that
    crosses 1 October spans two years. Given score_years, `df` must be the
    full frame: every trajectory with at least one row in score_years is
    built WHOLE, from lead 0, so the LSTM reads it from initialisation
    instead of restarting from a zero state at the year boundary with no
    knowledge of lead time. Rows outside score_years are input context only:
    their target is set to NaN, so the loss never sees them, and
    predict_trajectories does not return them. Their inputs are SEAS5 forcing
    and a0, both known at initialisation, so no held-out discharge enters.
    None keeps every row scored (df taken as already restricted).

    init_years: if given, only trajectories INITIALISED in these hydro years
    are built. Training and early-stopping calls pass the run's training
    years, so a trajectory initialised in a held-out year never trains, even
    on its rows that verify in a training year (they carry the held-out
    year's a0; see seas5ml.data.initialised_in). Prediction calls leave it
    None: a test year is scored on every trajectory that reaches it.
    """
    trajectories, fc_dyn_list = raw_trajectories(
        df, doy_clim, score_years=score_years, init_years=init_years)
    if is_train:
        dyn_scaler = fit_dyn_scaler(fc_dyn_list)
    apply_dyn_scaler(trajectories, dyn_scaler)
    return trajectories, dyn_scaler


def raw_trajectories(df, doy_clim, score_years=None, init_years=None):
    """
    The trajectories of `build_trajectories`, unstandardised, plus the list
    of their scored steps' dynamic features (what a training scaler is fit
    on). See `build_trajectories` for score_years and init_years.
    """
    if score_years is not None:
        score_years = set(score_years)
    if score_years is not None or init_years is not None:
        inits = (df.loc[df[GROUP_COL].isin(score_years), 'init_date'].unique()
                 if score_years is not None else df['init_date'].unique())
        if init_years is not None:
            inits = inits[np.isin(hydro_year_of(inits),
                                  [int(y) for y in init_years])]
        df = df[df['init_date'].isin(inits)]

    trajectories = []
    fc_dyn_list = []

    for (init_date, member), grp in df.groupby(['init_date', 'member'],
                                               sort=False):
        grp = grp.sort_values('lead_day').reset_index(drop=True)
        fc_dyn = grp[DYN_VARS].values.astype(np.float32)
        q_raw_fc = grp['Q'].values.astype(np.float64)
        doy_fc = grp['date'].dt.dayofyear.values
        # The climatology and observations stay float64: they are written
        # out as Q_clim / Q_raw, which must equal every other model's row for
        # row (the scoring layer pairs them), and Q_clim back-transforms the
        # prediction. Only the network's tensors are float32: the target is
        # built from the float32-rounded climatology, and the dataset's
        # climatology is cast to float32 by collate_fn, so training is
        # unchanged by the float64 copies.
        q_clim_fc = np.array([doy_clim.get(d, np.nan) for d in doy_fc],
                             dtype=np.float64)
        target_fc = (np.log1p(q_raw_fc)
                     - q_clim_fc.astype(np.float32)).astype(np.float32)
        hy = grp[GROUP_COL].values.astype(int)
        scored = (np.isin(hy, list(score_years)) if score_years is not None
                  else np.ones(len(grp), dtype=bool))
        target_fc[~scored] = np.nan

        # The scaler is fitted on scored steps only, i.e. the training years.
        fc_dyn_list.append(fc_dyn[scored])
        trajectories.append({
            'fc_dyn': fc_dyn,
            'target': target_fc,
            'q_clim': q_clim_fc,
            'q_raw': q_raw_fc,
            'date': grp['date'].values,
            'init_date': init_date,
            'member': member,
            'lead_day': grp['lead_day'].values,
            'hydro_year': hy,
            'scored': scored,
        })
    return trajectories, fc_dyn_list


def fit_dyn_scaler(fc_dyn_list):
    """Per-variable mean / std over the given scored steps (a zero std is
    set to 1)."""
    all_fc = np.concatenate(fc_dyn_list, axis=0)
    dyn_mean = np.nanmean(all_fc, axis=0)
    dyn_std = np.nanstd(all_fc, axis=0)
    dyn_std[dyn_std == 0] = 1.0
    return {'mean': dyn_mean, 'std': dyn_std}


def apply_dyn_scaler(trajectories, dyn_scaler):
    """Standardise every trajectory's dynamic features in place; non-finite
    values become 0 (the training mean)."""
    for traj in trajectories:
        fd = (traj['fc_dyn'] - dyn_scaler['mean']) / dyn_scaler['std']
        fd[~np.isfinite(fd)] = 0.0
        traj['fc_dyn'] = fd.astype(np.float32)


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajs = [t for t in trajectories if np.isfinite(t['target']).any()]

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        t = self.trajs[idx]
        return (torch.from_numpy(t['fc_dyn']),
                torch.from_numpy(t['target']),
                torch.from_numpy(t['q_clim']))


def collate_fn(batch):
    """Pad forecast sequences to the batch maximum length."""
    fc_list, tgt_list, clim_list = zip(*batch)
    B = len(batch)
    n_dyn = fc_list[0].shape[1]
    fc_lens = [f.shape[0] for f in fc_list]
    max_fc = max(fc_lens)

    fc_pad = torch.zeros(B, max_fc, n_dyn)
    tgt_pad = torch.full((B, max_fc), float('nan'))
    clim_pad = torch.zeros(B, max_fc)
    fc_mask = torch.zeros(B, max_fc, dtype=torch.bool)

    for i, (f, t, c, fl) in enumerate(zip(fc_list, tgt_list, clim_list, fc_lens)):
        fc_pad[i, :fl] = f
        tgt_pad[i, :fl] = t
        clim_pad[i, :fl] = c
        fc_mask[i, :fl] = True

    return fc_pad, tgt_pad, clim_pad, fc_mask


# ═══════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════

class DischargeLSTM(nn.Module):
    """LSTM with zero hidden-state initialisation."""

    def __init__(self, n_dyn, hidden_size, n_layers, dropout):
        super().__init__()
        self.dec_proj = nn.Linear(n_dyn, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, fc_dyn):
        out, _ = self.lstm(self.dec_proj(fc_dyn))
        return self.head(out).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# TRAIN / PREDICT
# ═══════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, optimizer, amp_scaler, is_train, loss_fn=None):
    """One pass over `loader`. Batches are (inputs, target, aux, mask);
    `loss_fn(pred, target, aux, valid)` defaults to the single-catchment loss
    (aux = climatology); the pooled model passes nse_star_loss (aux = per-step
    weight)."""
    loss_fn = _LOSS_FN if loss_fn is None else loss_fn
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for fc_dyn, tgt, clim, fc_mask in loader:
            fc_dyn = fc_dyn.to(DEVICE)
            tgt = tgt.to(DEVICE)
            clim = clim.to(DEVICE)
            fc_mask = fc_mask.to(DEVICE)

            valid = fc_mask & torch.isfinite(tgt)
            if not valid.any():
                continue

            if USE_AMP:
                with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                    pred = model(fc_dyn)
                    loss = loss_fn(pred, tgt, clim, valid)
            else:
                pred = model(fc_dyn)
                loss = loss_fn(pred, tgt, clim, valid)

            if is_train:
                optimizer.zero_grad()
                if USE_AMP:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def validation_loss(model, loader, loss_fn=None):
    """
    The early-stopping criterion: `loss_fn` evaluated over the whole
    validation set at once, accumulated in float64:
        nse_loss_anomaly / nse_loss_real   1 - NSE over every validation step
                                           (against the mean of the whole set)
        nse_star_loss                      weighted squared error averaged
                                           over every validation step
    Not the mean over batches of the batch loss (run_epoch): the per-batch
    NSE compares each batch with its own mean, and the unshuffled validation
    batches each cover only a few consecutive start dates, so that criterion
    measured skill against a local-mean reference (checked on wave 1,
    TODO.md). NaN if no step is valid.
    """
    loss_fn = _LOSS_FN if loss_fn is None else loss_fn
    if loss_fn not in (nse_loss_anomaly, nse_loss_real, nse_star_loss):
        raise ValueError(f"validation_loss: no whole-set form of {loss_fn}")
    model.eval()
    sse = n = s_o = s_oo = 0.0

    for fc_dyn, tgt, aux, fc_mask in loader:
        fc_dyn = fc_dyn.to(DEVICE)
        tgt = tgt.to(DEVICE)
        aux = aux.to(DEVICE)
        fc_mask = fc_mask.to(DEVICE)

        valid = fc_mask & torch.isfinite(tgt)
        if not valid.any():
            continue

        if USE_AMP:
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                pred = model(fc_dyn)
        else:
            pred = model(fc_dyn)
        p = pred[valid].double()
        o = tgt[valid].double()
        a = aux[valid].double()

        if loss_fn is nse_star_loss:          # aux = per-step weight
            sse += (a * (p - o) ** 2).sum().item()
            n += o.numel()
            continue
        if loss_fn is nse_loss_real:          # aux = climatology
            p = torch.expm1(p + a).clamp(min=0.0)
            o = torch.expm1(o + a).clamp(min=0.0)
        sse += ((p - o) ** 2).sum().item()
        n += o.numel()
        s_o += o.sum().item()
        s_oo += (o * o).sum().item()

    if n == 0:
        return np.nan
    if loss_fn is nse_star_loss:
        return sse / n
    ss_tot = s_oo - s_o ** 2 / n
    return sse / (ss_tot + 1e-6 * ss_tot + 1e-8)   # eps as in the batch losses


@torch.no_grad()
def predict_trajectories(model, trajectories):
    model.eval()
    records = []
    for traj in trajectories:
        fc_dyn = torch.from_numpy(traj['fc_dyn']).unsqueeze(0).to(DEVICE)
        if USE_AMP:
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                pred_anom = model(fc_dyn).squeeze(0).float().cpu().numpy()
        else:
            pred_anom = model(fc_dyn).squeeze(0).float().cpu().numpy()

        q_clim_fc = traj['q_clim']
        pred_q = np.clip(np.expm1(pred_anom + q_clim_fc), 0.0, None)

        for i, date in enumerate(traj['date']):
            if not traj['scored'][i]:
                continue    # context row from a neighbouring hydro year
            records.append({
                'date': pd.Timestamp(date),
                'init_date': traj['init_date'],
                'member': traj['member'],
                'lead_day': int(traj['lead_day'][i]),
                'hydro_year': int(traj['hydro_year'][i]),
                'Q_raw': float(traj['q_raw'][i]),
                'Q_pred': float(pred_q[i]),
                'Q_clim': float(np.expm1(q_clim_fc[i])),
            })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════
# FIT / PREDICT
# ═══════════════════════════════════════════════════════════════════════

def train_net(model, tr_loader, val_loader, config, loss_fn=None):
    """
    Train `model` with Adam and ReduceLROnPlateau, early-stopping on the
    validation loss (PATIENCE epochs, at most MAX_EPOCHS), and load the
    weights of the best validation epoch into it. Shared by the single and
    the pooled LSTM; `loss_fn` as in `run_epoch`. Training minimises the
    per-batch loss; the validation loss (scheduler and early stopping) is
    the same loss over the whole validation set (`validation_loss`).
    Returns (best_epoch, best_val_loss, n_epochs), n_epochs being the epochs
    actually run.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=config['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
    amp_scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val_loss = np.inf
    best_epoch = 0
    best_state = None
    patience_count = 0
    n_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        n_epochs = epoch
        run_epoch(model, tr_loader, optimizer, amp_scaler, True, loss_fn)
        val_loss = validation_loss(model, val_loader, loss_fn)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("LSTM fit: no epoch produced a finite validation "
                           "loss")
    model.load_state_dict(best_state)
    return best_epoch, best_val_loss, n_epochs


def fit_lstm(config, df, *, fit_clim, tr_yrs, val_yrs):
    """
    Train the LSTM for one configuration on `tr_yrs`, early-stopping on
    `val_yrs`, against the climatology of tr_yrs | val_yrs (`fit_clim`, a
    seas5ml.climatology.ClimatologyFitter; a0 is written into `df` in place).
    The weights of the best validation epoch are kept.

    Torch and NumPy are seeded with SEED at the start of every fit, so a fit
    depends only on its configuration and years, never on what ran before it
    in the same job.

    Returns the fitted-model dict (see seas5ml.models), or None if the
    training or validation years have no data.
    """
    if not (df[GROUP_COL].isin(tr_yrs).any()
            and df[GROUP_COL].isin(val_yrs).any()):
        return None
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_years = set(tr_yrs) | set(val_yrs)
    doy_clim = apply_fold_climatology(df, fit_clim, train_years)

    # Whole trajectories from the full frame; each call scores only its own
    # years (see build_trajectories). Only trajectories initialised in a
    # training year train or early-stop.
    tr_trajs, dyn_sc = build_trajectories(df, doy_clim, None, True,
                                          score_years=tr_yrs,
                                          init_years=train_years)
    val_trajs, _ = build_trajectories(df, doy_clim, dyn_sc, False,
                                      score_years=val_yrs,
                                      init_years=train_years)

    tr_loader = DataLoader(TrajectoryDataset(tr_trajs),
                           batch_size=config['batch'], shuffle=True,
                           collate_fn=collate_fn, num_workers=0,
                           pin_memory=True)
    val_loader = DataLoader(TrajectoryDataset(val_trajs),
                            batch_size=config['batch'], shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=True)

    arch = {'n_dyn': len(DYN_VARS), 'hidden': config['hidden'],
            'n_layers': N_LAYERS, 'dropout': DROPOUT}
    model = DischargeLSTM(arch['n_dyn'], arch['hidden'], arch['n_layers'],
                          arch['dropout']).to(DEVICE)
    best_epoch, best_val_loss, _ = train_net(model, tr_loader, val_loader,
                                             config)
    return {'kind': 'lstm', 'model': 'lstm', 'config': config,
            'net': model, 'dyn_scaler': dyn_sc, 'arch': arch,
            'doy_clim': doy_clim,
            'info': {'best_epoch': best_epoch,
                     'best_val_loss': float(best_val_loss),
                     'n_train': len(tr_trajs)}}


def predict_lstm(fitted, df_score, *, score_years):
    """
    One inference pass, for rows in `score_years` only; other rows of
    `df_score` (not modified) are read as trajectory context. a0, targets and
    Q_clim use the model's own climatology (`fitted['doy_clim']`).

    Inputs are standardised with the training scaler, exactly as at training
    time -- deliberately not re-standardised to their own statistics, so the
    sensitivity test's corrupted inputs reach the model as they are.
    Returns a per-member prediction frame, or None if nothing is scored.
    """
    doy_clim = fitted['doy_clim']
    df_score = set_a0(df_score.copy(), doy_clim)
    trajs, _ = build_trajectories(df_score, doy_clim, fitted['dyn_scaler'],
                                  False, score_years=score_years)
    if not trajs:
        return None
    return predict_trajectories(fitted['net'], trajs)


# ═══════════════════════════════════════════════════════════════════════
# POOLED LSTM (pooled_lstm): one network for every catchment of the pool
# ═══════════════════════════════════════════════════════════════════════
#
# Same network, training loop, constants and configurations as the single
# LSTM; the differences are confined to the inputs and the loss:
#   - one-hot catchment ID (POOLED_CATCHMENTS order) appended to the
#     standardised dynamic inputs at every step, raw 0/1 (not standardised);
#     through the input projection it acts as a learned per-catchment offset
#   - one dynamic-feature scaler over every catchment's training steps
#   - NSE* loss (nse_star_loss), each catchment's errors scaled by its own
#     training-years target spread
# Each catchment keeps its own climatology and a0, fit on its own training
# years, and its own training rule (only trajectories initialised in its
# training years).

class PooledTrajectoryDataset(TrajectoryDataset):
    """Like TrajectoryDataset, with the per-step NSE* weight in place of the
    climatology (collate_fn pads it the same way)."""

    def __getitem__(self, idx):
        t = self.trajs[idx]
        return (torch.from_numpy(t['fc_dyn']),
                torch.from_numpy(t['target']),
                torch.from_numpy(np.full(len(t['target']), t['weight'],
                                         dtype=np.float32)))


def add_catchment_id(trajectories, index, n_catchments):
    """Append the one-hot catchment ID (`index` of `n_catchments`) to every
    step of every trajectory, in place, after standardisation."""
    onehot = np.zeros(n_catchments, dtype=np.float32)
    onehot[index] = 1.0
    for traj in trajectories:
        ids = np.broadcast_to(onehot, (len(traj['fc_dyn']), n_catchments))
        traj['fc_dyn'] = np.concatenate([traj['fc_dyn'], ids],
                                        axis=1).astype(np.float32)


def _scored_targets(trajectories):
    """Every finite target of the trajectories' scored steps."""
    vals = [t['target'][t['scored'] & np.isfinite(t['target'])]
            for t in trajectories]
    return np.concatenate(vals) if vals else np.array([], dtype=np.float32)


def fit_pooled_lstm(config, pool, cfolds):
    """
    Train the pooled LSTM for one configuration on one pooled fold.

    `pool` is {catchment: seas5ml.pooled.PoolMember}; `cfolds` is
    {catchment: that catchment's restriction of the fold}
    (seas5ml.designs.restrict). For every catchment with training years:
    its climatology and a0 are fit on its own tr_yrs | val_yrs (a0 written
    into its frame in place), its training and early-stopping trajectories
    are built from its own frame (only trajectories initialised in its
    training years), and its NSE* weight 1 / (s_c + NSE_STAR_EPS)^2 is set
    from the spread s_c of its training targets (tr_yrs steps). One scaler
    is then fit over all catchments' training steps, the one-hot ID is
    appended, and the network is trained as the single LSTM is.

    Seeded with SEED at the start, like fit_lstm. Returns the fitted-model
    dict (kind 'pooled_lstm'; `doy_clim` is {catchment: climatology}), or
    None if no catchment has training data or none has early-stopping data.
    """
    from ..pooled import POOLED_CATCHMENTS, catchment_index

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    doy_clims, weights, sigmas, n_train = {}, {}, {}, {}
    tr_all, val_all, fc_all = [], [], []
    for c, f in cfolds.items():
        m = pool[c]
        train_years = set(f.tr_yrs) | set(f.val_yrs)
        if not m.df[GROUP_COL].isin(f.tr_yrs).any():
            continue            # no training data for this catchment here
        doy_clims[c] = apply_fold_climatology(m.df, m.fit_clim, train_years)
        tr, fc_list = raw_trajectories(m.df, doy_clims[c],
                                       score_years=f.tr_yrs,
                                       init_years=train_years)
        val, _ = raw_trajectories(m.df, doy_clims[c], score_years=f.val_yrs,
                                  init_years=train_years)
        sigmas[c] = float(np.std(_scored_targets(tr)))
        weights[c] = 1.0 / (sigmas[c] + NSE_STAR_EPS) ** 2
        for t in tr + val:
            t['weight'] = weights[c]
            t['catchment'] = c
        n_train[c] = len(tr)
        tr_all += tr
        val_all += val
        fc_all += fc_list
    if not tr_all or not val_all:
        return None

    dyn_sc = fit_dyn_scaler(fc_all)
    apply_dyn_scaler(tr_all + val_all, dyn_sc)
    n_id = len(POOLED_CATCHMENTS)
    for c in doy_clims:
        add_catchment_id([t for t in tr_all + val_all if t['catchment'] == c],
                         catchment_index(c), n_id)

    tr_loader = DataLoader(PooledTrajectoryDataset(tr_all),
                           batch_size=config['batch'], shuffle=True,
                           collate_fn=collate_fn, num_workers=0,
                           pin_memory=True)
    val_loader = DataLoader(PooledTrajectoryDataset(val_all),
                            batch_size=config['batch'], shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=True)

    # n_dyn is the network's input width (dynamic features + catchment ID),
    # so load_fitted rebuilds the network unchanged.
    arch = {'n_dyn': len(DYN_VARS) + n_id, 'hidden': config['hidden'],
            'n_layers': N_LAYERS, 'dropout': DROPOUT, 'n_id': n_id}
    model = DischargeLSTM(arch['n_dyn'], arch['hidden'], arch['n_layers'],
                          arch['dropout']).to(DEVICE)
    best_epoch, best_val_loss, n_epochs = train_net(
        model, tr_loader, val_loader, config, loss_fn=nse_star_loss)
    return {'kind': 'pooled_lstm', 'model': 'pooled_lstm', 'config': config,
            'net': model, 'dyn_scaler': dyn_sc, 'arch': arch,
            'doy_clim': doy_clims, 'catchments': list(POOLED_CATCHMENTS),
            'info': {'best_epoch': best_epoch,
                     'best_val_loss': float(best_val_loss),
                     'n_epochs': n_epochs,
                     'n_train': len(tr_all),
                     'n_train_by_catchment': n_train,
                     'sigma_by_catchment': sigmas}}


def predict_pooled_lstm(fitted, catchment, df_score, *, score_years):
    """
    `predict_lstm` for one catchment of a pooled model: that catchment's own
    climatology (a0, targets, Q_clim), the shared scaler and its one-hot ID.
    `df_score` is the catchment's frame (not modified). Returns a
    per-member prediction frame, or None if nothing is scored.
    """
    if catchment not in fitted['doy_clim']:
        raise ValueError(f"pooled model has no climatology for '{catchment}' "
                         f"(it had no training data in this fold)")
    doy_clim = fitted['doy_clim'][catchment]
    df_score = set_a0(df_score.copy(), doy_clim)
    trajs, _ = build_trajectories(df_score, doy_clim, fitted['dyn_scaler'],
                                  False, score_years=score_years)
    if not trajs:
        return None
    add_catchment_id(trajs, fitted['catchments'].index(catchment),
                     fitted['arch']['n_id'])
    return predict_trajectories(fitted['net'], trajs)

