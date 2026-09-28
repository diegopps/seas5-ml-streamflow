"""
flowcast_src.lstm – Baseline LSTM (zero hidden-state initialisation).

Each forecast trajectory (one ensemble member of one initialisation) is
processed as a sequence of dynamic features from a zero hidden state; there are
no static features. Trajectories are always read whole, from lead 0, even
where they cross into a hydro year outside the fold's scored years (see
build_trajectories). The network predicts a log1p(Q) anomaly relative to the DOY
climatology, which is back-transformed to discharge for scoring. All ensemble
members are treated as independent training sequences and averaged at
evaluation.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .cv import make_folds
from .features import DYN_VARS
from .metrics import calc_nse, calc_kge, calc_acc, crps_from_pred_df
from .evaluate import ensemble_mean, score_frame

# ── Architecture ─────────────────────────────────────────────────────────
HIDDEN_SIZE = 128
N_LAYERS = 2
DROPOUT = 0.2

# ── Training ─────────────────────────────────────────────────────────────
BATCH_SIZE = 512
MAX_EPOCHS = 50
PATIENCE = 20
LR = 3e-4
LR_PATIENCE = 5
LR_FACTOR = 0.5
WEIGHT_DECAY = 1e-4

SEED = 107
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


# ═══════════════════════════════════════════════════════════════════════
# TRAJECTORIES / DATASET
# ═══════════════════════════════════════════════════════════════════════

def build_trajectories(df, doy_clim, dyn_scaler, is_train, score_years=None):
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
    """
    if score_years is not None:
        score_years = set(score_years)
        inits = df.loc[df[GROUP_COL].isin(score_years), 'init_date'].unique()
        df = df[df['init_date'].isin(inits)]

    trajectories = []
    fc_dyn_list = []

    for (init_date, member), grp in df.groupby(['init_date', 'member'],
                                               sort=False):
        grp = grp.sort_values('lead_day').reset_index(drop=True)
        fc_dyn = grp[DYN_VARS].values.astype(np.float32)
        q_raw_fc = grp['Q'].values.astype(np.float64)
        doy_fc = grp['date'].dt.dayofyear.values
        q_clim_fc = np.array([doy_clim.get(d, np.nan) for d in doy_fc],
                             dtype=np.float32)
        target_fc = (np.log1p(q_raw_fc) - q_clim_fc).astype(np.float32)
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
            'q_raw': q_raw_fc.astype(np.float32),
            'date': grp['date'].values,
            'init_date': init_date,
            'member': member,
            'lead_day': grp['lead_day'].values,
            'hydro_year': hy,
            'scored': scored,
        })

    if is_train:
        all_fc = np.concatenate(fc_dyn_list, axis=0)
        dyn_mean = np.nanmean(all_fc, axis=0)
        dyn_std = np.nanstd(all_fc, axis=0)
        dyn_std[dyn_std == 0] = 1.0
        dyn_scaler = {'mean': dyn_mean, 'std': dyn_std}

    for traj in trajectories:
        fd = (traj['fc_dyn'] - dyn_scaler['mean']) / dyn_scaler['std']
        fd[~np.isfinite(fd)] = 0.0
        traj['fc_dyn'] = fd.astype(np.float32)

    return trajectories, dyn_scaler


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

def run_epoch(model, loader, optimizer, amp_scaler, is_train):
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
                    loss = _LOSS_FN(pred, tgt, clim, valid)
            else:
                pred = model(fc_dyn)
                loss = _LOSS_FN(pred, tgt, clim, valid)

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
# LOGO-CV RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_lstm(df, unique_years, doy_clim, out_dir):
    """
    LOGO-CV loop for the baseline LSTM. Saves the best checkpoint per fold to
    out_dir and returns per-fold metrics, ensemble-mean predictions across all
    folds, and per-member predictions (for CRPS).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(out_dir, exist_ok=True)

    fold_metrics = []
    all_pred_dfs = []

    print(f"  {'Fold':>4s}  {'HY':>6s}  {'BestEp':>6s}  {'NSE':>8s}  "
          f"{'KGE':>8s}  {'ACC':>8s}  {'CRPS':>8s}  {'Time':>6s}")
    print(f"  {'-' * 66}")

    for split in make_folds(unique_years):
        fold_i = split['fold_i']
        test_year = split['test_year']
        tr_yrs = split['tr_yrs']
        val_yrs = split['val_yrs']
        fold_t0 = time.time()

        if not (df[GROUP_COL] == test_year).any():
            continue

        # Whole trajectories from the full frame; each call scores only its
        # own years (see build_trajectories).
        tr_trajs, dyn_sc = build_trajectories(df, doy_clim, None, True,
                                              score_years=tr_yrs)
        val_trajs, _ = build_trajectories(df, doy_clim, dyn_sc, False,
                                          score_years=val_yrs)
        test_trajs, _ = build_trajectories(df, doy_clim, dyn_sc, False,
                                           score_years=[test_year])

        tr_loader = DataLoader(TrajectoryDataset(tr_trajs), batch_size=BATCH_SIZE,
                               shuffle=True, collate_fn=collate_fn,
                               num_workers=0, pin_memory=True)
        val_loader = DataLoader(TrajectoryDataset(val_trajs), batch_size=BATCH_SIZE,
                                shuffle=False, collate_fn=collate_fn,
                                num_workers=0, pin_memory=True)

        model = DischargeLSTM(len(DYN_VARS), HIDDEN_SIZE, N_LAYERS,
                              DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                     weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
        amp_scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        best_val_loss = np.inf
        best_epoch = 0
        patience_count = 0
        best_ckpt_path = os.path.join(out_dir, f'best_model_fold{fold_i+1}.pt')

        for epoch in range(1, MAX_EPOCHS + 1):
            run_epoch(model, tr_loader, optimizer, amp_scaler, True)
            val_loss = run_epoch(model, val_loader, optimizer, amp_scaler, False)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_count = 0
                torch.save(model.state_dict(), best_ckpt_path)
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    break

        model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE,
                                         weights_only=True))
        pred_df = predict_trajectories(model, test_trajs)
        all_pred_dfs.append(pred_df)

        ens = ensemble_mean(pred_df, test_year=test_year)
        scores = score_frame(ens)
        f_crps = crps_from_pred_df(pred_df)

        fold_metrics.append({
            'hydro_year': test_year,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'nse': scores['nse'], 'kge': scores['kge'],
            'acc': scores['acc'], 'crps': f_crps,
            'n_train_traj': len(tr_trajs),
            'n_test_traj': len(test_trajs),
            'time_s': time.time() - fold_t0,
        })

        print(f"  {fold_i+1:4d}  {test_year:6d}  {best_epoch:6d}  "
              f"{scores['nse']:+8.3f}  {scores['kge']:+8.3f}  "
              f"{scores['acc']:+8.3f}  {f_crps:8.3f}  "
              f"{fold_metrics[-1]['time_s']:5.0f}s", flush=True)

    mdf = pd.DataFrame(fold_metrics)
    all_preds = pd.concat(all_pred_dfs, ignore_index=True)
    return mdf, all_preds
