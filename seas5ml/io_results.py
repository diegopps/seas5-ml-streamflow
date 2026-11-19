"""
seas5ml.io_results – Output assembly and writing for stage-1 runs.

Every run writes the same artefacts to its results directory
(seas5ml.paths.run_dir / fold_dir):
    fold_metrics.csv          one row per scored hydro year (with its fold)
    predictions_daily.parquet ensemble-mean daily predictions
    summary.csv               single-row aggregate over the scored years
                              (default variant only)

Ensemble models write a fourth:
    predictions_members.parquet   per-member daily predictions

CRPS cannot be recomputed from the ensemble mean, so any CRPS scored over a
subset of lead days (the boxplot windows) needs the member frame kept on disk.
"""

import os
import pandas as pd

MEMBER_COLS = ['date', 'init_date', 'member', 'lead_day', 'hydro_year',
               'Q_raw', 'Q_pred', 'Q_clim']


def build_summary(catchment_id: str, model: str, mdf: pd.DataFrame,
                  runtime_s: float, has_crps: bool, extra: dict = None) -> dict:
    """
    Assemble the single-row summary dict from a per-year metrics frame.

    NSE/KGE/ACC aggregates are always included. CRPS aggregates are included
    when `has_crps` is True (ensemble models). `extra` supplies any
    model-specific columns (e.g. loss_space, phi, alpha_counts).
    """
    def agg(col):
        return {
            f'{col}_mean': round(mdf[col].mean(), 4),
            f'{col}_std':  round(mdf[col].std(), 4),
            f'{col}_min':  round(mdf[col].min(), 4),
            f'{col}_max':  round(mdf[col].max(), 4),
        }

    summary = {
        'catchment':   catchment_id,
        'model':       model,
        'runtime_s':   round(runtime_s, 1),
        'runtime_min': round(runtime_s / 60, 2),
        'n_folds':     int(mdf['fold'].nunique()),
        'n_years':     len(mdf),
    }
    summary.update(agg('nse'))
    summary.update({
        'kge_mean': round(mdf['kge'].mean(), 4),
        'kge_std':  round(mdf['kge'].std(), 4),
    })
    summary.update(agg('acc'))
    if has_crps:
        summary.update(agg('crps'))
    summary['years_nse_lt0'] = int((mdf['nse'] < 0).sum())
    summary['years_acc_lt0'] = int((mdf['acc'] < 0).sum())
    if extra:
        summary.update(extra)
    return summary


def write_outputs(out_dir, mdf: pd.DataFrame, preds: pd.DataFrame,
                  summary: dict = None, members: pd.DataFrame = None) -> None:
    """
    Write fold_metrics.csv, predictions_daily.parquet and, if given,
    summary.csv and predictions_members.parquet (the per-member frame of an
    ensemble model; deterministic baselines pass None, their members being
    identical by construction).
    """
    os.makedirs(out_dir, exist_ok=True)
    mdf.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)
    preds.to_parquet(os.path.join(out_dir, 'predictions_daily.parquet'),
                     index=False)
    if summary is not None:
        pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'summary.csv'),
                                       index=False)
    if members is not None and len(members) > 0:
        keep = [c for c in MEMBER_COLS if c in members.columns]
        members[keep].to_parquet(
            os.path.join(out_dir, 'predictions_members.parquet'), index=False)
