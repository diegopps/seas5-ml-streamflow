"""
flowcast_src.io_results – Output assembly and writing for stage-01 models.

Every model writes the same three artefacts to its results directory:
    fold_metrics.csv          per-fold metrics
    predictions_daily.parquet ensemble-mean daily predictions
    summary.csv               single-row aggregate (mean/std/min/max across folds)

Ensemble models write a fourth:
    predictions_members.parquet   per-member daily predictions

CRPS cannot be recomputed from the ensemble mean, so any CRPS scored over a
subset of lead days (the boxplot windows) needs the member frame kept on disk.
"""

import os
import numpy as np
import pandas as pd

from .paths import RESULTS_ROOT


def results_dir(catchment_id: str, model: str) -> str:
    """Standard output directory for one (catchment, model) run."""
    return str(RESULTS_ROOT / "01_exploration" /
               f"results_{catchment_id}_{model}")


def build_summary(catchment_id: str, model: str, mdf: pd.DataFrame,
                  runtime_s: float, has_crps: bool, extra: dict = None) -> dict:
    """
    Assemble the single-row summary dict from a per-fold metrics frame.

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
        'n_folds':     len(mdf),
    }
    summary.update(agg('nse'))
    summary.update({
        'kge_mean': round(mdf['kge'].mean(), 4),
        'kge_std':  round(mdf['kge'].std(), 4),
    })
    summary.update(agg('acc'))
    if has_crps:
        summary.update(agg('crps'))
    summary['folds_nse_lt0'] = int((mdf['nse'] < 0).sum())
    summary['folds_acc_lt0'] = int((mdf['acc'] < 0).sum())
    if extra:
        summary.update(extra)
    return summary


MEMBER_COLS = ['date', 'init_date', 'member', 'lead_day', 'hydro_year',
               'Q_raw', 'Q_pred', 'Q_clim']


def write_outputs(out_dir: str, mdf: pd.DataFrame, preds: pd.DataFrame,
                  summary: dict, members: pd.DataFrame = None) -> None:
    """
    Write fold_metrics.csv, predictions_daily.parquet, and summary.csv.

    `members` is the per-member prediction frame of an ensemble model; when
    given it is written as predictions_members.parquet. Deterministic
    baselines pass None -- their members are identical by construction.
    """
    os.makedirs(out_dir, exist_ok=True)
    mdf.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)
    preds.to_parquet(os.path.join(out_dir, 'predictions_daily.parquet'),
                     index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'summary.csv'),
                                   index=False)
    if members is not None and len(members) > 0:
        keep = [c for c in MEMBER_COLS if c in members.columns]
        members[keep].to_parquet(
            os.path.join(out_dir, 'predictions_members.parquet'), index=False)
