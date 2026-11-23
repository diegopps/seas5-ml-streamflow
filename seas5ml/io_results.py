"""
seas5ml.io_results – Output assembly and writing for stage-1 runs (and the
member-file format stage 2 shares, `write_members`).

Every run writes the same artefacts to its results directory
(seas5ml.paths.run_dir / fold_dir):
    fold_metrics.csv          one row per scored hydro year (with its fold)
    predictions_daily.parquet ensemble-mean daily predictions
    summary.csv               single-row aggregate over the scored years
                              (default variant only)
    folds.csv                 the folds the run used: id, test and training
                              years (default variant only; see write_folds)

Ensemble models also write:
    predictions_members.parquet   per-member daily predictions

CRPS cannot be recomputed from the ensemble mean, so any CRPS scored over a
subset of lead days (the boxplot windows) needs the member frame kept on disk.
"""

import os
import pandas as pd


def write_folds(out_dir, folds) -> None:
    """
    Write folds.csv: one row per fold (seas5ml.designs.Fold) with its id and
    its test and training hydro years (space-separated). Figures that need a
    year's training years read them from here (plots.common.load_folds)
    instead of rebuilding the design's folds.
    """
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{
        'fold': f.id,
        'test_years': ' '.join(str(int(y)) for y in f.test_years),
        'train_years': ' '.join(str(int(y)) for y in f.train_years),
    } for f in folds]).to_csv(os.path.join(out_dir, 'folds.csv'), index=False)

MEMBER_COLS = ['date', 'init_date', 'member', 'lead_day', 'hydro_year',
               'Q_raw', 'Q_pred', 'Q_clim']

DATE_COLS = ('date', 'init_date')


def _ns_dates(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with every date column as datetime64[ns]. Models build their
    frames differently (the LSTM from Python timestamps, the others from the
    feature table), which left init_date in ns for one and ms for the rest:
    merges cope, integer or numpy comparisons of dates do not."""
    cols = [c for c in DATE_COLS if c in df.columns]
    return df.astype({c: 'datetime64[ns]' for c in cols}) if cols else df


def write_members(path, members: pd.DataFrame) -> None:
    """A per-member prediction frame to `path` (parquet): the MEMBER_COLS it
    has, dates as datetime64[ns]. The one format of every member file
    (stage 1's predictions_members.parquet, stage 2's per-scenario files)."""
    keep = [c for c in MEMBER_COLS if c in members.columns]
    _ns_dates(members[keep]).to_parquet(path, index=False)


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
    identical by construction). Date columns are written as datetime64[ns]
    for every model (`_ns_dates`).
    """
    os.makedirs(out_dir, exist_ok=True)
    mdf.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)
    _ns_dates(preds).to_parquet(
        os.path.join(out_dir, 'predictions_daily.parquet'), index=False)
    if summary is not None:
        pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'summary.csv'),
                                       index=False)
    if members is not None and len(members) > 0:
        write_members(os.path.join(out_dir, 'predictions_members.parquet'),
                      members)
