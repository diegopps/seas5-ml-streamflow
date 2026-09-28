#!/usr/bin/env python3
"""
01_exploration.py – Stage-01 model exploration runner.

Trains and evaluates one model on one catchment under Leave-One-Hydro-Year-Out
cross-validation, and writes fold metrics, daily predictions, and a summary.

Usage:
    python 01_exploration.py <catchment_id> <model>

    <model> is one of:
        climatology
        persistence           (also writes damped_persistence)
        ridge
        rf
        xgb
        lstm

Input:
    data_processed/<catchment_id>_features.parquet   (from 00_preprocess_catchment.py)

Output:
    model_results/01_exploration/results_<catchment_id>_<model>/
        fold_metrics.csv
        predictions_daily.parquet
        predictions_members.parquet  (ensemble models only — for windowed CRPS)
        summary.csv
        best_model_fold<N>.pt        (lstm only)

All models predict a log1p(Q) anomaly relative to a single global smoothed
day-of-year climatology (fit once per catchment on the observed record
from clim_start_year (1981) onward, excluding skip_years), back-transformed to discharge and
scored with NSE, KGE and — for the ensemble models — CRPS in real Q space, and
with ACC on log1p anomalies (see flowcast_src.metrics.calc_acc).
"""

import sys
import time
import warnings
warnings.filterwarnings('ignore')

from catchments import CATCHMENTS, SHARED
from flowcast_src.data import load_obs_discharge, build_analysis_frame
from flowcast_src.climatology import compute_global_climatology
from flowcast_src.io_results import results_dir, build_summary, write_outputs
from flowcast_src.paths import PROCESSED_DIR
from flowcast_src.tabular import run_tabular_model
from flowcast_src.estimators import TABULAR_FACTORIES
from flowcast_src import references as ref
from flowcast_src import lstm as lstm_mod

CLIM_WINDOW     = SHARED.get('daily_clim_window', 31)
CLIM_START_YEAR = SHARED['clim_start_year']     # no .get() -- must be set
LEAD_BINS = SHARED['lead_bins_daily']

VALID_MODELS = ['climatology', 'persistence', 'ridge', 'rf', 'xgb', 'lstm']


# ═══════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <catchment_id> <model>")
        print(f"  catchments: {', '.join(CATCHMENTS.keys())}")
        print(f"  models    : {', '.join(VALID_MODELS)}")
        sys.exit(1)

    catchment_id = sys.argv[1].lower()
    model = sys.argv[2].lower()

    if catchment_id not in CATCHMENTS:
        print(f"ERROR: Unknown catchment '{catchment_id}'")
        print(f"Available: {', '.join(CATCHMENTS.keys())}")
        sys.exit(1)
    if model not in VALID_MODELS:
        print(f"ERROR: Unknown model '{model}'")
        print(f"Available: {', '.join(VALID_MODELS)}")
        sys.exit(1)

    return catchment_id, model


# ═══════════════════════════════════════════════════════════════════════
# PER-MODEL DISPATCH
# ═══════════════════════════════════════════════════════════════════════

def run_climatology(catchment_id, df, unique_years, doy_clim, t0):
    mdf, preds = ref.run_climatology(df, unique_years, doy_clim, LEAD_BINS)
    summary = build_summary(catchment_id, 'climatology', mdf,
                            time.time() - t0, has_crps=False,
                            extra={'clim_start_year': CLIM_START_YEAR})
    # Climatology's predicted anomaly is identically zero -> ACC undefined.
    summary['acc_mean'] = float('nan')
    summary['acc_std'] = float('nan')
    write_outputs(results_dir(catchment_id, 'climatology'), mdf, preds, summary)


def run_persistence(catchment_id, df, unique_years, doy_clim,
                    obs_daily_q, skip_years, t0):
    # Plain persistence (phi = 1).
    t_p = time.time()
    mdf_p, preds_p = ref.run_persistence(
        df, unique_years, doy_clim, phi=1.0,
        model_name='persistence')
    summary_p = build_summary(catchment_id, 'persistence', mdf_p,
                              time.time() - t_p, has_crps=False,
                              extra={'phi': 1.0,
                                     'clim_start_year': CLIM_START_YEAR})
    write_outputs(results_dir(catchment_id, 'persistence'),
                  mdf_p, preds_p, summary_p)

    # Damped persistence — phi estimated once from the anomaly record over the
    # same CLIM_START_YEAR window the climatology is fitted on, so the decay
    # rate and the climatology it decays towards describe one period.
    anomaly_full = ref.anomaly_record(obs_daily_q, doy_clim, skip_years,
                                      start_year=CLIM_START_YEAR)
    phi = ref.estimate_phi(anomaly_full)
    print(f"\nDamped persistence phi = {phi:.4f}\n")

    t_d = time.time()
    mdf_d, preds_d = ref.run_persistence(
        df, unique_years, doy_clim, phi=phi,
        model_name='damped_persistence')
    summary_d = build_summary(catchment_id, 'damped_persistence', mdf_d,
                              time.time() - t_d, has_crps=False,
                              extra={'phi': round(phi, 4),
                                     'clim_start_year': CLIM_START_YEAR})
    write_outputs(results_dir(catchment_id, 'damped_persistence'),
                  mdf_d, preds_d, summary_d)


def run_tabular(catchment_id, model, df, unique_years, doy_clim, t0):
    factory = TABULAR_FACTORIES[model]
    mdf, all_preds, all_members = run_tabular_model(
        model, df, unique_years, doy_clim, factory, LEAD_BINS)

    extra = {'clim_start_year': CLIM_START_YEAR}
    if model == 'ridge' and 'best_alpha' in mdf.columns:
        extra['alpha_counts'] = str(mdf['best_alpha'].value_counts().to_dict())
    summary = build_summary(catchment_id, model, mdf,
                            time.time() - t0, has_crps=True, extra=extra)
    write_outputs(results_dir(catchment_id, model), mdf, all_preds, summary,
                  members=all_members)


def run_lstm(catchment_id, df, unique_years, doy_clim, t0):
    out_dir = results_dir(catchment_id, 'lstm')
    mdf, all_preds = lstm_mod.run_lstm(df, unique_years, doy_clim, out_dir)

    # Ensemble-mean daily predictions across all folds, saved as the output.
    from flowcast_src.evaluate import ensemble_mean
    ens_daily = ensemble_mean(all_preds)

    summary = build_summary(catchment_id, 'lstm', mdf,
                            time.time() - t0, has_crps=True,
                            extra={'loss_space': lstm_mod.LOSS_SPACE,
                                   'clim_start_year': CLIM_START_YEAR})
    write_outputs(out_dir, mdf, ens_daily, summary, members=all_preds)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    catchment_id, model = parse_args()
    cfg = CATCHMENTS[catchment_id]
    skip_years = cfg.get('skip_years', set())
    in_path = str(PROCESSED_DIR / f"{catchment_id}_features.parquet")

    t0 = time.time()
    print("=" * 60)
    print(f"STAGE 01 EXPLORATION: {cfg['name']} ({catchment_id}) — {model}")
    print("=" * 60)

    # ── Load data and the single global climatology ──────────────────
    obs_daily_q = load_obs_discharge(cfg['discharge_csv'])
    df, unique_years = build_analysis_frame(in_path, obs_daily_q, skip_years)
    doy_clim = compute_global_climatology(obs_daily_q, skip_years, CLIM_WINDOW,
                                          start_year=CLIM_START_YEAR)

    print(f"  Rows: {len(df):,} | Hydro years: {len(unique_years)} "
          f"({min(unique_years)}-{max(unique_years)})")
    print(f"  Global climatology: {CLIM_START_YEAR}+ minus skip_years "
          f"({len(skip_years)} skipped), {CLIM_WINDOW}-day smoothing")

    # ── Dispatch ──────────────────────────────────────────────────────
    if model == 'climatology':
        run_climatology(catchment_id, df, unique_years, doy_clim, t0)
    elif model == 'persistence':
        run_persistence(catchment_id, df, unique_years, doy_clim,
                        obs_daily_q, skip_years, t0)
    elif model in TABULAR_FACTORIES:
        run_tabular(catchment_id, model, df, unique_years, doy_clim, t0)
    elif model == 'lstm':
        run_lstm(catchment_id, df, unique_years, doy_clim, t0)

    print(f"\n{'=' * 60}")
    print(f"DONE in {(time.time() - t0) / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
