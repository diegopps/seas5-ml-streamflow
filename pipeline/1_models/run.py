#!/usr/bin/env python3
"""
Stage 1 – Train and evaluate the models: the default run and the tuned run.

One job runs one (catchment, model) in one of two variants:

  default   Every fold of the design, each model with its default
            configuration (seas5ml.models.configs.DEFAULTS), plus the three
            baselines. One job runs all folds.
  tuned     ONE fold (--fold), with the nested-CV search of
            seas5ml.tuning: every configuration of the model's grid is
            scored by inner CV over the fold's training years, the best one
            is retrained on the whole fold and scored on its test years.
            rf, xgb and lstm only; one Slurm job per fold.

Both variants train through seas5ml.models.fit on the fold's tr_yrs /
val_yrs, so a tuned fold and the default run of the same fold differ only in
the configuration (for the LSTM, both early-stop on the same years). The two
variants are independent: either can run first.

All models predict a log1p(Q) anomaly relative to a smoothed day-of-year
climatology fit per training run on its training hydro years only (from
clim_start_year onward, skip_years excluded; see seas5ml.climatology), with
the antecedent anomaly a0 formed against the same climatology. Predictions
are back-transformed to discharge and scored per test year with NSE, KGE and
-- for the ensemble models -- the fair CRPS in real Q space, and with ACC on
log1p anomalies (seas5ml.metrics.calc_acc), each against its own
climatology (Q_clim).

Usage:
    python pipeline/1_models/run.py <catchment> <model> default
    python pipeline/1_models/run.py <catchment> <model> tuned --fold <id>
    (add --design <design> for another evaluation design; default loyo)

    <model>  default: climatology | persistence (also writes
                      damped_persistence) | ridge | rf | xgb | lstm
             tuned:   rf | xgb | lstm
    <id>     a fold id of the design (loyo: the test hydro year), as listed
             in data/processed_data/<catchment>_folds_<design>.txt

Input:
    data/processed_data/<catchment>_features.parquet   (stage 0)

Output (seas5ml.paths):
    data/results/<design>/default/<catchment>/<model>/
        fold_metrics.csv             one row per test year
        predictions_daily.parquet    ensemble-mean predictions
        predictions_members.parquet  per-member predictions (learned models)
        summary.csv
    data/results/<design>/tuned/<catchment>/<model>/fold_<id>/
        inner_cv_acc.csv             every config x inner fold
        winner.csv                   the selected config
        fold_metrics.csv, predictions_daily.parquet, predictions_members.parquet
                                     the winner, retrained on the fold and
                                     scored on its test years
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for seas5ml

import pandas as pd

from seas5ml import designs, models, paths, tuning
from seas5ml.catchments import CATCHMENTS, SHARED
from seas5ml.data import load_catchment
from seas5ml.evaluate import ensemble_mean, score_frame, score_members
from seas5ml.io_results import build_summary, write_outputs
from seas5ml.models import configs, references
from seas5ml.models import lstm as lstm_mod

DEFAULT_MODELS = ('climatology', 'persistence') + configs.LEARNED_MODELS
VARIANTS = ('default', 'tuned')


# ═══════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: train and evaluate one model on one catchment.")
    p.add_argument('catchment', type=str.lower, choices=sorted(CATCHMENTS))
    p.add_argument('model', type=str.lower)
    p.add_argument('variant', choices=VARIANTS)
    p.add_argument('--fold', help="fold id (tuned variant only)")
    p.add_argument('--design', default=designs.DEFAULT_DESIGN,
                   choices=designs.DESIGNS)
    args = p.parse_args()

    valid = DEFAULT_MODELS if args.variant == 'default' else configs.TUNED_MODELS
    if args.model not in valid:
        p.error(f"model '{args.model}' has no {args.variant} run "
                f"(one of: {', '.join(valid)})")
    if args.variant == 'tuned' and args.fold is None:
        p.error("the tuned variant needs --fold")
    if args.variant == 'default' and args.fold is not None:
        p.error("--fold applies to the tuned variant only (the default run "
                "covers every fold)")
    return args


# ═══════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════

def year_rows(fold, pred, *, ensemble, extra):
    """
    One metrics row per test year of `fold`. `pred` is a per-member frame
    (ensemble=True) or a baseline's deterministic frame; `extra` holds the
    per-fit values to record (config, n_train, best_epoch, phi, ...).
    """
    rows = []
    for year in fold.test_years:
        sub = pred[pred['hydro_year'] == year]
        if len(sub) == 0:
            continue
        if ensemble:
            scores = score_members(sub, test_year=year)
            n_test = len(sub[['init_date', 'date']].drop_duplicates())
        else:
            scores = score_frame(sub)
            n_test = len(sub)
        rows.append({'fold': fold.id, 'hydro_year': year, **scores,
                     'n_test': n_test, **extra})
    return rows


def print_header():
    print(f"  {'Fold':>9s}  {'HY':>6s}  {'NSE':>8s}  {'KGE':>8s}  "
          f"{'ACC':>8s}  {'CRPS':>8s}  {'Time':>6s}")
    print(f"  {'-' * 62}")


def print_row(row):
    crps = f"{row['crps']:8.3f}" if 'crps' in row else f"{'-':>8s}"
    print(f"  {row['fold']:>9s}  {row['hydro_year']:6d}  {row['nse']:+8.3f}  "
          f"{row['kge']:+8.3f}  {row['acc']:+8.3f}  {crps}  "
          f"{row['time_s']:5.0f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# DEFAULT VARIANT
# ═══════════════════════════════════════════════════════════════════════

def run_default_model(name, *, catchment, design, df, years, fit_clim):
    """Every fold of `design` for one model (a baseline or a learned model
    with its default configuration); writes the model's default results."""
    t0 = time.time()
    baseline = name in references.BASELINES
    config = None if baseline else configs.DEFAULTS[name]
    print(f"\n── {name} ({'baseline' if baseline else 'default config'}) ──")
    print_header()

    rows, daily, members = [], [], []
    for fold in designs.folds(design, years):
        t_fold = time.time()
        if baseline:
            pred, info = references.predict_baseline(name, df, fit_clim, fold)
            if pred is None:
                continue
            fold_rows = year_rows(fold, pred, ensemble=False, extra=info)
            daily.append(pred)
        else:
            fitted = models.fit(name, config, df, fit_clim=fit_clim,
                                tr_yrs=fold.tr_yrs, val_yrs=fold.val_yrs)
            if fitted is None:
                print(f"  {fold.id:>9s}  no training data -- skipped")
                continue
            pred = models.predict(fitted, df, score_years=fold.test_years)
            if pred is None:
                continue
            fold_rows = year_rows(fold, pred, ensemble=True,
                                  extra={'config': config['label'],
                                         **fitted['info']})
            daily.append(ensemble_mean(pred))
            members.append(pred)
        for row in fold_rows:
            row['time_s'] = time.time() - t_fold
            print_row(row)
        rows += fold_rows

    if not rows:
        print(f"ERROR: {name}: no fold produced results.")
        sys.exit(1)
    mdf = pd.DataFrame(rows)

    extra = {'clim_start_year': SHARED['clim_start_year']}
    if name == 'damped_persistence':
        phi = mdf['phi']
        print(f"\n  Damped persistence phi (per fold): mean {phi.mean():.4f}, "
              f"range {phi.min():.4f}-{phi.max():.4f}")
        extra.update({'phi_mean': round(phi.mean(), 4),
                      'phi_min': round(phi.min(), 4),
                      'phi_max': round(phi.max(), 4)})
    elif name == 'persistence':
        extra['phi'] = 1.0
    elif name == 'ridge':
        extra['alpha_counts'] = str(mdf['best_alpha'].value_counts().to_dict())
    elif name == 'lstm':
        extra['loss_space'] = lstm_mod.LOSS_SPACE

    summary = build_summary(catchment, name, mdf, time.time() - t0,
                            has_crps=not baseline, extra=extra)
    if name == 'climatology':
        # Climatology's predicted anomaly is identically zero -> ACC undefined.
        summary['acc_mean'] = float('nan')
        summary['acc_std'] = float('nan')

    out_dir = paths.run_dir(design, 'default', catchment, name)
    write_outputs(out_dir, mdf, pd.concat(daily, ignore_index=True), summary,
                  members=(pd.concat(members, ignore_index=True)
                           if members else None))
    print(f"  Saved {out_dir}/")


def run_default(model, **kw):
    # The persistence job writes both persistence models.
    names = (['persistence', 'damped_persistence'] if model == 'persistence'
             else [model])
    for name in names:
        run_default_model(name, **kw)


# ═══════════════════════════════════════════════════════════════════════
# TUNED VARIANT
# ═══════════════════════════════════════════════════════════════════════

def run_tuned(model, fold_id, *, catchment, design, df, years, fit_clim):
    """Nested-CV search, winner retrain and score for one fold."""
    fold = designs.get_fold(design, years, fold_id)
    out_dir = paths.fold_dir(design, catchment, model, fold.id)
    grid = configs.GRIDS[model]
    print(f"  Fold {fold.id}: train {len(fold.train_years)} years, test "
          f"{list(fold.test_years)}")

    # ── Inner CV: configs x inner folds ──────────────────────────────
    print("\n[1/3] Inner CV")
    inner_df = tuning.inner_cv(model, grid, df, fit_clim=fit_clim, fold=fold)
    if inner_df.empty:
        print(f"ERROR: fold {fold.id} produced no inner-CV results.")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    inner_df.to_csv(out_dir / 'inner_cv_acc.csv', index=False)

    # ── Winner = best mean inner-CV ACC ──────────────────────────────
    print("\n[2/3] Selecting winner (best mean inner-CV ACC)")
    label, mean_acc = tuning.select_winner(inner_df)
    if label is None:
        print(f"ERROR: fold {fold.id}: inner-CV ACC is NaN for every config "
              f"-- cannot select a winner.")
        sys.exit(1)
    winner = configs.grid_config(model, label)
    print(f"  winner = {label}  (mean inner ACC={mean_acc:+.4f})")
    pd.DataFrame([{
        'fold': fold.id, 'model': model, 'winner_config': label,
        'inner_cv_acc_mean': round(mean_acc, 4),
        'n_inner_folds': int(inner_df['inner_fold'].nunique()),
        **{k: v for k, v in winner.items() if k != 'label'},
    }]).to_csv(out_dir / 'winner.csv', index=False)

    # ── Retrain the winner on the whole fold, score its test years ───
    print(f"\n[3/3] Retraining {label} on the fold, scoring "
          f"{list(fold.test_years)}")
    t_fold = time.time()
    fitted = models.fit(model, winner, df, fit_clim=fit_clim,
                        tr_yrs=fold.tr_yrs, val_yrs=fold.val_yrs)
    pred = (models.predict(fitted, df, score_years=fold.test_years)
            if fitted is not None else None)
    if pred is None:
        print(f"ERROR: fold {fold.id}: the winner produced no test "
              f"predictions.")
        sys.exit(1)
    rows = year_rows(fold, pred, ensemble=True,
                     extra={'config': label,
                            'inner_cv_acc_mean': round(mean_acc, 4),
                            **fitted['info']})
    print_header()
    for row in rows:
        row['time_s'] = time.time() - t_fold
        print_row(row)
    write_outputs(out_dir, pd.DataFrame(rows), ensemble_mean(pred),
                  members=pred)
    print(f"  Saved {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()
    print("=" * 60)
    print(f"STAGE 1 ({args.variant}): {CATCHMENTS[args.catchment]['name']} "
          f"({args.catchment}) -- {args.model} -- design {args.design}"
          + (f" -- fold {args.fold}" if args.fold else ""))
    if args.model == 'lstm':
        print(f"  device={lstm_mod.DEVICE}, AMP={lstm_mod.USE_AMP}")
    print("=" * 60)

    df, years, fit_clim = load_catchment(args.catchment)
    kw = dict(catchment=args.catchment, design=args.design, df=df,
              years=years, fit_clim=fit_clim)
    if args.variant == 'default':
        run_default(args.model, **kw)
    else:
        run_tuned(args.model, args.fold, **kw)

    print(f"\n{'=' * 60}")
    print(f"DONE in {(time.time() - t0) / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
