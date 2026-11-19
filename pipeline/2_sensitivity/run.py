#!/usr/bin/env python3
"""
Stage 2 – Climate-input sensitivity test.

Tests whether the learned models actually use their meteorological forcing at
inference time. Each fold trains one model and then scores it twice on the
same test year, changing only the forcing it is shown:

    normal   real forcing for the target year
    mixed    forcing replaced by a donor hydro year's, matched on
             (init month, lead day, ensemble member) so only the source year
             differs

A model that scores the same under both was never using the forcing -- its
skill comes from the seasonal cycle, which the calendar encodings supply and
which the corruption does not touch. Scenario differences are read as paired
per-year deltas, so the same hydro year under real vs. donor-year forcing is
what gets compared.

The corruption is a real donor year rather than synthetic noise on purpose:
noise would push the inputs out of the training distribution, so a skill drop
could mean either "the model used the forcing" or "the model was thrown by
values it had never seen". A donor year is in-distribution by construction, so
a drop can only mean the model was using *this* year's weather. The donor is
drawn from the design's donor pool (seas5ml.designs.donor_pool; loyo: every
other hydro year; split: the other test years, operational forecasts like
the test year's own), keeping only candidates that can supply at least
sens.MIN_DONOR_SHARE of the test year's rows (partial years at the record's
edges are left out; seas5ml.sensitivity, "Partial donor years"). A test
year without a usable candidate is skipped with a warning.

Models and hyperparameters
    rf, xgb, lstm   each fold uses ITS OWN tuned winner (stage 1, tuned
                    variant: tuned/<catchment>/<model>/fold_<id>/winner.csv),
                    chosen by inner CV over that fold's training years, so the
                    test year played no part in it. A catchment-wide winner
                    would not do: under loyo every other fold's inner CV
                    scored configs on this fold's test year. The fold is
                    trained exactly as stage 1 trained the tuned winner (same
                    config, same years and early-stopping split, same seed),
                    so the normal scenario reproduces stage 1's tuned model.
                    Every fold needs its winner.csv: a missing one is an
                    error.
    ridge           its default configuration (RidgeCV selects its own alpha
                    by internal generalised cross-validation, so it has no
                    search). Runs on every fold.

Training happens once per fold and inference twice over those fixed weights.
Retraining per scenario would mix the effect being measured with ordinary
run-to-run training variation.

Usage:
    python pipeline/2_sensitivity/run.py <catchment> <model> [--design loyo|split]
    python pipeline/2_sensitivity/run.py zeravshan xgb

Input:
    data/processed_data/<catchment>_features.parquet
    data/results/<design>/tuned/<catchment>/<model>/fold_<id>/winner.csv
        (rf / xgb / lstm only)

Output (seas5ml.paths):
    data/results/<design>/sensitivity/<catchment>/<model>/
        sensitivity_fold_metrics.csv   one row per test year x scenario
        sensitivity_summary.csv        per-scenario aggregate
    data/trained_models/<design>/
        <catchment>_<model>_final.joblib   (ridge / rf / xgb)
        <catchment>_<model>_final.pt       (lstm)

The final model
    One model per (catchment, model) is saved to data/trained_models/<design>/
    with its provenance (catchment, config, seed, training years) and its
    climatology, reloadable via seas5ml.models.load_fitted.

    loyo   After the fold loop, one last model is fitted on EVERY hydro year
           using the catchment-wide winner config (most tuned-fold wins, the
           'best' of 03_configranking.pdf/.csv). No year is held out: it is
           never scored, so there is nothing to hold out for, and the votes
           of all folds can be pooled. The fold models answer "how well does
           the tuned model generalise"; this one is the catchment's deployable
           model fitted to the whole record (climatology fit on every year).
    split  The fold model itself is saved: trained on the hindcast years with
           the split's own tuned winner, it is exactly the model that was
           scored on the operational years, i.e. a model trained on hindcasts
           and deployable on operational forecasts. No extra fit and no vote.
           An all-years fit would not depend on the design, so it is left to
           loyo.

Climatology
    Each fold model is trained against a climatology (and a0) fit on that
    fold's training years only, and is scored against the same climatology
    in both scenarios (see seas5ml.sensitivity, "Climatology"): the same
    climatology as stage 1's run of that fold.
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for seas5ml

import numpy as np
import pandas as pd

from seas5ml import designs, models, paths, tuning
from seas5ml import sensitivity as sens
from seas5ml.catchments import CATCHMENTS
from seas5ml.data import load_catchment
from seas5ml.designs import SEED
from seas5ml.models import configs
from seas5ml.models import lstm as lstm_mod

# ridge is included alongside the three tuned families: it is a learned model
# that consumes the met forcing, so the question applies to it. climatology and
# persistence are not -- they never read the forcing, so corrupting it is a
# no-op for them by construction and the test would be vacuous.
VALID_MODELS = configs.LEARNED_MODELS
TUNED_MODELS = configs.TUNED_MODELS             # rf, xgb, lstm

FINAL_FIT_SALT = sens.FINAL_FIT_SALT


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2: climate-input sensitivity test.")
    p.add_argument('catchment', type=str.lower, choices=sorted(CATCHMENTS))
    p.add_argument('model', type=str.lower, choices=VALID_MODELS)
    p.add_argument('--design', default=designs.DEFAULT_DESIGN,
                   choices=designs.DESIGNS)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

def read_winner(design, catchment, model, fold_id):
    """A tuned fold's winner.csv as a frame, or None if missing/empty."""
    path = paths.fold_dir(design, catchment, model, fold_id) / 'winner.csv'
    if not path.exists():
        return None
    w = pd.read_csv(path)
    return w if len(w) else None


def fold_plan(design, catchment, model, folds):
    """
    The configuration of every fold: each fold's own tuned winner for rf, xgb
    and lstm (stops with a list of the folds missing winner.csv), the default
    configuration for ridge.
    """
    if model not in TUNED_MODELS:
        return [(fold, configs.DEFAULTS[model]) for fold in folds]

    plan, missing = [], []
    for fold in folds:
        w = read_winner(design, catchment, model, fold.id)
        if w is None:
            missing.append(fold.id)
            continue
        plan.append((fold, configs.grid_config(model,
                                               str(w['winner_config'].iloc[0]))))
    if missing:
        root = paths.run_dir(design, 'tuned', catchment, model)
        print(f"ERROR: no tuned winner.csv for {len(missing)} of {len(folds)} "
              f"folds under {root}:\n       {missing}\n"
              f"       Run pipeline/1_models/run.py {catchment} {model} tuned "
              f"--fold <id> for those folds first.")
        sys.exit(1)
    return plan


def final_config(design, catchment, model, folds):
    """The config of the final all-years model: the catchment-wide winner
    (most fold wins; see seas5ml.tuning.catchment_winner) for the tuned
    families, the default configuration for ridge."""
    if model not in TUNED_MODELS:
        return configs.DEFAULTS[model]
    grid_labels = {c['label'] for c in configs.GRIDS[model]}
    frames = []
    for fold in folds:
        w = read_winner(design, catchment, model, fold.id)
        if w is not None and str(w['winner_config'].iloc[0]) not in grid_labels:
            print(f"  WARNING: fold {fold.id} names config "
                  f"'{w['winner_config'].iloc[0]}', which is not in the "
                  f"{model} grid -- skipping this fold's vote.")
            continue
        frames.append(w)
    label = tuning.catchment_winner(frames)
    return configs.grid_config(model, label) if label else None


# ═══════════════════════════════════════════════════════════════════════
# FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════

def fit_final_model(model, config, *, df, fit_clim, years):
    """
    Train the catchment's final model on EVERY hydro year.

    No year is held out: nothing is scored, so withholding one would only
    discard data. The LSTM still needs a validation split for early
    stopping, so it reserves VAL_FRAC of the years under a dedicated seed;
    those years are trained against indirectly (they choose the stopping
    epoch). The tabular families train on every year.
    """
    tr_yrs, val_yrs = tuning.make_val_split(
        years,
        val_seed=int(np.random.SeedSequence(
            [SEED, FINAL_FIT_SALT]).generate_state(1)[0]))
    return models.fit(model, config, df, fit_clim=fit_clim,
                      tr_yrs=tr_yrs, val_yrs=val_yrs)


def save_final_model(fitted, *, design, catchment, model, config, years,
                     stage='2_sensitivity_final_fit', extra_meta=None):
    """Persist the final model under data/trained_models/<design>/."""
    out_dir = paths.trained_dir(design)
    os.makedirs(out_dir, exist_ok=True)
    ext = '.pt' if model == 'lstm' else '.joblib'
    path = out_dir / f"{catchment}_{model}_final{ext}"
    meta = {
        'catchment': catchment,
        'model': model,
        'design': design,
        'config': config['label'],
        'config_params': {k: v for k, v in config.items() if k != 'label'},
        'seed': SEED,
        'trained_on_years': [int(y) for y in years],
        'n_years': len(years),
        'stage': stage,
        **(extra_meta or {}),
    }
    models.save_fitted(fitted, path, meta=meta)
    return path


def fit_and_save_final(design, catchment, model, folds, *, df, fit_clim,
                       years):
    """loyo's final model: the catchment-wide winner (`final_config`) fitted
    on every year (`fit_final_model`) and saved."""
    config = final_config(design, catchment, model, folds)
    if config is None:
        print("  WARNING: no tuned winners to vote on -- no final model.")
        return
    print(f"  config={config['label']}"
          + (" (catchment-wide winner: most tuned-fold wins)"
             if model in TUNED_MODELS else ""))
    t0 = time.time()
    fitted = fit_final_model(model, config, df=df, fit_clim=fit_clim,
                             years=years)
    if fitted is None:
        print("  WARNING: final fit produced no model -- nothing saved.")
        return
    path = save_final_model(fitted, design=design, catchment=catchment,
                            model=model, config=config, years=years)
    print(f"  Saved {path}  ({time.time() - t0:.0f}s)")


def save_fold_model(fold, config, fitted, *, design, catchment, model):
    """Save a fold's model as the design's final model (split: the model
    trained on the hindcast years and scored on the operational ones)."""
    return save_final_model(
        fitted, design=design, catchment=catchment, model=model,
        config=config, years=fold.train_years, stage='2_sensitivity_fold_model',
        extra_meta={'fold': fold.id,
                    'test_years': [int(y) for y in fold.test_years],
                    'tr_yrs': sorted(int(y) for y in fold.tr_yrs),
                    'val_yrs': sorted(int(y) for y in fold.val_yrs)})


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    catchment, model, design = args.catchment, args.model, args.design
    out_dir = paths.run_dir(design, 'sensitivity', catchment, model)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print(f"CLIMATE-INPUT SENSITIVITY: {CATCHMENTS[catchment]['name']} "
          f"({catchment})  model={model}  design={design}  seed={SEED}")
    print(f"  scenarios: {', '.join(sens.SCENARIOS)}")
    if model == 'lstm':
        print(f"  device={lstm_mod.DEVICE}, AMP={lstm_mod.USE_AMP}")
    print("=" * 70)

    # ── 1. Data; the climatology (and a0) is fit per fold ────────────
    print("\n[1/4] Loading data...")
    df, years, fit_clim = load_catchment(catchment)
    folds = designs.folds(design, years)

    # ── 2. Fold plan ─────────────────────────────────────────────────
    print("\n[2/4] Resolving folds...")
    plan = fold_plan(design, catchment, model, folds)
    if model in TUNED_MODELS:
        counts = pd.Series([c['label'] for _, c in plan]).value_counts()
        print(f"  {len(plan)} folds, each using its own tuned winner: "
              + ", ".join(f"{k} x{v}" for k, v in counts.sort_index().items()))
    else:
        print(f"  {len(plan)} folds ({model} runs its default configuration "
              f"on every fold)")

    # ── 3. Per fold: fit once, score both scenarios per test year ────
    print(f"\n[3/4] Running folds...\n")
    print(f"  {'Fold':>9s}  {'HY':>6s}  {'Scenario':>8s}  {'NSE':>8s}  "
          f"{'KGE':>8s}  {'ACC':>8s}  {'CRPS':>9s}  {'Donor':>6s}  {'Time':>6s}")
    print(f"  {'-' * 91}")

    rows = []
    skipped = []        # test years without a usable donor
    fold_model = None   # split: (fold, config, fitted), saved as the final model
    for fold, config in plan:
        fold_t0 = time.time()

        # Scenario frames per test year, built before training so a fold
        # without a usable donor costs nothing.
        scenarios, skip = sens.test_year_scenarios(df, fold, design, years,
                                                   tag=fold.id)
        skipped += skip
        if not scenarios:
            continue

        fitted = models.fit(model, config, df, fit_clim=fit_clim,
                            tr_yrs=fold.tr_yrs, val_yrs=fold.val_yrs)
        if fitted is None:
            print(f"  {fold.id:>9s}  training produced no model -- skipping")
            continue
        if design == 'split':
            fold_model = (fold, config, fitted)

        rows += sens.scenario_rows(fitted, scenarios, fold=fold, model=model,
                                   config=config, t0=fold_t0, tag=fold.id)

    if skipped:
        print(f"\n  WARNING: {len(skipped)} test year(s) skipped, no usable "
              f"donor: {sorted(skipped)}")
    if not rows:
        print("\nERROR: no folds produced results.")
        sys.exit(1)

    # Fold metrics, plus the per-scenario summary the figure is built from.
    mdf = pd.DataFrame(rows)
    sens.write_results(out_dir, mdf, catchment=catchment, model=model)

    # ── 4. Final model ────────────────────────────────────────────────
    # split: the fold model itself (trained on the hindcast years, scored
    # on the operational ones). loyo: one fit on every year, saved not
    # scored.
    if design == 'split':
        print("\n[4/4] Saving the fold model as the final model...")
        if fold_model is None:
            print("  WARNING: the fold produced no model -- nothing saved.")
        else:
            fold, config, fitted = fold_model
            path = save_fold_model(fold, config, fitted, design=design,
                                   catchment=catchment, model=model)
            print(f"  Saved {path}  (config={config['label']}, trained on "
                  f"{len(fold.train_years)} years "
                  f"{fold.train_years[0]}-{fold.train_years[-1]})")
    else:
        print(f"\n[4/4] Training final model on all {len(years)} years...")
        fit_and_save_final(design, catchment, model, folds, df=df,
                           fit_clim=fit_clim, years=years)

    sens.print_summary(mdf, model)

    print(f"\nSaved outputs to {out_dir}/")
    print(f"DONE in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == '__main__':
    main()
