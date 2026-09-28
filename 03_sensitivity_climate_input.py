#!/usr/bin/env python3
"""
03_sensitivity_climate_input.py – Climate-input sensitivity test.

Tests whether the deployed models actually use their meteorological forcing at
inference time. Each fold trains one model and then scores it twice on the
same held-out year, changing only the forcing it is shown:

    normal   real forcing for the target year
    mixed    forcing replaced by a random donor hydro year's, matched on
             (init month, lead day, ensemble member) so only the source year
             differs

A model that scores the same under both was never using the forcing — its
skill comes from the seasonal cycle, which the calendar encodings supply and
which the corruption does not touch. That is the point of the test: it
separates "forecasts the weather's effect on discharge" from "knows what time
of year it is". Scenario differences are read as paired per-fold deltas, so
the same hydro year under real vs. donor-year forcing is what gets compared.

The corruption is a real donor year rather than synthetic noise on purpose:
noise would push the inputs out of the training distribution, so a skill drop
could mean either "the model used the forcing" or "the model was thrown by
values it had never seen". A donor year is in-distribution by construction, so
a drop can only mean the model was using *this* year's weather.

Models and hyperparameters
    rf, xgb, lstm   one fixed config per catchment: whichever config won the
                    most outer folds in stage 02's nested search (the 'best'
                    column of 03_configranking.pdf/.csv), applied to every
                    fold here. This tests the one config that would actually
                    be deployed for the catchment, not a different winner per
                    fold.
    ridge           its stage-01 definition (RidgeCV selects its own alpha by
                    internal generalised cross-validation, which is why it is
                    not part of the stage-02 search). Runs on every fold.

Training happens once per fold and inference twice over those fixed weights.
Retraining per scenario would mix the effect being measured with ordinary
run-to-run training variation.

Usage:
    python 03_sensitivity_climate_input.py <catchment_id> <model>
    python 03_sensitivity_climate_input.py zeravshan xgb

    Environment variables:
        SEED  – global seed, and the stage-02 search seed whose winners are
                read (default: 107)

Input:
    data_processed/<catchment_id>_features.parquet
    model_results/02_hyperparamtuning/results_<catchment>_<model>_nested_seed<SEED>/
        outer_<HY>/winner.csv          (rf / xgb / lstm only)

Output:
    model_results/03_sensitivity/results_<catchment_id>_<model>/
        sensitivity_fold_metrics.csv   one row per fold x scenario (normal, mixed)
        sensitivity_summary.csv        per-scenario aggregate
    trained_models/
        <catchment_id>_<model>_final_seed<SEED>.joblib   (ridge / rf / xgb)
        <catchment_id>_<model>_final_seed<SEED>.pt       (lstm)

The final model
    After the fold loop, one last model is fitted on EVERY hydro year using the
    catchment-wide winner config and saved to trained_models/. No year is held
    out: it is never scored, so there is nothing to hold out for. The fold
    models above answer "how well does this config generalise"; this one is
    simply that config fitted to the whole record, saved with its provenance
    (catchment, config, seed, training years) so it can be reloaded without
    retraining via flowcast_src.sensitivity.load_fitted.
"""

import os
import sys
import time
import glob
import warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

from catchments import CATCHMENTS, SHARED
from flowcast_src.data import load_obs_discharge, build_analysis_frame
from flowcast_src.climatology import compute_global_climatology
from flowcast_src import hptuning as hp
from flowcast_src import sensitivity as sens
from flowcast_src.paths import PROCESSED_DIR, RESULTS_ROOT, TRAINED_DIR

CLIM_WINDOW     = SHARED.get('daily_clim_window', 31)
CLIM_START_YEAR = SHARED['clim_start_year']     # no .get() -- must be set
SEED = int(os.environ.get('SEED', 107))

# ridge is included alongside the three tuned families: it is a learned model
# that consumes the met forcing, so the question applies to it. climatology and
# persistence are not — they never read the forcing, so corrupting it is a
# no-op for them by construction and the test would be vacuous.
VALID_MODELS = ['ridge', 'rf', 'xgb', 'lstm']
TUNED_MODELS = list(hp.CONFIGS.keys())          # rf, xgb, lstm

# Salt for the final model's val split, keeping it distinct from every
# fold's split seed while staying reproducible from SEED alone.
FINAL_FIT_SALT = 20260803


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
        sys.exit(1)
    if model not in VALID_MODELS:
        print(f"ERROR: Unknown model '{model}'. Available: "
              f"{', '.join(VALID_MODELS)}")
        sys.exit(1)
    return catchment_id, model


def hptune_root(catchment_id, model):
    return str(RESULTS_ROOT / "02_hyperparamtuning" /
               f"results_{catchment_id}_{model}_nested_seed{SEED}")


def load_catchment_winner_config(catchment_id, model):
    """
    The single config used for every fold: whichever config won the most
    outer folds in stage 02's nested search — the same 'best' reported in
    03_configranking.pdf/.csv. Ties are broken the same way: counts are
    sorted by config label before idxmax, so the alphabetically first of the
    tied configs wins, as in plot_configrank. Unlike a per-fold winner, this
    is one fixed hyperparameter set applied across the whole catchment, so the fold loop
    below tests one deployed model rather than a different one per fold.
    """
    configs = hp.CONFIGS[model]
    by_label = {c['label']: c for c in configs}

    counts = {}
    pattern = os.path.join(hptune_root(catchment_id, model), "outer_*",
                           "winner.csv")
    for path in sorted(glob.glob(pattern)):
        w = pd.read_csv(path)
        if len(w) == 0:
            continue
        label = str(w['winner_config'].iloc[0])
        if label not in by_label:
            print(f"  WARNING: {path} names config '{label}', which is not in "
                  f"the {model} grid — skipping this fold's vote.")
            continue
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        return None
    # sort_index: without it idxmax picks the tied config seen first in fold
    # order, which can differ from the one plot_configrank reports.
    best_label = pd.Series(counts).sort_index().idxmax()
    return by_label[best_label]


def fold_plan(catchment_id, model, unique_years):
    """
    Which folds to run, and with which config.

    Tuned families use ONE config for every fold: the catchment-wide stage-02
    winner (most outer-fold wins overall, i.e. 03_configranking.pdf's 'best'),
    not each fold's own inner-CV winner. Ridge has no search, so it runs every
    year with config None.
    """
    if model not in TUNED_MODELS:
        return [(y, None) for y in unique_years]

    config = load_catchment_winner_config(catchment_id, model)
    if config is None:
        print(f"ERROR: no stage-02 winners found under "
              f"{hptune_root(catchment_id, model)}\n"
              f"       Run 02_hyperparamtuning_nested.py for "
              f"{catchment_id}/{model} first.")
        sys.exit(1)
    return [(y, config) for y in unique_years]


def fit_for_fold(model, config, *, df, doy_clim, unique_years, test_year,
                 ckpt_path):
    """Train the fold's single model, shared by both scenarios."""
    outer_train_years = [y for y in unique_years if y != test_year]

    if model != 'lstm':
        # Tabular families do not early-stop, so every non-test year trains.
        return sens.fit_tabular(model, df, doy_clim=doy_clim, config=config,
                                fit_yrs=set(outer_train_years))

    # Reproduce stage 02's winner-retrain split exactly, so this is the same
    # model that stage 02 reported — same config, same train/val partition.
    outer_idx = unique_years.index(test_year)
    winner_idx = hp.CONFIGS['lstm'].index(config)
    tr_yrs, val_yrs = hp.make_val_split(
        outer_train_years,
        val_seed=hp.val_seed_for(outer_idx, hp.INNER_IDX_RETRAIN, winner_idx,
                                 seed=SEED))
    return sens.fit_lstm(df, doy_clim=doy_clim, config=config,
                         tr_yrs=tr_yrs, val_yrs=val_yrs, ckpt_path=ckpt_path)


def fit_final_model(model, config, *, df, doy_clim, unique_years, ckpt_path):
    """
    Train the catchment's final model on EVERY hydro year.

    This is not a fold: no year is held out, because there is nothing left to
    score against and withholding one would only discard data. It is the model
    the pipeline ends on — the catchment-wide winner config fitted to the full
    record — and it is saved rather than scored.

    The LSTM still needs a validation split for early stopping, so it reserves
    VAL_FRAC of the years under a dedicated seed. Those years are trained
    against indirectly (they choose the stopping epoch), which is the same
    compromise every LSTM fit in this project makes; the tabular families have
    no such requirement and fit every year directly.
    """
    if model != 'lstm':
        return sens.fit_tabular(model, df, doy_clim=doy_clim, config=config,
                                fit_yrs=set(unique_years))

    tr_yrs, val_yrs = hp.make_val_split(
        unique_years,
        val_seed=int(np.random.SeedSequence(
            [SEED, FINAL_FIT_SALT]).generate_state(1)[0]))
    return sens.fit_lstm(df, doy_clim=doy_clim, config=config,
                         tr_yrs=tr_yrs, val_yrs=val_yrs, ckpt_path=ckpt_path)


def save_final_model(fitted, *, catchment_id, model, config, unique_years):
    """Persist the final model under trained_models/, with provenance."""
    os.makedirs(TRAINED_DIR, exist_ok=True)
    ext = '.pt' if model == 'lstm' else '.joblib'
    path = os.path.join(TRAINED_DIR,
                        f"{catchment_id}_{model}_final_seed{SEED}{ext}")
    meta = {
        'catchment': catchment_id,
        'model': model,
        'config': config['label'] if config else 'ridgecv',
        'config_params': {k: v for k, v in config.items()
                          if k != 'label'} if config else {},
        'seed': SEED,
        'trained_on_years': [int(y) for y in unique_years],
        'n_years': len(unique_years),
        'stage': '03_final_fit',
    }
    sens.save_fitted(fitted, path, meta=meta)
    return path


def main():
    catchment_id, model = parse_args()
    cfg = CATCHMENTS[catchment_id]
    skip_years = cfg.get('skip_years', set())
    in_path = str(PROCESSED_DIR / f"{catchment_id}_features.parquet")
    out_dir = str(RESULTS_ROOT / "03_sensitivity" /
                  f"results_{catchment_id}_{model}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, '_ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print(f"CLIMATE-INPUT SENSITIVITY: {cfg['name']} ({catchment_id})  "
          f"model={model}  seed={SEED}")
    print(f"  scenarios: {', '.join(sens.SCENARIOS)}")
    if model == 'lstm':
        print(f"  device={sens.lstm_mod.DEVICE}, AMP={sens.lstm_mod.USE_AMP}")
    print("=" * 70)

    # ── 1. Data and the shared global climatology ────────────────────
    print("\n[1/4] Loading data...")
    obs_daily_q = load_obs_discharge(cfg['discharge_csv'])
    df, unique_years = build_analysis_frame(in_path, obs_daily_q, skip_years)
    doy_clim = compute_global_climatology(obs_daily_q, skip_years, CLIM_WINDOW,
                                          start_year=CLIM_START_YEAR)
    print(f"  Rows: {len(df):,} | Hydro years: {len(unique_years)} "
          f"({min(unique_years)}-{max(unique_years)})")
    print(f"  Global climatology: {CLIM_START_YEAR}+ minus skip_years "
          f"({len(skip_years)} skipped), {CLIM_WINDOW}-day smoothing")

    # ── 2. Fold plan ─────────────────────────────────────────────────
    print("\n[2/4] Resolving folds...")
    plan = fold_plan(catchment_id, model, unique_years)
    if not plan:
        print("ERROR: no folds to run.")
        sys.exit(1)
    if model in TUNED_MODELS:
        config_label = plan[0][1]['label']
        print(f"  {len(plan)} folds, all using catchment-wide winner "
              f"config {config_label}")
    else:
        print(f"  {len(plan)} folds (ridge runs every year; no stage-02 "
              f"search applies)")

    # ── 3. Per fold: fit once, score both scenarios ──────────────────
    print(f"\n[3/4] Running folds...\n")
    print(f"  {'Fold':>4s}  {'HY':>6s}  {'Scenario':>8s}  {'NSE':>8s}  "
          f"{'KGE':>8s}  {'ACC':>8s}  {'CRPS':>9s}  {'Donor':>6s}  {'Time':>6s}")
    print(f"  {'-' * 86}")

    rows = []
    for fold_i, (test_year, config) in enumerate(plan):
        fold_t0 = time.time()

        # Every trajectory reaching the test year, whole: the LSTM reads it
        # from initialisation; rows outside the test year are context only.
        test_inits = df.loc[df[sens.GROUP_COL] == test_year, 'init_date'].unique()
        if len(test_inits) == 0:
            continue
        df_test = df[df['init_date'].isin(test_inits)]

        # Seed the corruption per (seed, catchment fold), so the donor draw is
        # reproducible and independent of model.
        fold_seed = int(np.random.SeedSequence(
            [SEED, test_year]).generate_state(1)[0])

        frames, donor_year, n_rows, n_dropped = sens.build_scenarios(
            df_test, df, unique_years, test_year, fold_seed=fold_seed)
        if frames is None:
            print(f"  {fold_i+1:4d}  {test_year:6d}  "
                  f"no donor year available — skipping")
            continue
        if n_dropped:
            print(f"  {fold_i+1:4d}  {test_year:6d}  note: {n_dropped:,} rows "
                  f"dropped from all scenarios (donor {donor_year} lacks a "
                  f"match); {n_rows:,} retained")

        ckpt = os.path.join(ckpt_dir, f'fold{test_year}.pt')
        fitted = fit_for_fold(model, config, df=df, doy_clim=doy_clim,
                              unique_years=unique_years, test_year=test_year,
                              ckpt_path=ckpt)
        if os.path.exists(ckpt):
            os.remove(ckpt)
        if fitted is None:
            print(f"  {fold_i+1:4d}  {test_year:6d}  training produced no "
                  f"model — skipping")
            continue

        for scenario in sens.SCENARIOS:
            pred_df = sens.predict(fitted, frames[scenario], doy_clim=doy_clim,
                                   score_years=[test_year])
            scores = sens.score_predictions(pred_df, test_year)
            if scores is None:
                continue

            rows.append({
                'hydro_year': test_year,
                'scenario': scenario,
                'model': model,
                'config': config['label'] if config else 'ridgecv',
                'nse': scores['nse'], 'kge': scores['kge'],
                'acc': scores['acc'], 'crps': scores['crps'],
                'donor_year': donor_year if scenario == 'mixed' else np.nan,
                'n_rows': n_rows,
                'n_rows_dropped': n_dropped,
                'best_epoch': fitted.get('best_epoch', np.nan),
                'time_s': round(time.time() - fold_t0, 1),
            })

            donor_str = str(donor_year) if scenario == 'mixed' else '-'
            print(f"  {fold_i+1:4d}  {test_year:6d}  {scenario:>8s}  "
                  f"{scores['nse']:+8.3f}  {scores['kge']:+8.3f}  "
                  f"{scores['acc']:+8.3f}  {scores['crps']:9.3f}  "
                  f"{donor_str:>6s}  {time.time()-fold_t0:5.0f}s", flush=True)

    if not rows:
        print("\nERROR: no folds produced results.")
        sys.exit(1)

    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(out_dir, 'sensitivity_fold_metrics.csv'),
               index=False)

    # ── Summary + the paired deltas the figure is built from ─────────
    summary = []
    for scenario in sens.SCENARIOS:
        s = mdf[mdf['scenario'] == scenario]
        if len(s) == 0:
            continue
        row = {'catchment': catchment_id, 'model': model,
               'scenario': scenario, 'n_folds': len(s)}
        for metric in ('nse', 'kge', 'acc', 'crps'):
            row[f'{metric}_mean'] = round(s[metric].mean(), 4)
            row[f'{metric}_std'] = round(s[metric].std(), 4)
        summary.append(row)
    pd.DataFrame(summary).to_csv(
        os.path.join(out_dir, 'sensitivity_summary.csv'), index=False)

    # ── 4. Final model: one fit on every year, saved not scored ──────
    print(f"\n[4/4] Training final model on all {len(unique_years)} years...")
    final_config = plan[0][1]
    final_t0 = time.time()
    final_ckpt = os.path.join(ckpt_dir, 'final.pt')
    final_fitted = fit_final_model(model, final_config, df=df,
                                   doy_clim=doy_clim,
                                   unique_years=unique_years,
                                   ckpt_path=final_ckpt)
    if os.path.exists(final_ckpt):
        os.remove(final_ckpt)

    if final_fitted is None:
        print("  WARNING: final fit produced no model — nothing saved.")
    else:
        path = save_final_model(final_fitted, catchment_id=catchment_id,
                                model=model, config=final_config,
                                unique_years=unique_years)
        label = final_config['label'] if final_config else 'ridgecv'
        print(f"  config={label}  ({time.time() - final_t0:.0f}s)")
        print(f"  Saved {path}")

    if os.path.isdir(ckpt_dir) and not os.listdir(ckpt_dir):
        os.rmdir(ckpt_dir)

    print(f"\n  Scenario means ({model.upper()}):")
    pivot = mdf.pivot_table(index='scenario', values=['nse', 'kge', 'acc', 'crps'],
                            aggfunc='mean').reindex(list(sens.SCENARIOS))
    for scenario, r in pivot.iterrows():
        print(f"    {scenario:>8s}  NSE={r['nse']:+.4f}  KGE={r['kge']:+.4f}  "
              f"ACC={r['acc']:+.4f}  CRPS={r['crps']:.4f}")

    print(f"\n  Median paired delta vs. normal "
          f"(positive = corruption hurt skill):")
    base = mdf[mdf.scenario == 'normal'].set_index('hydro_year')
    for scenario in [s for s in sens.SCENARIOS if s != 'normal']:
        cor = mdf[mdf.scenario == scenario].set_index('hydro_year')
        common = base.index.intersection(cor.index)
        if len(common) == 0:
            continue
        parts = []
        for metric in ('nse', 'kge', 'acc'):
            d = np.median(base.loc[common, metric] - cor.loc[common, metric])
            parts.append(f"{metric.upper()}={d:+.4f}")
        d_crps = np.median(cor.loc[common, 'crps'] - base.loc[common, 'crps'])
        parts.append(f"CRPS={d_crps:+.4f}")
        print(f"    {scenario:>8s}  {'  '.join(parts)}")

    print(f"\nSaved outputs to {out_dir}/")
    print(f"DONE in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == '__main__':
    main()
