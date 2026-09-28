#!/usr/bin/env python3
"""
02_hyperparamtuning_nested.py – Nested-CV honest performance estimate.

Produces the leakage-free tuning result: the years used to SELECT the winning
config are kept separate from the year used to REPORT its score, so no year
ever does both jobs. This is the number reported as the model's skill.

One run = one OUTER fold (one held-out hydro year, `OUTER_HY`). For that
outer fold the script:

  1. Builds `N_INNER` (=5) inner folds by partitioning the outer-training years
     into 5 roughly-equal disjoint groups (fixed seed, reproducible).
  2. For each candidate config in the model's grid: trains on each inner fold's
     inner-training years — with the early-stopping split, where the model uses
     one, nested WITHIN that inner-training partition — and scores on the inner
     fold's held-out year(s). Averages ACC across the inner folds.
  3. Selects the config with the best mean inner-CV ACC = the winner for THIS
     outer fold. (The winner MAY differ across outer folds — that is expected
     and correct.)
  4. Retrains the winner on the FULL outer-training set and scores it on the
     outer-test year — the honest score.
  5. Reads the untuned reference for this outer fold from stage 01, rather
     than retraining it. Stage 01 fits the same default config on the same
     split (all years but the test year, scored on it), so its per-fold row IS
     the untuned baseline. Taking it from there guarantees that the number in
     outer_test_metrics.csv and the untuned box in the tuning figure are the
     same number — which a local retrain cannot guarantee for a model that
     draws a fresh early-stopping split — and saves one full retrain per fold.
     Stage 01 must therefore have been run before this stage.

Every model, every config, the winner, and the baseline share one single
global DOY climatology per catchment (fit once on the observed record
from clim_start_year (1981) onward, minus skip_years, exactly as stage 01 does) rather than a climatology refit
per fold, so ACC values are comparable both within this stage and against
stage 01.

Correctness invariants:
  - The inner-holdout year is used ONLY to score a config for selection; it is
    never in that inner training run or its early-stopping split.
  - The outer-test year is never touched during inner CV or any retrain's
    early-stopping split — only for the final score.

Model families, their config grids, baselines, and train/score functions are
defined in `flowcast_src/hptuning.py`. One Slurm job runs one (catchment,
model, outer fold) combination; `submit_02_hyperparamtuning_nested.sh` submits
one job per outer hydro year.

Usage:
    python 02_hyperparamtuning_nested.py <catchment_id> <model>
    python 02_hyperparamtuning_nested.py zeravshan lstm     # requires OUTER_HY env var

    Environment variables (set by submit_02_hyperparamtuning_nested.sh):
        OUTER_HY  – the held-out hydro year naming this outer fold (REQUIRED)
        SEED      – global random seed                (default: 107)
        N_INNER   – number of inner folds              (default: 5)

Input:
    data_processed/<catchment_id>_features.parquet   (from 00_preprocess_catchment.py)

Output:
    model_results/02_hyperparamtuning/results_<catchment_id>_<model>_nested_seed<SEED>/outer_<OUTER_HY>/
        inner_cv_acc.csv          (configs x inner folds)
        winner.csv                (the selected config for this outer fold)
        outer_test_metrics.csv    (winner's honest outer-test score + stage 01's
                                   untuned row for the same year)
        predictions_daily.parquet   (winner's outer-test predictions, ens. mean)
        predictions_members.parquet (winner's per-member predictions, for CRPS)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

from catchments import CATCHMENTS, SHARED
from flowcast_src.data import load_obs_discharge, build_analysis_frame
from flowcast_src.climatology import compute_global_climatology
from flowcast_src.evaluate import ensemble_mean
from flowcast_src.io_results import MEMBER_COLS
from flowcast_src.io_results import results_dir as stage01_results_dir
from flowcast_src import hptuning as hp
from flowcast_src.paths import PROCESSED_DIR, RESULTS_ROOT

CLIM_WINDOW     = SHARED.get('daily_clim_window', 31)
CLIM_START_YEAR = SHARED['clim_start_year']     # no .get() -- must be set

VALID_MODELS = list(hp.CONFIGS.keys())

SEED = int(os.environ.get('SEED', 107))
N_INNER = int(os.environ.get('N_INNER', 5))


def parse_args():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <catchment_id> <model>  "
              f"(set OUTER_HY env var)")
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
    if 'OUTER_HY' not in os.environ:
        print("ERROR: OUTER_HY env var is required (the held-out hydro year "
              "for this outer fold). Set by "
              "submit_02_hyperparamtuning_nested.sh.")
        sys.exit(1)

    return catchment_id, model, int(os.environ['OUTER_HY'])


def _cleanup_ckpts(ckpt_dir):
    """Remove the fold's scratch checkpoint directory, if it still exists."""
    if not os.path.isdir(ckpt_dir):
        return
    for f in os.listdir(ckpt_dir):
        os.remove(os.path.join(ckpt_dir, f))
    os.rmdir(ckpt_dir)


def stage01_baseline_row(catchment_id, model, outer_hy, baseline_label):
    """
    The untuned reference for this outer fold, read from stage 01.

    Stage 01 fits the default configuration on every year except its test year
    and scores it on that year — precisely the split this outer fold uses — so
    its per-fold row IS the untuned baseline and no retrain is needed here.
    Reading it rather than refitting also guarantees that this file and the
    untuned box in the tuning figure carry the same number: a separate retrain
    cannot guarantee that for a model that draws a fresh early-stopping split,
    since a different split means a different training run.

    Returns None if stage 01 has no result for this (model, year).
    """
    path = os.path.join(stage01_results_dir(catchment_id, model),
                        'fold_metrics.csv')
    if not os.path.exists(path):
        return None
    fm = pd.read_csv(path)
    hit = fm[fm['hydro_year'] == outer_hy]
    if len(hit) == 0:
        return None
    row = hit.iloc[0]
    return {
        'outer_hy': outer_hy, 'model': model, 'role': 'baseline',
        'config': baseline_label,
        'best_epoch': row.get('best_epoch', np.nan),
        'nse': row['nse'], 'kge': row['kge'], 'acc': row['acc'],
        'crps': row['crps'],
        'inner_cv_acc_mean': np.nan,
    }


def run_outer_fold(*, df, obs_daily_q, doy_clim, unique_years, outer_hy, model,
                   configs, baseline_hp, train_and_score, root_dir,
                   catchment_id):
    """
    Run inner CV and the winner retrain for one outer fold, and write that
    fold's outputs. The untuned baseline is not retrained here — it is taken
    from stage 01, which fits the same configuration on the same split.
    Returns the winner's outer-test metrics.
    """
    out_dir = os.path.join(root_dir, f"outer_{outer_hy}")
    ckpt_dir = os.path.join(out_dir, '_ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    outer_idx = unique_years.index(outer_hy)
    outer_train_years = [y for y in unique_years if y != outer_hy]

    # ── Inner CV: configs x N_INNER inner folds ───────────────────────
    inner_folds = hp.inner_partition(outer_train_years, n_inner=N_INNER,
                                     seed=SEED)
    n_inner_eff = len(inner_folds)
    print(f"\n[2/4] Inner CV — {len(configs)} configs x {n_inner_eff} inner "
          f"folds = {len(configs) * n_inner_eff} trainings")
    print(f"  inner-fold sizes: {[len(f) for f in inner_folds]}")

    inner_rows = []
    for cfg_idx, config in enumerate(configs):
        for inner_idx, inner_holdout in enumerate(inner_folds):
            it0 = time.time()
            inner_train_years = [y for y in outer_train_years
                                 if y not in inner_holdout]
            tr_yrs, val_yrs = hp.make_val_split(
                inner_train_years,
                val_seed=hp.val_seed_for(outer_idx, inner_idx, cfg_idx,
                                         seed=SEED))

            ckpt = os.path.join(
                ckpt_dir, f's{SEED}_o{outer_hy}_i{inner_idx}_c{cfg_idx}.pt')
            m, _ = train_and_score(
                df, obs_daily_q, doy_clim=doy_clim, hp=config,
                tr_yrs=tr_yrs, val_yrs=val_yrs,
                score_yrs=inner_holdout, ckpt_path=ckpt)
            if os.path.exists(ckpt):
                os.remove(ckpt)

            if m is None:
                continue
            inner_rows.append({
                'config': config['label'],
                'inner_fold': inner_idx,
                'acc': m['acc'], 'nse': m['nse'], 'kge': m['kge'],
                'best_epoch': m['best_epoch'],
                'time_s': round(time.time() - it0, 1),
            })
            print(f"    {config['label']}  inner {inner_idx+1}/{n_inner_eff}  "
                  f"ACC={m['acc']:+.3f}  ({time.time()-it0:.0f}s)", flush=True)

    inner_df = pd.DataFrame(inner_rows)
    if inner_df.empty:
        print(f"  WARNING: outer {outer_hy} produced no inner-CV results — "
              f"nothing to do.")
        _cleanup_ckpts(ckpt_dir)
        sys.exit(1)
    inner_df.to_csv(os.path.join(out_dir, 'inner_cv_acc.csv'), index=False)

    # ── Select winner = best MEAN inner ACC across inner folds ────────
    print(f"\n[3/4] Selecting winner (best mean inner-CV ACC)...")
    # dropna: a config whose ACC is NaN on every inner fold cannot win.
    mean_acc = inner_df.groupby('config')['acc'].mean().dropna()
    if mean_acc.empty:
        print(f"  ERROR: outer {outer_hy}: inner-CV ACC is NaN for every "
              f"config — cannot select a winner.")
        _cleanup_ckpts(ckpt_dir)
        sys.exit(1)
    winner_label = mean_acc.idxmax()
    winner_hp = next(c for c in configs if c['label'] == winner_label)
    winner_idx = configs.index(winner_hp)
    print(f"  winner = {winner_label}  "
          f"(mean inner ACC={mean_acc[winner_label]:+.4f})")

    winner_row = {
        'outer_hy': outer_hy,
        'model': model,
        'winner_config': winner_label,
        'inner_cv_acc_mean': round(float(mean_acc[winner_label]), 4),
        'n_inner_folds': n_inner_eff,
    }
    # Record the winning config's hyperparameters alongside its label.
    winner_row.update({k: v for k, v in winner_hp.items() if k != 'label'})
    pd.DataFrame([winner_row]).to_csv(
        os.path.join(out_dir, 'winner.csv'), index=False)

    # ── Retrain winner on full outer-train, score outer-test ──────────
    print(f"\n[4/4] Retraining winner on full outer-train, "
          f"scoring {outer_hy}...")

    tr_yrs, val_yrs = hp.make_val_split(
        outer_train_years,
        val_seed=hp.val_seed_for(outer_idx, hp.INNER_IDX_RETRAIN, winner_idx,
                                 seed=SEED))
    ckpt = os.path.join(ckpt_dir, f's{SEED}_o{outer_hy}_WINNER_c{winner_idx}.pt')
    m_winner, pred_df = train_and_score(
        df, obs_daily_q, doy_clim=doy_clim, hp=winner_hp,
        tr_yrs=tr_yrs, val_yrs=val_yrs,
        score_yrs={outer_hy}, ckpt_path=ckpt)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    if m_winner is None:
        print(f"ERROR: outer-test year {outer_hy} produced no predictions.")
        _cleanup_ckpts(ckpt_dir)
        sys.exit(1)

    baseline_row = stage01_baseline_row(catchment_id, model, outer_hy,
                                        baseline_hp['label'])
    if baseline_row is None:
        print(f"  WARNING: no stage-01 result for {model} {outer_hy} — "
              f"writing the winner row only. Run stage 01 first; the untuned "
              f"reference is taken from there, not retrained here.")

    rows = [{
        'outer_hy': outer_hy, 'model': model, 'role': 'winner',
        'config': winner_label, 'best_epoch': m_winner['best_epoch'],
        'nse': m_winner['nse'], 'kge': m_winner['kge'],
        'acc': m_winner['acc'], 'crps': m_winner['crps'],
        'inner_cv_acc_mean': round(float(mean_acc[winner_label]), 4),
    }]
    if baseline_row is not None:
        rows.append(baseline_row)
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'outer_test_metrics.csv'), index=False)

    # Ensemble-mean daily predictions, matching the stage-01 results contract
    # so the same plotting helpers can read either stage's output.
    ensemble_mean(pred_df).to_parquet(
        os.path.join(out_dir, 'predictions_daily.parquet'), index=False)

    # Per-member predictions, same contract as stage 01: CRPS cannot be
    # recovered from the ensemble mean, so the windowed CRPSS panel needs the
    # members kept.
    keep = [c for c in MEMBER_COLS if c in pred_df.columns]
    pred_df[keep].to_parquet(
        os.path.join(out_dir, 'predictions_members.parquet'), index=False)

    _cleanup_ckpts(ckpt_dir)

    print(f"\n  OUTER {outer_hy}: winner={winner_label}  "
          f"test ACC={m_winner['acc']:+.4f}  NSE={m_winner['nse']:+.4f}  "
          f"KGE={m_winner['kge']:+.4f}  CRPS={m_winner['crps']:.4f}")
    # baseline_row is the stage-01 untuned reference (None if stage 01 has no
    # result for this fold); it is not retrained here, so there is no metrics
    # dict of its own to report.
    if baseline_row is not None:
        print(f"  baseline: test ACC={baseline_row['acc']:+.4f}  "
              f"NSE={baseline_row['nse']:+.4f}")
    else:
        print("  baseline: unavailable (no stage-01 result for this fold)")
    return m_winner


def main():
    catchment_id, model, outer_hy = parse_args()
    cfg = CATCHMENTS[catchment_id]
    skip_years = cfg.get('skip_years', set())
    in_path = str(PROCESSED_DIR / f"{catchment_id}_features.parquet")
    root_dir = str(RESULTS_ROOT / "02_hyperparamtuning" /
                   f"results_{catchment_id}_{model}_nested_seed{SEED}")
    os.makedirs(root_dir, exist_ok=True)
    t0 = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    configs = hp.CONFIGS[model]
    baseline_hp = hp.BASELINE_CONFIG[model]
    train_and_score = hp.TRAIN_AND_SCORE[model]

    print("=" * 60)
    print(f"NESTED CV: {cfg['name']} ({catchment_id})  model={model}  "
          f"outer_HY={outer_hy}  seed={SEED}  n_inner={N_INNER}")
    print(f"  {len(configs)} configs x {N_INNER} inner folds "
          f"+ winner retrain (baseline read from stage 01)")
    if model == 'lstm':
        print(f"  device={hp.lstm_mod.DEVICE}, AMP={hp.lstm_mod.USE_AMP}")
    print("=" * 60)

    # ── 1. Load data and the single global climatology ───────────────
    print("\n[1/4] Loading data...")
    obs_daily_q = load_obs_discharge(cfg['discharge_csv'])
    df, unique_years = build_analysis_frame(in_path, obs_daily_q, skip_years)
    doy_clim = compute_global_climatology(obs_daily_q, skip_years, CLIM_WINDOW,
                                          start_year=CLIM_START_YEAR)
    print(f"  Global climatology: {CLIM_START_YEAR}+ minus skip_years "
          f"({len(skip_years)} skipped), {CLIM_WINDOW}-day smoothing "
          f"— identical to stage 01's")

    if outer_hy not in unique_years:
        print(f"ERROR: OUTER_HY={outer_hy} not among this catchment's hydro "
              f"years {unique_years}. Nothing to do.")
        sys.exit(1)

    print(f"  Rows: {len(df):,} | Hydro years: {len(unique_years)} "
          f"| outer-test={outer_hy} | outer-train="
          f"{len(unique_years) - 1} yrs")

    # ── 2-4. Inner CV, winner selection, winner retrain ──────────────
    m_winner = run_outer_fold(
        df=df, obs_daily_q=obs_daily_q, doy_clim=doy_clim,
        unique_years=unique_years, outer_hy=outer_hy, model=model,
        configs=configs, baseline_hp=baseline_hp,
        train_and_score=train_and_score, root_dir=root_dir,
        catchment_id=catchment_id)

    elapsed = time.time() - t0
    print(f"\nSaved outputs to {root_dir}/outer_{outer_hy}/")
    print(f"DONE in {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
