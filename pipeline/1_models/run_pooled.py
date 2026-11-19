#!/usr/bin/env python3
"""
Stage 1 – Train and evaluate the pooled models (pooled_lstm).

A pooled model is trained on every catchment of the pool at once
(seas5ml.pooled) and scored per catchment. Its results are written to the
same per-catchment folders and files as every other model's, so the figures
and comparisons treat it as one more model:

  default   Every pooled fold of the design with the default configuration
            (configs.DEFAULTS), in one job. Each fold is checkpointed once
            done, so a restarted job resumes at the first unfinished fold.
            At the end each catchment's standard results are assembled and
            the checkpoints removed.
  tuned     ONE pooled fold (--fold), with the nested-CV search: every
            configuration of the grid is trained on every inner fold of the
            fold's (pooled) training years and scored per catchment on that
            catchment's held-out years. Each (config, inner fold) result is
            appended to the pool-level record as soon as it exists, so a
            restarted job skips it. Winner = best mean over catchments of
            each catchment's mean inner-CV ACC (every catchment weighs the
            same). The winner is retrained on the fold's own early-stopping
            split and scored on every catchment's test years.

Folds, as for every model, come from the design (seas5ml.designs), built over
the union of the pool's hydro years and restricted to each catchment's own:
under loyo a catchment's training years in fold Y are exactly those of its
own single-catchment fold Y (same climatology, a0, Q_clim, scored rows). The
early-stopping years are drawn once per pooled fold and shared by the pool.
See seas5ml.pooled and seas5ml.models.lstm (pooled LSTM) for the model.

Usage:
    python pipeline/1_models/run_pooled.py pooled_lstm default
    python pipeline/1_models/run_pooled.py pooled_lstm tuned --fold <id>
    python pipeline/1_models/run_pooled.py pooled_lstm timing [--fold <id>]
    (add --design <design> for another evaluation design; default loyo)

  timing    Run before the real runs, on the CPUs they will get: times one
            full default fit of one pooled fold plus a short probe per hidden
            size of the grid, and prints the projected runtimes, suggested
            --time and --mem of the default and tuned jobs. Writes no results.

    <id>  a pooled fold id (loyo: a test hydro year of any catchment; split:
          hindcast)

Input:
    data/processed_data/<catchment>_features.parquet   (stage 0, every
                                                        catchment of the pool)

Output (seas5ml.paths), for every catchment scored in the run:
    data/results/<design>/default/<catchment>/<model>/
        fold_metrics.csv, predictions_daily.parquet,
        predictions_members.parquet, summary.csv, folds.csv
    data/results/<design>/tuned/<catchment>/<model>/fold_<id>/
        inner_cv_acc.csv   this catchment's scores, every config x inner fold
        winner.csv         the pooled winner (same for every catchment)
        fold_metrics.csv, predictions_daily.parquet, predictions_members.parquet
    data/results/<design>/<kind>/_pooled/<model>/          (pool level)
        default: fold_<id>/ checkpoints, removed after assembly
        tuned:   fold_<id>/pool_inner_cv.csv, every catchment's inner-CV scores;
                 fold_<id>/COMPLETE once the fold is fully written
"""

import argparse
import os
import shutil
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for seas5ml

import numpy as np
import pandas as pd

from seas5ml import designs, models, paths, pooled, tuning
from seas5ml.catchments import SHARED
from seas5ml.designs import SEED
from seas5ml.evaluate import ensemble_mean, score_members, year_rows
from seas5ml.io_results import build_summary, write_folds, write_outputs
from seas5ml.models import configs
from seas5ml.models import lstm as lstm_mod

VARIANTS = ('default', 'tuned', 'timing')

# Timing variant: epochs of the short per-hidden-size probes, the share of a
# fold's years an inner run trains on (N_INNER = 5 -> 4/5), and the margin
# on the suggested walltimes.
PROBE_EPOCHS = 3
INNER_SHARE = (tuning.N_INNER - 1) / tuning.N_INNER
WALLTIME_MARGIN = 1.5
MAX_WALLTIME_H = 7 * 24       # QOS medium

DONE = 'DONE'                       # checkpoint marker of a finished fold
COMPLETE = 'COMPLETE'               # tuned: every catchment of the fold
                                    # written (submit_pooled.sh skips it)
POOL_INNER_CV = 'pool_inner_cv.csv' # tuned: pool-level inner-CV record
# Columns of the pool-level record (appended run by run, so fixed); a
# catchment's inner_cv_acc.csv is the same without `catchment`, as written
# by the single-catchment search.
INNER_COLS = ['catchment', 'config', 'inner_fold', 'acc', 'nse', 'kge',
              'crps', 'best_epoch', 'time_s']


# ═══════════════════════════════════════════════════════════════════════
# ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: train a pooled model on every catchment, "
                    "score it per catchment.")
    p.add_argument('model', type=str.lower, choices=configs.POOLED_MODELS)
    p.add_argument('variant', choices=VARIANTS)
    p.add_argument('--fold', help="pooled fold id (tuned; optional for "
                                  "timing, default: the first fold)")
    p.add_argument('--design', default=designs.DEFAULT_DESIGN,
                   choices=designs.DESIGNS)
    args = p.parse_args()
    if args.variant == 'tuned' and args.fold is None:
        p.error("the tuned variant needs --fold")
    if args.variant == 'default' and args.fold is not None:
        p.error("--fold applies to the tuned and timing variants only (the "
                "default run covers every fold)")
    return args


# ═══════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════

def fit_extra(fitted, catchment, label, extra=None):
    """Per-fit values recorded in a catchment's fold_metrics rows."""
    info = fitted['info']
    return {'config': label,
            'best_epoch': info['best_epoch'],
            'best_val_loss': info['best_val_loss'],
            'n_epochs': info['n_epochs'],
            'n_train': info['n_train_by_catchment'].get(catchment, 0),
            'n_train_pool': info['n_train'],
            'nse_star_sigma': round(info['sigma_by_catchment']
                                    .get(catchment, np.nan), 4),
            **(extra or {})}


def score_fold(fitted, pool, cfolds, label, extra=None):
    """
    Predict and score every catchment with test years in the fold. Returns
    {catchment: (fold_metrics rows, member predictions)}.
    """
    out = {}
    for c, cf in cfolds.items():
        if not cf.test_years:
            continue
        pred = models.predict(fitted, pool[c].df, score_years=cf.test_years,
                              catchment=c)
        if pred is None or len(pred) == 0:
            continue
        rows = year_rows(cf, pred, ensemble=True,
                         extra=fit_extra(fitted, c, label, extra))
        if rows:
            out[c] = (rows, pred)
    return out


def print_header():
    print(f"  {'Fold':>9s}  {'Catchment':>11s}  {'HY':>6s}  {'NSE':>8s}  "
          f"{'KGE':>8s}  {'ACC':>8s}  {'CRPS':>8s}  {'Time':>6s}")
    print(f"  {'-' * 76}")


def print_rows(c, rows):
    for r in rows:
        print(f"  {r['fold']:>9s}  {c:>11s}  {r['hydro_year']:6d}  "
              f"{r['nse']:+8.3f}  {r['kge']:+8.3f}  {r['acc']:+8.3f}  "
              f"{r['crps']:8.3f}  {r['time_s']:5.0f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# DEFAULT VARIANT
# ═══════════════════════════════════════════════════════════════════════

def write_checkpoint(fdir, scored):
    """One finished fold: each scored catchment's rows and member
    predictions, then the DONE marker (written last, so a fold interrupted
    while writing is redone)."""
    os.makedirs(fdir, exist_ok=True)
    for c, (rows, pred) in scored.items():
        pd.DataFrame(rows).to_csv(fdir / f"ckpt_rows_{c}.csv", index=False)
        pred.to_parquet(fdir / f"ckpt_members_{c}.parquet", index=False)
    (fdir / DONE).write_text("done\n")


def assemble_default(model, *, design, pool, folds, work, runtime_s):
    """Every catchment's standard default-run files from the fold
    checkpoints; the checkpoints are removed once every catchment is
    written."""
    missing = [f.id for f, _ in folds if not (work / f"fold_{f.id}" / DONE).exists()]
    if missing:
        print(f"ERROR: {len(missing)} fold(s) unfinished, not assembling: "
              f"{missing}")
        sys.exit(1)

    for c in pool:
        rows, daily, members, c_folds = [], [], [], []
        for fold, cfolds in folds:
            fdir = work / f"fold_{fold.id}"
            p_rows = fdir / f"ckpt_rows_{c}.csv"
            if not p_rows.exists():
                continue
            rows.append(pd.read_csv(p_rows, dtype={'fold': str}))
            mem = pd.read_parquet(fdir / f"ckpt_members_{c}.parquet")
            daily.append(ensemble_mean(mem))
            members.append(mem)
            c_folds.append(cfolds[c])
        if not rows:
            print(f"  WARNING: {c}: no scored fold -- nothing written")
            continue
        mdf = pd.concat(rows, ignore_index=True)
        summary = build_summary(
            c, model, mdf, runtime_s, has_crps=True,
            extra={'clim_start_year': SHARED['clim_start_year'],
                   'loss': 'nse_star', 'nse_star_eps': lstm_mod.NSE_STAR_EPS,
                   'pool': ' '.join(pool)})
        out_dir = paths.run_dir(design, 'default', c, model)
        write_outputs(out_dir, mdf, pd.concat(daily, ignore_index=True),
                      summary, members=pd.concat(members, ignore_index=True))
        write_folds(out_dir, c_folds)
        print(f"  {c}: {len(mdf)} test years -> {out_dir}/")

    shutil.rmtree(work)
    print(f"  Removed checkpoints {work}/")


def run_default(model, *, design, pool):
    config = configs.DEFAULTS[model]
    work = paths.pooled_work_dir(design, 'default', model)
    folds = pooled.pooled_folds(design, pool)
    t0 = time.time()
    print(f"\n── {model} (default config), {len(folds)} pooled folds ──")
    print_header()

    for fold, cfolds in folds:
        fdir = work / f"fold_{fold.id}"
        if (fdir / DONE).exists():
            print(f"  {fold.id:>9s}  checkpoint found -- skipped")
            continue
        if fdir.exists():
            shutil.rmtree(fdir)         # interrupted while writing: redo
        t_fold = time.time()
        fitted = models.fit_pooled(model, config, pool, cfolds)
        if fitted is None:
            print(f"  {fold.id:>9s}  no training data -- skipped")
            write_checkpoint(fdir, {})
            continue
        scored = score_fold(fitted, pool, cfolds, config['label'])
        for c, (rows, _) in scored.items():
            for r in rows:
                r['time_s'] = time.time() - t_fold
            print_rows(c, rows)
        write_checkpoint(fdir, scored)

    print("\nAssembling per-catchment results...")
    assemble_default(model, design=design, pool=pool, folds=folds, work=work,
                     runtime_s=time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════
# TUNED VARIANT
# ═══════════════════════════════════════════════════════════════════════

def inner_fold_of(fold, holdout, inner_idx, cfg_idx):
    """One inner run of the search: the fold's training years minus the
    holdout, with the inner run's own early-stopping split (tuning's seed
    rule, shared by the pool)."""
    inner_train = [y for y in fold.train_years if y not in holdout]
    tr_yrs, val_yrs = tuning.make_val_split(
        inner_train, val_seed=tuning.val_seed_for(fold.index, inner_idx,
                                                  cfg_idx))
    return designs.Fold(index=fold.index, id=f"{fold.id}_inner{inner_idx}",
                        train_years=tuple(inner_train),
                        test_years=tuple(sorted(holdout)),
                        tr_yrs=frozenset(tr_yrs), val_yrs=frozenset(val_yrs))


def run_inner_cv(model, grid, *, fold, pool, record):
    """
    Every (config, inner fold) of the search, each scored per catchment on
    that catchment's held-out years; rows appended to `record` as soon as a
    run is done. Runs already in `record` (a restarted job) are skipped.
    Returns the full record.
    """
    inner_folds = tuning.inner_partition(fold.train_years,
                                         n_inner=tuning.N_INNER, seed=SEED)
    done = set()
    if record.exists():
        prev = pd.read_csv(record, dtype={'config': str})
        done = set(zip(prev['config'], prev['inner_fold'].astype(int)))
        print(f"  Resuming: {len(done)} of {len(grid) * len(inner_folds)} "
              f"inner runs already in {record.name}")
    print(f"  {len(grid)} configs x {len(inner_folds)} inner folds = "
          f"{len(grid) * len(inner_folds)} trainings; inner-fold sizes "
          f"{[len(f) for f in inner_folds]}")

    for cfg_idx, config in enumerate(grid):
        for inner_idx, holdout in enumerate(inner_folds):
            if (config['label'], inner_idx) in done:
                continue
            t0 = time.time()
            inner = inner_fold_of(fold, holdout, inner_idx, cfg_idx)
            icfolds = {c: designs.restrict(inner, m.years)
                       for c, m in pool.items()}
            fitted = models.fit_pooled(model, config, pool, icfolds)
            rows = []
            for c, icf in icfolds.items():
                if fitted is None or not icf.test_years:
                    continue
                scores = score_members(models.predict(
                    fitted, pool[c].df, score_years=icf.test_years,
                    catchment=c))
                if scores is None:
                    continue
                rows.append({'catchment': c, 'config': config['label'],
                             'inner_fold': inner_idx,
                             'acc': scores['acc'], 'nse': scores['nse'],
                             'kge': scores['kge'], 'crps': scores['crps'],
                             'best_epoch': fitted['info']['best_epoch'],
                             'time_s': round(time.time() - t0, 1)})
            if not rows:
                # Mark the run as done even though nothing was scored.
                rows = [{'catchment': '', 'config': config['label'],
                         'inner_fold': inner_idx, 'acc': np.nan,
                         'time_s': round(time.time() - t0, 1)}]
            (pd.DataFrame(rows).reindex(columns=INNER_COLS)
             .to_csv(record, mode='a', index=False,
                     header=not record.exists()))
            accs = [r['acc'] for r in rows if np.isfinite(r['acc'])]
            print(f"    {config['label']}  inner {inner_idx + 1}/"
                  f"{len(inner_folds)}  mean ACC over {len(accs)} catchments="
                  f"{np.mean(accs) if accs else np.nan:+.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    inner = pd.read_csv(record, dtype={'catchment': str, 'config': str})
    inner['catchment'] = inner['catchment'].fillna('')
    return inner


def select_pooled_winner(inner):
    """
    The configuration with the best mean over catchments of each
    catchment's mean inner-CV ACC (every catchment weighs the same), as
    (label, criterion). Ties as in tuning.select_winner (alphabetical).
    """
    per_catchment = (inner[inner['catchment'] != '']
                     .groupby(['config', 'catchment'])['acc'].mean()
                     .reset_index())
    return tuning.select_winner(per_catchment)


def run_tuned(model, fold_id, *, design, pool):
    """Nested-CV search, winner retrain and per-catchment scores for one
    pooled fold."""
    fold, cfolds = pooled.get_pooled_fold(design, pool, fold_id)
    grid = configs.GRIDS[model]
    work = paths.pooled_work_dir(design, 'tuned', model) / f"fold_{fold.id}"
    os.makedirs(work, exist_ok=True)
    scored_cs = [c for c, cf in cfolds.items() if cf.test_years]
    print(f"  Pooled fold {fold.id}: train {len(fold.train_years)} years, "
          f"test {list(fold.test_years)}; scored catchments: "
          f"{', '.join(scored_cs)}")

    # ── Inner CV: configs x inner folds, scored per catchment ────────
    print("\n[1/3] Inner CV")
    inner = run_inner_cv(model, grid, fold=fold, pool=pool,
                         record=work / POOL_INNER_CV)

    # ── Winner ───────────────────────────────────────────────────────
    print("\n[2/3] Selecting winner (best mean over catchments of the "
          "mean inner-CV ACC)")
    label, crit = select_pooled_winner(inner)
    if label is None:
        print(f"ERROR: fold {fold.id}: inner-CV ACC is NaN for every config "
              f"-- cannot select a winner.")
        sys.exit(1)
    winner = configs.grid_config(model, label)
    print(f"  winner = {label}  (mean over catchments of mean inner ACC="
          f"{crit:+.4f})")

    # ── Retrain the winner on the fold, score every catchment ────────
    print(f"\n[3/3] Retraining {label} on the fold, scoring "
          f"{list(fold.test_years)}")
    t_fold = time.time()
    fitted = models.fit_pooled(model, winner, pool, cfolds)
    if fitted is None:
        print(f"ERROR: fold {fold.id}: the winner produced no model.")
        sys.exit(1)
    scored = score_fold(fitted, pool, cfolds, label,
                        extra={'inner_cv_acc_mean': round(crit, 4)})
    if not scored:
        print(f"ERROR: fold {fold.id}: the winner produced no test "
              f"predictions.")
        sys.exit(1)

    n_inner = int(inner['inner_fold'].nunique())
    print_header()
    for c, (rows, pred) in scored.items():
        for r in rows:
            r['time_s'] = time.time() - t_fold
        print_rows(c, rows)
        out_dir = paths.fold_dir(design, c, model, fold.id)
        os.makedirs(out_dir, exist_ok=True)
        (inner[inner['catchment'] == c].drop(columns='catchment')
         .to_csv(out_dir / 'inner_cv_acc.csv', index=False))
        pd.DataFrame([{
            'fold': fold.id, 'model': model, 'winner_config': label,
            'inner_cv_acc_mean': round(crit, 4),
            'n_inner_folds': n_inner,
            'selection': 'mean over catchments of mean inner-CV ACC',
            **{k: v for k, v in winner.items() if k != 'label'},
        }]).to_csv(out_dir / 'winner.csv', index=False)
        write_outputs(out_dir, pd.DataFrame(rows), ensemble_mean(pred),
                      members=pred)
        print(f"  Saved {out_dir}/")
    # Written last: the fold is finished, a resubmission skips it.
    (work / COMPLETE).write_text("complete\n")


# ═══════════════════════════════════════════════════════════════════════
# TIMING VARIANT
# ═══════════════════════════════════════════════════════════════════════

def _timed_fit(model, config, pool, cfolds):
    t0 = time.time()
    fitted = models.fit_pooled(model, config, pool, cfolds)
    if fitted is None:
        print("ERROR: the timing fold produced no model.")
        sys.exit(1)
    return fitted, time.time() - t0


def _walltime(hours):
    """Suggested --time for a projected runtime: margin, rounded up to the
    hour, as D-HH:00:00."""
    h = int(np.ceil(hours * WALLTIME_MARGIN))
    return f"{h // 24}-{h % 24:02d}:00:00", h


def run_timing(model, fold_id, *, design, pool, t_load):
    """
    Time one pooled fold on this job's CPUs and project the default and
    tuned runs of `design`; writes no results.

    1. One full default fit (to early stopping) and its predictions.
    2. A PROBE_EPOCHS-epoch fit per hidden size of the grid.
    A fit costs setup (trajectories, scaler) + epochs x time per epoch: the
    full fit and the probe at the default hidden size give both, and each
    probe then gives its hidden size's time per epoch. The projection
    assumes every configuration stops after as many epochs as the default
    fit (upper bound: MAX_EPOCHS).
    """
    import resource
    import torch

    folds = pooled.pooled_folds(design, pool)
    if fold_id is None:
        fold, cfolds = folds[0]
    else:
        fold, cfolds = pooled.get_pooled_fold(design, pool, fold_id)
    config = configs.DEFAULTS[model]
    grid = configs.GRIDS[model]
    print(f"\n── Timing: pooled fold {fold.id} ({len(fold.train_years)} "
          f"training years), torch threads={torch.get_num_threads()} ──")

    # 1. Full default fit and predictions.
    fitted, t_fit = _timed_fit(model, config, pool, cfolds)
    n_ep = fitted['info']['n_epochs']
    t0 = time.time()
    score_fold(fitted, pool, cfolds, config['label'])
    t_pred = time.time() - t0
    print(f"  full default fit (hidden {config['hidden']}): {t_fit / 60:.1f} "
          f"min, {n_ep} epochs (best {fitted['info']['best_epoch']}); "
          f"predictions {t_pred / 60:.1f} min")

    # 2. Short probes, one per hidden size of the grid.
    hidden_sizes = sorted({c['hidden'] for c in grid} | {config['hidden']})
    t_probe = {}
    max_epochs = lstm_mod.MAX_EPOCHS
    lstm_mod.MAX_EPOCHS = PROBE_EPOCHS
    try:
        for h in hidden_sizes:
            _, t_probe[h] = _timed_fit(model, dict(config, hidden=h,
                                                   label=f"probe_h{h}"),
                                       pool, cfolds)
            print(f"  probe hidden {h:4d}: {PROBE_EPOCHS} epochs in "
                  f"{t_probe[h] / 60:.1f} min")
    finally:
        lstm_mod.MAX_EPOCHS = max_epochs

    h0 = config['hidden']
    epoch_s = {h0: (t_fit - t_probe[h0]) / (n_ep - PROBE_EPOCHS)}
    setup_s = t_probe[h0] - PROBE_EPOCHS * epoch_s[h0]
    for h in hidden_sizes:
        if h != h0:
            epoch_s[h] = (t_probe[h] - setup_s) / PROBE_EPOCHS
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2

    print(f"\n  setup per fit: {setup_s / 60:.1f} min; time per epoch: "
          + ", ".join(f"hidden {h} {epoch_s[h] / 60:.2f} min"
                      for h in hidden_sizes))
    print(f"  loading the pool: {t_load / 60:.1f} min; peak memory "
          f"{peak_gb:.1f} GB")

    # Projections (hours).
    def fit_h(hidden, epochs, share=1.0):
        return share * (setup_s + epochs * epoch_s[hidden]) / 3600

    n_folds = len(folds)
    default_h = (t_load / 3600
                 + n_folds * (fit_h(h0, n_ep) + t_pred / 3600))
    default_max = (t_load / 3600
                   + n_folds * (fit_h(h0, lstm_mod.MAX_EPOCHS) + t_pred / 3600))
    worst_h = max(hidden_sizes)

    def tuned_hours(epochs):
        inner = sum(tuning.N_INNER * fit_h(c['hidden'], epochs, INNER_SHARE)
                    for c in grid)
        # Inner-CV scoring ~ one prediction pass per inner run, on ~1/5 of
        # the years; retrain with the slowest hidden size as the worst case.
        scoring = len(grid) * t_pred / 3600
        return (t_load / 3600 + inner + scoring
                + fit_h(worst_h, epochs) + t_pred / 3600)

    tuned_h, tuned_max = tuned_hours(n_ep), tuned_hours(lstm_mod.MAX_EPOCHS)
    print(f"\n  Projection, design {design} ({n_folds} pooled folds), this "
          f"many CPUs:")
    for name, exp_h, max_h in (
            ("default (one job, all folds)", default_h, default_max),
            ("tuned (per fold job)", tuned_h, tuned_max)):
        wt, wt_h = _walltime(exp_h)
        flag = (f"  WARNING: above the {MAX_WALLTIME_H} h QOS medium limit"
                if wt_h > MAX_WALLTIME_H else "")
        print(f"    {name}: expected {exp_h:.1f} h (upper bound "
              f"{max_h:.1f} h at {lstm_mod.MAX_EPOCHS} epochs) -> suggested "
              f"--time={wt}{flag}")
    print(f"    tuned, all {n_folds} folds: {n_folds * tuned_h:.0f} job-hours")
    print(f"    suggested --mem: {int(np.ceil(peak_gb * WALLTIME_MARGIN))}G")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()
    print("=" * 60)
    print(f"STAGE 1 ({args.variant}): pooled model {args.model} -- design "
          f"{args.design}" + (f" -- fold {args.fold}" if args.fold else ""))
    print(f"  pool: {', '.join(pooled.POOLED_CATCHMENTS)}")
    print(f"  device={lstm_mod.DEVICE}, AMP={lstm_mod.USE_AMP}")
    print("=" * 60)

    t_load = time.time()
    pool = pooled.load_pool()
    t_load = time.time() - t_load
    if args.variant == 'default':
        run_default(args.model, design=args.design, pool=pool)
    elif args.variant == 'tuned':
        run_tuned(args.model, args.fold, design=args.design, pool=pool)
    else:
        run_timing(args.model, args.fold, design=args.design, pool=pool,
                   t_load=t_load)

    print(f"\n{'=' * 60}")
    print(f"DONE in {(time.time() - t0) / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
