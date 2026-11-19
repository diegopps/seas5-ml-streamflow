#!/usr/bin/env python3
"""
Stage 2 – Climate-input sensitivity test, pooled models (pooled_lstm).

The same test as pipeline/2_sensitivity/run.py (see there and
seas5ml.sensitivity), for a model trained on every catchment of the pool at
once: each pooled fold trains ONE model, with that fold's own tuned winner
(stage 1, run_pooled.py tuned: tuned/<catchment>/pooled_lstm/fold_<id>/
winner.csv, the same for every catchment), exactly as stage 1 retrained it,
so the normal scenario reproduces stage 1's tuned pooled model. Every
catchment with test years in the fold is then scored twice per test year
(real and donor-year forcing), with its scenarios built exactly as for the
single-catchment models: same donor pool, same usable-donor filter, same
seeded draw, hence the same donor year as that catchment's single models.

Each finished fold is checkpointed, so a restarted job resumes at the first
unfinished fold. At the end each catchment's standard stage-2 files are
assembled and the checkpoints removed.

The final model (one file for the pool)
    loyo   one pooled fit on every hydro year of every catchment, with the
           most frequent winner across the pooled folds (tie-break as
           seas5ml.tuning.catchment_winner); early-stopping years drawn once
           from the union of years (FINAL_FIT_SALT) and shared by the pool.
           Never scored.
    split  the fold model itself (trained on the hindcast years).

Usage:
    python pipeline/2_sensitivity/run_pooled.py pooled_lstm [--design loyo|split]

Input:
    data/processed_data/<catchment>_features.parquet   (every catchment)
    data/results/<design>/tuned/<catchment>/pooled_lstm/fold_<id>/winner.csv

Output (seas5ml.paths):
    data/results/<design>/sensitivity/<catchment>/pooled_lstm/
        sensitivity_fold_metrics.csv, sensitivity_summary.csv
    data/results/<design>/sensitivity/_pooled/pooled_lstm/
        fold_<id>/ checkpoints, removed after assembly
    data/trained_models/<design>/pooled_lstm_final.pt
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
from seas5ml import sensitivity as sens
from seas5ml.designs import SEED
from seas5ml.models import configs
from seas5ml.models import lstm as lstm_mod

DONE = 'DONE'     # checkpoint marker of a finished fold


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 2: climate-input sensitivity test, pooled model.")
    p.add_argument('model', type=str.lower, choices=configs.POOLED_MODELS)
    p.add_argument('--design', default=designs.DEFAULT_DESIGN,
                   choices=designs.DESIGNS)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════

def fold_plan(design, model, folds):
    """
    (fold, cfolds, winner config) of every pooled fold with scored
    catchments. The winner is read from the scored catchments' tuned
    winner.csv (one per catchment, all the same); stops with the folds whose
    winner is missing, and raises if the catchments disagree.
    """
    plan, missing = [], []
    for fold, cfolds in folds:
        scored = [c for c, cf in cfolds.items() if cf.test_years]
        if not scored:
            continue
        labels = set()
        for c in scored:
            path = paths.fold_dir(design, c, model, fold.id) / 'winner.csv'
            if path.exists():
                w = pd.read_csv(path)
                if len(w):
                    labels.add(str(w['winner_config'].iloc[0]))
        if not labels:
            missing.append(fold.id)
            continue
        if len(labels) > 1:
            raise ValueError(f"pooled fold {fold.id}: the catchments' "
                             f"winner.csv disagree: {sorted(labels)}")
        plan.append((fold, cfolds, configs.grid_config(model, labels.pop())))
    if missing:
        root = paths.kind_dir(design, 'tuned')
        print(f"ERROR: no tuned winner.csv for {len(missing)} pooled folds "
              f"under {root}/<catchment>/{model}/:\n       {missing}\n"
              f"       Run pipeline/1_models/run_pooled.py {model} tuned "
              f"--fold <id> for those folds first.")
        sys.exit(1)
    return plan


# ═══════════════════════════════════════════════════════════════════════
# FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════

def all_years_cfolds(pool):
    """Every catchment trains on all its years; early-stopping years drawn
    once from the union (FINAL_FIT_SALT), shared by the pool."""
    union = pooled.pool_years(pool)
    tr_yrs, val_yrs = tuning.make_val_split(
        union, val_seed=int(np.random.SeedSequence(
            [SEED, sens.FINAL_FIT_SALT]).generate_state(1)[0]))
    final = designs.Fold(index=-1, id='final', train_years=tuple(union),
                         test_years=(), tr_yrs=frozenset(tr_yrs),
                         val_yrs=frozenset(val_yrs))
    return {c: designs.restrict(final, m.years) for c, m in pool.items()}


def save_final(fitted, *, design, model, config, cfolds, stage, extra=None):
    """Persist the pool's final model under data/trained_models/<design>/."""
    out_dir = paths.trained_dir(design)
    os.makedirs(out_dir, exist_ok=True)
    path = out_dir / f"{model}_final.pt"
    meta = {
        'model': model,
        'design': design,
        'pool': list(pooled.POOLED_CATCHMENTS),
        'config': config['label'],
        'config_params': {k: v for k, v in config.items() if k != 'label'},
        'seed': SEED,
        'trained_on_years': {c: [int(y) for y in cf.train_years]
                             for c, cf in cfolds.items()},
        'val_yrs': {c: sorted(int(y) for y in cf.val_yrs)
                    for c, cf in cfolds.items()},
        'stage': stage,
        **(extra or {}),
    }
    models.save_fitted(fitted, path, meta=meta)
    return path


def final_model(design, model, plan, pool, fold_model):
    """loyo: the all-years fit with the voted winner. split: the fold model
    (refit, identically seeded, if this job resumed past its fold)."""
    if design == 'split':
        fold, cfolds, config = plan[0]
        print("\n[4/4] Saving the fold model as the final model...")
        if fold_model is None:
            print("  (fold finished in an earlier job: refitting it)")
            fold_model = models.fit_pooled(model, config, pool, cfolds)
        if fold_model is None:
            print("  WARNING: the fold produced no model -- nothing saved.")
            return
        path = save_final(fold_model, design=design, model=model,
                          config=config, cfolds=cfolds,
                          stage='2_sensitivity_fold_model',
                          extra={'fold': fold.id,
                                 'test_years': [int(y) for y in fold.test_years]})
        print(f"  Saved {path}  (config={config['label']})")
        return

    print("\n[4/4] Training the final model on every year of every "
          "catchment...")
    label = tuning.catchment_winner(
        [pd.DataFrame([{'winner_config': c['label']}]) for _, _, c in plan])
    if label is None:
        print("  WARNING: no tuned winners to vote on -- no final model.")
        return
    config = configs.grid_config(model, label)
    print(f"  config={label} (most pooled-fold wins)")
    t0 = time.time()
    cfolds = all_years_cfolds(pool)
    fitted = models.fit_pooled(model, config, pool, cfolds)
    if fitted is None:
        print("  WARNING: final fit produced no model -- nothing saved.")
        return
    path = save_final(fitted, design=design, model=model, config=config,
                      cfolds=cfolds, stage='2_sensitivity_final_fit')
    print(f"  Saved {path}  ({time.time() - t0:.0f}s)")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    model, design = args.model, args.design
    t0 = time.time()

    print("=" * 70)
    print(f"CLIMATE-INPUT SENSITIVITY (pooled): {model}  design={design}  "
          f"seed={SEED}")
    print(f"  pool: {', '.join(pooled.POOLED_CATCHMENTS)}")
    print(f"  scenarios: {', '.join(sens.SCENARIOS)}")
    print(f"  device={lstm_mod.DEVICE}, AMP={lstm_mod.USE_AMP}")
    print("=" * 70)

    print("\n[1/4] Loading the pool...")
    pool = pooled.load_pool()
    folds = pooled.pooled_folds(design, pool)

    print("\n[2/4] Resolving folds...")
    plan = fold_plan(design, model, folds)
    counts = pd.Series([c['label'] for _, _, c in plan]).value_counts()
    print(f"  {len(plan)} pooled folds, each using its own tuned winner: "
          + ", ".join(f"{k} x{v}" for k, v in counts.sort_index().items()))

    print("\n[3/4] Running folds...\n")
    print(f"  {'Fold/catch.':>9s}  {'HY':>6s}  {'Scenario':>8s}  {'NSE':>8s}  "
          f"{'KGE':>8s}  {'ACC':>8s}  {'CRPS':>9s}  {'Donor':>6s}  {'Time':>6s}")
    print(f"  {'-' * 91}")

    work = paths.pooled_work_dir(design, 'sensitivity', model)
    skipped = []
    fold_model = None
    for fold, cfolds, config in plan:
        fdir = work / f"fold_{fold.id}"
        if (fdir / DONE).exists():
            print(f"  {fold.id:>9s}  checkpoint found -- skipped")
            continue
        if fdir.exists():
            shutil.rmtree(fdir)         # interrupted while writing: redo
        fold_t0 = time.time()

        # Every scored catchment's scenarios, built before training.
        scenarios = {}
        for c, cf in cfolds.items():
            if not cf.test_years:
                continue
            s, skip = sens.test_year_scenarios(pool[c].df, cf, design,
                                               pool[c].years,
                                               tag=f"{c}/{fold.id}")
            skipped += [f"{c} {y}" for y in skip]
            if s:
                scenarios[c] = s

        os.makedirs(fdir, exist_ok=True)
        if scenarios:
            fitted = models.fit_pooled(model, config, pool, cfolds)
            if fitted is None:
                print(f"  {fold.id:>9s}  training produced no model -- "
                      f"skipping")
            else:
                if design == 'split':
                    fold_model = fitted
                for c, s in scenarios.items():
                    rows = sens.scenario_rows(
                        fitted, s, fold=cfolds[c], model=model, config=config,
                        t0=fold_t0, tag=f"{c}/{fold.id}", catchment=c)
                    if rows:
                        pd.DataFrame(rows).to_csv(fdir / f"ckpt_rows_{c}.csv",
                                                  index=False)
        (fdir / DONE).write_text("done\n")

    if skipped:
        print(f"\n  WARNING: {len(skipped)} test year(s) skipped, no usable "
              f"donor: {skipped}")

    # ── Assemble every catchment's stage-2 files ─────────────────────
    print("\nAssembling per-catchment results...")
    written = 0
    for c in pool:
        parts = [pd.read_csv(p, dtype={'fold': str})
                 for p in sorted(work.glob(f"fold_*/ckpt_rows_{c}.csv"))]
        if not parts:
            print(f"  WARNING: {c}: no scored fold -- nothing written")
            continue
        # Fold order (the checkpoint folders sort as the fold ids), rows
        # within a fold as scored: the layout of the single models' file.
        mdf = pd.concat(parts, ignore_index=True)
        out_dir = paths.run_dir(design, 'sensitivity', c, model)
        sens.write_results(out_dir, mdf, catchment=c, model=model)
        print(f"\n  {c}: {mdf['hydro_year'].nunique()} test years -> "
              f"{out_dir}/")
        sens.print_summary(mdf, model)
        written += 1
    if not written:
        print("\nERROR: no folds produced results.")
        sys.exit(1)
    shutil.rmtree(work)
    print(f"\n  Removed checkpoints {work}/")

    final_model(design, model, plan, pool, fold_model)
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == '__main__':
    main()
