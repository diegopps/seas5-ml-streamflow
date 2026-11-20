#!/usr/bin/env python3
"""
Stage 1 – Score the models at daily, weekly and monthly resolution.

The verification layer (seas5ml.aggregate, seas5ml.scores) on top of the
stage-1 predictions: no model is trained or rerun. Each model's daily
predictions are aggregated to every resolution and scored on the common
lead-month grid; the figures read the tables written here.

One job scores one (catchment, variant) line-up:

  default   every model from its default run: the baselines (climatology,
            persistence, damped_persistence, doy_reference), ridge, rf, xgb,
            lstm, pooled_lstm.
  tuned     the tuned models (rf, xgb, lstm, pooled_lstm) from their tuned
            folds, plus the models without a tuned variant from their
            default run: ridge (RidgeCV selects its own alpha) and the
            baselines. The line-up of the main results.

A model's predictions are read from predictions_members.parquet where it
exists (the learned models: CRPS over the members) and from
predictions_daily.parquet otherwise (the deterministic baselines). The tuned
models' folds are concatenated; a tuned model with folds missing against the
catchment's stage-0 fold manifest stops the job (scores on a subset of years
would pass unnoticed into the figures) unless --partial. Models without
results are skipped with a note.

Usage:
    python pipeline/1_models/score.py <catchment> default|tuned
        [--model <m> ...] [--design loyo|split] [--partial]

Input (seas5ml.paths):
    data/results/<design>/default/<catchment>/<model>/predictions_*.parquet
    data/results/<design>/tuned/<catchment>/<model>/fold_<id>/predictions_members.parquet
    data/processed_data/<catchment>_folds_<design>.txt      (tuned: fold check)

Output, per model (seas5ml.paths.scores_dir):
    data/results/<design>/scores/<variant>/<catchment>/<model>/
        yearly.csv                    one row per (resolution, hydro year,
                                      lead month): seas5ml.scores.yearly_scores
        cells.csv                     one row per (resolution, start month,
                                      lead month): seas5ml.scores.cell_scores
        ensemble_<resolution>.parquet the aggregated ensemble-mean frame
                                      (significance tests at figure time)
        stats.csv                     per resolution: periods kept / dropped
                                      (seas5ml.aggregate), where the
                                      predictions came from, folds read
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

from seas5ml import designs, paths
from seas5ml.aggregate import RESOLUTIONS
from seas5ml.catchments import CATCHMENTS
from seas5ml.models import configs, references
from seas5ml.scores import score_model

# The line-ups: (model, run it is read from).
FROM_DEFAULT = (references.BASELINES + configs.LEARNED_MODELS
                + configs.POOLED_MODELS)
LINEUPS = {
    'default': [(m, 'default') for m in FROM_DEFAULT],
    'tuned': ([(m, 'default') for m in references.BASELINES + ('ridge',)]
              + [(m, 'tuned') for m in configs.TUNED_MODELS
                 + configs.POOLED_MODELS]),
}

MEMBERS_FILE = 'predictions_members.parquet'
DAILY_FILE = 'predictions_daily.parquet'


def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: score every model of a line-up at daily, "
                    "weekly and monthly resolution.")
    p.add_argument('catchment', type=str.lower, choices=sorted(CATCHMENTS))
    p.add_argument('variant', choices=paths.SCORE_VARIANTS)
    p.add_argument('--model', nargs='+', type=str.lower,
                   help="score only these models of the line-up")
    p.add_argument('--design', default=designs.DEFAULT_DESIGN,
                   choices=designs.DESIGNS)
    p.add_argument('--partial', action='store_true',
                   help="score tuned models even if some folds are missing")
    args = p.parse_args()
    names = [m for m, _ in LINEUPS[args.variant]]
    if args.model:
        unknown = [m for m in args.model if m not in names]
        if unknown:
            p.error(f"not in the {args.variant} line-up: {unknown} "
                    f"(one of: {', '.join(names)})")
    return args


# ═══════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════

def _read(path):
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    df['init_date'] = pd.to_datetime(df['init_date'])
    return df


def load_default(design, catchment, model):
    """A default run's member frame, else its daily frame; (frame, info) or
    (None, reason)."""
    run = paths.run_dir(design, 'default', catchment, model)
    for name in (MEMBERS_FILE, DAILY_FILE):
        if (run / name).exists():
            return _read(run / name), {'source': f'default/{name}'}
    return None, f"no predictions in {run}"


def manifest_folds(design, catchment):
    """The catchment's fold ids from its stage-0 manifest, or None."""
    path = paths.fold_manifest_path(design, catchment)
    return path.read_text().split() if path.exists() else None


def load_tuned(design, catchment, model, *, partial):
    """A tuned model's member frames, every fold concatenated; (frame, info)
    or (None, reason). Stops on folds missing against the manifest unless
    `partial`."""
    root = paths.run_dir(design, 'tuned', catchment, model)
    found = {p.parent.name[len('fold_'):]: p
             for p in sorted(root.glob(f'fold_*/{MEMBERS_FILE}'))}
    if not found:
        return None, f"no tuned folds in {root}"
    expected = manifest_folds(design, catchment)
    missing = ([] if expected is None
               else [f for f in expected if f not in found])
    if missing:
        msg = (f"{model}: {len(missing)} of {len(expected)} tuned folds "
               f"missing under {root}: {missing}")
        if not partial:
            print(f"ERROR: {msg}\n       Finish them, or pass --partial.")
            sys.exit(1)
        print(f"  WARNING: {msg} -- scoring the {len(found)} present "
              f"(--partial)")
    frame = pd.concat([_read(p) for p in found.values()], ignore_index=True)
    return frame, {'source': f'tuned/fold_*/{MEMBERS_FILE}',
                   'n_folds': len(found),
                   'n_folds_missing': len(missing)}


# ═══════════════════════════════════════════════════════════════════════
# WRITING
# ═══════════════════════════════════════════════════════════════════════

def write_scores(out_dir, result, info):
    """The tables of seas5ml.scores.score_model's `result` into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    pd.concat([r['yearly'] for r in result.values()], ignore_index=True
              ).to_csv(out_dir / 'yearly.csv', index=False)
    pd.concat([r['cells'] for r in result.values()], ignore_index=True
              ).to_csv(out_dir / 'cells.csv', index=False)
    for res, r in result.items():
        r['ens'].to_parquet(out_dir / f'ensemble_{res}.parquet', index=False)
    pd.DataFrame([{'resolution': res, **r['stats'], **info}
                  for res, r in result.items()]
                 ).to_csv(out_dir / 'stats.csv', index=False)


def print_result(model, result):
    for res, r in result.items():
        y, c, s = r['yearly'], r['cells'], r['stats']
        acc = y['acc'].median() if len(y) else float('nan')
        print(f"    {res:>7s}: kept {s['kept']:,} periods (dropped: "
              f"{s['incomplete']:,} incomplete, {s['crosses_hydro_year']:,} "
              f"crossing 1 Oct, {s['beyond_grid']:,} beyond lead month 7) | "
              f"{y['hydro_year'].nunique()} years, {len(y)} yearly rows, "
              f"median ACC {acc:+.3f} | {int(c['acc'].notna().sum())}/"
              f"{len(c)} cells", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    t0 = time.time()
    print("=" * 70)
    print(f"STAGE 1 SCORES ({args.variant}): {CATCHMENTS[args.catchment]['name']}"
          f" ({args.catchment}) -- design {args.design} -- resolutions "
          f"{', '.join(RESOLUTIONS)}")
    print("=" * 70)

    lineup = [(m, src) for m, src in LINEUPS[args.variant]
              if not args.model or m in args.model]
    scored, skipped = [], []
    for model, src in lineup:
        t_model = time.time()
        print(f"\n── {model} ({src} run) ──", flush=True)
        if src == 'default':
            pred, info = load_default(args.design, args.catchment, model)
        else:
            pred, info = load_tuned(args.design, args.catchment, model,
                                    partial=args.partial)
        if pred is None:
            print(f"  skipped: {info}")
            skipped.append(model)
            continue
        print(f"  {len(pred):,} daily rows from {info['source']}")
        result = score_model(pred, design=args.design)
        print_result(model, result)
        out_dir = paths.scores_dir(args.design, args.variant,
                                   args.catchment, model)
        write_scores(out_dir, result, info)
        print(f"  Saved {out_dir}/  ({time.time() - t_model:.0f}s)")
        scored.append(model)

    print(f"\n{'=' * 70}")
    print(f"Scored {len(scored)}: {', '.join(scored) or '-'}")
    if skipped:
        print(f"Skipped {len(skipped)} (no results): {', '.join(skipped)}")
    print(f"DONE in {(time.time() - t0) / 60:.1f} minutes")
    print("=" * 70)
    if not scored:
        sys.exit(1)


if __name__ == '__main__':
    main()
