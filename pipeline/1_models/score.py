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
  sensitivity
            stage 2's scenarios (ridge, rf, xgb, lstm, pooled_lstm): each
            scenario's member predictions (predictions_members_<scenario>
            .parquet, checked to be paired row for row) scored like any run,
            plus deltas.csv, the paired per-year differences per (resolution,
            lead month) against the normal scenario, signed so that positive
            = the donor-year forcing hurt skill (seas5ml.scores
            .scenario_deltas). Run after stage 2.

A model's predictions are read from predictions_members.parquet where it
exists (the learned models: CRPS over the members) and from
predictions_daily.parquet otherwise (the deterministic baselines). The tuned
models' folds are concatenated; a tuned model with folds missing against the
catchment's stage-0 fold manifest stops the job (scores on a subset of years
would pass unnoticed into the figures) unless --partial. Models without
results are skipped with a note.

Usage:
    python pipeline/1_models/score.py <catchment> default|tuned|sensitivity
        [--model <m> ...] [--design loyo|split] [--partial]
    (--partial applies to tuned only)

Input (seas5ml.paths):
    data/results/<design>/default/<catchment>/<model>/predictions_*.parquet
    data/results/<design>/tuned/<catchment>/<model>/fold_<id>/predictions_members.parquet
    data/processed_data/<catchment>_folds_<design>.txt      (tuned: fold check)
    data/results/<design>/sensitivity/<catchment>/<model>/predictions_members_<scenario>.parquet

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
    sensitivity: the same four files with a leading `scenario` column
    (normal, mixed), plus
        deltas.csv                    one row per (corrupted scenario,
                                      resolution, hydro year, lead month):
                                      d_nse, d_kge, d_acc, d_crps, d_crpss
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
from seas5ml import sensitivity as sens
from seas5ml.aggregate import RESOLUTIONS
from seas5ml.catchments import CATCHMENTS
from seas5ml.models import configs, references
from seas5ml.scores import scenario_deltas, score_model

# The line-ups: (model, run it is read from).
FROM_DEFAULT = (references.BASELINES + configs.LEARNED_MODELS
                + configs.POOLED_MODELS)
LINEUPS = {
    'default': [(m, 'default') for m in FROM_DEFAULT],
    'tuned': ([(m, 'default') for m in references.BASELINES + ('ridge',)]
              + [(m, 'tuned') for m in configs.TUNED_MODELS
                 + configs.POOLED_MODELS]),
    # Stage 2: the models that read the forcing (the baselines never do).
    'sensitivity': [(m, 'sensitivity') for m in configs.LEARNED_MODELS
                    + configs.POOLED_MODELS],
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


def load_sensitivity(design, catchment, model):
    """A stage-2 run's member frame per scenario, checked to be paired
    (sens.check_paired); ({scenario: frame}, info) or (None, reason)."""
    run = paths.run_dir(design, 'sensitivity', catchment, model)
    files = {s: run / sens.members_file(s) for s in sens.SCENARIOS}
    absent = [p.name for p in files.values() if not p.exists()]
    if absent:
        return None, f"no {', '.join(absent)} in {run}"
    frames = {s: _read(p) for s, p in files.items()}
    sens.check_paired(frames)
    return frames, {'source': 'sensitivity/predictions_members_<scenario>'
                              '.parquet'}


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


def write_sensitivity_scores(out_dir, results, info):
    """
    The tables of every scenario's `score_model` result ({scenario: result})
    into out_dir, each with a leading `scenario` column (one file per table,
    as for the other variants), plus deltas.csv: the paired per-year
    differences of every corrupted scenario against normal
    (seas5ml.scores.scenario_deltas). Returns the deltas.
    """
    os.makedirs(out_dir, exist_ok=True)

    def tagged(frames):
        return pd.concat([f.assign(scenario=s)[['scenario'] + list(f.columns)]
                          for s, f in frames], ignore_index=True)

    yearly = {s: pd.concat([r['yearly'] for r in res.values()],
                           ignore_index=True) for s, res in results.items()}
    tagged(yearly.items()).to_csv(out_dir / 'yearly.csv', index=False)
    tagged([(s, r['cells']) for s, res in results.items()
            for r in res.values()]).to_csv(out_dir / 'cells.csv', index=False)
    for res_name in RESOLUTIONS:
        tagged([(s, res[res_name]['ens']) for s, res in results.items()]
               ).to_parquet(out_dir / f'ensemble_{res_name}.parquet',
                            index=False)
    pd.DataFrame([{'scenario': s, 'resolution': res_name,
                   **res[res_name]['stats'], **info}
                  for s, res in results.items() for res_name in res]
                 ).to_csv(out_dir / 'stats.csv', index=False)
    deltas = pd.concat([scenario_deltas(yearly['normal'], yearly[s], s)
                        for s in sens.SCENARIOS if s != 'normal'],
                       ignore_index=True)
    deltas.to_csv(out_dir / 'deltas.csv', index=False)
    return deltas


def print_deltas(deltas):
    """Median paired delta per lead month (ACC, CRPSS), to the log."""
    for (scen, res), d in deltas.groupby(['scenario', 'resolution'],
                                         sort=False):
        med = d.groupby('lead_month')[['d_acc', 'd_crpss']].median()
        print(f"    {scen} {res:>7s} median d_acc by lead month: "
              + " ".join(f"{v:+.3f}" for v in med['d_acc'])
              + " | d_crpss: "
              + " ".join(f"{v:+.3f}" for v in med['d_crpss']))


def print_result(model, result):
    for res, r in result.items():
        y, c, s = r['yearly'], r['cells'], r['stats']
        acc = y['acc'].median() if len(y) else float('nan')
        print(f"    {res:>7s}: kept {s['kept']:,} periods (dropped: "
              f"{s['incomplete']:,} incomplete, {s['crosses_hydro_year']:,} "
              f"crossing 1 Oct, {s['beyond_grid']:,} beyond lead month 7; "
              f"{s['contrib_dropped']:,} start-months not covered at every "
              f"resolution) | "
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
        out_dir = paths.scores_dir(args.design, args.variant,
                                   args.catchment, model)
        if src == 'sensitivity':
            frames, info = load_sensitivity(args.design, args.catchment,
                                            model)
            if frames is None:
                print(f"  skipped: {info}")
                skipped.append(model)
                continue
            results = {}
            for scen, pred in frames.items():
                print(f"  {scen}: {len(pred):,} member rows", flush=True)
                results[scen] = score_model(pred, design=args.design,
                                            model=model)
                print_result(model, results[scen])
            print_deltas(write_sensitivity_scores(out_dir, results, info))
            print(f"  Saved {out_dir}/  ({time.time() - t_model:.0f}s)")
            scored.append(model)
            continue
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
        result = score_model(pred, design=args.design, model=model)
        print_result(model, result)
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
