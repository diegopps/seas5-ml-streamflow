"""
seas5ml.sensitivity – Climate-input sensitivity test (stage 2): the
donor-year forcing swap. Models are fitted and scored through
seas5ml.models and seas5ml.evaluate, as in stage 1.

Asks whether the learned models actually use the meteorological forcing at
inference time, or whether they are really tracking the seasonal cycle and
would score the same on forcing that carries no information about the target
year. A model whose skill barely moves when its forcing is replaced by another
year's was never using that forcing.

Two inference scenarios per fold, both scored on the same held-out year:

  normal  real forcing for the target year — the reference score.
  mixed   the seven meteorological variables are replaced wholesale by a
          randomly chosen donor hydro year's, matched on the exact structural
          analogue (init month, lead day, ensemble member slot) so the swapped
          series keeps the real forecast's seasonal position, lead-time
          structure, and ensemble spread. Only the year the weather came from
          changes.

Donor-year swapping rather than added noise
-------------------------------------------
The corruption is deliberately a real year's weather rather than synthetic
noise. Noise-corrupted forcing pushes the inputs out of the distribution the
model was trained on, so a skill drop has two possible causes that cannot be
told apart: the model was genuinely using the forcing, or it was merely thrown
off by values it had never seen. A donor year is in-distribution by
construction — real weather, real ensemble spread, real lead-time structure —
so a skill drop can only mean the model was using *this* year's weather
specifically. That is the question, and swapping years asks it cleanly.

The calendar encodings (sin_doy, cos_doy) are never corrupted. They carry no
weather information — they are the model's legitimate access to "what time of
year is it", and corrupting them would test something else entirely (whether
the model knows the season), which is not the question. The antecedent anomaly
(a0) is likewise never corrupted: it is the observed log1p(Q) anomaly at
initialisation, not meteorological forcing. The corruption writes only
MET_VARS, so logq0 passes through untouched, and both scenarios get the same
a0 (formed from logq0 against the model's own climatology at prediction
time); the test measures use of the forcing rather than use of the initial
condition.

Climatology
-----------
Every fitted model carries the climatology it was trained against
(`fitted['doy_clim']`), fit on exactly its training years (the fold's
training years; every year for loyo's final model). Prediction reuses it for
a0, the back-transform and Q_clim, so a model is never scored against a
climatology that includes its test year, and a saved model needs nothing
else to predict.

Train once, infer twice
-----------------------
Each fold trains exactly one model and then runs two inference passes over it.
Retraining per scenario would confound the effect of corrupting the input with
ordinary training stochasticity — for the LSTM especially, two runs of the same
config on the same data differ enough to swamp the effect being measured.
Scenario deltas are therefore differences in inference input alone, with model
weights held fixed.

Which hyperparameters
---------------------
rf, xgb, and lstm use, in each fold, that fold's own tuned winner (the
config chosen by the inner CV of stage 1's tuned run for the same fold), so
the test year played no part in choosing it. The catchment-wide winner (most
fold wins) is used only for loyo's final all-years model, which is never
scored. Under the split there is one fold, and its model is the final one.
Ridge has no search (RidgeCV selects its own regularisation by internal
generalised cross-validation) and uses its default configuration.

Paired-row guarantee
--------------------
The two scenarios are compared per fold as paired differences, so they must be
scored over identical rows. A donor year can be missing an (init month, lead
day, member) combination that the target year has — rare, but real at the
record's edges. Rows that fail to find a donor match are dropped from both
scenarios, not just from `mixed`, so every comparison stays paired.

Per-member predictions
----------------------
Each scenario's per-member predictions are written alongside the
full-horizon scores (`write_results`, `members_file`), over exactly the
paired rows above, so the paired differences can be scored afterwards for
any lead month and resolution by the verification layer
(pipeline/1_models/score.py sensitivity; seas5ml.scores.scenario_deltas).
Both scenarios share the observations and Q_clim row for row; only Q_pred
differs.

Partial donor years
-------------------
A year at the record's edge can hold only part of its rows (e.g. Oct-Dec
when discharge ends in December, or no late summer when the last SEAS5 init
is in December). Drawn as a donor, it would leave most of the test year
unscored by the rule above. A candidate therefore stays in the donor pool
only if it can supply at least MIN_DONOR_SHARE of the test year's rows
(`usable_donors`), measured with the same matching as the swap itself. The
rule is the same under every design. A test year with no usable candidate
is skipped (the caller warns).
"""

import time

import numpy as np
import pandas as pd

from .data import GROUP_COL

# The seven meteorological drivers — the only columns the corruption touches.
# `corrupt_mixed` writes these by explicit loop, so every other dynamic
# predictor (sin_doy, cos_doy, a0) passes through untouched by construction.
MET_VARS = ['tas', 'tasmax', 'tasmin', 'ssrd', 'tp', 'sd', 'sf']

SCENARIOS = ('normal', 'mixed')

# Fixed salt keeping the donor draw reproducible from the fold seed alone.
DONOR_SALT = 811

# Salt for the loyo final model's val split (stage 2), keeping it distinct
# from every fold's split seed while staying reproducible from SEED alone.
FINAL_FIT_SALT = 20260803

# A candidate donor year must be able to supply at least this share of the
# test year's rows (see `usable_donors`); others leave the pool.
MIN_DONOR_SHARE = 0.9


# ═══════════════════════════════════════════════════════════════════════
# INPUT CORRUPTION
# ═══════════════════════════════════════════════════════════════════════

def pick_donor_year(candidates, *, fold_seed):
    """Choose the donor hydro year for the `mixed` scenario from
    `candidates` (seas5ml.designs.donor_pool), reproducibly."""
    candidates = list(candidates)
    if not candidates:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([fold_seed, DONOR_SALT]))
    return int(rng.choice(candidates))


def _member_slots(df_test, donor):
    """Map each of the target's members onto a donor member by position,
    wrapping if the donor has fewer (see `corrupt_mixed`)."""
    test_members = np.sort(df_test['member'].unique())
    donor_members = np.sort(donor['member'].unique())
    return {int(m): int(donor_members[i % len(donor_members)])
            for i, m in enumerate(test_members)}


def donor_share(df_test, df_full, donor_year, test_year):
    """
    Share of `test_year`'s rows in `df_test` that `donor_year` can supply:
    rows whose (init month, lead day, member slot) exists in the donor year,
    matched exactly as `corrupt_mixed` matches them. The rest would be
    dropped from both scenarios. 0.0 if the donor has no rows.
    """
    donor = df_full[df_full[GROUP_COL] == donor_year]
    target = df_test[df_test[GROUP_COL] == test_year]
    if len(donor) == 0 or len(target) == 0:
        return 0.0
    slot = _member_slots(df_test, donor)
    have = pd.MultiIndex.from_arrays([donor['init_date'].dt.month,
                                      donor['lead_day'], donor['member']])
    need = pd.MultiIndex.from_arrays([target['init_date'].dt.month,
                                      target['lead_day'],
                                      target['member'].map(slot)])
    return float(need.isin(have).mean())


def usable_donors(df_test, df_full, candidates, test_year, *,
                  min_share=MIN_DONOR_SHARE):
    """
    The candidates (seas5ml.designs.donor_pool) that can supply at least
    `min_share` of the test year's rows (`donor_share`), in their original
    order, plus every candidate's share as {year: share}. Keeps partial
    years (record edges: Oct-Dec only, or missing the late summer) out of
    the draw, where they would leave most of the test year unscored. Same
    rule under every design.
    """
    shares = {int(y): donor_share(df_test, df_full, int(y), test_year)
              for y in candidates}
    return [int(y) for y in candidates if shares[int(y)] >= min_share], shares


def corrupt_mixed(df_test, df_full, donor_year):
    """
    Replace MET_VARS with the donor hydro year's values.

    Matching is on (init month, lead day, member slot), which is the donor's
    structurally equivalent forecast: same point in the seasonal cycle, same
    forecast horizon, same position in the ensemble. Verified unique on this
    dataset — (hydro_year, init_month, lead_day, member) has no duplicates.

    Member counts differ by era (25 in the hindcast years, 51 in the forecast
    years), so the target's members are mapped onto the donor's by position,
    wrapping if the donor has fewer. This keeps genuine member-to-member
    variation in the swapped forcing rather than collapsing every member onto
    one averaged series, which would silently shrink ensemble spread and, via
    the pairwise-spread term, flatter CRPS for reasons unrelated to skill.

    Returns the corrupted frame; rows with no donor match carry NaN met values
    and are filtered by the caller (see `paired_row_mask`).
    """
    donor = df_full[df_full[GROUP_COL] == donor_year]
    if len(donor) == 0:
        return None

    slot = _member_slots(df_test, donor)

    donor = donor.copy()
    donor['_im'] = donor['init_date'].dt.month
    lut = donor.set_index(['_im', 'lead_day', 'member'])[MET_VARS]

    out = df_test.copy()
    out['_im'] = out['init_date'].dt.month
    out['_dm'] = out['member'].map(slot)

    idx = pd.MultiIndex.from_arrays([out['_im'], out['lead_day'], out['_dm']])
    swapped = lut.reindex(idx)
    for v in MET_VARS:
        out[v] = swapped[v].values

    return out.drop(columns=['_im', '_dm'])


def paired_row_mask(df_mixed):
    """Rows whose donor swap produced complete met values. Applied to both
    scenarios so they are scored over identical rows (see module docstring)."""
    return df_mixed[MET_VARS].notna().all(axis=1).values


def build_scenarios(df_test, df_full, donor_candidates, test_year, *,
                    fold_seed):
    """
    Build the two scenario frames for one test year, restricted to the rows
    both scenarios can supply. The donor year is drawn from those
    `donor_candidates` (seas5ml.designs.donor_pool) that can supply at least
    MIN_DONOR_SHARE of the test year's rows (`usable_donors`).

    `df_test` holds every trajectory that reaches `test_year`, whole, so the
    LSTM can read each from initialisation (see lstm.build_trajectories);
    rows from neighbouring hydro years are input context and never scored.
    The donor swap covers the context rows too, so a corrupted trajectory is
    corrupted from lead 0. Only test-year rows are subject to the paired-row
    filter: dropping a context row would open a gap inside a sequence. A
    context row without a donor match keeps NaN forcing, which the LSTM's
    standardisation maps to the training mean.

    Returns (frames, donor_year, n_rows, n_dropped, donors) where `frames`
    maps scenario name to its frame and the counts refer to test-year rows,
    or (None, ...) if no candidate is usable. `donors` describes the pool:
    n_candidates, n_excluded, excluded ({year: share} of the candidates
    below MIN_DONOR_SHARE) and donor_share (the drawn donor's share).
    """
    usable, shares = usable_donors(df_test, df_full, donor_candidates,
                                   test_year)
    donors = {'n_candidates': len(shares),
              'n_excluded': len(shares) - len(usable),
              'excluded': {y: s for y, s in shares.items() if y not in usable},
              'donor_share': np.nan}
    donor_year = pick_donor_year(usable, fold_seed=fold_seed)
    if donor_year is None:
        return None, None, 0, 0, donors
    donors['donor_share'] = shares[donor_year]

    df_mixed_full = corrupt_mixed(df_test, df_full, donor_year)
    if df_mixed_full is None:
        return None, donor_year, 0, 0, donors

    in_test = (df_mixed_full[GROUP_COL] == test_year).values
    matched = paired_row_mask(df_mixed_full)
    keep = matched | ~in_test
    n_dropped = int((in_test & ~matched).sum())

    df_base = df_test.loc[keep].reset_index(drop=True)
    df_mixed = df_mixed_full.loc[keep].reset_index(drop=True)

    frames = {'normal': df_base, 'mixed': df_mixed}
    return frames, donor_year, int((in_test & matched).sum()), n_dropped, donors


# ═══════════════════════════════════════════════════════════════════════
# ONE FOLD, ONE CATCHMENT (shared by stage 2's single and pooled drivers)
# ═══════════════════════════════════════════════════════════════════════

def test_year_scenarios(df, fold, design, years, *, tag):
    """
    The scenario frames of every test year of `fold` for one catchment
    (`df` its analysis frame, `years` its hydro years), built before any
    training so a fold without a usable donor costs nothing. The donor pool
    is designs.donor_pool's, filtered by `usable_donors`; the draw is seeded
    per (SEED, test year), independent of the model. `tag` labels the
    printed notes (the fold id; the pooled driver adds the catchment).

    Returns ({test_year: (frames, donor_year, n_rows, n_dropped, donors)},
    [test years skipped for want of a usable donor]).
    """
    from .designs import SEED, donor_pool

    scenarios, skipped = {}, []
    for test_year in fold.test_years:
        # Every trajectory reaching the test year, whole: the LSTM reads
        # it from initialisation; rows outside the test year are context.
        test_inits = df.loc[df[GROUP_COL] == test_year, 'init_date'].unique()
        if len(test_inits) == 0:
            continue
        df_test = df[df['init_date'].isin(test_inits)]
        # Seeded per (seed, test year): the donor draw is reproducible
        # and independent of the model.
        fold_seed = int(np.random.SeedSequence(
            [SEED, test_year]).generate_state(1)[0])
        frames, donor_year, n_rows, n_dropped, donors = build_scenarios(
            df_test, df, donor_pool(design, fold, test_year, years),
            test_year, fold_seed=fold_seed)
        if donors['excluded']:
            print(f"  {tag:>9s}  {test_year:6d}  note: "
                  f"{donors['n_excluded']} of {donors['n_candidates']} "
                  f"donor candidates supply < {MIN_DONOR_SHARE:.0%} "
                  f"of the year's rows, excluded: "
                  + ", ".join(f"{y} ({s:.0%})" for y, s
                              in sorted(donors['excluded'].items())))
        if frames is None:
            print(f"  {tag:>9s}  {test_year:6d}  WARNING: no usable "
                  f"donor year ({donors['n_candidates']} candidates, none "
                  f"supplies >= {MIN_DONOR_SHARE:.0%} of the year's "
                  f"rows) -- test year SKIPPED, absent from the "
                  f"sensitivity results")
            skipped.append(test_year)
            continue
        if n_dropped:
            print(f"  {tag:>9s}  {test_year:6d}  note: {n_dropped:,} "
                  f"rows dropped from all scenarios (donor {donor_year} "
                  f"lacks a match); {n_rows:,} retained")
        scenarios[test_year] = (frames, donor_year, n_rows, n_dropped, donors)
    return scenarios, skipped


def scenario_rows(fitted, scenarios, *, fold, model, config, t0, tag,
                  catchment=None):
    """
    Score both scenarios of every test year in `scenarios`
    (`test_year_scenarios`) with the fitted model, against its own (fold)
    climatology. One sensitivity_fold_metrics.csv row per (test year,
    scenario), printed as it comes; `t0` is the fold's start time, `tag` the
    printed label. `catchment` is passed to a pooled model's predict.

    Returns (rows, members): `members` maps each scenario to its per-member
    prediction frames, one per scored test year (the frames the rows were
    scored from; `write_results` writes them, so the paired differences can
    be scored per lead month afterwards: pipeline/1_models/score.py
    sensitivity). A test year is kept only if every scenario scored it, so
    the scenarios' frames stay paired.
    """
    from . import models
    from .evaluate import score_members

    predict_kw = {} if catchment is None else {'catchment': catchment}
    rows = []
    members = {scenario: [] for scenario in SCENARIOS}
    for test_year, (frames, donor_year, n_rows, n_dropped,
                    donors) in scenarios.items():
        year_rows, year_preds = [], {}
        for scenario in SCENARIOS:
            pred = models.predict(fitted, frames[scenario],
                                  score_years=[test_year], **predict_kw)
            scores = score_members(pred, test_year=test_year)
            if scores is None:
                continue
            year_preds[scenario] = pred
            year_rows.append({
                'fold': fold.id,
                'hydro_year': test_year,
                'scenario': scenario,
                'model': model,
                'config': config['label'],
                'nse': scores['nse'], 'kge': scores['kge'],
                'acc': scores['acc'], 'crps': scores['crps'],
                'donor_year': donor_year if scenario == 'mixed' else np.nan,
                'n_rows': n_rows,
                'n_rows_dropped': n_dropped,
                'donor_share': round(donors['donor_share'], 4),
                'n_donor_candidates': donors['n_candidates'],
                'n_donors_excluded': donors['n_excluded'],
                'best_epoch': fitted['info'].get('best_epoch', np.nan),
                'time_s': round(time.time() - t0, 1),
            })
            donor_str = str(donor_year) if scenario == 'mixed' else '-'
            print(f"  {tag:>9s}  {test_year:6d}  {scenario:>8s}  "
                  f"{scores['nse']:+8.3f}  {scores['kge']:+8.3f}  "
                  f"{scores['acc']:+8.3f}  {scores['crps']:9.3f}  "
                  f"{donor_str:>6s}  {time.time() - t0:5.0f}s",
                  flush=True)
        if len(year_preds) < len(SCENARIOS):
            print(f"  {tag:>9s}  {test_year:6d}  WARNING: scored under "
                  f"{sorted(year_preds) or 'no scenario'} only -- test year "
                  f"left out (scenarios must be paired)")
            continue
        rows += year_rows
        for scenario, pred in year_preds.items():
            members[scenario].append(pred)
    return rows, members


PAIR_KEYS = ['init_date', 'date', 'member']


def check_paired(frames: dict) -> None:
    """
    Raise ValueError unless every scenario's member frame ({scenario: frame},
    as written by `write_results`) holds the same (init_date, date, member)
    rows with the same Q_raw and Q_clim: the paired-row guarantee, checked
    on what was written. Only Q_pred may differ between scenarios.
    """
    names = list(frames)
    ref = frames[names[0]].sort_values(PAIR_KEYS).reset_index(drop=True)
    for name in names[1:]:
        other = frames[name].sort_values(PAIR_KEYS).reset_index(drop=True)
        if (len(other) != len(ref)
                or not ref[PAIR_KEYS].equals(other[PAIR_KEYS])):
            raise ValueError(f"scenario frames {names[0]} ({len(ref):,} rows) "
                             f"and {name} ({len(other):,} rows) do not hold "
                             f"the same (init_date, date, member) rows")
        for col in ('Q_raw', 'Q_clim'):
            if not np.array_equal(ref[col].to_numpy(float),
                                  other[col].to_numpy(float), equal_nan=True):
                raise ValueError(f"scenario frames {names[0]} and {name} "
                                 f"differ in {col}")


def members_file(scenario: str) -> str:
    """File name of one scenario's per-member predictions in a stage-2 run
    folder."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}' (one of {SCENARIOS})")
    return f"predictions_members_{scenario}.parquet"


def write_results(out_dir, mdf, *, catchment, model, members):
    """
    Into `out_dir`: sensitivity_fold_metrics.csv (the rows),
    sensitivity_summary.csv (per-scenario mean and std of every metric) and
    one member file per scenario (`members_file`), from `members`
    ({scenario: frame}, every scored test year; seas5ml.io_results
    .write_members format).
    """
    import os
    from .io_results import write_members
    os.makedirs(out_dir, exist_ok=True)
    for scenario in SCENARIOS:
        write_members(os.path.join(out_dir, members_file(scenario)),
                      members[scenario])
    mdf.to_csv(os.path.join(out_dir, 'sensitivity_fold_metrics.csv'),
               index=False)
    summary = []
    for scenario in SCENARIOS:
        s = mdf[mdf['scenario'] == scenario]
        if len(s) == 0:
            continue
        row = {'catchment': catchment, 'model': model,
               'scenario': scenario, 'n_years': len(s)}
        for metric in ('nse', 'kge', 'acc', 'crps'):
            row[f'{metric}_mean'] = round(s[metric].mean(), 4)
            row[f'{metric}_std'] = round(s[metric].std(), 4)
        summary.append(row)
    pd.DataFrame(summary).to_csv(
        os.path.join(out_dir, 'sensitivity_summary.csv'), index=False)


def print_summary(mdf, model):
    """Scenario means and the median paired delta vs. normal, to the log."""
    print(f"\n  Scenario means ({model.upper()}):")
    pivot = mdf.pivot_table(index='scenario',
                            values=['nse', 'kge', 'acc', 'crps'],
                            aggfunc='mean').reindex(list(SCENARIOS))
    for scenario, r in pivot.iterrows():
        print(f"    {scenario:>8s}  NSE={r['nse']:+.4f}  KGE={r['kge']:+.4f}  "
              f"ACC={r['acc']:+.4f}  CRPS={r['crps']:.4f}")

    print(f"\n  Median paired delta vs. normal "
          f"(positive = corruption hurt skill):")
    base = mdf[mdf.scenario == 'normal'].set_index('hydro_year')
    for scenario in [s for s in SCENARIOS if s != 'normal']:
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

