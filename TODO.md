# Revision TODO

Changes for the major revision. Nothing gets rerun until this list is complete.
Tags give the reviewer comment each item answers (R1–R4 = major comments, m1… = minor comments).

## TRAINING

### Old method: leave-one-year-out CV (stays the primary analysis; fix leakage) [R1]

- [ ] Fit the DOY climatology per fold on the training years only, including the inner folds of the Stage 2 nested CV.
      Select years by hydro year (1 Oct–30 Sep), not calendar year: the climatology and φ code currently apply the start year and skip_years by calendar year, while folds are hydro years. Excluding a test year by calendar year would leave Oct–Dec of the test hydro year in the climatology.
      `src/utils/climatology.py`, `src/utils/references.py` (`anomaly_record`); callers in `src/01_exploration/01_exploration.py`, `src/02_hyperparamtuning/02_hyperparamtuning_nested.py`, `src/03_sensitivity/03_sensitivity_climate_input.py`
- [ ] Compute `a0` per fold: store log1p(Q) at the init date in the feature parquet instead of `a0`, and subtract the fold's climatology at runtime.
      `src/00_preprocessing/00_preprocess_catchment.py`, `src/utils/features.py`, `src/utils/lstm.py`, `src/utils/references.py`
- [ ] Estimate damped-persistence φ on the training years only.
      `src/utils/references.py`, `src/01_exploration/01_exploration.py`
- [ ] Drop from training every trajectory initialised in the test year (its rows verifying in the next year carry the test year's `a0`).
      `src/utils/tabular.py`, `src/utils/lstm.py`, `src/utils/hptuning.py`, `src/utils/sensitivity.py`
- [ ] Stage 3: use each fold's own Stage 2 winner (`outer_<Y>/winner.csv`), not the catchment-wide most frequent winner: the other folds' votes come from inner CV that scored configs on the test year. The catchment-wide winner can stay for the final all-years model (never scored).
      `src/03_sensitivity/03_sensitivity_climate_input.py` (`load_catchment_winner_config`, `fold_plan`)
- [ ] Use the fair CRPS (Ferro 2014) so CRPS doesn't depend on ensemble size (25 hindcast vs 51 operational members); keep all members. Spread term divided by M(M−1) instead of M².
      `src/utils/metrics.py` (`calc_crps`); CRPS equation in Methods
- [ ] Methods: document that operational years (51 members) carry ~2× the training weight of hindcast years (25 members) in the CV. Not corrected.
- [ ] Check: in every ACC (models, baselines, day-of-year reference; per-year and pooled), forecast and observed anomalies use the same fold climatology. Mixing references (forecast built on training years, scored against a climatology that includes the test year) causes the cross-validation degeneracy (Barnston & van den Dool 1993), which biases ACC negative. The current setup has a mild form of it (day-of-year reference and learned models vs the global climatology); the per-fold climatology removes it.
      `src/utils/metrics.py` (`calc_acc`), `src/utils/plot_boxplots.py` (`doy_reference_acc`), `src/utils/plot_heatmap.py`

### New method: train on hindcasts, test on operational forecasts [R1]

A second, independent evaluation run in parallel with the CV, not a replacement. Same pipeline, with one split instead of N folds: same catchments, features, models, baselines, metrics and three stages.

**Split**
- [ ] Train on hydro years ≤ 2016 (all rows come from hindcasts); test on hydro years ≥ 2018 (all rows come from operational forecasts, inits from Mar 2017). Hydro year 2017 is a buffer, neither trained nor scored: it is the only year mixing hindcast and operational inits, and leaving it out keeps test-year `a0` out of training (same role as dropping test-year trajectories in the CV).
      **Decide (at implementation):** keep hydro year 2017 as a buffer, or use it.
- [ ] Fit the climatology, `a0` and φ on the training hydro years only, with the same rules as the CV (from 1981, skip_years excluded, hydro years).
- [ ] Count test years per catchment from the data (expected from file names: 8 Central Asia (2018–2025), 6 Kirchbichl (2018–2023), 3 Diepoldsau (2018–2020)).
- [ ] Check whether newer discharge records exist for the Alpine gauges (GRDC; BAFU for the Rhine, eHYD for the Inn) to lengthen their test period.

**Stage 1 (untuned)**
- [ ] Same default configs as the CV; the LSTM early-stops on 15% of the training hydro years.

**Stage 2 (tuning)**
- [ ] Same 15-config grid; 5 inner folds over the training hydro years; select by mean inner ACC; retrain the winner on all training years; score on the test years. Equivalent to one outer fold of the CV. Untuned reference = the split's Stage 1.

**Stage 3 (sensitivity)**
- [ ] Use the split's Stage 2 winner (one config per catchment, chosen without test data). For each test year, draw the donor from the other test years (same era, same 51 members).

**Pooled LSTM**
- [ ] Run the pooled 9-catchment LSTM (subsection below) on the split too, in parallel with the single-catchment LSTM.

**Evaluation**
- [ ] Same metrics (NSE, KGE, per-year ACC, CRPSS with fair CRPS), lead windows (0–30, 185–214 d) and day-of-year reference line; score all 51 test members.
- [ ] Heatmap: `MIN_YEARS = 10` blanks every cell with 3–8 test years. Decide: lower the threshold for the split, pool catchments within a regime, or drop the heatmap for the split.
- [ ] Config-rank figure: no outer folds to count wins over. Decide: show ranks across the inner folds instead, or drop it for the split.
- [ ] Significance: a per-catchment bootstrap on 3–8 years has little power. Report per catchment descriptively; for significance tests, pool catchment-years within each regime.

### ESP-style benchmark (Stage 3 extension) [R2]

- [ ] **Discuss with co-authors.** Same trained model and real `a0`, but forcing replaced by every other year's (not one random donor): an ensemble of climatological weather from this year's initial state, compared with the SEAS5-forced forecast. Inference only, no retraining. Needs the real year's initial snow state kept (Stage 3 currently swaps SEAS5 snow depth, including lead 0, with the weather). Optional mirror test (reverse ESP, Wood & Lettenmaier 2008): real forcing, `a0` from other years.

### Pooled LSTM on the 9 study catchments [R3]

A parallel run to the single-catchment LSTM: one LSTM trained on all 9 catchments at once, scored per catchment with the same folds, metrics and baselines. No new data needed beyond static attributes.

- [ ] Run Stages 1–3 as for the single-catchment LSTM (default config, nested tuning with the same grid, donor-year sensitivity), under both the CV and the hindcast→operational split.
- [ ] Folds by year across all catchments: when Y is the test year, remove it from all 9 catchments (neighbours share that year's weather and anomalies; Kirchbichl and Diepoldsau have overlapping SEAS5 boxes). One model per year then scores every catchment that has year Y.
- [ ] Add static catchment attributes as inputs (e.g. area, mean elevation, glacier fraction, aridity, snow fraction), with the same definitions for all 9 (HydroATLAS as in Caravan; glacier fraction from RGI).
- [ ] Standardise dynamic inputs with one scaler over the pooled training data, so catchment identity comes from the static attributes.
- [ ] Loss: normalise each catchment's errors by its own anomaly variance (NSE* loss, Kratzert et al. 2019, HESS); otherwise the steppe catchments, with much larger anomalies, dominate training.
- [ ] Per-catchment climatology, `a0`, φ and all R1 leakage rules apply unchanged.
- [ ] Optional: a 7-catchment Central Asian version, to see whether adding the data-rich Alpine catchments helps the Central Asian ones.
- [ ] Optional: fine-tune the pooled model on each catchment's training years (the reviewer's "fine-tuned" variant).
- [ ] Manuscript: move regional training from future work (Discussion, `main.tex` L421) into the Results, and discuss the limits of single-catchment training.
- [ ] **Discuss with co-authors.** External donors: a regional LSTM that also trains on snow-/glacier-fed catchments outside the study set, then evaluated (or fine-tuned) on the 9. Donors must lie inside the already-downloaded SEAS5 domains, so their forcing needs no new download: Alpine ones from Caravan (LamaH-CE, CAMELS-CH), Central Asian ones from GRDC, CA-discharge (Marti et al. 2023) or extra Kazhydromet gauges. Needs donor discharge, outlines and attributes, and the same cross-catchment year exclusion. Tension with the paper's data-scarcity framing.

### Monthly models [R4]

- [ ] **Discuss with co-authors.** Train monthly models directly (monthly-mean anomaly target, monthly SEAS5 inputs, 7-step LSTM sequence), as in Swift-LaPointe et al. (2025), and compare with the aggregated daily predictions (see EVALUATION). Goes beyond the reviewer's request; would make monthly the main result and change the title and framing.

## EVALUATION

### Weekly, monthly and seasonal aggregation [R4]

A verification layer on top of the existing daily predictions (no retraining), applied to every run: CV and split, single-catchment and pooled LSTM, all baselines.

- [ ] Aggregate daily predictions per member into weekly and monthly means for each forecast start. Forecasts start on the 1st, so lead months are calendar months; 215 days always cover 7 full months.
- [ ] Compute anomalies of the aggregated means against a weekly/monthly climatology from training years only (R1 rules). Decide: log of the aggregated mean, or mean of the daily log anomalies.
- [ ] Score per start month and lead week/month: ACC across years (year-to-year skill) and CRPSS (fair CRPS), against damped persistence aggregated the same way; significance by paired bootstrap over years, as now.
- [ ] Seasonal target: April–September mean flow from the 1 March and 1 April starts (the only ones covering the whole season). This is the quantity the hydromet agencies forecast operationally (Apel et al. 2018).
- [ ] Probabilistic: above-/below-normal (tercile) forecasts, with terciles from training years only; RPSS against climatological 1/3 probabilities, and ROC area for each event.
- [ ] Decide: how to give forecasts a realistic spread for the probabilistic scores. Model ensembles are near-identical (the models barely respond to forcing) and damped persistence is deterministic; e.g. dress every forecast, baselines included, with its training-period errors.
- [ ] Figures: a lead-month version of the heatmap (start month × lead month, ACC across years) could replace Fig. 5.
- [ ] Manuscript: justify practical relevance (daily at short leads; weekly, monthly and April–September at longer leads, for reservoir operations and water allocation).

### Forcing sensitivity by lead time [m3]

- [ ] Stage 3: save per-member predictions for both scenarios (real and donor forcing), so differences can be scored for any lead window afterwards.
      `src/03_sensitivity/03_sensitivity_climate_input.py`, `src/utils/sensitivity.py`
- [ ] Score the paired differences per Fig. 5 lead bin (0–14, 15–30, 31–60, 61–90, 91–120, 121–150, 151–215 d): ACC, CRPSS, NSE, KGE.
- [ ] Figure: median difference with IQR per lead bin, one line per model (replaces Fig. 9). Decide: add a start-month × lead-bin heatmap of the change in pooled ACC.
      `src/utils/plot_sensitivity.py`
- [ ] Optional: test difference ≠ 0 per bin (paired bootstrap, BH across bins and models).
- [ ] Decide: keep the real year's SEAS5 snow depth in the donor swap, so short-lead differences reflect weather only (same fix as the ESP-style item).

## SEAS5 VS REANALYSIS COMPARISON

### SEAS5 meteorological skill against ERA5-Land [R2]

- [ ] Download ERA5-Land for both domains, 1981–2025, from the CDS: daily mean/max/min 2 m temperature, total precipitation, snowfall, snow water equivalent, surface solar radiation downwards. Use the CDS daily-statistics product if it covers these variables (avoids hourly downloads).
- [ ] Match units and accumulation conventions with the SEAS5 preprocessing (e.g. snow as water equivalent in both).
- [ ] Average to the same 1° catchment boxes as SEAS5 (cos-latitude weighted). Optional: also over the catchment polygons, to test the Discussion claim that box averaging washes out orographic gradients.
- [ ] Compute anomalies against each dataset's own climatology (SEAS5 per init month and lead, ERA5-Land per day of year), using training years only (same leakage rules as the CV / split), so skill reflects information rather than model bias. Report raw bias separately.
- [ ] Score SEAS5 the same way as streamflow: same aggregation levels (daily, weekly, monthly, April–September), lead windows, metrics (ACC, CRPSS with fair CRPS) and figure layouts, so each SEAS5 skill figure sits next to its streamflow counterpart. Reference is climatology (significance against zero skill).
- [ ] Link the result to the Discussion: at which leads SEAS5 carries skill, and whether that explains the models' insensitivity to the forcing.

## MANUSCRIPT

### Tuned models as the main results [m1]

- [ ] Main results on tuned models (nested-CV winners; Ridge via RidgeCV) and baselines; Stage 1 no longer a separate stage. Figs. 3–5 redrawn with tuned models.
- [ ] Appendix: default vs tuned comparison (Figs. 7–8) and configuration ranking (Fig. 6).
- [ ] Methods keeps the nested-CV description. Update the Fig. 2 schematic, the stage numbering and the abstract.
- [ ] Code: `src/01_exploration/01_exploration.py` still runs (baselines and the untuned reference that Stage 2 reads); rename stages in the code and README to match the paper.

## TO REVISE (discuss with co-authors before doing anything)

### Bias correction and downscaling of SEAS5 [m2]

- [ ] **Discuss with co-authors:** small change or major work. Suggestions so far:
  - Catchment-scale quantile mapping: map the SEAS5 box mean onto the ERA5-Land catchment-outline mean, per variable, start month and lead window (bias correction and downscaling in one step; reuses the R2 ERA5-Land download). Alternatives: grid-based BCSD at 0.1° (Wood et al. 2002; Lorenz et al. 2021), with much larger data volume; dynamical downscaling is out of scope.
  - Fit the quantile mapping on training years only (per fold in the CV, ≤ 2016 in the split).
  - Rerun the tuned models with the corrected forcing (per-fold winners, no new search), plus Stage 3.
  - Score raw vs corrected SEAS5 in the R2 comparison.
  - Expected effect is small: statistical downscaling can't add year-to-year information missing from the 1° forecast, and the models already ignore the forcing. A null result makes "the limit isn't SEAS5's bias or grid" a tested claim.
  - Manuscript, regardless of the decision: reword the Discussion (`main.tex` L415, L419) and Conclusion bullet 3 (downscaling recovers average gradients, not forecast information). In the response, note that the other seasonal systems on the CDS are also 1°.
  - Optional resolution test with perfect forcing: ERA5-Land averaged over the catchment outline vs over the 1° box.
