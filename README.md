# Can SEAS5 seasonal forecasts drive daily streamflow prediction? Machine learning experiments in Central Asian catchments

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23015467.svg)](https://doi.org/10.5281/zenodo.23015467)

Code for the article of the same title (Pineda Schwarz, Didovets, Fallah and
Tillakarim, *Journal of Hydrology: Regional Studies*).

The study tests whether ECMWF's openly available SEAS5 seasonal forecast
carries usable information for daily discharge prediction at lead times up to
214 days. Ridge regression, Random Forest, XGBoost and an LSTM are trained to
predict the log-transformed discharge anomaly relative to the day-of-year
climatology. They are evaluated by leave-one-hydrological-year-out
cross-validation against climatology, persistence and damped-persistence
baselines, and compared untuned, after nested-CV tuning, and under corrupted
forcing.

- Article: [-] — DOI
- This code (Zenodo): [10.5281/zenodo.23015467](https://doi.org/10.5281/zenodo.23015467)

**Contents:** [Quick start](#quick-start) · [What is included](#what-is-included) ·
[Catchments](#catchments) · [Data](#data) · [Installation and configuration](#installation-and-configuration) ·
[Pipeline](#pipeline) · [Paper figures](#paper-figures) · [Compute notes](#compute-notes) ·
[Repository layout](#repository-layout) · [Citation and licence](#citation-and-licence)

---

## Quick start

All steps run from the repository folder on a Slurm cluster. Each `submit_*`
command sends jobs to the queue; wait until they have finished
(`squeue -u $USER` shows nothing left) before starting the next step.

```bash
# 1. Create the Python environment (once)
conda env create -f environment.yml

# 2. Put the input data in place (see "Data" below)
#      data/discharge_obs/   one discharge CSV per catchment
#      data/seas5/           SEAS5 NetCDF files
#    or point config.toml at wherever the data already lives.

# 3. Check config.toml: data locations, and the [slurm] env_setup lines
#    that activate the environment inside each job.

# 4. Run the four stages, one after the other
bash submit_00_preprocesscatchment.sh                  # build feature tables
bash submit_01_exploration.sh all                      # Stage 1: all models, untuned
bash submit_02_hyperparamtuning_nested.sh --model all all   # Stage 2: nested-CV tuning
bash submit_03_sensitivity.sh --model all all          # Stage 3: forcing sensitivity

# 5. Draw the paper figures
bash submit_paper_figures.sh
```

Job logs are written to `slurm_log/`. The partition and QOS names in the
`submit_*.sh` scripts (`standard`, `short`, `medium`) are those of the cluster
the study was run on; change them if yours uses different names.

---

## What is included

- The full pipeline: SEAS5 download and conversion, preprocessing, the three
  experimental stages, and all code-produced figures.
- The paper's figures (`model_results/00_plots/`, `model_results/00_appendix/`)
  and manuscript source (`LaTeX_report/main.tex`, `references.bib`, and the
  Elsevier class files `elsarticle.cls` and `elsarticle-harv.bst` it compiles
  with).

Not included: discharge observations, SEAS5 data, model results and trained
models. See [Data](#data) for how to obtain the inputs.

---

## Catchments

Nine catchments in three regimes. The main text of the paper shows one per
regime (Zeravshan, Ayat, Kirchbichl-Bichlwang); the appendices show all nine.

| id (`catchments.py`) | River | Station | Regime | Box lat (°N) | Box lon (°E) | Skipped hydro years | Folds |
|---|---|---|---|---|---|---|---|
| `zeravshan`  | Zeravshan   | Dupuli                 | Central Asian mountain (nival–glacial) | 38–41 | 67–71 | 2005–2010 | 33 |
| `ulkenboken` | Ulken Boken | Zhumba                 | Central Asian mountain (nival–glacial) | 48–50 | 82–84 | –         | 31 |
| `kurshim`    | Kurshim     | Voznesenskoje          | Central Asian mountain (nival–glacial) | 48–50 | 83–87 | 1997, 2002 | 28 |
| `kalkutan`   | Kalkutan    | Qalaotan               | Central Asian steppe (nival, rainfed)  | 51–53 | 69–72 | –         | 31 |
| `esil`       | Esil        | Turgenevka             | Central Asian steppe (nival, rainfed)  | 50–51 | 72–74 | –         | 31 |
| `ayat`       | Ayat        | Varvarinka             | Central Asian steppe (nival, rainfed)  | 52–54 | 59–63 | 2006, 2009 | 29 |
| `zhabay`     | Zhabay      | Atbasar                | Central Asian steppe (nival, rainfed)  | 51–53 | 67–70 | –         | 31 |
| `kirchbichl` | Inn         | Kirchbichl-Bichlwang   | Alpine (nival–glacial)                 | 46–48 | 9–13  | –         | 44 |
| `diepoldsau` | Rhine       | Diepoldsau-Rietbrücke  | Alpine (nival–glacial)                 | 46–48 | 8–11  | –         | 41 |

- **Box:** the SEAS5 bounding box whose 1° grid cells are averaged
  (cosine-latitude weighted) into each catchment's forcing.
- **Skipped hydro years:** years whose observations exist but are not
  trustworthy (e.g. a gauge under calibration). Years with missing data need
  no entry; they drop out automatically.
- **Folds:** hydrological years available for cross-validation (one fold
  each). A hydrological year starts on 1 October.

Catchment definitions, including the discharge file names, live in
`catchments.py`. A new catchment is added there.

---

## Data

### SEAS5 seasonal forecasts

Source: Copernicus Climate Data Store (CDS), dataset
`seasonal-original-single-levels`, ECMWF system 51, 1° × 1° grid.

- **Variables (7):** 2 m temperature (mean, maximum, minimum), surface solar
  radiation downwards, total precipitation, snow depth, snowfall.
- **Initialisations:** the 1st of every month, 1981–2025. Hindcasts
  (1981–2016) have 25 ensemble members, operational forecasts (2017 onward)
  51. All members are kept.
- **Lead time:** 6-hourly steps up to 5160 h, averaged to 215 daily steps.
- **Regions** (set in `config.toml`): `centralasia` (34–58°N, 46–88°E) for the
  seven Central Asian catchments and `europe` (35–59°N, 11°W–26°E) for the two
  Alpine ones.

**CDS access.** Create a CDS account, accept the dataset's licence on its CDS
page, and put your API key in `~/.cdsapirc` as described at
<https://cds.climate.copernicus.eu/how-to-api>. Never commit this file.

**Download and conversion**, once per region. Set `region` in `config.toml`
(`centralasia` or `europe`), then run the download on the login node (compute
nodes usually have no internet access). It takes long, so start it inside
`tmux` (or `screen`) to survive a dropped connection:

```bash
tmux new -s ecmwf
bash download_ecmwf/download_interactive.sh    # GRIB download
```

The download skips files already present, so it can simply be restarted if
it stops. When it has finished, convert the GRIB files to daily NetCDF with
one Slurm array job:

```bash
bash download_ecmwf/submit_02_grib2netcdf.sh
```

Files are written under `seas5_dir` (see `config.toml`):

```
data/seas5/
├── grib/<region>/                 {variable}_{YYYY}_{MM}.grib
├── netcdf_hindcast/<region>/      forecast_{YYYY}_{MM}_daily_member{NN}.nc   (1981–2016)
└── netcdf_forecast/<region>/      forecast_{YYYY}_{MM}_daily_member{NN}.nc   (2017–)
```

### Discharge observations

Daily discharge was obtained from the Agency for Hydrometeorology of the
Republic of Kazakhstan (Kazhydromet), the Agency for Hydrometeorology of the
Republic of Tajikistan (Tajik Hydromet) and the Global Runoff Data Centre
(GRDC). The Central Asian records were provided by the two agencies and are
not publicly available. The Alpine records are available from GRDC.

Each catchment needs one CSV file in `discharge_dir` (see `config.toml`),
named as in `catchments.py` (e.g. `Zeravshan_1936-2025_Q.csv`): a header row,
then exactly two columns, the date and the daily discharge in m³/s. For
example (illustrative values):

```
date,Q
1981-01-01,45.2
1981-01-02,44.8
```

---

## Installation and configuration

```bash
conda env create -f environment.yml     # creates the environment "FlowCast"
```

Python 3.11 or newer is required; `environment.yml` pins the exact versions
used for the paper. All models run on CPU.

**`config.toml`** is the only file to edit:

| Section | Key | Meaning |
|---|---|---|
| `[paths]` | `discharge_dir` | Folder with the discharge CSVs (default `data/discharge_obs`) |
| | `seas5_dir` | Folder with the SEAS5 data tree (default `data/seas5`) |
| `[seas5_download]` | `region` | Region to download/convert: `centralasia` or `europe` |
| | `first_year`, `last_year` | Initialisation years to download (1981–2025) |
| | `regions` | Bounding box of each region (N, W, S, E) |
| `[slurm]` | `env_setup` | Shell commands run at the start of every job |

Relative paths are relative to the repository folder; absolute paths also
work. Instead of editing the paths, the default folders can be symbolic links
to existing data, e.g. `ln -s /path/to/discharge data/discharge_obs`.

**`env_setup`** holds whatever makes `python` available in a fresh shell on
your cluster, one command per line, for example:

```toml
[slurm]
env_setup = """
module load miniforge
source activate FlowCast
"""
```

Every job script runs these lines through `load_env.sh` before starting
Python. Leave the block empty if nothing is needed.

---

## Pipeline

Run every command from the repository folder. The `submit_*` scripts are the
normal entry point; the `python` commands underneath show what each job runs,
for use without Slurm (activate the environment first).

### Stage 00 — Preprocessing

Crops each catchment's SEAS5 fields to its box and averages them, derives
daily totals for precipitation, radiation and snowfall, adds the seasonal
encodings (sin/cos of day of year), the antecedent discharge anomaly `a0` and
the observed discharge, and writes one feature table per catchment.

```bash
bash submit_00_preprocesscatchment.sh        # all nine catchments
python 00_preprocess_catchment.py zeravshan  # one catchment
```

Output in `data_processed/`: `<catchment>_features.parquet` (one row per
verification date × initialisation × ensemble member) and `<catchment>_folds.txt`
(the hydrological years used as cross-validation folds).

### Stage 01 — Model exploration (paper Stage 1)

Every model and baseline at default settings, under leave-one-hydrological-year-out
cross-validation: `climatology`, `persistence` (also writes
`damped_persistence`), `ridge`, `rf`, `xgb`, `lstm`. One job per catchment and
model.

```bash
bash submit_01_exploration.sh all                 # all catchments
bash submit_01_exploration.sh zeravshan ayat      # chosen catchments
python 01_exploration.py zeravshan xgb
```

Output in `model_results/01_exploration/results_<catchment>_<model>/`:
`fold_metrics.csv` (NSE, KGE, ACC, CRPS per fold), `summary.csv`,
`predictions_daily.parquet` (ensemble mean), `predictions_members.parquet`
(per member; learned models) and `best_model_fold<N>.pt` (LSTM).

### Stage 02 — Hyperparameter tuning (paper Stage 2)

Nested cross-validation for `rf`, `xgb` and `lstm` (Ridge tunes its own
regularisation within Stage 01). For each outer fold, 15 candidate
configurations are scored on 5 inner folds; the best is retrained on all
outer-training years and scored on the held-out year. One job per catchment,
model and outer year, read from `data_processed/<catchment>_folds.txt`.
**Requires Stage 01**, whose results serve as the untuned reference.

```bash
bash submit_02_hyperparamtuning_nested.sh --model all all     # everything
bash submit_02_hyperparamtuning_nested.sh --model rf zeravshan
OUTER_HY=2001 python 02_hyperparamtuning_nested.py zeravshan rf
```

Without options the script tunes `lstm` for `zeravshan ayat kirchbichl`.
Environment variables: `OUTER_HY` (held-out year, required for direct runs),
`SEED` (default 107), `N_INNER` (default 5).

Output in `model_results/02_hyperparamtuning/results_<catchment>_<model>_nested_seed<SEED>/outer_<year>/`:
`inner_cv_acc.csv`, `winner.csv`, `outer_test_metrics.csv` (tuned and untuned
scores), `predictions_daily.parquet`, `predictions_members.parquet`.

### Stage 03 — Sensitivity test (paper Stage 3)

Scores each learned model (`ridge`, `rf`, `xgb`, `lstm`) twice per fold on the
same held-out year: with its real SEAS5 forcing, and with the forcing of a
randomly chosen donor year (matched on initialisation month, lead day and
ensemble member). The seasonal encodings and `a0` are left untouched. `rf`,
`xgb` and `lstm` use one configuration per catchment: the one that won the
most Stage 02 outer folds. **Requires Stage 02** to have finished.

```bash
bash submit_03_sensitivity.sh --model all all
bash submit_03_sensitivity.sh --model xgb zeravshan
python 03_sensitivity_climate_input.py zeravshan xgb
```

Without options the script runs all four models for
`zeravshan ayat kirchbichl`.

Output in `model_results/03_sensitivity/results_<catchment>_<model>/`:
`sensitivity_fold_metrics.csv` and `sensitivity_summary.csv`. Each job also
fits the chosen configuration on all years and saves it as
`trained_models/<catchment>_<model>_final_seed<SEED>.joblib` (`.pt` for the
LSTM); `flowcast_src.sensitivity.load_fitted` reloads it.

---

## Paper figures

```bash
bash submit_paper_figures.sh
```

One Slurm job (a few minutes) that regenerates every code-produced figure of
the paper under the file names `LaTeX_report/main.tex` uses. It needs the
results of all three stages for all nine catchments, and the discharge CSVs.

| Plot script | Raw output (`model_results/00_plots/`) | Main text | Appendix (mountain, steppe, Alpine) |
|---|---|---|---|
| `01_plot_exploration.py` | `01a_boxplotcomparison_short` | Fig. 3 | A1–A3 |
| | `01b_boxplotcomparison_long` | Fig. 4 | A4–A6 |
| | `02_leadtimeheatmap` | Fig. 5 | A7–A9 |
| `02_plot_hyperparamtuning.py` | `03_configranking` | Fig. 6 | B1–B3 |
| | `04a_tuningvsreference_short` | Fig. 7 | B4–B6 |
| | `04b_tuningvsreference_long` | Fig. 8 | B7–B9 |
| `03_plot_sensitivity.py` | `07_sensitivity` | Fig. 9 | C1–C3 |

The results are written to `model_results/00_plots/fig03.pdf`–`fig09.pdf` and
`model_results/00_appendix/figA01.pdf`–`figC03.pdf`. Fig. 1 (catchment map),
Fig. 2 (workflow schematic) and the graphical abstract were made outside this
code.

The plot scripts can also be run on their own for any set of catchments, e.g.
`python 01_plot_exploration.py zeravshan kurshim`; they write the raw file
names above.

---

## Compute notes

- **Number of jobs** for all nine catchments: 9 (Stage 00), 54 (Stage 01),
  897 (Stage 02: one per model × catchment × outer year) and 36 (Stage 03).
- **Run times** on the study's cluster (16 cores): a Stage 02 outer fold
  takes about 50 min for XGBoost, 2.5 h for Random Forest and up to 16 h for
  the LSTM (Kirchbichl-Bichlwang, the catchment with most years). A Stage 03
  LSTM job can take about a day.
- **Reproducibility.** Seeds are fixed (`SEED=107`). Rerunning the baselines
  and tabular models reproduces the paper's fold scores.

---

## Repository layout

```
├── config.toml                      the only file to edit (data paths, download, Slurm setup)
├── environment.yml                  conda environment
├── catchments.py                    catchment definitions
├── load_env.sh                      runs config.toml's env_setup inside each job
├── 00_preprocess_catchment.py       Stage 00
├── 01_exploration.py                Stage 01
├── 02_hyperparamtuning_nested.py    Stage 02
├── 03_sensitivity_climate_input.py  Stage 03
├── 0{1,2,3}_plot_*.py               figures of each stage
├── submit_*.sh / run_*.sh           Slurm submission / job scripts per stage and for the figures
├── flowcast_src/                    shared library (data, features, CV, models, metrics, plotting)
├── download_ecmwf/                  SEAS5 download (CDS) and GRIB -> NetCDF conversion
├── model_results/
│   ├── 00_plots/                    paper figures (main text)
│   └── 00_appendix/                 paper figures (appendices)
├── LaTeX_report/                    manuscript source (main.tex, references.bib, Elsevier class files)
├── CITATION.cff
└── LICENSE
```

Not part of the repository: `data/` (the inputs, by default; see
[Data](#data)) and the folders the pipeline creates, `data_processed/`,
`model_results/01_exploration/`, `02_hyperparamtuning/`, `03_sensitivity/`,
`trained_models/` and `slurm_log/`.

---

## Citation and licence

If you use this code, please cite the article (reference to follow on
publication) and this software, <https://doi.org/10.5281/zenodo.23015467>. Citation metadata is in
`CITATION.cff`.

- **Code:** MIT licence, see `LICENSE`.
- **Figures and manuscript source** (`model_results/00_plots/`,
  `model_results/00_appendix/`, `LaTeX_report/main.tex`,
  `LaTeX_report/references.bib`): Creative Commons Attribution 4.0
  International (CC BY 4.0), <https://creativecommons.org/licenses/by/4.0/>.
- **Elsevier class files** (`LaTeX_report/elsarticle.cls`,
  `LaTeX_report/elsarticle-harv.bst`): © Elsevier Ltd, distributed under the
  LaTeX Project Public License 1.3 or later, <https://www.latex-project.org/lppl.txt>.
