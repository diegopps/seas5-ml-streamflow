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

All commands run from the repository root on a Slurm cluster. Each `submit_*`
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
bash src/00_preprocessing/submit_00_preprocesscatchment.sh                # build feature tables
bash src/01_exploration/submit_01_exploration.sh all                      # Stage 1: all models, untuned
bash src/02_hyperparamtuning/submit_02_hyperparamtuning_nested.sh --model all all   # Stage 2: nested-CV tuning
bash src/03_sensitivity/submit_03_sensitivity.sh --model all all          # Stage 3: forcing sensitivity

# 5. Draw the paper figures
bash src/submit_paper_figures.sh
```

Job logs are written to `logs/`. The partition and QOS names in the
`submit_*.sh` scripts (`standard`, `short`, `medium`) are those of the cluster
the study was run on; change them if yours uses different names.

---

## What is included

The full pipeline: SEAS5 download and conversion, preprocessing, the three
experimental stages, and all code-produced figures.

Not included: discharge observations, SEAS5 data, model results, trained
models and the figures themselves. See [Data](#data) for how to obtain the inputs.

---

## Catchments

Nine catchments in three regimes. The main text of the paper shows one per
regime (Zeravshan, Ayat, Kirchbichl-Bichlwang); the appendices show all nine.

| id (`src/utils/catchments.py`) | River | Station | Regime | Box lat (°N) | Box lon (°E) | Skipped hydro years | Folds |
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
`src/utils/catchments.py`. A new catchment is added there.

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
bash src/download_ecmwf/download_interactive.sh    # GRIB download
```

The download skips files already present, so it can simply be restarted if
it stops; its progress is logged to `logs/ecmwf_download.log`. When it has finished, convert the GRIB files to daily NetCDF with
one Slurm array job:

```bash
bash src/download_ecmwf/submit_02_grib2netcdf.sh
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
named as in `src/utils/catchments.py` (e.g. `Zeravshan_1936-2025_Q.csv`): a header row,
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

Every job script runs these lines through `src/load_env.sh` before starting
Python. Leave the block empty if nothing is needed.

---

## Pipeline

Run every command from the repository root. The `submit_*` scripts are the
normal entry point (they can be called from any folder); the `python` commands underneath show what each job runs,
for use without Slurm (activate the environment first).

### Stage 00 — Preprocessing

Crops each catchment's SEAS5 fields to its box and averages them, derives
daily totals for precipitation, radiation and snowfall, adds the seasonal
encodings (sin/cos of day of year), the antecedent discharge anomaly `a0` and
the observed discharge, and writes one feature table per catchment.

```bash
bash src/00_preprocessing/submit_00_preprocesscatchment.sh        # all nine catchments
python src/00_preprocessing/00_preprocess_catchment.py zeravshan  # one catchment
```

Output in `data/processed_data/`: `<catchment>_features.parquet` (one row per
verification date × initialisation × ensemble member) and `<catchment>_folds.txt`
(the hydrological years used as cross-validation folds).

### Stage 01 — Model exploration (paper Stage 1)

Every model and baseline at default settings, under leave-one-hydrological-year-out
cross-validation: `climatology`, `persistence` (also writes
`damped_persistence`), `ridge`, `rf`, `xgb`, `lstm`. One job per catchment and
model.

```bash
bash src/01_exploration/submit_01_exploration.sh all                 # all catchments
bash src/01_exploration/submit_01_exploration.sh zeravshan ayat      # chosen catchments
python src/01_exploration/01_exploration.py zeravshan xgb
```

Output in `data/results/01_exploration/results_<catchment>_<model>/`:
`fold_metrics.csv` (NSE, KGE, ACC, CRPS per fold), `summary.csv`,
`predictions_daily.parquet` (ensemble mean), `predictions_members.parquet`
(per member; learned models) and `best_model_fold<N>.pt` (LSTM).

### Stage 02 — Hyperparameter tuning (paper Stage 2)

Nested cross-validation for `rf`, `xgb` and `lstm` (Ridge tunes its own
regularisation within Stage 01). For each outer fold, 15 candidate
configurations are scored on 5 inner folds; the best is retrained on all
outer-training years and scored on the held-out year. One job per catchment,
model and outer year, read from `data/processed_data/<catchment>_folds.txt`.
**Requires Stage 01**, whose results serve as the untuned reference.

```bash
bash src/02_hyperparamtuning/submit_02_hyperparamtuning_nested.sh --model all all     # everything
bash src/02_hyperparamtuning/submit_02_hyperparamtuning_nested.sh --model rf zeravshan
OUTER_HY=2001 python src/02_hyperparamtuning/02_hyperparamtuning_nested.py zeravshan rf
```

Without options the script tunes `lstm` for `zeravshan ayat kirchbichl`.
Environment variables: `OUTER_HY` (held-out year, required for direct runs),
`SEED` (default 107), `N_INNER` (default 5).

Output in `data/results/02_hyperparamtuning/results_<catchment>_<model>_nested_seed<SEED>/outer_<year>/`:
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
bash src/03_sensitivity/submit_03_sensitivity.sh --model all all
bash src/03_sensitivity/submit_03_sensitivity.sh --model xgb zeravshan
python src/03_sensitivity/03_sensitivity_climate_input.py zeravshan xgb
```

Without options the script runs all four models for
`zeravshan ayat kirchbichl`.

Output in `data/results/03_sensitivity/results_<catchment>_<model>/`:
`sensitivity_fold_metrics.csv` and `sensitivity_summary.csv`. Each job also
fits the chosen configuration on all years and saves it as
`data/trained_models/<catchment>_<model>_final_seed<SEED>.joblib` (`.pt` for the
LSTM); `utils.sensitivity.load_fitted` (`src/utils/sensitivity.py`) reloads it.

---

## Paper figures

```bash
bash src/submit_paper_figures.sh
```

One Slurm job (a few minutes) that regenerates every code-produced figure of
the paper under the file names the manuscript uses. It needs the
results of all three stages for all nine catchments, and the discharge CSVs.

| Plot script (`src/<stage>/`) | Raw output (`data/results/figures/`) | Main text | Appendix (mountain, steppe, Alpine) |
|---|---|---|---|
| `01_plot_exploration.py` | `01a_boxplotcomparison_short` | Fig. 3 | A1–A3 |
| | `01b_boxplotcomparison_long` | Fig. 4 | A4–A6 |
| | `02_leadtimeheatmap` | Fig. 5 | A7–A9 |
| `02_plot_hyperparamtuning.py` | `03_configranking` | Fig. 6 | B1–B3 |
| | `04a_tuningvsreference_short` | Fig. 7 | B4–B6 |
| | `04b_tuningvsreference_long` | Fig. 8 | B7–B9 |
| `03_plot_sensitivity.py` | `07_sensitivity` | Fig. 9 | C1–C3 |

The results are written to `data/results/figures/main/fig03.pdf`–`fig09.pdf` and
`data/results/figures/appendix/figA01.pdf`–`figC03.pdf`. Fig. 1 (catchment map),
Fig. 2 (workflow schematic) and the graphical abstract were made outside this
code.

The plot scripts can also be run on their own for any set of catchments, e.g.
`python src/01_exploration/01_plot_exploration.py zeravshan kurshim`; they
write the raw file names above to `data/results/figures/`.

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
.
├── README.md
├── LICENSE                          MIT License
├── CITATION.cff                     citation metadata of this code
├── config.toml                      the only file to edit (data paths, download, Slurm setup)
├── environment.yml                  conda environment, with the versions used for the paper
├── src/
│   ├── download_ecmwf/              SEAS5 download (CDS) and GRIB -> NetCDF conversion
│   ├── 00_preprocessing/            Stage 00: feature tables
│   ├── 01_exploration/              Stage 01: untuned models and baselines, and their figures
│   ├── 02_hyperparamtuning/         Stage 02: nested-CV tuning, and its figures
│   ├── 03_sensitivity/              Stage 03: forcing sensitivity, and its figures
│   ├── utils/                       shared library (catchments, data, features, CV, models, metrics, plotting)
│   ├── load_env.sh                  runs config.toml's env_setup inside each job
│   └── submit_paper_figures.sh / run_paper_figures.sh
│
│   not in the repository:
├── data/                            inputs and every output; each subfolder may be a symbolic link
│   ├── discharge_obs/  seas5/       inputs (default locations, see Data)
│   ├── processed_data/              Stage 00 feature tables and fold lists
│   ├── trained_models/              final models of Stage 03
│   └── results/
│       ├── 01_exploration/ ...      tables of each stage
│       └── figures/                 plot outputs; paper figures in main/ and appendix/
└── logs/                            Slurm and download logs
```

Each stage folder holds the stage's Python script(s), its `submit_*.sh`
(submits the jobs) and `run_*.sh` (the body of one job). The scripts put
`src/` on the Python path, so they run from any folder.

---

## Citation and licence

If you use this code, please cite the article (reference to follow on
publication) and this software, <https://doi.org/10.5281/zenodo.23015467>. Citation metadata is in
`CITATION.cff`.

- **Code:** MIT licence, see `LICENSE`.
