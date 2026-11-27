"""
seas5ml.paths – Input and output locations for the whole pipeline.

Input locations (discharge CSVs, SEAS5 NetCDF, catchment polygons) are read from the [paths]
section of config.toml at the repository root; relative entries there are
resolved against the repository root. Output locations are fixed, all under
data/ (gitignored; each subfolder may be a symbolic link to project storage).

Every output path is built here and nowhere else:

    data/processed_data/
        <catchment>_features.parquet        stage 0 feature table
        <catchment>_folds_<design>.txt      stage 0 fold manifest (fold ids)
        <catchment>_forcing_cells.csv       stage 0 grid cells and weights of the forcing mean
        <catchment>_era5.parquet            ERA5 daily catchment series on the SEAS5 cells (R2 comparison)
        era5_parts/<catchment>_<year>.parquet   its per-year pieces (pipeline/0_data/era5/extract_era5.py)
    data/results/<design>/                  one tree per evaluation design
        default/<catchment>/<model>/        stage 1, default configs (all folds)
        tuned/<catchment>/<model>/fold_<id>/  stage 1, nested-CV tuning (one fold)
        sensitivity/<catchment>/<model>/    stage 2
        <kind>/_pooled/<model>/             pooled models: checkpoints, pool-level records
        scores/<variant>/<catchment>/<model>/  scores at daily, weekly and
                                            monthly resolution (score.py; variant
                                            default, tuned or sensitivity)
        figures/                            plot-script outputs
    data/results/paper_figures/{main,appendix}/   stage 3
    data/trained_models/<design>/           stage 2 final all-years models
"""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # seas5ml/paths.py -> repo root
CONFIG_FILE = REPO_ROOT / "config.toml"

with open(CONFIG_FILE, "rb") as _fh:
    CONFIG = tomllib.load(_fh)

# ── Inputs (config.toml) ─────────────────────────────────────────────────
DISCHARGE_DIR = REPO_ROOT / CONFIG["paths"]["discharge_dir"]
SEAS5_DIR     = REPO_ROOT / CONFIG["paths"]["seas5_dir"]
POLYGON_DIR   = REPO_ROOT / CONFIG["paths"]["polygon_dir"]
ERA5_DIR      = REPO_ROOT / CONFIG["paths"]["era5_dir"]       # absolute in config.toml: kept as is
ERA5_SD_DIR   = REPO_ROOT / CONFIG["paths"]["era5_sd_dir"]

# ── Outputs (repo-relative) ──────────────────────────────────────────────
DATA_DIR      = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed_data"
TRAINED_DIR   = DATA_DIR / "trained_models"
RESULTS_ROOT  = DATA_DIR / "results"
PAPER_FIG_DIR = RESULTS_ROOT / "paper_figures"   # main/, appendix/

# Result kinds under data/results/<design>/.
RUN_KINDS = ("default", "tuned", "sensitivity")


def features_path(catchment: str) -> Path:
    """Stage 0 feature table of one catchment."""
    return PROCESSED_DIR / f"{catchment}_features.parquet"


def forcing_cells_path(catchment: str) -> Path:
    """Stage 0 record of the grid cells averaged into one catchment's forcing:
    one row per cell of the box (lat, lon, overlap with the catchment polygon
    in km², weight)."""
    return PROCESSED_DIR / f"{catchment}_forcing_cells.csv"


def era5_path(catchment: str) -> Path:
    """ERA5 daily series of one catchment, averaged over the same SEAS5 cells
    and weights as its forcing (pipeline/0_data/era5/extract_era5.py)."""
    return PROCESSED_DIR / f"{catchment}_era5.parquet"


def era5_part_path(catchment: str, year: int) -> Path:
    """One year of era5_path, written by one task of the extraction array."""
    return PROCESSED_DIR / "era5_parts" / f"{catchment}_{year}.parquet"


def fold_manifest_path(design: str, catchment: str) -> Path:
    """Stage 0 list of one catchment's fold ids under `design` (read by the
    tuned submit script, which starts one job per fold)."""
    return PROCESSED_DIR / f"{catchment}_folds_{design}.txt"


def design_dir(design: str) -> Path:
    """Root of every result of one evaluation design."""
    return RESULTS_ROOT / design


def kind_dir(design: str, kind: str) -> Path:
    """data/results/<design>/<kind>/, holding one folder per catchment."""
    if kind not in RUN_KINDS:
        raise ValueError(f"unknown result kind '{kind}' (one of {RUN_KINDS})")
    return design_dir(design) / kind


def run_dir(design: str, kind: str, catchment: str, model: str) -> Path:
    """Results of one (design, kind, catchment, model) run."""
    return kind_dir(design, kind) / catchment / model


def fold_dir(design: str, catchment: str, model: str, fold_id: str) -> Path:
    """Results of one tuned fold (inner CV, winner and its test score)."""
    return run_dir(design, "tuned", catchment, model) / f"fold_{fold_id}"


def pooled_work_dir(design: str, kind: str, model: str) -> Path:
    """
    Pool-level working folder of a pooled model (seas5ml.pooled):
    data/results/<design>/<kind>/_pooled/<model>/. Holds the per-fold
    checkpoints a restarted job resumes from (removed once the per-catchment
    results are assembled) and the tuned run's pool-level inner-CV record.
    The per-catchment results go to run_dir / fold_dir as for every model.
    """
    return kind_dir(design, kind) / "_pooled" / model


SCORE_VARIANTS = ("default", "tuned", "sensitivity")


def scores_dir(design: str, variant: str, catchment: str, model: str) -> Path:
    """
    Scores of one model at every resolution (pipeline/1_models/score.py):
    data/results/<design>/scores/<variant>/<catchment>/<model>/. `variant`
    names the line-up scored (default: every default run; tuned: the tuned
    models plus ridge and the baselines; sensitivity: the stage-2 scenarios
    of the learned models), not where each model's predictions came from.
    """
    if variant not in SCORE_VARIANTS:
        raise ValueError(f"unknown score variant '{variant}' "
                         f"(one of {SCORE_VARIANTS})")
    return design_dir(design) / "scores" / variant / catchment / model


def metskill_dir(design: str, catchment: str) -> Path:
    """SEAS5 meteorological skill against ERA5 of one catchment
    (pipeline/0_data/seas5_skill/score.py): data/results/<design>/seas5_skill/<catchment>/."""
    return design_dir(design) / "seas5_skill" / catchment


def figures_dir(design: str) -> Path:
    """Where the plot scripts of one design write their figures."""
    return design_dir(design) / "figures"


def trained_dir(design: str) -> Path:
    """Saved final models of one design."""
    return TRAINED_DIR / design
