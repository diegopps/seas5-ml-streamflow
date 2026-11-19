"""
seas5ml.paths – Input and output locations for the whole pipeline.

Input locations (discharge CSVs, SEAS5 NetCDF) are read from the [paths]
section of config.toml at the repository root; relative entries there are
resolved against the repository root. Output locations are fixed, all under
data/ (gitignored; each subfolder may be a symbolic link to project storage).

Every output path is built here and nowhere else:

    data/processed_data/
        <catchment>_features.parquet        stage 0 feature table
        <catchment>_folds_<design>.txt      stage 0 fold manifest (fold ids)
    data/results/<design>/                  one tree per evaluation design
        default/<catchment>/<model>/        stage 1, default configs (all folds)
        tuned/<catchment>/<model>/fold_<id>/  stage 1, nested-CV tuning (one fold)
        sensitivity/<catchment>/<model>/    stage 2
        <kind>/_pooled/<model>/             pooled models: checkpoints, pool-level records
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


def figures_dir(design: str) -> Path:
    """Where the plot scripts of one design write their figures."""
    return design_dir(design) / "figures"


def trained_dir(design: str) -> Path:
    """Saved final models of one design."""
    return TRAINED_DIR / design
