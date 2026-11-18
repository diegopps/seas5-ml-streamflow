"""
utils.paths – Input and output locations for the whole pipeline.

Input locations (discharge CSVs, SEAS5 NetCDF) are read from the [paths]
section of config.toml at the repository root; relative entries there are
resolved against the repository root. Output locations are fixed, all under
data/ (gitignored; each subfolder may be a symbolic link to project storage):
data/processed_data/, data/trained_models/ and data/results/ (tables of each
stage, and the figures in data/results/figures/).
"""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # src/utils/paths.py -> repo root
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
PLOT_DIR      = RESULTS_ROOT / "figures"    # plot-script outputs; paper figures in main/, appendix/
