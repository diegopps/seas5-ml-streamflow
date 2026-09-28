"""
flowcast_src.paths – Input and output locations for the whole pipeline.

Input locations (discharge CSVs, SEAS5 NetCDF) are read from the [paths]
section of config.toml at the repository root; relative entries there are
resolved against the repository root. Output locations are fixed and
repo-relative.
"""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "config.toml"

with open(CONFIG_FILE, "rb") as _fh:
    CONFIG = tomllib.load(_fh)

# ── Inputs (config.toml) ─────────────────────────────────────────────────
DISCHARGE_DIR = REPO_ROOT / CONFIG["paths"]["discharge_dir"]
SEAS5_DIR     = REPO_ROOT / CONFIG["paths"]["seas5_dir"]

# ── Outputs (repo-relative) ──────────────────────────────────────────────
PROCESSED_DIR = REPO_ROOT / "data_processed"
RESULTS_ROOT  = REPO_ROOT / "model_results"
PLOT_DIR      = RESULTS_ROOT / "00_plots"
TRAINED_DIR   = REPO_ROOT / "trained_models"
