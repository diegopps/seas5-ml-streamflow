#!/usr/bin/env python3
"""
ECMWF seasonal forecast GRIB → NetCDF converter.

Input:  {grib_dir}/{var}_{YEAR}_{MONTH}.grib   (one file per variable per year,
                                         containing all 12 init months)
Output: {nc_dir}/forecast_{YEAR}_{MM}_daily_member{NN}.nc
        (one file per year × month × ensemble member, all variables merged)

Filter: only messages where dataDate month == validityDate month == target month
        (i.e. data initialised in month M that is also valid in month M).

Day alignment (checked, see check_daily_alignment / check_merged):
    On the CDS only 2m_temperature is 6-hourly (steps 6 ... 5160 h); every
    other variable (tp, sf, ssrd, mx2t24, mn2t24, sd) comes at 24-h steps
    only (24 ... 5160 h, i.e. 00 UTC of the next day). `cdo daymean` gives
    2m_temperature 216 days (start date ... start + 215 d, the last holding
    only the 00 UTC step) and the 24-h variables 215 days (start + 1 d ...
    start + 215 d). `cdo merge` takes the time axis from the FIRST input
    (2m_temperature, first in VARIABLES), keeps 215 steps and pairs the
    other variables by step index, so the day labelled D carries their value
    at 00 UTC of D+1: running totals through the end of UTC day D (deaccumulated
    in preprocess.py), the extremes over UTC day D, the snow depth at the end
    of day D. This is correct, but implicit: reordering VARIABLES or a change
    in what the CDS delivers would shift the 24-h variables by a day without
    any error, so every conversion checks these facts and stops if one fails.
"""
import argparse
import os
import re
import subprocess
import logging
import eccodes
import sys
import netCDF4 as nc4

from datetime import date, timedelta

from seas5_settings import REGION, YEARS, MONTHS, LEADTIMES, n_members, nc_subdir, SEAS5_DIR

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─── SETTINGS ────────────────────────────────────────────────────────────────
SAVE_INDIVIDUAL_MEMBERS = True   # False → ensemble mean instead
# Ensemble size is per-year (25 hindcast / 51 forecast) — see seas5_settings.n_members().

VARIABLES = [
    "2m_temperature",
    "maximum_2m_temperature_in_the_last_24_hours",
    "minimum_2m_temperature_in_the_last_24_hours",
    "surface_solar_radiation_downwards",
    "total_precipitation",
    "snow_depth",
    "snowfall",
]

# 2m_temperature must stay first: cdo merge takes the time axis from the first
# input (see "Day alignment" in the module docstring).
TIME_AXIS_VAR = "2m_temperature"
if VARIABLES[0] != TIME_AXIS_VAR:
    raise RuntimeError(f"VARIABLES must start with {TIME_AXIS_VAR!r} (cdo merge takes "
                       f"its time axis from the first input); got {VARIABLES[0]!r}")

# Days in a merged file: the last lead (5160 h) is 215 days after the start.
N_DAYS = LEADTIMES[-1] // 24

VARIABLE_MAPPING = {
    "2m_temperature":                              {"short_name": "tas",    "long_name": "Mean 2m Temperature",              "units": "K",                     "grib_code": 167},
    "maximum_2m_temperature_in_the_last_24_hours": {"short_name": "tasmax", "long_name": "Maximum 2m Temperature in 24h",    "units": "K",                     "grib_code": 51},
    "minimum_2m_temperature_in_the_last_24_hours": {"short_name": "tasmin", "long_name": "Minimum 2m Temperature in 24h",    "units": "K",                     "grib_code": 52},
    "surface_solar_radiation_downwards":           {"short_name": "ssrd",   "long_name": "Surface Solar Radiation Downwards","units": "J m**-2",               "grib_code": 169},
    "total_precipitation":                         {"short_name": "tp",     "long_name": "Total Precipitation",              "units": "m",                     "grib_code": 228},
    "snow_depth":                                  {"short_name": "sd",     "long_name": "Snow Depth",                       "units": "m of water equivalent",  "grib_code": 141},
    "snowfall":                                    {"short_name": "sf",     "long_name": "Snowfall",                         "units": "m of water equivalent",  "grib_code": 144},
}

ECMWF_DIR = str(SEAS5_DIR)
GRIB_DIR  = f"{ECMWF_DIR}/grib/{REGION}"
TEMP_DIR  = f"{ECMWF_DIR}/temp_netcdf"   # next to the SEAS5 data, not in the repository


def nc_dir_for(year) -> str:
    """Output dir for `year` — netcdf_hindcast/ or netcdf_forecast/ per seas5_settings."""
    return os.path.join(ECMWF_DIR, nc_subdir(year), REGION)

# Matches ONLY {varname}_{YYYY}_{MM}.grib — ignores any other files in the folder 
_GRIB_PATTERN = re.compile(r"^(.+)_(\d{4})_(\d{2})\.grib$")


# ─── SETUP ───────────────────────────────────────────────────────────────────
# NC output dirs are created per-year in process_year_month (hindcast/forecast).
os.makedirs(TEMP_DIR, exist_ok=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _month_of(yyyymmdd: int) -> int:
    """Extract month (1–12) from an integer date like 19950301."""
    return (yyyymmdd // 100) % 100


def extract_month_grib(src_grib: str, dst_grib: str, target_month: int) -> int:
    """
    Copy from src_grib → dst_grib only messages where:
        month(dataDate) == month(validityDate) == target_month

    Returns the number of messages written.
    """
    written = 0
    with open(src_grib, "rb") as fin, open(dst_grib, "wb") as fout:
        while True:
            msg = eccodes.codes_grib_new_from_file(fin)
            if msg is None:
                break
            try:
                data_month     = _month_of(eccodes.codes_get(msg, "dataDate"))
                validity_month = _month_of(eccodes.codes_get(msg, "validityDate"))
                if data_month == target_month == validity_month:
                    eccodes.codes_write(msg, fout)
                    written += 1
            finally:
                eccodes.codes_release(msg)
    return written


def split_member_grib(src_grib: str, dst_grib: str, member_idx: int) -> None:
    """Extract a single perturbationNumber from a GRIB file via grib_copy."""
    subprocess.run(
        ["grib_copy", "-w", f"perturbationNumber={member_idx}", src_grib, dst_grib],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def grib_to_netcdf_daily(src_grib: str, dst_nc: str) -> None:
    """Convert a single-member GRIB to daily-mean NetCDF via CDO."""
    subprocess.run(
        ["cdo", "-f", "nc", "-daymean", src_grib, dst_nc],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def grib_to_netcdf_ensmean(src_grib: str, dst_nc: str) -> None:
    """Convert a multi-member GRIB to daily-mean ensemble-mean NetCDF via CDO."""
    subprocess.run(
        ["cdo", "-f", "nc", "-daymean", "-ensmean", src_grib, dst_nc],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def merge_variables(var_nc_files: list, dst_nc: str) -> bool:
    """Merge multiple single-variable NetCDF files into one with CDO."""
    try:
        subprocess.run(
            ["cdo", "merge"] + var_nc_files + [dst_nc],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"    merge failed: {e.stderr.decode().strip()}")
        return False


def rename_variables(nc_file: str) -> None:
    """Rename variables in a NetCDF file from CDO defaults to CF short names."""
    grib_code_map = {info["grib_code"]: info for info in VARIABLE_MAPPING.values()}
    with nc4.Dataset(nc_file, "r+") as ds:
        data_vars = [v for v in ds.variables if v not in ("time", "lat", "lon", "latitude", "longitude")]
        for old_name in data_vars:
            code = getattr(ds[old_name], "code", None)
            if code in grib_code_map:
                info = grib_code_map[code]
                ds.renameVariable(old_name, info["short_name"])
                ds[info["short_name"]].long_name = info["long_name"]
                ds[info["short_name"]].units      = info["units"]


class AlignmentError(RuntimeError):
    """The daily files do not line up as the merge assumes ("Day alignment"
    in the module docstring). Stops the task: nothing is written for it."""


def _nc_dates(nc_file: str) -> list:
    """Calendar dates of a NetCDF file's time steps."""
    with nc4.Dataset(nc_file) as ds:
        t = ds["time"]
        times = nc4.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
    return [date(x.year, x.month, x.day) for x in times]


def check_daily_alignment(var_files: list, init: date) -> None:
    """
    Before the merge: `var_files` = [(variable, daily NetCDF), ...] in merge
    order. The first must be 2m_temperature, starting on the start date with
    N_DAYS + 1 days; every other variable must start one day later with
    N_DAYS days and end on the same date. Raises AlignmentError otherwise.
    """
    first_var, first_file = var_files[0]
    if first_var != TIME_AXIS_VAR:
        raise AlignmentError(f"merge order starts with {first_var!r}, not {TIME_AXIS_VAR!r}")
    t_dates = _nc_dates(first_file)
    if t_dates[0] != init or len(t_dates) != N_DAYS + 1:
        raise AlignmentError(
            f"{first_var}: {len(t_dates)} days from {t_dates[0]}, expected "
            f"{N_DAYS + 1} days from {init} (6-hourly steps 6 ... {LEADTIMES[-1]} h)")
    for var, path in var_files[1:]:
        d = _nc_dates(path)
        if d[0] != init + timedelta(days=1) or len(d) != N_DAYS or d[-1] != t_dates[-1]:
            raise AlignmentError(
                f"{var}: {len(d)} days {d[0]} ... {d[-1]}, expected {N_DAYS} days "
                f"{init + timedelta(days=1)} ... {t_dates[-1]} (24-h steps only); "
                f"the merge would pair it with the wrong days")


def check_merged(nc_file: str, init: date) -> None:
    """After the merge: N_DAYS days from the start date and every variable."""
    d = _nc_dates(nc_file)
    if d[0] != init or len(d) != N_DAYS:
        raise AlignmentError(f"{os.path.basename(nc_file)}: {len(d)} days from {d[0]}, "
                             f"expected {N_DAYS} from {init}")
    with nc4.Dataset(nc_file) as ds:
        missing = [VARIABLE_MAPPING[v]["short_name"] for v in VARIABLES
                   if VARIABLE_MAPPING[v]["short_name"] not in ds.variables]
    if missing:
        raise AlignmentError(f"{os.path.basename(nc_file)}: variables missing after merge: {missing}")


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ─── CORE PROCESSING ─────────────────────────────────────────────────────────
def process_year_month(year: str, month: str) -> None:
    # Hindcast (25 members) vs operational forecast (51) — resolved per year.
    n_mem  = n_members(year)
    nc_dir = nc_dir_for(year)
    init   = date(int(year), int(month), 1)
    log.info(f"\n> {year}-{month}  [{nc_subdir(year)}, {n_mem} members]")

    # ── Skip if all outputs already exist ─────────────────────────────────
    if SAVE_INDIVIDUAL_MEMBERS:
        all_exist = all(
            os.path.exists(os.path.join(nc_dir, f"forecast_{year}_{month}_daily_member{m:02d}.nc"))
            for m in range(n_mem)
        )
    else:
        all_exist = os.path.exists(os.path.join(nc_dir, f"forecast_{year}_{month}_daily_ensmean.nc"))

    if all_exist:
        log.info("  All output files already exist — skipping.")
        return

    # ── Locate input GRIB files (strict pattern match) ────────────────────
    existing_grib: dict[str, str] = {}
    missing: list[str] = []

    for var in VARIABLES:
        candidate = os.path.join(GRIB_DIR, f"{var}_{year}_{month}.grib")
        if _GRIB_PATTERN.match(os.path.basename(candidate)) and os.path.exists(candidate):
            existing_grib[var] = candidate
        else:
            missing.append(var)

    if missing:
        log.warning(f"  Missing GRIB files for: {', '.join(missing)} — skipping {year}-{month}")
        return

    # ── Step 1: Now just pass full grib data directly! [Extract target-month messages from each yearly GRIB] ───────
    month_gribs: dict[str, str] = {}
    month_gribs = existing_grib

    # ── Step 2a: INDIVIDUAL MEMBERS ───────────────────────────────────────
    if SAVE_INDIVIDUAL_MEMBERS:
        os.makedirs(nc_dir, exist_ok=True)
        # per_member_nc[member_idx] = [(var1, var1.nc), (var2, var2.nc), ...], in VARIABLES order
        per_member_nc: dict[int, list[tuple[str, str]]] = {m: [] for m in range(n_mem)}

        for var, month_grib in month_gribs.items():
            for member_idx in range(n_mem):
                member_str  = f"{member_idx:02d}"
                member_grib = os.path.join(TEMP_DIR, f"{var}_{year}_{month}_member{member_str}.grib")
                member_nc   = os.path.join(TEMP_DIR, f"{var}_{year}_{month}_member{member_str}.nc")
                try:
                    split_member_grib(month_grib, member_grib, member_idx)
                    grib_to_netcdf_daily(member_grib, member_nc)
                    per_member_nc[member_idx].append((var, member_nc))
                except subprocess.CalledProcessError as e:
                    log.error(f"    member {member_str} / {var}: {e.stderr.decode().strip()}")
                finally:
                    _remove(member_grib)

        log.info("  Merging variables per member...")
        for member_idx in range(n_mem):
            member_str = f"{member_idx:02d}"
            pairs      = per_member_nc[member_idx]
            var_ncs    = [f for _, f in pairs]
            if len(var_ncs) != len(VARIABLES):
                log.warning(f"  member {member_str}: {len(var_ncs)}/{len(VARIABLES)} variables — skipping")
                for f in var_ncs:
                    _remove(f)
                continue
            out_nc = os.path.join(nc_dir, f"forecast_{year}_{month}_daily_member{member_str}.nc")
            try:
                check_daily_alignment(pairs, init)
                if merge_variables(var_ncs, out_nc):
                    rename_variables(out_nc)
                    check_merged(out_nc, init)
                    log.info(f"    ✓ {os.path.basename(out_nc)}")
            except AlignmentError as e:
                _remove(out_nc)
                for _, other in per_member_nc.items():
                    for _, f in other:
                        _remove(f)
                log.error(f"  {year}-{month} member {member_str}: {e}")
                raise
            finally:
                for f in var_ncs:
                    _remove(f)

    # ── Step 2b: ENSEMBLE MEAN ────────────────────────────────────────────
    else:
        os.makedirs(nc_dir, exist_ok=True)
        pairs: list[tuple[str, str]] = []
        for var, month_grib in month_gribs.items():
            var_nc = os.path.join(TEMP_DIR, f"{var}_{year}_{month}_ensmean.nc")
            try:
                grib_to_netcdf_ensmean(month_grib, var_nc)
                pairs.append((var, var_nc))
            except subprocess.CalledProcessError as e:
                log.error(f"    ensmean / {var}: {e.stderr.decode().strip()}")
        var_ncs = [f for _, f in pairs]

        try:
            if len(var_ncs) == len(VARIABLES):
                out_nc = os.path.join(nc_dir, f"forecast_{year}_{month}_daily_ensmean.nc")
                try:
                    check_daily_alignment(pairs, init)
                    if merge_variables(var_ncs, out_nc):
                        rename_variables(out_nc)
                        check_merged(out_nc, init)
                        log.info(f"  ✓ {os.path.basename(out_nc)}")
                except AlignmentError as e:
                    _remove(out_nc)
                    log.error(f"  {year}-{month} ensmean: {e}")
                    raise
        finally:
            for f in var_ncs:
                _remove(f)



# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--n-tasks", action="store_true",
                        help="Print the number of year-month tasks and exit "
                             "(used by submit_sbatch.sh to size the array).")
    args = parser.parse_args()

    combos = [(y, m) for y in YEARS for m in MONTHS]  # len(YEARS) x len(MONTHS), see seas5_settings.py

    if args.n_tasks:
        print(len(combos))
        return

    try:
        if args.task_id is not None:
            # Single task: process just one year-month
            if not 0 <= args.task_id < len(combos):
                log.error(f"--task-id {args.task_id} out of range "
                          f"(0-{len(combos) - 1} for {YEARS[0]}-{YEARS[-1]})")
                sys.exit(1)
            year, month = combos[args.task_id]
            process_year_month(year, month)
        else:
            # Fallback: sequential (original behavior)
            for year, month in combos:
                process_year_month(year, month)
    finally:
        # TEMP_DIR is shared across concurrent array tasks, so only remove it
        # once empty rather than rmtree'ing files another task may still be using.
        try:
            os.rmdir(TEMP_DIR)
        except OSError:
            pass


if __name__ == "__main__":
    main()
