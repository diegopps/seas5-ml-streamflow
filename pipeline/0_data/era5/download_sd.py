#!/usr/bin/env python3
"""
ERA5 snow depth from the CDS, for the SEAS5 skill evaluation (TODO.md, R2).

Snow depth (ERA5 `sd`, m of water equivalent, as SEAS5's) is the one
variable missing from the PIK ERA5 copy that extract_era5.py reads. This
downloads its 00 UTC value of every day, over one small area per region
(seas5ml.era5.SD_REGIONS: catchments with overlapping boxes share one), one
request per region and year, 1980-2025, plus 1 Jan 2026 (the end-of-day
value of 31 Dec 2025). Files already present are skipped, so re-running
resumes.

Output: <config.toml era5_sd_dir>/<region>/snow_depth_<region>_<year>.nc
Then:   python pipeline/0_data/era5/extract_era5.py --combine   (adds sd)

NOT a Slurm job: compute nodes have no network access. Run on the login
node, AFTER the SEAS5 download (same CDS account and queue):
    bash pipeline/0_data/era5/download_interactive.sh
"""
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, get_ident

import cdsapi

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root, for seas5ml
from seas5ml import era5                                         # noqa: E402
from seas5ml.catchments import CATCHMENTS                        # noqa: E402
from seas5ml.paths import ERA5_SD_DIR, REPO_ROOT                 # noqa: E402

DATASET = "reanalysis-era5-single-levels"
MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_BACKOFF = 30          # seconds, doubles on each retry

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "era5_sd_download.log")])
log = logging.getLogger(__name__)

_lock = Lock()
_clients: dict = {}


def client() -> cdsapi.Client:
    """One cdsapi.Client per thread (not thread-safe)."""
    tid = get_ident()
    with _lock:
        if tid not in _clients:
            _clients[tid] = cdsapi.Client(quiet=True)
    return _clients[tid]


def region_area(region: str) -> list:
    """[N, W, S, E] covering every member catchment's ERA5 window."""
    wins = [era5.era5_window(CATCHMENTS[c]) for c in era5.SD_REGIONS[region]]
    return [max(w[0][0] for w in wins), min(w[1][0] for w in wins),
            min(w[0][1] for w in wins), max(w[1][1] for w in wins)]


def tasks():
    days = [f"{d:02d}" for d in range(1, 32)]           # the CDS skips dates that don't exist
    months = [f"{m:02d}" for m in range(1, 13)]
    out = []
    for region in era5.SD_REGIONS:
        for y in range(era5.FIRST_YEAR, era5.LAST_YEAR + 1):
            out.append((region, y, months, days))
        out.append((region, era5.LAST_YEAR + 1, ["01"], ["01"]))
    return out


def download(region, year, months, days):
    target = ERA5_SD_DIR / region / f"snow_depth_{region}_{year}.nc"
    if target.exists():
        return target, "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": ["reanalysis"],
        "variable": ["snow_depth"],
        "year": [str(year)],
        "month": months,
        "day": days,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": region_area(region),
    }
    tmp = target.with_suffix(".nc.part")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"START {region} {year} (attempt {attempt}/{MAX_RETRIES})")
            client().retrieve(DATASET, request).download(str(tmp))
            tmp.rename(target)                            # only complete files get the final name
            log.info(f"DONE  {region} {year}")
            return target, "done"
        except Exception as e:                            # noqa: BLE001
            log.warning(f"FAIL  {region} {year} attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * 2 ** (attempt - 1))
    log.error(f"GIVE UP {region} {year}")
    return target, "failed"


def main():
    todo = tasks()
    log.info(f"=== ERA5 snow depth: {len(todo)} requests, {MAX_WORKERS} workers -> {ERA5_SD_DIR}")
    for region in era5.SD_REGIONS:
        log.info(f"    {region}: area [N, W, S, E] = {region_area(region)}")
    counts = {"done": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download, *t) for t in todo]
        for i, f in enumerate(as_completed(futures), 1):
            _, status = f.result()
            counts[status] += 1
            log.info(f"    progress {i}/{len(todo)} {counts}")
    log.info(f"=== FINISHED {counts}")
    if counts["failed"]:
        log.error("some requests failed: run the script again (finished files are skipped)")
        sys.exit(1)


if __name__ == "__main__":
    main()
