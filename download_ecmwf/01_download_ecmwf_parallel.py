#!/usr/bin/env python3
"""
Optimized ECMWF seasonal forecast downloader.
- Parallel downloads via ThreadPoolExecutor
- Retry logic with exponential backoff
- Batched requests (one per var/year/month)
- Progress tracking
- Single script handles multiple variables
"""

import cdsapi
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from seas5_settings import REGION, AREA, YEARS, MONTHS, DAYS, LEADTIMES, SEAS5_DIR

# ─── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ecmwf_download.log"),
    ],
)
log = logging.getLogger(__name__)


# ─── SETTINGS ────────────────────────────────────────────────────────────────
DATASET = "seasonal-original-single-levels"

VARIABLES = [
    "snow_depth",
    "snowfall",
    "2m_temperature",
    "maximum_2m_temperature_in_the_last_24_hours",
    "minimum_2m_temperature_in_the_last_24_hours",
    "surface_solar_radiation_downwards",
    "total_precipitation"
]

GRIB_DIR = str(SEAS5_DIR / "grib" / REGION)

# ─── PARALLELISM & RETRY
MAX_WORKERS   = 5    # concurrent CDS API requests
MAX_RETRIES   = 3
RETRY_BACKOFF = 30   # seconds; doubles on each retry


# ─── SETUP ───────────────────────────────────────────────────────────────────
os.makedirs(GRIB_DIR, exist_ok=True)

# One client per thread (cdsapi.Client is not thread-safe)
_client_lock = Lock()
_thread_clients: dict = {}

def get_client() -> cdsapi.Client:
    """Return a per-thread cdsapi.Client instance."""
    import threading
    tid = threading.get_ident()
    if tid not in _thread_clients:
        with _client_lock:
            if tid not in _thread_clients:
                _thread_clients[tid] = cdsapi.Client(quiet=True)
    return _thread_clients[tid]


# ─── DOWNLOAD TASK ───────────────────────────────────────────────────────────
def download_month(var: str, year: str, month: str) -> tuple[str, bool]:
    """
    Download a single month for a given variable/year/month combination.
    Returns (filename, was_skipped).
    """
    filename = os.path.join(GRIB_DIR, f"{var}_{year}_{month}.grib")

    if os.path.exists(filename):
        log.info(f"  SKIP  {var} {year}-{month} (already exists)")
        return filename, True

    request = {
        "originating_centre": "ecmwf",
        "system": "51",
        "variable": [var],
        "year": [year],
        "month": [month],
        "day": DAYS,
        "leadtime_hour": LEADTIMES,
        "data_format": "grib",
        "area": AREA,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"  START {var} {year}-{month} (attempt {attempt}/{MAX_RETRIES})")
            client = get_client()
            client.retrieve(DATASET, request).download(filename)
            log.info(f"  DONE  {var} {year}-{month} → {filename}")
            return filename, False
        except Exception as e:
            log.warning(f"  FAIL  {var} {year}-{month} attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                log.info(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"  GIVE UP {var} {year}-{month} after {MAX_RETRIES} attempts")
                return filename, False

    return filename, False


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    tasks = [
        (var, year, month)
        for var in VARIABLES
        for year in YEARS
        for month in MONTHS
    ]

    total    = len(tasks)
    done     = 0
    skipped  = 0
    failed   = 0

    log.info(f"=== ECMWF DOWNLOAD: {total} tasks, {MAX_WORKERS} workers ===")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_month, var, year, month): (var, year, month)
            for var, year, month in tasks
        }

        for future in as_completed(futures):
            var, year, month = futures[future]
            try:
                _, was_skipped = future.result()
                done += 1
                if was_skipped:
                    skipped += 1
            except Exception as e:
                log.error(f"  Unhandled error for {var} {year}-{month}: {e}")
                failed += 1
                done += 1

            log.info(f"  Progress: {done}/{total}  (skipped={skipped}, failed={failed})")

    log.info("=== ALL PROCESSING COMPLETE ===")
    log.info(f"  Total: {total} | Done: {done-failed-skipped} | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    main()
