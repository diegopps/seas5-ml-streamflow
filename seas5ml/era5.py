"""
seas5ml.era5 – ERA5 daily catchment series matched to the SEAS5 forcing.

The reference for the SEAS5 meteorological skill evaluation (TODO.md, R2).
For each catchment, ERA5 is averaged over exactly the area its SEAS5 forcing
averages, and reduced to days exactly as stage 0 reduces SEAS5, so the two
series differ only by what the forecast got wrong.

Area. Stage 0 weights each 1 degree SEAS5 cell of the catchment's box by its
overlap with the catchment polygon (seas5ml.spatial.cell_weights). Here each
ERA5 cell (0.25 degree, centre +- 0.125) gets its area share of every SEAS5
cell it lies in, times that SEAS5 cell's weight (era5_weights): the ERA5 mean
over a SEAS5 cell, weighted as stage 0 weights the cell. ERA5 covers land and
water, so no cell is missing.

Days (UTC), as in the stage-0 SEAS5 table (grib2netcdf.py, "Day alignment"):
    tas            mean of the 00, 06, 12, 18 UTC values of day D
    tasmax, tasmin max / min of the hourly t2m over the 24 h ending 00 UTC of
                   D+1 (hours 01 ... 24), per cell, then averaged (SEAS5:
                   the box mean of each cell's mx2t24 / mn2t24; SEAS5 takes
                   the extreme over model time steps, hourly values here)
    tp, sf, ssrd   sum of the hourly accumulations over the same 24 h (ERA5
                   hourly accumulations cover the hour ending at the
                   validity time)
    sd             value at 00 UTC of D+1 (snow at the end of day D)
Units as SEAS5: K, m, m of water equivalent, J m-2.
"""

import numpy as np
import pandas as pd
from shapely.geometry import box

from .spatial import area_km2, cell_overlaps, cell_weights, load_polygon

SEAS5_CELL = 1.0
ERA5_CELL = 0.25
FIRST_YEAR, LAST_YEAR = 1980, 2025      # verification period Oct 1980 - Dec 2025

# output variable -> (ERA5 folder and file stem, NetCDF variable, daily rule)
VARIABLES = {
    "tas":    ("2m_temperature", "t2m", "synoptic_mean"),
    "tasmax": ("2m_temperature", "t2m", "max24"),
    "tasmin": ("2m_temperature", "t2m", "min24"),
    "tp":     ("total_precipitation", "tp", "sum24"),
    "sf":     ("snowfall", "sf", "sum24"),
    "ssrd":   ("surface_solar_radiation_downwards", "ssrd", "sum24"),
}
SD_VARIABLE = "sd"                     # snow depth (m w.e.), CDS download (00 UTC daily): day D = 00 UTC of D+1
OUTPUT_COLUMNS = ["tas", "tasmax", "tasmin", "ssrd", "tp", "sd", "sf"]   # as SHARED['met_variables']

# Regions for the snow-depth download: catchments whose boxes overlap share
# one CDS request.
SD_REGIONS = {
    "alps": ["kirchbichl", "diepoldsau"],
    "altai": ["ulkenboken", "kurshim"],
    "akmola": ["kalkutan", "zhabay"],
    "esil": ["esil"],
    "ayat": ["ayat"],
    "zeravshan": ["zeravshan"],
}


def seas5_cells(cfg):
    """Centres of the SEAS5 1 degree cells in a catchment's box (lat
    descending, as in the SEAS5 files). The box bounds are whole degrees and
    the SEAS5 grid is centred on half degrees, so the cells tile the box."""
    (la, lb), (lo, lr) = cfg["catch_lat"], cfg["catch_lon"]
    if any(float(v) != int(v) for v in (la, lb, lo, lr)):
        raise ValueError(f"box bounds must be whole degrees: {cfg['catch_lat']}, {cfg['catch_lon']}")
    return np.arange(lb - 0.5, la, -SEAS5_CELL), np.arange(lo + 0.5, lr, SEAS5_CELL)


def seas5_weights(cfg):
    """(lat, lon, weights) of the SEAS5 cells, exactly as stage 0 computes them."""
    lat, lon = seas5_cells(cfg)
    w, _, _ = cell_weights(lat, lon, load_polygon(cfg["polygon"]), SEAS5_CELL)
    return lat, lon, w


def era5_window(cfg):
    """(lat_max, lat_min), (lon_min, lon_max) of the ERA5 points to read: the
    box plus one ERA5 cell on every side."""
    (la, lb), (lo, lr) = cfg["catch_lat"], cfg["catch_lon"]
    return (lb + ERA5_CELL, la - ERA5_CELL), (lo - ERA5_CELL, lr + ERA5_CELL)


def era5_weights(cfg, era_lat, era_lon):
    """
    Weights (len(era_lat) x len(era_lon), sum 1) of the ERA5 cells that
    reproduce the SEAS5 forcing mean: each SEAS5 cell's weight spread over
    the ERA5 cells in it by area. Raises if a weighted SEAS5 cell is not
    fully covered by the ERA5 points given.
    """
    s_lat, s_lon, s_w = seas5_weights(cfg)
    W = np.zeros((len(era_lat), len(era_lon)))
    half = SEAS5_CELL / 2
    for i, y in enumerate(s_lat):
        for j, x in enumerate(s_lon):
            if s_w[i, j] == 0:
                continue
            cell = box(x - half, y - half, x + half, y + half)
            ov = cell_overlaps(era_lat, era_lon, cell, ERA5_CELL)
            if abs(ov.sum() / area_km2(cell) - 1) > 1e-6:
                raise ValueError(f"SEAS5 cell ({y}, {x}) only {ov.sum() / area_km2(cell):.4%} "
                                 f"covered by the ERA5 points read")
            W += s_w[i, j] * ov / ov.sum()
    return W / W.sum()


def daily_fields(hourly: np.ndarray, times, rule: str):
    """
    Daily fields from an hourly series running from 00 UTC of the first day
    to 00 UTC after the last day (n_days * 24 + 1 steps, hourly, no gaps).
    hourly: (time, lat, lon). Returns (dates, (n_days, lat, lon)).
    """
    times = pd.DatetimeIndex(times)
    if times[0].hour != 0 or times[-1].hour != 0 or (len(times) - 1) % 24:
        raise ValueError(f"hourly series must run 00 UTC to 00 UTC: {times[0]} .. {times[-1]}")
    if not (np.diff(times.values) == np.timedelta64(1, "h")).all():
        raise ValueError("hourly series has gaps or repeated hours")
    n = (len(times) - 1) // 24
    dates = times[:-1:24].normalize()
    rest = hourly.shape[1:]
    if rule == "synoptic_mean":
        out = hourly[:n * 24].reshape(n, 24, *rest)[:, [0, 6, 12, 18]].mean(axis=1)
    elif rule in ("max24", "min24", "sum24"):
        win = hourly[1:n * 24 + 1].reshape(n, 24, *rest)          # hours 01 ... 24 of each day
        out = {"max24": win.max, "min24": win.min, "sum24": win.sum}[rule](axis=1)
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return dates, out


def weighted_series(daily: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Catchment mean of daily fields (n_days, lat, lon) with weights W."""
    used = W > 0
    if np.isnan(daily[:, used]).any():
        raise ValueError("missing ERA5 values in a weighted cell")
    return np.einsum("tij,ij->t", np.where(used, daily, 0.0), W)
