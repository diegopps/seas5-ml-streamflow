"""
catchments.py – Catchment definitions for seasonal discharge prediction.

Add new catchments here. The pipeline scripts all read from this file.
Data locations come from config.toml (see flowcast_src/paths.py).
"""

from flowcast_src.paths import DISCHARGE_DIR, SEAS5_DIR

CATCHMENTS = {
    'ulkenboken': {
        'name': 'Ulken Boken',
        'discharge_csv': str(DISCHARGE_DIR / 'Ulkenboken_1995-2025_Q.csv'),
        # Catchment-scale bounding box (Bol'shoy Bukon' / Ulken Boken, draining to
        # the Buktyrma reservoir in the Kazakh Altai; catchment polygon of
        # GRDC station 2311527 "ZHUMBA", lon [82.5921, 83.0362],
        # lat [49.0079, 49.3704]. Extended one degree south of the tightest
        # integer box (lat 49-50), whose southern edge sat 0.008 deg from the
        # polygon — too tight to be comfortable against grid-cell edges.
        'catch_lat': (48.0, 50.0),
        'catch_lon': (82.0, 84.0),
        # Hydro years to skip: observations EXIST but are untrustworthy (e.g. gauge
        # under calibration). Years with missing/sparse data need no entry — those
        # rows are removed automatically by the a0/Q availability filters in
        # 00_preprocess_catchment.py / flowcast_src.data.build_analysis_frame.
        #
        # Applied in two places, for two different dates:
        #   - init side  : 00_preprocess_catchment.py drops rows whose INIT date falls
        #                  in a skip year (baked into the parquet), so a0 never carries
        #                  untrustworthy discharge.
        #   - valid side : build_analysis_frame drops rows whose VALID date's hydro_year
        #                  is a skip year (applied fresh at analysis time).
        # Both are needed: hydro_year is derived from the valid date, so the valid-side
        # filter alone does not stop a trajectory initialised inside a skip year from
        # contributing rows that fall outside it.
        'skip_years': set(),
    },
    'zeravshan': {
        'name': 'Zeravshan',
        'discharge_csv': str(DISCHARGE_DIR / 'Zeravshan_1936-2025_Q.csv'),
        # Catchment-scale bounding box (Zeravshan basin, Tajikistan/Uzbekistan).
        # Trimmed from lon (66, 72) to (67, 71): the SEAS5 grid is centred on
        # half-integers, so the old box selected 3 lat x 6 lon cells
        # (lon 66.5 ... 71.5) and the new one selects 3 x 4 (lon 67.5 ... 70.5),
        # dropping the westernmost and easternmost column. Bounds are given as
        # integers for consistency with the other catchments; any value in
        # (66.5, 67.5] and [70.5, 71.5) selects the same four columns.
        'catch_lat': (38.0, 41.0),
        'catch_lon': (67.0, 71.0),
        # 2005-2010 are complete, present discharge records excluded on a
        # data-quality judgement (station still being calibrated). See
        # 'ulkenboken' above for the general skip_years convention.
        'skip_years': {2005, 2006, 2007, 2008, 2009, 2010},
    },
    'esil': {
        'name': 'Esil',
        'discharge_csv': str(DISCHARGE_DIR / 'Esil_1995-2025_Q.csv'),
        # Catchment-scale bounding box (Ishim river, Kazakh steppe; catchment
        # polygon of GRDC station 2311302 "TURGENEVKA",
        # lon [72.0779, 73.2721], lat [50.4021, 50.9304], rounded outward)
        'catch_lat': (50.0, 51.0),
        'catch_lon': (72.0, 74.0),
        'skip_years': set(),
    },
    'ayat': {
        'name': 'Ayat',
        'discharge_csv': str(DISCHARGE_DIR / 'Ayat_1995-2025_Q.csv'),
        # Catchment-scale bounding box (Ayat river, Kazakh/Russian steppe;
        # catchment polygon of GRDC station 2911230 "VARVARINKA",
        # lon [59.8657, 62.3852], lat [52.4489, 53.5261], rounded outward)
        'catch_lat': (52.0, 54.0),
        'catch_lon': (59.0, 63.0),
        'skip_years': {2006, 2009},
    },
    'kalkutan': {
        'name': 'Kalkutan',
        'discharge_csv': str(DISCHARGE_DIR / 'Kalkutan_1995-2025_Q.csv'),
        # Catchment-scale bounding box (Koluton river, Akmola steppe; catchment
        # polygon of GRDC station 2311360 "QALAOTAN",
        # lon [69.0398, 71.4802], lat [51.4413, 52.8918], rounded outward)
        'catch_lat': (51.0, 53.0),
        'catch_lon': (69.0, 72.0),
        'skip_years': set(),
    },
    'kurshim': {
        'name': 'Kurshim',
        'discharge_csv': str(DISCHARGE_DIR / 'Kurshim_1995-2025_Q.csv'),
        # Catchment-scale bounding box (Kurshim river, Kazakh Altai; catchment
        # polygon of GRDC station 2311840 "VOZNESENSKOJE",
        # lon [83.8404, 85.8270], lat [48.4936, 49.0934]. Tightest integer box
        # would be lon (83, 86); extended one degree east to (83, 87) to keep
        # the polygon roughly centred rather than hard against the east edge.
        'catch_lat': (48.0, 50.0),
        'catch_lon': (83.0, 87.0),
        'skip_years': {1997,2002},
    },
    'zhabay': {
        'name': 'Zhabay',
        'discharge_csv': str(DISCHARGE_DIR / 'Zhabay_1995-2025_Q.csv'),
        # Catchment-scale bounding box (Zhabay river, Akmola steppe; catchment
        # polygon of GRDC station 2311330 "ATBASAR",
        # lon [67.6998, 69.3174], lat [51.8080, 52.8496], rounded outward)
        'catch_lat': (51.0, 53.0),
        'catch_lon': (67.0, 70.0),
        'skip_years': set(),
    },
    'kirchbichl': {
        'name': 'Kirchbichl-Bichlwang',
        'discharge_csv': str(DISCHARGE_DIR / 'kirchbichl_bichlwang_1951-2023_Q.csv'),
        # Catchment-scale bounding box (Inn basin, Austria; catchment polygon,
        # lon [9.6638, 12.3129], lat [46.3354, 47.6763], rounded outward to integers)
        'catch_lat': (46.0, 48.0),
        'catch_lon': (9.0, 13.0),
        'skip_years': set(),
        # Overrides SHARED's centralasia dirs — this catchment's NetCDF source is Europe
        'hindcast_dir': str(SEAS5_DIR / 'netcdf_hindcast' / 'europe'),
        'forecast_dir': str(SEAS5_DIR / 'netcdf_forecast' / 'europe'),
    },
    'diepoldsau': {
        'name': 'Diepoldsau-Rietbrücke',
        'discharge_csv': str(DISCHARGE_DIR / 'diepoldsau_1919-2020_Q.csv'),
        # Catchment-scale bounding box (Rhine basin, Switzerland/Austria;
        # catchment polygon, lon [8.6521, 10.2221], lat [46.3663, 47.3854],
        # rounded outward to integers)
        'catch_lat': (46.0, 48.0),
        'catch_lon': (8.0, 11.0),
        'skip_years': set(),
        # Overrides SHARED's centralasia dirs — this catchment's NetCDF source is Europe
        'hindcast_dir': str(SEAS5_DIR / 'netcdf_hindcast' / 'europe'),
        'forecast_dir': str(SEAS5_DIR / 'netcdf_forecast' / 'europe'),
    },
}

# Shared configuration
SHARED = {
    # Data paths for CA catchments.
    'hindcast_dir': str(SEAS5_DIR / 'netcdf_hindcast' / 'centralasia'),
    'forecast_dir': str(SEAS5_DIR / 'netcdf_forecast' / 'centralasia'),
    # Processing
    'met_variables': ['tas', 'tasmax', 'tasmin', 'ssrd', 'tp', 'sd', 'sf'],
    'lead_bins_daily': [
        (0, 14, '0-14d'),
        (15, 30, '15-30d'),
        (31, 60, '31-60d'),
        (61, 90, '61-90d'),
        (91, 120, '91-120d'),
        (121, 150, '121-150d'),
        (151, 215, '151-215d'),
    ],
    'daily_clim_window': 31,
    # First year of the DOY discharge climatology, shared by every catchment.
    # Set to the first year of the SEAS5 archive (see config.toml,
    # [seas5_download] first_year = 1981) so the climatology
    # describes the same era the models are scored in.
    #
    # Deliberately global, NOT per-catchment: the climatology must mean the
    # same thing at every gauge. Catchments whose record starts later (the six
    # Central Asian gauges, 1995) are simply unaffected by it.
    'clim_start_year': 1981,
}
