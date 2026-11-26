"""
seas5ml.spatial – Catchment polygons and the weights of the forcing grid cells.

Stage 0 reduces each SEAS5 field to one series per catchment as a weighted
mean over the grid cells of the catchment's box (`catch_lat`, `catch_lon` in
seas5ml.catchments). Each cell's weight is the area it shares with the
catchment polygon (`polygon`, a GeoJSON file per catchment), so cells the
catchment does not touch get weight 0 and the mean is the areal mean over the
catchment as the 1 degree grid resolves it. The overlap area already accounts
for the cells' size, so no separate cos-latitude factor is applied. The same
weights serve any field on the same grid (the ERA5-Land comparison remaps
ERA5-Land onto the SEAS5 grid and reuses them).

Cells are centred on the grid points, with edges half a cell away (the SEAS5
CDS grid has centres on half degrees, so cell edges fall on whole degrees and
a box with whole-degree bounds is exactly the union of its cells). Areas are
computed in an equal-area cylindrical projection (EPSG:6933), where the
cells' edges (parallels, meridians) stay straight lines, so cell areas are
exact and the polygon's area is accurate to its vertex spacing.
"""

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely import ops
from shapely.geometry import box, shape
from shapely.validation import make_valid

# lon/lat (WGS84) -> equal-area cylindrical, metres
_TO_EQUAL_AREA = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
_WGS84_NAMES = ("urn:ogc:def:crs:OGC:1.3:CRS84", "EPSG:4326", "urn:ogc:def:crs:EPSG::4326")

# Share of the polygon's area that must lie inside its box (whatever lies
# outside would be lost from the mean).
MIN_POLYGON_IN_BOX = 0.999


def load_polygon(path):
    """
    The catchment polygon of a GeoJSON file, as one shapely geometry in
    lon/lat (WGS84). All features are merged. A file declaring any other CRS
    raises (GeoJSON without a `crs` member is WGS84 by the standard).
    """
    path = Path(path)
    with open(path) as fh:
        gj = json.load(fh)
    crs = (gj.get("crs") or {}).get("properties", {}).get("name")
    if crs is not None and crs not in _WGS84_NAMES:
        raise ValueError(f"{path}: CRS {crs!r} is not WGS84 lon/lat; reproject the file")
    if gj.get("type") == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in gj["features"] if f.get("geometry")]
    elif gj.get("type") == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    if not geoms:
        raise ValueError(f"{path}: no geometry")
    geom = ops.unary_union([g if g.is_valid else make_valid(g) for g in geoms])
    if geom.is_empty or geom.area == 0:
        raise ValueError(f"{path}: empty polygon")
    return geom


def area_km2(geom) -> float:
    """Area of a lon/lat geometry in km² (equal-area projection)."""
    if geom.is_empty:
        return 0.0
    return ops.transform(_TO_EQUAL_AREA.transform, geom).area / 1e6


def cell_overlaps(lat, lon, polygon, cell_size: float = 1.0) -> np.ndarray:
    """
    Area (km²) each grid cell shares with `polygon`, shape (len(lat), len(lon)).
    Cell (i, j) spans lat[i] ± cell_size/2, lon[j] ± cell_size/2.
    """
    half = cell_size / 2
    out = np.zeros((len(lat), len(lon)))
    for i, y in enumerate(np.asarray(lat, dtype=float)):
        for j, x in enumerate(np.asarray(lon, dtype=float)):
            cell = box(x - half, y - half, x + half, y + half)
            if cell.intersects(polygon):
                out[i, j] = area_km2(cell.intersection(polygon))
    return out


def cell_weights(lat, lon, polygon, cell_size: float = 1.0):
    """
    Overlap weights of the grid cells (lat × lon), normalised to sum to 1,
    plus the overlap areas in km² and the share of the polygon's area the
    cells cover. Raises if that share is below MIN_POLYGON_IN_BOX (part of the
    catchment outside the cells) or if no cell overlaps the polygon.
    """
    overlaps = cell_overlaps(lat, lon, polygon, cell_size)
    total = overlaps.sum()
    if total <= 0:
        raise ValueError("no grid cell overlaps the catchment polygon")
    covered = total / area_km2(polygon)
    if covered < MIN_POLYGON_IN_BOX:
        raise ValueError(
            f"the grid cells cover only {covered:.2%} of the catchment polygon "
            f"(need >= {MIN_POLYGON_IN_BOX:.1%}): widen the catchment's box "
            f"(catch_lat / catch_lon in seas5ml/catchments.py)")
    return overlaps / total, overlaps, covered
