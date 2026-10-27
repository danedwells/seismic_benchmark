"""
ShakeAlert alert footprint (A. Williamson, USGS): the area alerted for an event of magnitude M is the MMI-contour
distance from the BA08/Worden table (SA-20200401-BA08_worden_contourdist_av.csv, in seismic_benchmark/), drawn around the
epicenter as a circle (quick plot) or as an 8-vertex polygon (what phone alerts use). Alerts go out only once M >= 4.5.
"""
import os

import numpy as np
import pandas as pd

TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SA-20200401-BA08_worden_contourdist_av.csv')
EARTH_R_KM = 6371.0
# Williamson's snippet puts the first octagon vertex due north (bearings 0, 45, ...), so no side is parallel to latitude.
# Kept as in her snippet (0). Setting 22.5 gives flat top/bottom sides (exactly parallel to latitude) and roughly N-S
# left/right sides, with the same vertex distance (same area).
OCTAGON_ROTATION_DEG = 0.0
_table = pd.read_csv(TABLE, skiprows=1, index_col=False, names=['cmag', 'cmmi', 'notneeded1', 'notneeded2', 'cdist-km'])


def contour_distance_km(magnitude, mmi_level):
    """Distance (km) to the MMI contour at the table magnitude nearest `magnitude` (the table has -1 where there is no contour)."""
    cmag = _table['cmag'][np.argmin(np.abs(_table['cmag'] - magnitude))]
    return float(_table['cdist-km'][np.where((_table['cmag'] == cmag) & (_table['cmmi'] == mmi_level))[0][0]])


def terminal_location(lon, lat, bearing_deg, distance_km):
    """(lon, lat) reached from (lon, lat) travelling distance_km along bearing_deg (spherical Earth)."""
    d, b = distance_km / EARTH_R_KM, np.radians(bearing_deg)
    la1, lo1 = np.radians(lat), np.radians(lon)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(b))
    lo2 = lo1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la1), np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(lo2), np.degrees(la2)


def footprint(lon, lat, magnitude, mmi_level, shape='octagon'):
    """Closed (lon, lat) outline of the alert footprint, or None if there is no alert (M < 4.5 or no contour)."""
    if magnitude < 4.5:
        return None
    dist = contour_distance_km(magnitude, mmi_level)
    if dist <= 0:
        return None
    bearings = OCTAGON_ROTATION_DEG + np.arange(0, 360 + 45, 45) if shape == 'octagon' else np.arange(361)
    lons, lats = terminal_location(lon, lat, bearings, dist)
    return np.column_stack([lons, lats])
