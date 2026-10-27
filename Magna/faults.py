# faults.py — Wasatch fault zone surface trace (USGS Quaternary Fault & Fold Database) for the Magna figures
import os

import pandas as pd
import shapely
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

QFAULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qfaults.csv')   # seismic_benchmark/qfaults.csv


def wasatch_fault():
    """Shapely geometries of every 'Wasatch fault zone' trace segment (lon, lat)."""
    f = pd.read_csv(QFAULTS, usecols=['name', 'WKT'])
    return list(shapely.from_wkt(f.loc[f['name'] == 'Wasatch fault zone', 'WKT'].values))


def draw_wasatch(ax, crs, zorder=4):
    """Draw the trace in black with a white halo (readable over the red/blue maps and the earthquake dots)."""
    ax.add_geometries(wasatch_fault(), crs=crs, facecolor='none', edgecolor='k', linewidth=1.6, zorder=zorder,
                      path_effects=[pe.withStroke(linewidth=3.4, foreground='white')])


# legend handle matching draw_wasatch()
WASATCH_HANDLE = Line2D([], [], color='k', linewidth=1.6, path_effects=[pe.withStroke(linewidth=3.4, foreground='white')],
                        label='Wasatch fault (surface trace)')


def east_of_wasatch(lat, lon, window_deg=0.05):
    """True where (lat, lon) lies east of the Wasatch fault trace: the event's longitude exceeds the median longitude of the
    trace vertices within +/-window_deg of its latitude. NaN where the trace has no vertices that close in latitude
    (beyond its northern/southern end), so the side is undefined there."""
    import numpy as np
    v = np.vstack([np.array(g.coords) for gg in wasatch_fault() for g in (gg.geoms if hasattr(gg, 'geoms') else [gg])])   # (lon, lat)
    out = []
    for la, lo in zip(np.atleast_1d(lat), np.atleast_1d(lon)):
        near = v[np.abs(v[:, 1] - la) <= window_deg]
        out.append(np.nan if len(near) == 0 else float(lo > np.median(near[:, 0])))
    return np.array(out)
