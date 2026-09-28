#%%
# =============================================================================
# data_california_map.py — paper figure: California catalog map + 3 side panels
# =============================================================================
# Centerpiece: map of USGS/ANSS event locations (colored by magnitude, sized
# by magnitude, background seismicity in gray, case-study mainshocks starred),
# adapted from data_examination_scripts/examine_catalog.py, section 2 (the
# logic behind data/california/reference/catalog_map.png).
#
# Layout: the map is the centerpiece on the left; three panels are stacked
# vertically to the right, currently left blank as placeholders for future
# content. As in paper_figures/prior_comparison.py, every panel's position is
# set explicitly via AX_POS so the layout can be hand-tuned directly.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from benchmark.runner import load_reference_catalog
from benchmark.background import load_background_seismicity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
SEIS_CACHE   = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
})

proj = ccrs.PlateCarree()

#%%
# ── Load catalog and background seismicity ──────────────────────────────────
raw = pd.read_csv(CATALOG_PATH, sep='\t')

# Parse 'ANSS date' — format: '2018-09-30-14:41:29.510-GMT'
raw['datetime'] = pd.to_datetime(
    raw['ANSS date'].str.replace(r'-GMT$', '', regex=True),
    format='%Y-%m-%d-%H:%M:%S.%f',
    utc=True,
)

catalog = load_reference_catalog(CATALOG_PATH)
catalog['datetime'] = pd.to_datetime(raw['datetime'], utc=True)
catalog = catalog.sort_values('datetime').reset_index(drop=True)

print(f"Catalog loaded: {len(catalog)} events")

map_extent = [catalog['usgs_lon'].min() - 1, catalog['usgs_lon'].max() + 1,
              catalog['usgs_lat'].min() - 1, catalog['usgs_lat'].max() + 1]

# force_refresh left at its default (False) — SEIS_CACHE already exists on
# disk, so this loads instantly from the parquet cache instead of re-querying
# USGS ComCat every time the figure is rerun.
bg = load_background_seismicity(
    cache_path = SEIS_CACHE,
    bounds     = (map_extent[0], map_extent[1], map_extent[2], map_extent[3]),
    start_year = 2000,
    end_year   = 2018,
    min_mag    = 3.5,
)

#%%
# ── Build the figure ──────────────────────────────────────────────────────
# Exact per-panel placement [left, bottom, width, height] (figure fraction).
# Edit these directly to hand-tune the layout.
AX_POS = {
    'map':      [0.04, 0.06, 0.52, 0.88],  # centerpiece
    'map_cbar': [0.575, 0.06, 0.015, 0.88],
    'panel2':   [0.68, 0.70, 0.28, 0.24],  # top-right    (blank for now)
    'panel3':   [0.68, 0.38, 0.28, 0.24],  # middle-right (blank for now)
    'panel4':   [0.68, 0.06, 0.28, 0.24],  # bottom-right (blank for now)
}

fig = plt.figure(figsize=(16, 9))

# -- Centerpiece: map of USGS/ANSS event locations ---------------------------
ax_map = fig.add_axes(AX_POS['map'], projection=proj)
ax_map.set_extent(map_extent, crs=proj)
ax_map.add_feature(cfeature.STATES,    linewidth=0.5, edgecolor='black')
ax_map.add_feature(cfeature.BORDERS,   linewidth=0.7, edgecolor='black')
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax_map.add_feature(cfeature.OCEAN,     facecolor='lightcyan', alpha=0.4)
ax_map.add_feature(cfeature.LAND,      facecolor='whitesmoke')
gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
gl.top_labels   = False
gl.right_labels = False

if bg is not None:
    ax_map.scatter(bg['longitude'], bg['latitude'],
                    s=6, c='gray', alpha=0.1, transform=proj, zorder=1, linewidths=0)

sc = ax_map.scatter(
    catalog['usgs_lon'], catalog['usgs_lat'],
    c=catalog['usgs_mag'], cmap='plasma',
    s=2 * (catalog['usgs_mag'] - catalog['usgs_mag'].min() + 0.1) ** 3,
    alpha=0.4, transform=proj, zorder=5,
)

# Case-study mainshocks — labeled stars
_MAINSHOCKS = {
    'Ridgecrest\nM7.1 (2019)':  (35.7695, -117.5990),
    'Ferndale\nM6.4 (2022)':    (40.5268, -124.4227),
    'El Mayor\nM7.2 (2010)':    (32.2863, -115.2950),
}
_star_color = 'cyan'
for _label, (_lat, _lon) in _MAINSHOCKS.items():
    ax_map.plot(_lon, _lat, marker='*', markersize=14, color=_star_color,
                markeredgecolor='black', markeredgewidth=0.4,
                transform=proj, zorder=10)
    ax_map.annotate(
        _label,
        xy=(_lon, _lat), xycoords=proj._as_mpl_transform(ax_map),
        fontsize=7.5, color=_star_color, fontweight='bold',
        xytext=(6, 6), textcoords='offset points',
        ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, lw=0),
    )

cbar_ax = fig.add_axes(AX_POS['map_cbar'])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label('ANSS Magnitude')
ax_map.set_title(f'USGS/ANSS Event Locations  (n={len(catalog)})')

# -- Side panels: placeholders ------------------------------------------------
for key in ('panel2', 'panel3', 'panel4'):
    ax = fig.add_axes(AX_POS[key])
    ax.text(0.5, 0.5, 'TBD', ha='center', va='center', color='lightgray', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(os.path.join(OUTPUT_DIR, 'data_california_map.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'data_california_map.pdf'), bbox_inches='tight')
plt.show()

# %%
