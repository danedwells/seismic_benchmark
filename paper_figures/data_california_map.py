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
from benchmark.metrics import location_error_km

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
SEIS_CACHE   = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
RUN_DIR      = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
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
# ── Distance to the 4th triggering station ──────────────────────────────────
# .run files carry a station/order/lat/lon row per channel trigger, repeated
# identically across every 'version' block. 'order' is per-channel, not
# per-station (the same station can trigger on 2+ channels within ~a second),
# so the 4th *unique* station requires deduplicating by station code first.
def _fourth_station_distance_km(event_id, event_lat, event_lon):
    run_path = os.path.join(RUN_DIR, f'{int(event_id)}.run')
    if not os.path.exists(run_path):
        return np.nan
    df = pd.read_csv(run_path)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    last_version = df[df['version'] == df['version'].max()].sort_values('order')
    unique_stations = last_version.drop_duplicates(subset='station', keep='first')
    if len(unique_stations) < 4:
        return np.nan
    fourth = unique_stations.iloc[3]
    return location_error_km(event_lat, event_lon, fourth['latitude'], fourth['longitude'])

catalog['dist_4th_station_km'] = [
    _fourth_station_distance_km(eid, lat, lon)
    for eid, lat, lon in zip(catalog['event_id'], catalog['usgs_lat'], catalog['usgs_lon'])
]

_n_valid = catalog['dist_4th_station_km'].notna().sum()
print(f"4th-station distance computed for {_n_valid}/{len(catalog)} events "
      f"({len(catalog) - _n_valid} skipped: no .run file or <4 unique stations)")

#%%
# ── Build the figure ──────────────────────────────────────────────────────
# Exact per-panel placement [left, bottom, width, height] (figure fraction).
# Edit these directly to hand-tune the layout.
AX_POS = {
    'map':      [0.04, 0.06, 0.52, 0.88],  # centerpiece
    'map_cbar': [0.525, 0.06, 0.015, 0.88],
    'panel2':   [0.64, 0.70, 0.28, 0.22],  # top-right    (blank for now)
    'panel3':   [0.64, 0.38, 0.28, 0.22],  # middle-right (blank for now)
    'panel4':   [0.64, 0.06, 0.28, 0.22],  # bottom-right (blank for now)
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

# Case-study mainshocks — labeled stars. MTJ sits close to Ferndale on this
# map (40.37N -125.02W vs. 40.53N -124.42W), so their label offsets point in
# opposite directions (Ferndale NE, MTJ SW) to keep the two boxes apart.
_MAINSHOCKS = {
    'Ridgecrest\nM7.1 (2019)':  {'loc': (35.7695, -117.5990), 'offset': (6, 6),   'ha': 'left',  'va': 'bottom'},
    'Ferndale\nM6.4 (2022)':    {'loc': (40.5268, -124.4227), 'offset': (6, 6),   'ha': 'left',  'va': 'bottom'},
    'El Mayor\nM7.2 (2010)':    {'loc': (32.2863, -115.2950), 'offset': (6, 6),   'ha': 'left',  'va': 'bottom'},
    'MTJ\nM7.0 (2024)':         {'loc': (40.3740, -125.0217), 'offset': (-8, -10), 'ha': 'right', 'va': 'top'},
}
_star_color = 'cyan'
_MAINSHOCK_ZORDER = 20  # above every other map layer (scatter zorder=5, stars/labels included)
for _label, _spec in _MAINSHOCKS.items():
    _lat, _lon = _spec['loc']
    ax_map.plot(_lon, _lat, marker='*', markersize=14, color=_star_color,
                markeredgecolor='black', markeredgewidth=0.4,
                transform=proj, zorder=_MAINSHOCK_ZORDER)
    ax_map.annotate(
        _label,
        xy=(_lon, _lat), xycoords=proj._as_mpl_transform(ax_map),
        fontsize=7.5, color=_star_color, fontweight='bold',
        xytext=_spec['offset'], textcoords='offset points',
        ha=_spec['ha'], va=_spec['va'], zorder=_MAINSHOCK_ZORDER + 1,
        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55, lw=0),
    )

cbar_ax = fig.add_axes(AX_POS['map_cbar'])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label('ANSS Magnitude')
ax_map.set_title(f'USGS/ANSS Event Locations  (n={len(catalog)})')

# -- panel2: magnitude histogram ----------------------------------------------
ax2 = fig.add_axes(AX_POS['panel2'])
_mag_bins = np.arange(np.floor(catalog['usgs_mag'].min() * 2) / 2,
                       np.ceil(catalog['usgs_mag'].max() * 2) / 2 + 0.25, 0.25)
ax2.hist(catalog['usgs_mag'], bins=_mag_bins, color='steelblue', edgecolor='white', linewidth=0.4)
ax2.set_xlabel('ANSS Magnitude')
ax2.set_ylabel('Count')
ax2.set_title('Magnitude Distribution')
ax2.grid(True, linewidth=0.3, alpha=0.4, axis='y')

# -- panel3: depth histogram ---------------------------------------------------
ax3 = fig.add_axes(AX_POS['panel3'])
ax3.hist(catalog['usgs_depth'], bins=30, color='darkorange', edgecolor='white', linewidth=0.4)
ax3.set_xlabel('Depth (km)')
ax3.set_ylabel('Count')
ax3.set_title('Depth Distribution')
ax3.grid(True, linewidth=0.3, alpha=0.4, axis='y')

# -- panel4: distance to the 4th triggering station histogram ----------------
ax4 = fig.add_axes(AX_POS['panel4'])
ax4.hist(catalog['dist_4th_station_km'].dropna(), bins=30,
         color='seagreen', edgecolor='white', linewidth=0.4)
ax4.set_xlabel('Distance to 4th station (km)')
ax4.set_ylabel('Count')
ax4.set_title('4th-Station Distance Distribution')
ax4.grid(True, linewidth=0.3, alpha=0.4, axis='y')

fig.savefig(os.path.join(OUTPUT_DIR, 'data_california_map.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'data_california_map.pdf'), bbox_inches='tight')
plt.show()

# %%
