#%%
# =============================================================================
# pocatello_historic_earthquakes.py — Historic M>=6 earthquakes near Pocatello, ID
# =============================================================================
# Stand-alone: no catalog needed. Epicenters are hard coded (approximate, from
# USGS / Idaho Geological Survey / Utah Geological Survey summaries). Background is
# Esri shaded relief (downloaded on first run via cartopy tiles, needs internet).
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'figure.dpi': 150})

proj = ccrs.PlateCarree()

# name, year, magnitude, lat, lon, label offset (points), ha
EVENTS = [
    ('Hansel Valley, UT',        1934, 6.6, 41.70, -112.80, (-12, -14), 'right'),
    ('Hebgen Lake, MT',          1959, 7.3, 44.83, -111.08, (10, 0),    'left'),
    ('Pocatello Valley, ID',     1975, 6.0, 42.07, -112.50, (12, -14),  'left'),
    ('Borah Peak, ID',           1983, 6.9, 44.06, -113.86, (14, -10),  'left'),
    ('Challis (Stanley), ID',    2020, 6.5, 44.40, -115.11, (12, 12),   'left'),
]
CITIES = [
    ('Pocatello',    42.871, -112.446),
    ('Idaho Falls',  43.492, -112.034),
    ('Twin Falls',   42.563, -114.461),
]
EXTENT = [-116.0, -110.0, 41.2, 45.6]

#%%
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(EXTENT, crs=proj)

# Shaded-relief terrain at alpha 0.3
terrain = cimgt.GoogleTiles(
    url='https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}')
ax.add_image(terrain, 8, alpha=0.45, zorder=0)

ax.add_feature(cfeature.STATES, linewidth=1.8, edgecolor='black', facecolor='none', zorder=2)

for name, year, mag, lat, lon, off, ha in EVENTS:
    ax.scatter(lon, lat, s=(mag - 4) ** 3 * 18, marker='o', color='red', alpha=0.75,
               edgecolor='black', linewidth=0.8, transform=proj, zorder=5)
    ax.annotate(f'{name}\nM{mag} ({year})', (lon, lat), xytext=off, textcoords='offset points',
                ha=ha, va='center', fontsize=11, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7),
                transform=proj, zorder=6)

for name, lat, lon in CITIES:
    is_poc = name == 'Pocatello'
    ax.plot(lon, lat, marker='*' if is_poc else 's', markersize=18 if is_poc else 9,
            color='gold' if is_poc else 'black', markeredgecolor='black',
            transform=proj, zorder=7)
    ax.annotate(name, (lon, lat), xytext=(8, 8), textcoords='offset points',
                fontsize=13, fontweight='bold', transform=proj, zorder=7)

ax.set_title('Historic M≥6 earthquakes near Pocatello, Idaho', fontsize=14)

out = os.path.join(OUTPUT_DIR, 'figures', 'pocatello_historic_earthquakes')
fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
fig.savefig(out + '.pdf', bbox_inches='tight')
plt.show()

# %%
