#%%
# =============================================================================
# data_cascadia_map.py — Cascadia catalog map + 3 side panels (future-work slide)
# =============================================================================
# Same layout/styling as data_california_map.py, for the Cascadia benchmark catalog
# (data/cascadia/reference/Cascadia_Benchmark_catalog.parquet, USGS ComCat):
# map of event locations (colored/sized by magnitude, M>=6 starred) on the left;
# magnitude, depth and events-per-year histograms stacked on the right.
# Differences from the California figure: no background seismicity (no Cascadia cache) and
# events per year in place of distance to the 4th station (only one Cascadia .run file exists).
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference', 'Cascadia_Benchmark_catalog.parquet')
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.labelsize':   11,
    'figure.dpi':       150,
    'savefig.transparent': True,
})

proj = ccrs.PlateCarree()

#%%
# ── Load catalog ─────────────────────────────────────────────────────────────
catalog = pd.read_parquet(CATALOG_PATH)
catalog['time'] = pd.to_datetime(catalog['time'], utc=True)
catalog['year'] = catalog['time'].dt.year + (catalog['time'].dt.dayofyear - 1) / 365.25
print(f"Catalog loaded: {len(catalog)} events, {catalog['time'].min():%Y-%m-%d} to {catalog['time'].max():%Y-%m-%d}")

map_extent = [catalog['longitude'].min() - 1, catalog['longitude'].max() + 1,
              catalog['latitude'].min() - 1, catalog['latitude'].max() + 1]

#%%
# ── Build the figure ──────────────────────────────────────────────────────
AX_POS = {
    'map':    [0.04, 0.06, 0.52, 0.88],
    'panel2': [0.71, 0.70, 0.26, 0.22],
    'panel3': [0.71, 0.38, 0.26, 0.22],
    'panel4': [0.71, 0.06, 0.26, 0.22],
}

fig = plt.figure(figsize=(16, 9))

# -- Map of USGS event locations ---------------------------------------------
ax_map = fig.add_axes(AX_POS['map'], projection=proj)
ax_map.set_extent(map_extent, crs=proj)
ax_map.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=2)
ax_map.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor='gray', zorder=2)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=2)
ax_map.add_feature(cfeature.OCEAN,     facecolor='lightcyan', zorder=0)
ax_map.add_feature(cfeature.LAND,      facecolor='lightgray', zorder=0)

sc = ax_map.scatter(
    catalog['longitude'], catalog['latitude'],
    c=catalog['mag'], cmap='Reds',
    s=2 * (catalog['mag'] - catalog['mag'].min() + 0.1) ** 3,
    alpha=0.6, edgecolors='black', linewidths=0.3, transform=proj, zorder=5,
)

# M>=6 events starred; the two M6.4 events labeled
STAR_MAG = 6.0
for _, r in catalog[catalog['mag'] >= STAR_MAG].iterrows():
    ax_map.plot(r['longitude'], r['latitude'], marker='*', markersize=14, color='gold',
                markeredgecolor='black', markeredgewidth=0.4, transform=proj, zorder=20)
LABELS = {   # event id -> (label, offset points, ha, va)
    'us7000mykj': ('Tofino\nM6.4 (2024)',   (30, 20), 'left',  'bottom'),
    'nc73821036': ('Ferndale\nM6.4 (2022)', (35, 30), 'left',  'bottom'),
}
for eid, (lab, off, ha, va) in LABELS.items():
    r = catalog.set_index('id').loc[eid]
    ax_map.annotate(lab, xy=(r['longitude'], r['latitude']), xycoords=proj._as_mpl_transform(ax_map),
                    fontsize=9, color='white', fontweight='bold', xytext=off, textcoords='offset points',
                    ha=ha, va=va, zorder=21, bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.8, lw=0),
                    arrowprops=dict(arrowstyle='-', color='black', lw=0.8, shrinkA=0, shrinkB=6))

# Colorbar hugs the map's actual (aspect-constrained) box
ax_map.apply_aspect()
_p = ax_map.get_position()
cbar_ax = fig.add_axes([_p.x1 + 0.012, _p.y0, 0.012, _p.height])
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=sc.norm, cmap=sc.cmap), cax=cbar_ax)   # opaque (sc has alpha)
cbar.set_label('ANSS Magnitude')

# -- Histograms ---------------------------------------------------------------
def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Count')

hist_kw = dict(color='black', alpha=0.8, edgecolor='white', linewidth=0.4)

ax2 = fig.add_axes(AX_POS['panel2'])
ax2.hist(catalog['mag'], bins=np.arange(np.floor(catalog['mag'].min() * 4) / 4,
                                        np.ceil(catalog['mag'].max() * 4) / 4 + 0.25, 0.25), **hist_kw)
ax2.set_xlabel('ANSS Magnitude'); style(ax2)

ax3 = fig.add_axes(AX_POS['panel3'])
ax3.hist(catalog['depth'], bins=30, **hist_kw)
ax3.set_xlabel('Depth (km)'); style(ax3)

ax4 = fig.add_axes(AX_POS['panel4'])
ax4.hist(catalog['year'], bins=np.arange(2022, np.ceil(catalog['year'].max()) + 0.5, 0.5), **hist_kw)
ax4.set_xlabel('Year'); style(ax4)

fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'data_cascadia_map.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'data_cascadia_map.pdf'), bbox_inches='tight')
