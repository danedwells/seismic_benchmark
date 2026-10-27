#%%
# =============================================================================
# plot_nshm.py — USGS NSHM moment-rate prior for Utah (presentation figure)
# =============================================================================
# The Utah NSHM prior (data/priors/nshm_Magna.tt3, built by run_nshm_benchmarks.py) as a map of
# probability density relative to a uniform prior: red = more probable than uniform, blue = less.
# The Wasatch fault shows up as the bright N-S ridge east of Magna. No title/colorbar
# (see paper_figures/skills.md); the red/blue read-out is a small direct label.
#
# Usage: python Magna/plot_nshm.py
# =============================================================================
import os

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

from priors import SeismicPrior
from faults import draw_wasatch, WASATCH_HANDLE
from variant import NSHM_TAG   # MAGNA_NSHM_RES=0.02 -> finer-grid prior (data/priors/nshm_Magna_0.02deg.tt3)

MAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(MAGNA_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
NSHM_TT3  = os.path.join(MAGNA_DIR, 'data', 'priors', f'nshm_Magna{NSHM_TAG}.tt3')
MAGNA     = (40.75, -112.08)          # (lat, lon) of the 2020 M5.7 mainshock
EXTENT = [-113.0, -111.0, 39.0, 42.4]    # zoomed on the Wasatch fault trace (the NSHM grid ends at 42.4N; the full prior covers -114.5..-109, 37..42.4)
HALF_DECADES = 2.0                    # color range about the uniform level (orders of magnitude)
CITIES = {'Salt Lake City': (40.7608, -111.8910, (8, 6)), 'Ogden': (41.2230, -111.9738, (8, 6)),
          'Provo': (40.2338, -111.6585, (8, 6))}

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'figure.dpi': 150, 'savefig.transparent': True})

#%%
p = SeismicPrior.from_tt3(NSHM_TT3)
# cells with no moment rate (grid value 0) are floored at the bottom of the color range
ratio = np.log10(np.maximum(p.grid * p.grid.size, 10 ** -HALF_DECADES))    # log10(density / uniform density)

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(4.8, 9.2))
ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection=proj)
ax.set_extent(EXTENT, crs=proj)
ax.pcolormesh(p.lons, p.lats, ratio, cmap='RdBu_r', norm=TwoSlopeNorm(0, -HALF_DECADES, HALF_DECADES),
              transform=proj, shading='auto', zorder=0)
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'), facecolor='lightcyan',
               edgecolor='gray', linewidth=0.4, zorder=2)
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='k', zorder=2)
for name, (lat, lon, off) in CITIES.items():
    ax.scatter(lon, lat, s=22, c='k', marker='s', transform=proj, zorder=5)
    ax.annotate(name, (lon, lat), xycoords=proj._as_mpl_transform(ax), xytext=off, textcoords='offset points',
                fontsize=10, fontweight='bold', zorder=6, bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.85, ec='none'))
draw_wasatch(ax, proj, zorder=4)
ax.scatter(MAGNA[1], MAGNA[0], marker='*', s=420, c='gold', edgecolors='k', linewidths=1, transform=proj, zorder=7)
ax.annotate('Magna M5.7', (MAGNA[1], MAGNA[0]), xycoords=proj._as_mpl_transform(ax), xytext=(-12, -22),
            textcoords='offset points', ha='right', fontsize=12, fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9, ec='none'))
ax.text(0.03, 0.03, 'NSHM moment-rate prior\nred: more probable than uniform\nblue: less', transform=ax.transAxes,
        fontsize=11, fontweight='bold', va='bottom', zorder=9, bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='none'))

ax.legend(handles=[WASATCH_HANDLE], loc='upper right', fontsize=9.5, framealpha=0.9, edgecolor='none')
fig.savefig(os.path.join(FIG_DIR, f'nshm_utah{NSHM_TAG}.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, f'nshm_utah{NSHM_TAG}.pdf'), bbox_inches='tight')
