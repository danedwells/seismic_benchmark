#%%
# =============================================================================
# location_hist_isolated.py — location-error histograms + map for the isolated Wasatch Front events
# =============================================================================
# Same layout as location_hist_magna.py / paper_figures/location_hist_case_study.py, for the 26 isolated M>=3 events of
# run_isolated.py (results/output_isolated). (a) Map: USGS location of each event (black x), where Uniform, static NSHM, ETAS and
# ETAS + NSHM background put it at TRIGGER_NUMBER triggers (colored dots, arrows to the USGS location), and the
# Wasatch fault trace. (b) Stacked location-error histograms with medians.
# Usage: python Magna/location_hist_isolated.py
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter

from faults import draw_wasatch, east_of_wasatch, WASATCH_HANDLE
from variant import ISO_MIN_MAG, ISO_TAG, PRIOR_TAG, PRIOR_ALPHA, FLOOR_FRAC   # MAGNA_ISO_MAG=2.5 -> the M>=2.5 set and its own figure name

MAGNA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(MAGNA_DIR, 'results', 'figures')
OUT_DIR   = os.path.join(MAGNA_DIR, 'results', f'output_isolated{ISO_TAG}{PRIOR_TAG}')
CATALOG   = os.path.join(MAGNA_DIR, 'data', f'wasatch_isolated{ISO_TAG}', 'catalog.parquet')
TRIGGER_NUMBER = 4
WEST_ONLY      = True      # drop events east of the Wasatch fault surface trace (the trace's longitude at the event's latitude)
MAP_PAD_DEG    = 0.30
CITIES = {'Ogden': (41.223, -111.974), 'Salt Lake City': (40.761, -111.891), 'Provo': (40.234, -111.659)}

# (results file stem, label, color); drawn in this order, histogram rows top to bottom. Colors: skills.md (NSHM = orange)
PRIORS = [('uniform', 'Uniform', 'gray'), ('nshm', 'NSHM', 'tab:orange'), ('etas', 'ETAS', 'red')]
# ETAS + NSHM background left out of the presentation version (restore by adding it back to PRIORS):
#           ('etas_nshmbg', 'ETAS + NSHM background', 'tab:brown')
SIZE_SCALE = {'Uniform': 2.5, 'NSHM': 1.7, 'ETAS': 1.0, 'ETAS + NSHM background': 0.6}      # concentric dots where priors coincide

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.labelsize': 11, 'figure.dpi': 150,
                     'savefig.transparent': True})
proj = ccrs.PlateCarree()

#%%
ref = pd.read_parquet(CATALOG).rename(columns={'id': 'event_id', 'latitude': 'usgs_lat', 'longitude': 'usgs_lon'})[['event_id', 'usgs_lat', 'usgs_lon', 'mag']]
if WEST_ONLY:
    east = east_of_wasatch(ref['usgs_lat'].values, ref['usgs_lon'].values) == 1
    print(f'excluding {east.sum()} events east of the Wasatch fault trace:', ', '.join(ref.loc[east, 'event_id']))
    ref = ref[~east]
res = {}
for stem, label, _ in PRIORS:
    d = pd.read_csv(os.path.join(OUT_DIR, f'{stem}_benchmark_results.csv'))
    res[label] = d[d['n_trigs'] == TRIGGER_NUMBER].dropna(subset=['map_err_km']).merge(ref, on='event_id')

#%%
FIG_W, FIG_H = 12.0, 7.6
AX_POS = {'map': [0.02, 0.04, 0.54, 0.92], 'hists': [0.62, 0.10, 0.36, 0.84]}
fig = plt.figure(figsize=(FIG_W, FIG_H))

# -- (b) stacked histograms, shared log x-axis ------------------------------------------------
bins = np.logspace(-1, 2, 19)
hl, hb, hw, hh = AX_POS['hists']
h_each = hh / len(PRIORS)
axes_h = []
for i, (stem, label, color) in enumerate(PRIORS):
    ax = fig.add_axes([hl, hb + hh - (i + 1) * h_each, hw, h_each], sharex=axes_h[0] if axes_h else None)
    axes_h.append(ax)
    e = res[label]['map_err_km'].values
    ax.hist(e, bins=bins, rwidth=0.9, color=color, alpha=0.7)
    ax.axvline(np.median(e), ymax=0.55, color=color, ls='--', lw=1.5)          # stops above the bars, below the labels
    ax.text(0.03, 0.92, label, transform=ax.transAxes, ha='left', va='top', fontsize=13, color=color, fontweight='bold')
    ax.text(0.03, 0.70, f'median: {np.median(e):.1f} km', transform=ax.transAxes, ha='left', va='top', fontsize=11, color=color, fontweight='bold')
    ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
    if i < len(PRIORS) - 1:
        ax.tick_params(labelbottom=False, bottom=False)
axes_h[-1].set_xscale('log')
axes_h[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
axes_h[-1].set_xlabel('Location error (km)')
ymax = max(a.get_ylim()[1] for a in axes_h)
for a in axes_h:
    a.set_ylim(0, 2.0 * ymax)                                                   # headroom so the labels clear the bars
axes_h[0].text(-0.12, 0.87, '(b)', transform=axes_h[0].transAxes, ha='left', va='bottom', fontsize=13, fontweight='bold')

# -- (a) map ------------------------------------------------------------------------------------
ax = fig.add_axes(AX_POS['map'], projection=proj)
# Zoomed on the USGS locations (the Wasatch Front); the few far-off posterior locations run off the map on purpose
lat_all = res['ETAS']['usgs_lat'].values; lon_all = res['ETAS']['usgs_lon'].values
ext = [lon_all.min() - MAP_PAD_DEG, lon_all.max() + MAP_PAD_DEG, lat_all.min() - MAP_PAD_DEG, lat_all.max() + MAP_PAD_DEG]
_target = AX_POS['map'][2] * FIG_W / (AX_POS['map'][3] * FIG_H)                       # box width/height in degrees (equal aspect)
_w, _h = ext[1] - ext[0], ext[3] - ext[2]
if _w / _h < _target:
    c = (ext[0] + ext[1]) / 2; ext[0], ext[1] = c - _target * _h / 2, c + _target * _h / 2
else:
    c = (ext[2] + ext[3]) / 2; ext[2], ext[3] = c - _w / _target / 2, c + _w / _target / 2
ax.set_extent(ext, crs=proj)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='lightcyan', zorder=0)
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '10m'), facecolor='lightcyan', edgecolor='gray', linewidth=0.4, zorder=1)
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=1)
draw_wasatch(ax, proj, zorder=3)

for stem, label, color in PRIORS:
    d = res[label]
    ax.quiver(d['posterior_lon'], d['posterior_lat'], d['usgs_lon'] - d['posterior_lon'], d['usgs_lat'] - d['posterior_lat'],
              angles='xy', scale_units='xy', scale=1, color=color, alpha=0.5, width=0.003, headwidth=3, headlength=3,
              headaxislength=2.5, transform=proj, zorder=4)
for stem, label, color in PRIORS:
    d = res[label]
    ax.scatter(d['posterior_lon'], d['posterior_lat'], s=45 * SIZE_SCALE[label], color=color, alpha=0.95, edgecolors='white',
               linewidths=0.5, transform=proj, zorder=5, label=label)
u = res['ETAS']
ax.scatter(u['usgs_lon'], u['usgs_lat'], marker='x', c='k', s=45, linewidths=1.3, transform=proj, zorder=6, label='USGS/ANSS (true)')
for name, (la, lo) in CITIES.items():
    ax.scatter(lo, la, s=22, c='k', marker='s', transform=proj, zorder=7)
    ax.annotate(name, (lo, la), xycoords=proj._as_mpl_transform(ax), xytext=(6, -12), textcoords='offset points', fontsize=10,
                fontweight='bold', zorder=8, bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.85, ec='none'))
# Lat/lon ticks and axis labels left off for the presentation version (restore for the paper).
ax.text(0.02, 0.10, f'{TRIGGER_NUMBER} triggers\n{len(u)} isolated M≥{ISO_MIN_MAG:g} events\nwest of the fault', transform=ax.transAxes,
        ha='left', va='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'))
ax.text(0.01, 0.99, '(a)', transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')
h, l = ax.get_legend_handles_labels()
ax.legend(h + [WASATCH_HANDLE], l + [WASATCH_HANDLE.get_label()], loc='upper right', fontsize=9, framealpha=0.9, edgecolor='none')

fig.savefig(os.path.join(FIG_DIR, f'location_hist_isolated{ISO_TAG}{PRIOR_TAG}.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, f'location_hist_isolated{ISO_TAG}{PRIOR_TAG}.pdf'), bbox_inches='tight')
