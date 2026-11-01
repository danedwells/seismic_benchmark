#%%
# =============================================================================
# location_hist_nshm_bg.py — California benchmark: ETAS with the NSHM as background grid vs pure ETAS vs Uniform
# =============================================================================
# Reads results/california/output/time_dependent_nshm_bg/sig_0.35/max_trigs_10 (time_dependent_scripts/run_benchmarks_nshm_bg.py).
# (a) Map: change in MAP location error at TRIGGER_NUMBER triggers when the ETAS background is replaced by the NSHM
#     (ETAS + NSHM bg minus pure ETAS, km; blue = NSHM background better, red = worse). Events whose location did not change are small gray dots.
# (b) Stacked location-error histograms (Uniform / ETAS / ETAS + NSHM background) with medians.
# Usage: python paper_figures/location_hist_nshm_bg.py
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

from benchmark.runner import load_reference_catalog

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'output', 'time_dependent_nshm_bg', 'sig_0.35', 'max_trigs_10')
CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
TRIGGER_NUMBER = 4
EXTENT = [-125.5, -114, 32, 42.5]
LIM = 10                      # color limit for the change in error (km); saturates beyond
PRIORS = [('uniform', 'Uniform', 'gray'), ('etas', 'ETAS', 'red'), ('etas_nshmbg', 'ETAS + NSHM background', 'tab:orange')]   # skills.md colors

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.labelsize': 11, 'figure.dpi': 150, 'savefig.transparent': True})
proj = ccrs.PlateCarree()

#%%
cat = load_reference_catalog(CATALOG)[['event_id', 'usgs_lat', 'usgs_lon']]
res = {}
for stem, label, _ in PRIORS:
    d = pd.read_csv(os.path.join(RES_DIR, f'{stem}_benchmark_results.csv'))
    res[label] = d[d['n_trigs'] == TRIGGER_NUMBER].dropna(subset=['map_err_km']).merge(cat, on='event_id')
e = res['ETAS'].set_index('event_id'); n = res['ETAS + NSHM background'].set_index('event_id'); u = res['Uniform'].set_index('event_id')
delta = (n.map_err_km - e.map_err_km).dropna()
ev = pd.DataFrame({'delta': delta}).join(e[['usgs_lat', 'usgs_lon']])
changed = ev[ev.delta.abs() > 0.05]
print(f'{len(ev)} events at {TRIGGER_NUMBER} triggers; location changed in {len(changed)}: better {(delta < -0.05).sum()}, worse {(delta > 0.05).sum()}')
for k, d in res.items():
    print(f'  {k:24s} median {d.map_err_km.median():.2f} km  mean {d.map_err_km.mean():.1f} km  >10 km: {(d.map_err_km > 10).sum()}')

#%%
fig = plt.figure(figsize=(14, 8.5))

# -- (b) stacked histograms -------------------------------------------------------------------
bins = np.logspace(-1, 3, 20)
hl, hb, hw, hh = 0.62, 0.09, 0.35, 0.86
h_each = hh / len(PRIORS)
axes_h = []
for i, (stem, label, color) in enumerate(PRIORS):
    ax = fig.add_axes([hl, hb + hh - (i + 1) * h_each, hw, h_each], sharex=axes_h[0] if axes_h else None)
    axes_h.append(ax)
    x = res[label]['map_err_km'].values
    ax.hist(x, bins=bins, rwidth=0.9, color=color, alpha=0.7)
    ax.axvline(np.median(x), ymax=0.55, color=color, ls='--', lw=1.5)
    ax.text(0.03, 0.92, label, transform=ax.transAxes, ha='left', va='top', fontsize=13, color=color, fontweight='bold')
    ax.text(0.03, 0.70, f'median: {np.median(x):.1f} km', transform=ax.transAxes, ha='left', va='top', fontsize=11, color=color, fontweight='bold')
    ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
    if i < len(PRIORS) - 1:
        ax.tick_params(labelbottom=False, bottom=False)
axes_h[-1].set_xscale('log')
axes_h[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
axes_h[-1].set_xlabel('Location error (km)')
ymax = max(a.get_ylim()[1] for a in axes_h)
for a in axes_h:
    a.set_ylim(0, 2.0 * ymax)
axes_h[0].text(-0.12, 0.87, '(b)', transform=axes_h[0].transAxes, ha='left', va='bottom', fontsize=13, fontweight='bold')

# -- (a) map of the change in error ------------------------------------------------------------
ax = fig.add_axes([0.03, 0.17, 0.55, 0.78], projection=proj)
ax.set_extent(EXTENT, crs=proj)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='lightcyan', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=2)
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=2)
same = ev[ev.delta.abs() <= 0.05]
ax.scatter(same.usgs_lon, same.usgs_lat, s=6, c='gray', alpha=0.6, linewidths=0, transform=proj, zorder=3)
cmap = plt.get_cmap('RdBu_r'); norm = TwoSlopeNorm(0, -LIM, LIM)
sc = ax.scatter(changed.usgs_lon, changed.usgs_lat, c=changed.delta, cmap=cmap, norm=norm, s=40, edgecolors='k', linewidths=0.4, transform=proj, zorder=4)
ax.text(0.97, 0.97, f'{TRIGGER_NUMBER} triggers\n{len(ev)} events\nlocation changed in {len(changed)}', transform=ax.transAxes, ha='right', va='top',
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='none'))
ax.text(0.01, 0.99, '(a)', transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')
cax = fig.add_axes([0.10, 0.085, 0.41, 0.022])
cb = fig.colorbar(sc, cax=cax, orientation='horizontal', extend='both')
cb.set_label('Change in location error, NSHM background − ETAS (km)')
for txt, x0, ha, v in (('← better', 0.0, 'left', -0.8 * LIM), ('worse →', 1.0, 'right', 0.8 * LIM)):
    cax.annotate(txt, xy=(x0, 0), xycoords='axes fraction', xytext=(0, -52), textcoords='offset points', ha=ha, va='top',
                 fontsize=16, fontweight='bold', color=cmap(norm(v)))

fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'location_hist_nshm_bg.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'location_hist_nshm_bg.pdf'), bbox_inches='tight')
