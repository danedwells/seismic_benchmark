#%%
# =============================================================================
# location_hist_nshm_blend_bg.py — California benchmark: ETAS with a 50/50 NSHM + ETAS background vs ETAS vs Uniform
# =============================================================================
# Same style as the 'benchmark' case of location_hist_case_study.py: (a) map of posterior locations at TRIGGER_NUMBER
# triggers with arrows to the USGS location, (b) stacked location-error histograms with medians.
# Reads results/california/output/time_dependent_nshm_blend_bg/sig_0.35/max_trigs_10_grid50_migrate3 (time_dependent_scripts/run_benchmarks_nshm_blend_bg.py).
# Usage: python paper_figures/location_hist_nshm_blend_bg.py
# =============================================================================
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from benchmark.runner import load_reference_catalog

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'output', 'time_dependent_nshm_blend_bg', 'sig_0.35', 'max_trigs_10_grid50_migrate3_mtjprobe')
CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
TRIGGER_NUMBER = 4
MAX_LAT   = 42.2            # trim to just above the CA/OR border
BUF       = 1.0             # degrees around the posterior locations
SIZE_MULT = 1.5             # map marker scale
MIN_MAP_ASPECT = 8.4 / 7.3  # map box width/height
# (csv stem, label, color, map marker size scale); drawn in this order, sizes shrink so the dots read as concentric
PRIORS = [('uniform', 'Uniform', 'gray', 1.6), ('etas', 'ETAS', 'red', 1.3), ('etas_nshmblend', 'ETAS + NSHM blend', 'tab:green', 1.1)]
REFERENCE_PRIOR = 'Uniform'  # top histogram row
AX_POS = {'map': [0.06, 0.08, 0.60, 0.86], 'hists': [0.74, 0.183, 0.23, 0.654]}

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'figure.dpi': 150})
proj = ccrs.PlateCarree()

#%%
ref = load_reference_catalog(CATALOG)[['event_id', 'usgs_lat', 'usgs_lon']]
ref = ref[ref.usgs_lat <= MAX_LAT]
res = {}
for stem, label, _, _ in PRIORS:
    d = pd.read_csv(os.path.join(RES_DIR, f'{stem}_benchmark_results.csv'))
    d = d[d['n_trigs'] == TRIGGER_NUMBER].dropna(subset=['map_err_km', 'posterior_lat', 'posterior_lon'])
    res[label] = d.merge(ref, on='event_id')
for k, d in res.items():
    print(f'  {k:20s} {len(d)} events  median {d.map_err_km.median():.2f} km  mean {d.map_err_km.mean():.1f} km  >10 km: {(d.map_err_km > 10).sum()}')

all_lats = pd.concat([d.posterior_lat for d in res.values()]); all_lons = pd.concat([d.posterior_lon for d in res.values()])
extent = [all_lons.min() - BUF, all_lons.max() + BUF, all_lats.min() - BUF, min(all_lats.max() + BUF, MAX_LAT)]
if (extent[1] - extent[0]) < MIN_MAP_ASPECT * (extent[3] - extent[2]):
    half = 0.5 * MIN_MAP_ASPECT * (extent[3] - extent[2]); c = 0.5 * (extent[0] + extent[1])
    extent[0], extent[1] = c - half, c + half

#%%
def make_figure(res, extent, out_name):
    fig = plt.figure(figsize=(14, 8.5))

    # -- (b) stacked histograms -------------------------------------------------------------------
    bins = np.logspace(-1, 3, 20)
    labels = [p[1] for p in PRIORS]; colors = {p[1]: p[2] for p in PRIORS}
    order = [REFERENCE_PRIOR] + [l for l in labels if l != REFERENCE_PRIOR]
    hl, hb, hw, hh = AX_POS['hists']
    h_each = hh / len(order)
    axes_h = []
    for i, name in enumerate(order):
        ax = fig.add_axes([hl, hb + hh - (i + 1) * h_each, hw, h_each], sharex=axes_h[0] if axes_h else None)
        axes_h.append(ax)
        x = res[name]['map_err_km'].values; color = colors[name]
        ax.hist(x, bins=bins, rwidth=0.9, color=color, alpha=0.7)
        ax.axvline(np.median(x), ymax=0.55, color=color, ls='--', lw=1.5)
        ax.text(0.03, 0.70, f'median: {np.median(x):.1f} km', transform=ax.transAxes, ha='left', va='top', fontsize=11, color=color, fontweight='bold')
        ax.text(0.03, 0.92, name, transform=ax.transAxes, ha='left', va='top', fontsize=13, color=color, fontweight='bold')
        ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
        if i < len(order) - 1:
            ax.tick_params(labelbottom=False, bottom=False)
    axes_h[-1].set_xscale('log')
    axes_h[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axes_h[-1].set_xlabel('Location error (km)')
    ymax = max(a.get_ylim()[1] for a in axes_h)
    for a in axes_h:
        a.set_ylim(0, 2.0 * ymax)
    axes_h[0].text(-0.15, 0.87, '(b)', transform=axes_h[0].transAxes, ha='left', va='bottom', fontsize=13, fontweight='bold')

    # -- (a) map of posterior locations, coloured by prior -------------------------------------------
    ax_map = fig.add_axes(AX_POS['map'], projection=proj)
    ax_map.set_extent(extent, crs=proj)
    ax_map.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
    ax_map.add_feature(cfeature.OCEAN, facecolor='#d6eaf8', zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    ax_map.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray', zorder=1)
    for _, name, color, _ in PRIORS:
        m = res[name]
        ax_map.quiver(m.posterior_lon.values, m.posterior_lat.values, (m.usgs_lon - m.posterior_lon).values, (m.usgs_lat - m.posterior_lat).values,
                      angles='xy', scale_units='xy', scale=1, color=color, alpha=0.3, width=0.0025,
                      headwidth=3, headlength=3, headaxislength=4., transform=proj, zorder=4)
    for _, name, color, sz in PRIORS:
        m = res[name]
        ax_map.scatter(m.posterior_lon, m.posterior_lat, color=color, s=30 * sz * SIZE_MULT, alpha=0.5,
                       edgecolors='white', linewidths=0.4, transform=proj, zorder=5, label=name)
    true = pd.concat(res.values())[['event_id', 'usgs_lat', 'usgs_lon']].drop_duplicates('event_id')
    ax_map.scatter(true.usgs_lon, true.usgs_lat, marker='x', c='black', s=35 * SIZE_MULT, linewidths=1.1 * SIZE_MULT ** 0.5,
                   transform=proj, zorder=6, alpha=0.5, label='USGS (true)')

    ax_map.set_xticks(np.linspace(extent[0], extent[1], 5), crs=proj)
    ax_map.set_yticks(np.linspace(extent[2], extent[3], 5), crs=proj)
    ax_map.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
    ax_map.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
    ax_map.set_xlabel('Longitude'); ax_map.set_ylabel('Latitude')
    ax_map.text(0.98, 0.02, f'Posterior locations at {TRIGGER_NUMBER} triggers', transform=ax_map.transAxes, ha='right', va='bottom',
                fontsize=13, fontweight='bold', color='black', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
    ax_map.text(0.01, 0.98, '(a)', transform=ax_map.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')
    ax_map.legend(loc='lower left', fontsize=9, framealpha=0.9)

    fig.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{out_name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{out_name}.pdf'), bbox_inches='tight')
    plt.show()

#%%
make_figure(res, extent, 'location_hist_nshm_blend_bg')

#%%
# ── MTJ zoom: same figure, events whose USGS location is in the Mendocino Triple Junction window ──
ZOOM_EXTENT = [-126.0, -122.0, 38.4, 41.2]   # [lon0, lon1, lat0, lat1]
in_zoom = ref[ref.usgs_lon.between(*ZOOM_EXTENT[:2]) & ref.usgs_lat.between(*ZOOM_EXTENT[2:])].event_id
res_zoom = {k: d[d.event_id.isin(in_zoom)] for k, d in res.items()}
for k, d in res_zoom.items():
    print(f'  MTJ {k:20s} {len(d)} events  median {d.map_err_km.median():.2f} km  mean {d.map_err_km.mean():.1f} km  >50 km: {(d.map_err_km > 50).sum()}')
make_figure(res_zoom, ZOOM_EXTENT, 'location_hist_nshm_blend_bg_MTJ')

# %%
