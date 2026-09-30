#%%
# =============================================================================
# location_hist_case_study.py — paper figure: location-error histograms +
# combined location map for ONE case study (default: 2024 Mendocino Triple
# Junction M7)
# =============================================================================
# Same figure as location_hist_large_events.py, but fed from a case-study run
# (results/california/case_studies/{CASE_STUDY}/) instead of the large-event
# catalog. Change CASE_STUDY below; the output filename follows it.
#
# NOTE: the rest of this header describes the original large-events figure.
# =============================================================================
# Top row (3 panels): per-prior histogram of posterior (MAP) location error at
# a fixed trigger count, each overlaid with the reference prior's distribution
# in blue — adapted from plot_scripts/california_large_events_plot_comparison.py,
# "Figure 8" (the logic behind hist_location_error_4_map_ref_kde_seismicity.png).
#
# Bottom row (1 panel): posterior locations of all three priors combined on a
# single map, colored by PRIOR IDENTITY rather than by location-error bin —
# adapted from that same script's "Figure 9" (error_map_4trigs_map.png), which
# instead drew one map per prior colored by error magnitude.
#
# As in the other paper_figures/ scripts, every panel's position is set
# explicitly via AX_POS so the layout can be hand-tuned directly.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from benchmark.metrics import load_final_values

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config — must match the case_studies.py runs these CSVs came from.
# ---------------------------------------------------------------------------
CASE_STUDY      = 'MTJ_2024_M7'
MAINSHOCK_ID    = 'nc75095651'   # highlighted on the map (USGS location + each prior's estimate)
MAX_TRIGS       = 10
TRIGGER_NUMBER  = 4              # trigger count at which locations are compared
REFERENCE_PRIOR = 'KDE_Seismicity'  # shown in blue in every histogram panel

column_lat = 'posterior_lat'
column_lon = 'posterior_lon'
column_err = 'map_err_km'

CS_DIR             = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', CASE_STUDY)
OUTPUT_DIR_STATIC  = os.path.join(CS_DIR, 'output', 'time_independent', f'max_trigs_{MAX_TRIGS}')
OUTPUT_DIR_DYNAMIC = os.path.join(CS_DIR, 'output', 'time_dependent',   f'max_trigs_{MAX_TRIGS}')
CATALOG_PATH       = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', CASE_STUDY,
                                  'mtj_2024_m7.csv')   # USGS ComCat download (id, latitude, longitude, ...)

PRIOR_SPECS = [
    {'name': name, 'csv': os.path.join(OUTPUT_DIR_STATIC, f'{name.lower()}_benchmark_results.csv')}
    for name in ('KDE_Seismicity', 'Uniform')
] + [
    {'name': 'ETAS', 'csv': os.path.join(OUTPUT_DIR_DYNAMIC, 'etas_dynamic_benchmark_results.csv')}
]

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
})

# Consistent prior -> color mapping, shared by the histograms (reference
# overlay excepted) and the combined map.
_colors = plt.cm.tab10.colors
COLOR_LOOKUP = {spec['name']: _colors[i % len(_colors)] for i, spec in enumerate(PRIOR_SPECS)}

#%%
# ── Load benchmark results and reference catalog ────────────────────────────
loaded = {}
for spec in PRIOR_SPECS:
    if not os.path.exists(spec['csv']):
        print(f"  [{spec['name']}] CSV not found — skipping: {spec['csv']}")
        continue
    df = pd.read_csv(spec['csv'])
    df['event_id'] = df['event_id'].astype(str)
    loaded[spec['name']] = df

ref_catalog = pd.read_csv(CATALOG_PATH).rename(
    columns={'id': 'event_id', 'latitude': 'usgs_lat', 'longitude': 'usgs_lon'}
)[['event_id', 'usgs_lat', 'usgs_lon']].copy()
ref_catalog['event_id'] = ref_catalog['event_id'].astype(str)

#%%
# ── Build the figure ──────────────────────────────────────────────────────
# Exact per-panel placement [left, bottom, width, height] (figure fraction).
# Edit these directly to hand-tune the layout.
AX_POS = {
    'hist_kde':     [0.06, 0.66, 0.27, 0.24],
    'hist_uniform': [0.38, 0.66, 0.27, 0.24],
    'hist_etas':    [0.70, 0.66, 0.27, 0.24],
    'map':          [0.08, 0.06, 0.84, 0.51],
}

fig = plt.figure(figsize=(11, 11))

# -- Top row: per-prior location-error histograms -----------------------------
bins = np.logspace(-1, 3, 20)
_ref_csv = next(s['csv'] for s in PRIOR_SPECS if s['name'] == REFERENCE_PRIOR)
ref_stats = load_final_values(_ref_csv, column_err, n_trigs=TRIGGER_NUMBER)

_hist_keys = {'KDE_Seismicity': 'hist_kde', 'Uniform': 'hist_uniform', 'ETAS': 'hist_etas'}
_letters   = {'KDE_Seismicity': 'a', 'Uniform': 'b', 'ETAS': 'c'}

for spec in PRIOR_SPECS:
    name = spec['name']
    ax = fig.add_axes(AX_POS[_hist_keys[name]])
    stats = load_final_values(spec['csv'], column_err, n_trigs=TRIGGER_NUMBER)

    if ref_stats is not None:
        ax.hist(ref_stats, bins=bins, rwidth=0.9, color='b', alpha=0.4, label=f'{REFERENCE_PRIOR}  (ref)')
    if stats is not None:
        ax.hist(stats, bins=bins, rwidth=0.9, color='r', alpha=0.6)

    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.97, f'({_letters[name]})', transform=ax.transAxes,
            ha='left', va='top', fontsize=13, fontweight='bold')

    if stats is not None and len(stats) > 0:
        median_err = np.median(stats)
        counts, edges = np.histogram(stats, bins=bins)
        mode_lo, mode_hi = edges[np.argmax(counts)], edges[np.argmax(counts) + 1]
        ax.text(0.95, 0.95, f"median: {median_err:.1f} km\nmode: {mode_lo:.1f}–{mode_hi:.1f} km",
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel('Location error (km)')
    if name == 'KDE_Seismicity':
        ax.set_ylabel('Event count')
        ax.legend(loc='lower left', fontsize=9)
    else:
        ax.set_yticklabels([])

# Consistent y-axis across the three histogram panels
_hist_axes = [fig.axes[i] for i in range(3)]
_y_max = max(ax.get_ylim()[1] for ax in _hist_axes)
for ax in _hist_axes:
    ax.set_ylim(0, _y_max)

# -- Bottom: combined map of posterior locations, colored by prior -----------
proj = ccrs.PlateCarree()
ax_map = fig.add_axes(AX_POS['map'], projection=proj)

map_data = {}
for spec in PRIOR_SPECS:
    name = spec['name']
    if name not in loaded:
        continue
    df = loaded[name]
    sub = df[df['n_trigs'] == TRIGGER_NUMBER][['event_id', column_lat, column_lon]].dropna()
    if len(sub) == 0:
        continue
    map_data[name] = sub

all_lats = pd.concat([d[column_lat] for d in map_data.values()])
all_lons = pd.concat([d[column_lon] for d in map_data.values()])
buf = 0.1
extent = [all_lons.min() - buf, all_lons.max() + buf,
          all_lats.min() - buf, all_lats.max() + buf]

ax_map.set_extent(extent, crs=proj)
ax_map.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
ax_map.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
ax_map.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)

# Arrows from each prior's posterior location to the true USGS location,
# colored to match that prior's scatter points.
for name, sub in map_data.items():
    matched = sub.merge(ref_catalog, on='event_id', how='inner')
    if matched.empty:
        continue
    ax_map.quiver(matched[column_lon].values, matched[column_lat].values,
                  (matched['usgs_lon'] - matched[column_lon]).values,
                  (matched['usgs_lat'] - matched[column_lat]).values,
                  angles='xy', scale_units='xy', scale=1,
                  color=COLOR_LOOKUP[name], alpha=0.5, width=0.004,
                  headwidth=5, headlength=5, headaxislength=5.,
                  transform=proj, zorder=4)

for name, sub in map_data.items():
    ax_map.scatter(sub[column_lon], sub[column_lat],
                    color=COLOR_LOOKUP[name], s=30, alpha=0.65,
                    edgecolors='white', linewidths=0.4,
                    transform=proj, zorder=5, label=name)

# USGS reference (true) locations — one marker per event, deduplicated across
# priors since the true location doesn't depend on which prior located it.
_all_event_ids = set().union(*(set(sub['event_id']) for sub in map_data.values()))
usgs_pts = ref_catalog[ref_catalog['event_id'].isin(_all_event_ids)]
aftershocks = usgs_pts[usgs_pts['event_id'] != MAINSHOCK_ID]
ax_map.scatter(aftershocks['usgs_lon'], aftershocks['usgs_lat'],
                marker='x', c='black', s=35, linewidths=1.1,
                transform=proj, zorder=6, alpha=0.3, label='USGS (true)')

# Mainshock: each prior's estimate (larger, bolder, in front) + USGS location
# as a red star.
for name, sub in map_data.items():
    ms = sub[sub['event_id'] == MAINSHOCK_ID]
    ax_map.scatter(ms[column_lon], ms[column_lat],
                    color=COLOR_LOOKUP[name], s=90, alpha=1.0,
                    edgecolors='black', linewidths=1.0,
                    transform=proj, zorder=8)
ms_true = usgs_pts[usgs_pts['event_id'] == MAINSHOCK_ID]
ax_map.scatter(ms_true['usgs_lon'], ms_true['usgs_lat'],
                marker='*', c='red', s=200, edgecolors='black', linewidths=0.8,
                transform=proj, zorder=9, label='USGS mainshock (true)')

_lon_ticks = np.linspace(extent[0], extent[1], 5)
_lat_ticks = np.linspace(extent[2], extent[3], 5)
ax_map.set_xticks(_lon_ticks, crs=proj)
ax_map.set_yticks(_lat_ticks, crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
ax_map.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
ax_map.set_xlabel('Longitude')
ax_map.set_ylabel('Latitude')
ax_map.set_title(f'Posterior locations at {TRIGGER_NUMBER} triggers — all priors')
ax_map.text(0.01, 0.98, '(d)', transform=ax_map.transAxes,
            ha='left', va='top', fontsize=13, fontweight='bold')
ax_map.legend(loc='lower left', fontsize=9, framealpha=0.9)

fig.savefig(os.path.join(OUTPUT_DIR, f'location_hist_{CASE_STUDY}.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, f'location_hist_{CASE_STUDY}.pdf'), bbox_inches='tight')
plt.show()

# %%
