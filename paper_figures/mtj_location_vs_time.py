#%%
# =============================================================================
# mtj_location_vs_time.py — paper figure: location error vs. time for the
# 2024 Mendocino Triple Junction M7 mainshock
# =============================================================================
# (a) Epicentral (MAP location) error of each prior vs. seconds since origin
#     time, one point per trigger version.
# (b) Map of the catalog epicenter, stations available at event time, and
#     stations used (the first N_USED triggers, i.e. the versions in (a)).
# (c) Placeholder for station magnitude vs. epicentral distance — no magnitude
#     results yet.
#
# Modeled on Fig. 7 (Anchorage) of the EPIC magnitude paper. Panel positions
# are set explicitly via AX_POS so the layout can be hand-tuned directly.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from benchmark.runner import load_station_availability_cache

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config — must match the case_studies.py runs these CSVs came from.
# ---------------------------------------------------------------------------
CASE_STUDY   = 'MTJ_2024_M7'
EVENT_ID     = 'nc75095651'   # the M7 mainshock
MAX_TRIGS    = 10
N_USED       = 10             # stations from the first N_USED triggers are "used" in (b)

column_err = 'map_err_km'

CS_DIR             = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', CASE_STUDY)
OUTPUT_DIR_STATIC  = os.path.join(CS_DIR, 'output', 'time_independent', f'max_trigs_{MAX_TRIGS}')
OUTPUT_DIR_DYNAMIC = os.path.join(CS_DIR, 'output', 'time_dependent',   f'max_trigs_{MAX_TRIGS}')
DATA_DIR           = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', CASE_STUDY)
CATALOG_PATH       = os.path.join(DATA_DIR, 'mtj_2024_m7.csv')
RUN_PATH           = os.path.join(DATA_DIR, 'run_files', f'{EVENT_ID}.run')
AVAIL_PATH         = os.path.join(DATA_DIR, 'station_availability_cache.parquet')

# Same prior order / colors as location_hist_case_study.py
PRIOR_SPECS = [
    {'name': name, 'csv': os.path.join(OUTPUT_DIR_STATIC, f'{name.lower()}_benchmark_results.csv')}
    for name in ('KDE_Seismicity', 'Uniform')
] + [
    {'name': 'ETAS', 'csv': os.path.join(OUTPUT_DIR_DYNAMIC, 'etas_dynamic_benchmark_results.csv')}
]
_colors = plt.cm.tab10.colors
COLOR_LOOKUP = {spec['name']: _colors[i % len(_colors)] for i, spec in enumerate(PRIOR_SPECS)}

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
})

#%%
# ── Load catalog, trigger times, errors, stations ───────────────────────────
catalog = pd.read_csv(CATALOG_PATH)
event   = catalog[catalog['id'] == EVENT_ID].iloc[0]
origin_epoch = pd.Timestamp(event['time']).timestamp()
epi_lat, epi_lon = event['latitude'], event['longitude']

run_df = pd.read_csv(RUN_PATH)
run_df.columns = [c.replace(' ', '_') for c in run_df.columns]

# Version v's time = arrival of its newest trigger
version_time = run_df.groupby('version')['trigger_time'].max() - origin_epoch

errors = {}
for spec in PRIOR_SPECS:
    df = pd.read_csv(spec['csv'])
    df = df[df['event_id'] == EVENT_ID].sort_values('version')
    errors[spec['name']] = (version_time.reindex(df['version']).values, df[column_err].values)

used = run_df[run_df['version'] == N_USED - 1].drop_duplicates(subset='station')
available = load_station_availability_cache(AVAIL_PATH)[EVENT_ID]

#%%
# ── Build the figure ──────────────────────────────────────────────────────
# Exact per-panel placement [left, bottom, width, height] (figure fraction).
# Edit these directly to hand-tune the layout.
AX_POS = {
    'time': [0.08, 0.58, 0.88, 0.34],
    'map':  [0.06, 0.06, 0.44, 0.40],
    'mag':  [0.60, 0.06, 0.36, 0.40],
}

fig = plt.figure(figsize=(11, 9))

# -- (a) error vs. time -------------------------------------------------------
ax = fig.add_axes(AX_POS['time'])
for name, (t, err) in errors.items():
    ax.plot(t, err, '-o', color=COLOR_LOOKUP[name], lw=2.5, ms=7, alpha=0.8, label=name)
ax.set_xlabel('Seconds since origin time')
ax.set_ylabel('Epicentral error (km)')
ax.set_title(f"{pd.Timestamp(event['time']):%-d %B %Y} M {event['mag']:.1f} Mendocino Triple Junction earthquake")
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.text(0.01, 0.97, '(a)', transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')

# -- (b) stations map ---------------------------------------------------------
proj = ccrs.PlateCarree()
ax_map = fig.add_axes(AX_POS['map'], projection=proj)

all_lons = np.concatenate([available['longitude'], used['longitude'], [epi_lon]])
all_lats = np.concatenate([available['latitude'],  used['latitude'],  [epi_lat]])
buf = 0.5
extent = [all_lons.min() - buf, all_lons.max() + buf, all_lats.min() - buf, all_lats.max() + buf]

ax_map.set_extent(extent, crs=proj)
ax_map.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
ax_map.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
ax_map.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
ax_map.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)

ax_map.scatter(available['longitude'], available['latitude'], marker='^', s=30,
               facecolors='none', edgecolors='k', linewidths=0.8,
               transform=proj, zorder=4, label='Available station')
ax_map.scatter(used['longitude'], used['latitude'], marker='^', s=40,
               color='steelblue', edgecolors='k', linewidths=0.6,
               transform=proj, zorder=5, label='Station used')
ax_map.scatter([epi_lon], [epi_lat], marker='*', s=200, facecolors='none',
               edgecolors='k', linewidths=1.2, transform=proj, zorder=6, label='Catalog epicenter')

ax_map.set_xticks(np.linspace(extent[0], extent[1], 4), crs=proj)
ax_map.set_yticks(np.linspace(extent[2], extent[3], 4), crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
ax_map.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
ax_map.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax_map.text(0.02, 0.98, '(b)', transform=ax_map.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')

# -- (c) magnitude placeholder ------------------------------------------------
ax_mag = fig.add_axes(AX_POS['mag'])
ax_mag.axhline(0, color='gray', lw=1.5)
ax_mag.set_xlim(0, 200)
ax_mag.set_ylim(-1.5, 1.5)
ax_mag.set_xlabel('Station epicentral distance (km)')
ax_mag.set_ylabel('Station magnitude − catalog magnitude')
ax_mag.text(0.5, 0.7, 'Magnitude results pending', transform=ax_mag.transAxes,
            ha='center', va='center', fontsize=12, color='gray')
ax_mag.text(0.02, 0.98, '(c)', transform=ax_mag.transAxes, ha='left', va='top', fontsize=13, fontweight='bold')

fig.savefig(os.path.join(OUTPUT_DIR, 'mtj_location_vs_time.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'mtj_location_vs_time.pdf'), bbox_inches='tight')
plt.show()
