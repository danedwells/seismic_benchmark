#%%
# =============================================================================
# location_hist_case_study.py — paper figure: location-error histograms +
# combined location map for ONE case study (default: 2024 Mendocino Triple
# Junction M7)
# =============================================================================
# Case-study figures (MTJ_2024_M7, Ridgecrest, ElMayor) and the large-events figure
# ('large_events', M>=5 California catalog) share this one script and style; pass the
# name as the first argument. The output filename follows it.
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
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config — must match the case_studies.py runs these CSVs came from.
# ---------------------------------------------------------------------------
# Usage: python location_hist_case_study.py [MTJ_2024_M7 | Ridgecrest | ElMayor | large_events | benchmark | catalog [MIN_MAG]]
#   catalog = the full California benchmark catalog (results/california/output), filtered to USGS M >= MIN_MAG (default 4)
CASES = {
    'MTJ_2024_M7':  dict(mainshock='nc75095651', catalog='mtj_2024_m7.csv', min_mag=4.0, ns_pad=0.5),
    'Ridgecrest':   dict(mainshock='ci38457511', catalog='Ridgecrest_2019_catalog.parquet', min_mag=4.0, ns_pad=0),
    'ElMayor':      dict(mainshock='ci14607652', catalog='El_Mayor-Cucapah_2010_catalog.parquet', min_mag=4.0, ns_pad=0),
    # all M>=5 events in the California benchmark catalog (no single mainshock; the results are already M>=5)
    'large_events': dict(mainshock=None, results_dir='M5', min_mag=None, ns_pad=0, inset=False, buf=1.0),
    # the entire benchmark, no magnitude cut
    'benchmark':    dict(mainshock=None, results_dir='', min_mag=None, ns_pad=0, inset=False, buf=1.0, max_lat=42.2, size_mult=1.5),
    # the whole catalog with a magnitude threshold: python location_hist_case_study.py catalog 4.5
    'catalog':      dict(mainshock=None, results_dir='', min_mag=float(sys.argv[2]) if len(sys.argv) > 2 else 4.0,
                         ns_pad=0, inset=False, buf=1.0, max_lat=42.2, size_mult=1.5),   # trimmed to just above the CA/OR border
}
CASE_STUDY      = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in CASES else 'MTJ_2024_M7'
MAINSHOCK_ID    = CASES[CASE_STUDY]['mainshock']   # highlighted on the map (USGS location + each prior's estimate)
MIN_USGS_MAG    = CASES[CASE_STUDY]['min_mag']     # only plot events with USGS magnitude >= this (None = all)
MAX_TRIGS       = 10
TRIGGER_NUMBER  = 4              # trigger count at which locations are compared
REFERENCE_PRIOR = 'Uniform'  # top histogram row
MAP_NS_PAD      = CASES[CASE_STUDY]['ns_pad']   # extra north-south map extent, as a fraction of the data's latitude span (each side)
MIN_MAP_ASPECT  = 8.4 / 7.3      # map box width/height; tall regions are widened east-west to fill it (axes are equal-aspect in degrees)

# Inset locator map: [left, bottom, width, height] in map-axes fraction.
# Move it per case study so it sits where there are no arrows.
INSET_POS    = [0.74, 0.72, 0.24, 0.26]
INSET_EXTENT = [-128, -113.5, 30, 42.5]   # California region [lon0, lon1, lat0, lat1]

column_lat = 'posterior_lat'
column_lon = 'posterior_lon'
column_err = 'map_err_km'

if CASE_STUDY in ('large_events', 'catalog', 'benchmark'):
    CS_DIR       = os.path.join(PROJECT_ROOT, 'results', 'california', CASES[CASE_STUDY]['results_dir'])
    CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
else:
    CS_DIR       = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', CASE_STUDY)
    CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', CASE_STUDY,
                                CASES[CASE_STUDY]['catalog'])   # USGS ComCat download (id, latitude, longitude, mag, ...)
OUTPUT_DIR_STATIC  = os.path.join(CS_DIR, 'output', 'time_independent', f'max_trigs_{MAX_TRIGS}')
OUTPUT_DIR_DYNAMIC = os.path.join(CS_DIR, 'output', 'time_dependent',   f'max_trigs_{MAX_TRIGS}')

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
COLOR_LOOKUP = {'KDE_Seismicity': 'darkblue', 'Uniform': 'gray', 'ETAS': 'red'}

# Map marker size multiplier per prior. Priors often land on nearly the same
# spot, and they're drawn in PRIOR_SPECS order, so sizes shrink with draw order:
# the result reads as concentric dots instead of ETAS hiding the others.
SIZE_SCALE = {'KDE_Seismicity': 1.6, 'Uniform': 1.3, 'ETAS': 1.1}

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

if CASE_STUDY in ('large_events', 'catalog', 'benchmark'):
    from benchmark.runner import load_reference_catalog
    ref_catalog = load_reference_catalog(CATALOG_PATH)[['event_id', 'usgs_lat', 'usgs_lon', 'usgs_mag']].rename(columns={'usgs_mag': 'mag'})
else:
    _read_catalog = pd.read_parquet if CATALOG_PATH.endswith('.parquet') else pd.read_csv
    ref_catalog = _read_catalog(CATALOG_PATH).rename(
        columns={'id': 'event_id', 'latitude': 'usgs_lat', 'longitude': 'usgs_lon'}
    )[['event_id', 'usgs_lat', 'usgs_lon', 'mag']].copy()
ref_catalog['event_id'] = ref_catalog['event_id'].astype(str)

# Optional magnitude cutoff (USGS magnitude), applied to every prior's results and the map
if MIN_USGS_MAG is not None:
    keep_ids = set(ref_catalog.loc[ref_catalog['mag'] >= MIN_USGS_MAG, 'event_id'])
    loaded = {n: d[d['event_id'].isin(keep_ids)] for n, d in loaded.items()}
    ref_catalog = ref_catalog[ref_catalog['event_id'].isin(keep_ids)]

# Optional northern cutoff (USGS latitude), e.g. to trim the full catalog to California
MAX_LAT = CASES[CASE_STUDY].get('max_lat')
if MAX_LAT is not None:
    keep_ids = set(ref_catalog.loc[ref_catalog['usgs_lat'] <= MAX_LAT, 'event_id'])
    loaded = {n: d[d['event_id'].isin(keep_ids)] for n, d in loaded.items()}
    ref_catalog = ref_catalog[ref_catalog['event_id'].isin(keep_ids)]
SIZE_MULT = CASES[CASE_STUDY].get('size_mult', 1.0)   # scales the map markers

def errors_at_trigger_number(name):
    """Location errors (one per event) at TRIGGER_NUMBER for a prior, or None if unavailable."""
    if name not in loaded:
        return None
    vals = loaded[name].query('n_trigs == @TRIGGER_NUMBER')[column_err].dropna().values
    return vals if len(vals) else None

#%%
# ── Map data and extent ──────────────────────────────────────────────────
proj = ccrs.PlateCarree()
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
buf = CASES[CASE_STUDY].get('buf', -0.02)   # degrees added around the posterior locations
lat_pad = MAP_NS_PAD * (all_lats.max() - all_lats.min())   # extra north-south room -> taller map
extent = [all_lons.min() - buf, all_lons.max() + buf,
          all_lats.min() - buf - lat_pad, all_lats.max() + buf + lat_pad]
if MAX_LAT is not None:
    extent[3] = min(extent[3], MAX_LAT)
# Tall regions: widen the map east-west to fill the map box
if (extent[1] - extent[0]) < MIN_MAP_ASPECT * (extent[3] - extent[2]):
    _half = 0.5 * MIN_MAP_ASPECT * (extent[3] - extent[2])
    _lon_c = 0.5 * (extent[0] + extent[1])
    extent[0], extent[1] = _lon_c - _half, _lon_c + _half


# ── Build the figure ──────────────────────────────────────────────────────
# Exact per-panel placement [left, bottom, width, height] (figure fraction).
# Edit these directly to hand-tune the layout. The map box is 8.4 x 7.3 in (aspect 1.15).
AX_POS = {
    'map':   [0.06, 0.08, 0.60, 0.86],
    'hists': [0.74, 0.183, 0.23, 0.654],   # box shared by the three stacked histograms
}

fig = plt.figure(figsize=(14, 8.5))

# -- Right: stacked location-error histograms, one per prior, shared x-axis ---
# Reference prior on top, then the rest in PRIOR_SPECS order. Colors match the map.
bins = np.logspace(-1, 3, 20)
hist_order = [REFERENCE_PRIOR] + [s['name'] for s in PRIOR_SPECS if s['name'] != REFERENCE_PRIOR]

hl, hb, hw, hh = AX_POS['hists']
h_each = hh / len(hist_order)
hist_axes = []
for i, name in enumerate(hist_order):
    ax = fig.add_axes([hl, hb + hh - (i + 1) * h_each, hw, h_each],
                      sharex=hist_axes[0] if hist_axes else None)
    hist_axes.append(ax)
    stats = errors_at_trigger_number(name)
    color = COLOR_LOOKUP[name]

    if stats is not None:
        ax.hist(stats, bins=bins, rwidth=0.9, color=color, alpha=0.7)
        median_err = np.median(stats)
        ax.axvline(median_err, ymax=0.55, color=color, ls='--', lw=1.5)   # stops above the bars, below the labels
        ax.text(0.03, 0.70, f'median: {median_err:.1f} km', transform=ax.transAxes,
                ha='left', va='top', fontsize=11, color=color, fontweight='bold')
    ax.text(0.03, 0.92, name, transform=ax.transAxes, ha='left', va='top',
            fontsize=13, color=color, fontweight='bold')

    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)
    if i < len(hist_order) - 1:
        ax.tick_params(labelbottom=False, bottom=False)

hist_axes[-1].set_xscale('log')
hist_axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
hist_axes[-1].set_xlabel('Location error (km)')

# Same vertical scale in every row so bar heights are comparable
_y_max = max(ax.get_ylim()[1] for ax in hist_axes)
for ax in hist_axes:
    ax.set_ylim(0, 2.0 * _y_max)   # headroom: bars top out at half height so the name/median labels sit clear above them
hist_axes[0].text(-0.15, 0.87, '(b)', transform=hist_axes[0].transAxes,
                  ha='left', va='bottom', fontsize=13, fontweight='bold')

# -- Bottom: combined map of posterior locations, colored by prior -----------
ax_map = fig.add_axes(AX_POS['map'], projection=proj)

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
    # Mainshock arrow drawn thicker and in front of the rest.
    for is_main, width, alpha, z in ((False, 0.0025, 0.3, 4), (True, 0.004, 0.8, 6)):
        m = matched[(matched['event_id'] == MAINSHOCK_ID) == is_main]
        if m.empty:
            continue
        ax_map.quiver(m[column_lon].values, m[column_lat].values,
                      (m['usgs_lon'] - m[column_lon]).values,
                      (m['usgs_lat'] - m[column_lat]).values,
                      angles='xy', scale_units='xy', scale=1,
                      color=COLOR_LOOKUP[name], alpha=alpha, width=width,
                      headwidth=3, headlength=3, headaxislength=4.,
                      transform=proj, zorder=z)

for name, sub in map_data.items():
    ax_map.scatter(sub[column_lon], sub[column_lat],
                    color=COLOR_LOOKUP[name], s=30 * SIZE_SCALE[name] * SIZE_MULT, alpha=0.5,
                    edgecolors='white', linewidths=0.4,
                    transform=proj, zorder=5, label=name)

# USGS reference (true) locations — one marker per event, deduplicated across
# priors since the true location doesn't depend on which prior located it.
_all_event_ids = set().union(*(set(sub['event_id']) for sub in map_data.values()))
usgs_pts = ref_catalog[ref_catalog['event_id'].isin(_all_event_ids)]
aftershocks = usgs_pts[usgs_pts['event_id'] != MAINSHOCK_ID]
ax_map.scatter(aftershocks['usgs_lon'], aftershocks['usgs_lat'],
                marker='x', c='black', s=35 * SIZE_MULT, linewidths=1.1 * SIZE_MULT ** 0.5,
                transform=proj, zorder=6, alpha=0.5, label='USGS (true)')

# Mainshock: each prior's estimate (larger, bolder, in front) + USGS location
# as a red star.
for name, sub in map_data.items():
    ms = sub[sub['event_id'] == MAINSHOCK_ID]
    ax_map.scatter(ms[column_lon], ms[column_lat],
                    color=COLOR_LOOKUP[name], s=350, alpha=1.0,
                    edgecolors='black', linewidths=1.0, marker='*',
                    transform=proj, zorder=8)
ms_true = usgs_pts[usgs_pts['event_id'] == MAINSHOCK_ID]
ax_map.scatter(ms_true['usgs_lon'], ms_true['usgs_lat'],
                marker='*', c='yellow', s=400, edgecolors='black', linewidths=0.8,
                transform=proj, zorder=9, label='USGS mainshock (true)' if MAINSHOCK_ID else None)

_lon_ticks = np.linspace(extent[0], extent[1], 5)
_lat_ticks = np.linspace(extent[2], extent[3], 5)
ax_map.set_xticks(_lon_ticks, crs=proj)
ax_map.set_yticks(_lat_ticks, crs=proj)
ax_map.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
ax_map.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
ax_map.set_xlabel('Longitude')
ax_map.set_ylabel('Latitude')
mag_note = f', USGS M >= {MIN_USGS_MAG}' if MIN_USGS_MAG is not None else ''
ax_map.text(0.98, 0.02, f'Posterior locations at {TRIGGER_NUMBER} triggers', transform=ax_map.transAxes,
            ha='right', va='bottom', fontsize=13, fontweight='bold', color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'))
ax_map.text(0.01, 0.98, '(a)', transform=ax_map.transAxes,
            ha='left', va='top', fontsize=13, fontweight='bold')
ax_map.legend(loc='lower left', fontsize=9, framealpha=0.9)

# -- Inset: where the map extent sits within California (skipped when the map already covers it) --
if CASES[CASE_STUDY].get('inset', True):
    ax_in = ax_map.inset_axes(INSET_POS, projection=proj)
    ax_in.set_extent(INSET_EXTENT, crs=proj)
    ax_in.add_feature(cfeature.LAND,      facecolor='#f0f0f0')
    ax_in.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8')
    ax_in.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax_in.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray')
    ax_in.plot([extent[0], extent[1], extent[1], extent[0], extent[0]],
               [extent[2], extent[2], extent[3], extent[3], extent[2]],
               color='red', linewidth=1.2, transform=proj)
    ax_in.set_zorder(10)

out_name = f'location_hist_{CASE_STUDY}' + (f'_M{MIN_USGS_MAG:g}' if CASE_STUDY == 'catalog' else '')
fig.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{out_name}.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'figures', f'{out_name}.pdf'), bbox_inches='tight')
plt.show()

#%%
# ── Second figure (benchmark case only): zoom on the Mendocino Triple Junction ──
# Extends southeast far enough to include ETAS's high-error events, with the ETAS free_background
# spatial prior shaded behind the points (cached by prior_comparison.py, panel (c)).
if CASE_STUDY == 'benchmark':
    from matplotlib.colors import LogNorm
    ZOOM_EXTENT  = [-126.0, -122.0, 38.4, 41.2]   # [lon0, lon1, lat0, lat1]
    ETAS_BG_CACHE = os.path.join(OUTPUT_DIR, '.cache', 'row1.pkl')

    fig2 = plt.figure(figsize=(9, 8))
    ax2 = fig2.add_axes([0.09, 0.08, 0.88, 0.86], projection=proj)
    ax2.set_extent(ZOOM_EXTENT, crs=proj)
    ax2.add_feature(cfeature.LAND,      facecolor='#f0f0f0', zorder=0)
    ax2.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
    ax2.add_feature(cfeature.STATES,    linewidth=0.4, edgecolor='gray', zorder=1)

    with open(ETAS_BG_CACHE, 'rb') as fh:
        bg = pickle.load(fh)[2]      # (kde_prior, prior_flat, prior_spatial, ...)
    lon_m = (bg.lons >= ZOOM_EXTENT[0] - 0.5) & (bg.lons <= ZOOM_EXTENT[1] + 0.5)
    lat_m = (bg.lats >= ZOOM_EXTENT[2] - 0.5) & (bg.lats <= ZOOM_EXTENT[3] + 0.5)
    grid = bg.grid[np.ix_(lat_m, lon_m)].astype(float)
    pos = grid[grid > 0]
    ax2.pcolormesh(*np.meshgrid(bg.lons[lon_m], bg.lats[lat_m]), grid, transform=proj,
                          cmap='YlOrBr', shading='auto', alpha=0.6, zorder=2,
                          norm=LogNorm(vmin=np.percentile(pos, 5), vmax=np.percentile(pos, 99)))

    in_zoom = lambda lon, lat: ((lon >= ZOOM_EXTENT[0]) & (lon <= ZOOM_EXTENT[1]) &
                                (lat >= ZOOM_EXTENT[2]) & (lat <= ZOOM_EXTENT[3]))
    # Keep events whose USGS location is in the zoom window; arrows may leave it (clipped by the axes)
    zoom_ids = set(ref_catalog.loc[in_zoom(ref_catalog['usgs_lon'], ref_catalog['usgs_lat']), 'event_id'])
    for name, sub in map_data.items():
        m = sub[sub['event_id'].isin(zoom_ids)].merge(ref_catalog, on='event_id')
        ax2.quiver(m[column_lon].values, m[column_lat].values,
                   (m['usgs_lon'] - m[column_lon]).values, (m['usgs_lat'] - m[column_lat]).values,
                   angles='xy', scale_units='xy', scale=1, color=COLOR_LOOKUP[name], alpha=0.35,
                   width=0.0025, headwidth=3, headlength=3, headaxislength=4., transform=proj, zorder=4)
        ax2.scatter(m[column_lon], m[column_lat], color=COLOR_LOOKUP[name],
                    s=45 * SIZE_SCALE[name], alpha=0.55, edgecolors='white', linewidths=0.4,
                    transform=proj, zorder=5, label=name)
    zoom_true = ref_catalog[ref_catalog['event_id'].isin(zoom_ids)]
    ax2.scatter(zoom_true['usgs_lon'], zoom_true['usgs_lat'], marker='x', c='black', s=40,
                linewidths=1.1, alpha=0.6, transform=proj, zorder=6, label='USGS (true)')

    ax2.set_xticks(np.linspace(ZOOM_EXTENT[0], ZOOM_EXTENT[1], 5), crs=proj)
    ax2.set_yticks(np.linspace(ZOOM_EXTENT[2], ZOOM_EXTENT[3], 5), crs=proj)
    ax2.xaxis.set_major_formatter(LongitudeFormatter(number_format='.2f'))
    ax2.yaxis.set_major_formatter(LatitudeFormatter(number_format='.2f'))
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.9)

    fig2.savefig(os.path.join(OUTPUT_DIR, 'figures', 'location_hist_benchmark_MTJ.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(OUTPUT_DIR, 'figures', 'location_hist_benchmark_MTJ.pdf'), bbox_inches='tight')
    plt.show()

# %%
