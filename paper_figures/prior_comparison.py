#%%
# =============================================================================
# prior_comparison.py — paper figure: KDE seismicity vs ETAS flat/spatial mu
# =============================================================================
# Clean, publication-ready comparison of the spatial priors evaluated in the
# benchmark:
#
# Row 1 — statewide priors at the benchmark timewindow_end:
#   (a) KDE Seismicity        — static kernel-density prior
#   (b) ETAS — flat background   — ETAS prior with a single scalar mu
#   (c) ETAS — spatial background — ETAS prior with free_background's
#                                   locally-smoothed mu(x, y)
#
# Row 2 — zoomed on the 2024 Mendocino Triple Junction M7, illustrating the
# effect of the ETAS mathematics directly:
#   (d) ETAS prior immediately before the mainshock
#   (e) ETAS prior immediately after the mainshock (once it has been fed into
#       the rolling catalog) — shows the sharp, localized excitation the
#       triggering term adds around the new event.
#
# Row 1 adapted from plot_scripts/spatial_free_bg_productivity.py (cell 4),
# dropping the diagnostic log-ratio panel and adding the KDE_Seismicity panel.
# Row 2 uses the same MTJ_2024_M7 case-study inversion consumed by
# time_dependent_scripts/case_studies.py.
#
# Usage: run cell-by-cell, or top-to-bottom.
# =============================================================================

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from priors import SeismicPrior, EtasPriorUpdater
from benchmark import config
from benchmark.runner import repair_inversion_json_paths
from benchmark.usgs import download_case_study_catalog

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONTEXT_NAME   = 'benchmark'  # California context
_ETAS_ID       = config.etas_output_id(CONTEXT_NAME)
INVERSION_JSON = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion', f'parameters_{_ETAS_ID}.json')
KDE_TT3        = os.path.join(SeismicPrior.data_dir, f'kde_seismicity_{CONTEXT_NAME}.tt3')
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))

EPS = 1e-12

# Everything computationally expensive (ETAS updater construction, haversine
# precompute, .update() calls) is cached to disk below, keyed by cell. Delete
# the relevant file under CACHE_DIR (or set FORCE_RECOMPUTE = True) to rebuild
# after changing anything upstream of the figure (inversion params, context,
# mainshock delta, etc.) — pure figure/styling edits never need a rebuild.
CACHE_DIR = os.path.join(OUTPUT_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)
FORCE_RECOMPUTE = False

plt.rcParams.update({
    'font.size':        12,
    'font.family':      'sans-serif',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       150,
})

proj = ccrs.PlateCarree()


def _add_basemap(ax):
    ax.add_feature(cfeature.STATES,    linewidth=0.6, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND,      facecolor='lightgray',   zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='lightyellow', zorder=0)


#%%
# ── Load the three priors (cached) ──────────────────────────────────────────
ROW1_CACHE = os.path.join(CACHE_DIR, 'row1.pkl')

if not FORCE_RECOMPUTE and os.path.exists(ROW1_CACHE):
    with open(ROW1_CACHE, 'rb') as f:
        (kde_prior, prior_flat, prior_spatial, forecast_time,
         grid_lons_masked, grid_lats_masked) = pickle.load(f)
else:
    kde_prior = SeismicPrior.from_tt3(KDE_TT3)

    repair_inversion_json_paths(INVERSION_JSON)

    flat_cfg = dict(config.ETAS_UPDATER_CONFIG)
    flat_cfg['use_spatial_background']   = False
    flat_cfg['use_spatial_productivity'] = False
    flat_cfg['out_of_bounds_fill']       = 1e-8  # panel (b) only; config default is 1e-9
    flat_updater = EtasPriorUpdater.from_inversion_json(json_path=INVERSION_JSON, **flat_cfg)

    spatial_cfg = dict(config.ETAS_UPDATER_CONFIG)
    spatial_cfg['use_spatial_background']   = True
    spatial_cfg['use_spatial_productivity'] = False
    spatial_updater = EtasPriorUpdater.from_inversion_json(json_path=INVERSION_JSON, **spatial_cfg)

    forecast_time = pd.Timestamp(flat_updater.metadata_base['timewindow_end'])
    prior_flat    = flat_updater.update(forecast_time)
    prior_spatial = spatial_updater.update(forecast_time)
    grid_lons_masked = spatial_updater.grid_lons_masked
    grid_lats_masked = spatial_updater.grid_lats_masked

    with open(ROW1_CACHE, 'wb') as f:
        pickle.dump((kde_prior, prior_flat, prior_spatial, forecast_time,
                     grid_lons_masked, grid_lats_masked), f)

#%%
# ── MTJ 2024 M7 — ETAS prior immediately before/after the mainshock (cached) ─
MTJ_CASE  = 'MTJ_2024_M7'
MTJ_CS    = config.CASE_STUDIES[MTJ_CASE]
MTJ_AFTER_DELTA = pd.Timedelta(hours=1)  # how long after the mainshock to evaluate the "after" panel

MTJ_CACHE = os.path.join(CACHE_DIR, 'mtj_row2.pkl')

if not FORCE_RECOMPUTE and os.path.exists(MTJ_CACHE):
    with open(MTJ_CACHE, 'rb') as f:
        (prior_mtj_before, prior_mtj_after,
         MAINSHOCK_TIME, MAINSHOCK_LAT, MAINSHOCK_LON, MAINSHOCK_MAG) = pickle.load(f)
else:
    MTJ_DATA_DIR       = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', MTJ_CASE)
    MTJ_INVERSION_JSON = os.path.join(MTJ_DATA_DIR, 'etas_inversion', f'parameters_{config.etas_output_id(MTJ_CASE)}.json')
    MTJ_HIST_CATALOG   = os.path.join(MTJ_DATA_DIR, 'etas_inversion', 'input', f'catalog_{config.etas_catalog_tag(MTJ_CASE)}.csv')

    # Mainshock time/location — read from the case-study catalog rather than
    # hardcoding, keyed by the ANSS id already on record in config_california.py.
    mtj_catalog_df = download_case_study_catalog(MTJ_CS, cache_dir=MTJ_DATA_DIR, REDOWNLOAD=False)
    mainshock = mtj_catalog_df.set_index('id').loc[config.FOCUS_EVENTS_MAINSHOCK[MTJ_CASE]]
    MAINSHOCK_TIME = pd.Timestamp(mainshock['time']).tz_localize(None)
    MAINSHOCK_LAT, MAINSHOCK_LON, MAINSHOCK_MAG = mainshock['latitude'], mainshock['longitude'], mainshock['mag']

    repair_inversion_json_paths(MTJ_INVERSION_JSON)
    mtj_hist_catalog = pd.read_csv(MTJ_HIST_CATALOG, index_col=0, parse_dates=['time'])

    mtj_updater = EtasPriorUpdater.from_inversion_json(
        json_path  = MTJ_INVERSION_JSON,
        catalog_df = mtj_hist_catalog,
        **config.ETAS_UPDATER_CONFIG,
    )

    prior_mtj_before = mtj_updater.update(MAINSHOCK_TIME)

    mtj_updater.append_events(pd.DataFrame([{
        'time':      MAINSHOCK_TIME,
        'latitude':  MAINSHOCK_LAT,
        'longitude': MAINSHOCK_LON,
        'magnitude': MAINSHOCK_MAG,
    }]))

    prior_mtj_after = mtj_updater.update(MAINSHOCK_TIME + MTJ_AFTER_DELTA)

    with open(MTJ_CACHE, 'wb') as f:
        pickle.dump((prior_mtj_before, prior_mtj_after,
                     MAINSHOCK_TIME, MAINSHOCK_LAT, MAINSHOCK_LON, MAINSHOCK_MAG), f)

#%%
# ── Build the figure ──────────────────────────────────────────────────────
cmap = 'viridis'

fig, axes = plt.subplots(2, 3, figsize=(15, 9), subplot_kw={'projection': proj})

# -- Row 1: statewide prior comparison ---------------------------------------
panels = [
    ('KDE Seismicity',            kde_prior),
    ('ETAS — flat background',    prior_flat),
    ('ETAS — spatial background', prior_spatial),
]
log_grids = [np.log10(obj.grid + EPS) for _, obj in panels]

vlo, vhi = np.percentile(np.concatenate([g.ravel() for g in log_grids]), [1, 99.5])
norm = mcolors.Normalize(vmin=vlo, vmax=vhi)

# Zoom to the ETAS polygon's extent (with padding) — shared across all three
# panels since KDE_Seismicity covers the same California region.
lon_pad = 1.0
lat_pad = 1.5
_extent = [
    grid_lons_masked.min() - lon_pad,
    grid_lons_masked.max() + lon_pad,
    grid_lats_masked.min() - lat_pad,
    grid_lats_masked.max() + lat_pad,
]

for ax, (title, obj), log_grid, letter in zip(axes[0, :], panels, log_grids, 'abc'):
    ax.set_extent(_extent, crs=proj)
    _add_basemap(ax)
    ax.pcolormesh(obj.lons, obj.lats, log_grid, cmap=cmap, norm=norm, transform=proj)
    ax.set_title(title)
    ax.text(0.02, 0.97, f'({letter})', transform=ax.transAxes,
            ha='left', va='top', fontsize=13, fontweight='bold', color='white')

cbar_ax_row1 = fig.add_axes([0.945, 0.54, 0.015, 0.42])  # [left, bottom, width, height]
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax_row1,
             label=r'$\log_{10}$ prior density')

# -- Row 2: MTJ 2024 M7 — before vs after the mainshock ----------------------
axes[1, 2].remove()  # reserved for a future 3rd panel in this row

mtj_panels    = [('Before mainshock', prior_mtj_before), ('After mainshock', prior_mtj_after)]
mtj_log_grids = [np.log10(obj.grid + EPS) for _, obj in mtj_panels]

mtj_vlo, mtj_vhi = np.percentile(np.concatenate([g.ravel() for g in mtj_log_grids]), [1, 99.9])
mtj_norm = mcolors.Normalize(vmin=mtj_vlo, vmax=mtj_vhi)

# Custom mendecino triple junction extent
_mtj_extent = [
    -127, -123, 39, 42
]

for ax, (title, obj), log_grid, letter in zip(axes[1, :2], mtj_panels, mtj_log_grids, 'de'):
    ax.set_extent(_mtj_extent, crs=proj)
    _add_basemap(ax)
    ax.pcolormesh(obj.lons, obj.lats, log_grid, cmap=cmap, norm=mtj_norm, transform=proj)
    ax.plot(MAINSHOCK_LON, MAINSHOCK_LAT, marker='*', color='red', markersize=16,
            markeredgecolor='black', markeredgewidth=0.8, transform=proj, zorder=5)
    ax.set_title(title)
    ax.text(0.02, 0.97, f'({letter})', transform=ax.transAxes,
            ha='left', va='top', fontsize=13, fontweight='bold', color='white')

cbar_ax_row2 = fig.add_axes([0.65, 0.04, 0.015, 0.42])  # [left, bottom, width, height]
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=mtj_norm), cax=cbar_ax_row2,
             label=r'$\log_{10}$ prior density')

fig.suptitle(
    f'Spatial prior comparison  ({forecast_time.date()})   |   '
    f'MTJ M{MAINSHOCK_MAG:g}  {MAINSHOCK_TIME:%Y-%m-%d %H:%M} UTC',
    y=0.98,
)

# ── Exact per-panel placement [left, bottom, width, height] (figure fraction) ──
# Edit these directly to hand-tune the layout — set_position overrides
# whatever matplotlib's default subplot grid computed above.
axes[0, 0].set_position([0.03, 0.52, 0.28, 0.42])  # (a) KDE Seismicity
axes[0, 1].set_position([0.34, 0.52, 0.28, 0.42])  # (b) ETAS — flat background
axes[0, 2].set_position([0.65, 0.52, 0.28, 0.42])  # (c) ETAS — spatial background
axes[1, 0].set_position([0.03, 0.04, 0.28, 0.42])  # (d) Before mainshock
axes[1, 1].set_position([0.34, 0.04, 0.28, 0.42])  # (e) After mainshock

# X-axis ticks/labels — bottom-most panel in each column (axes[0,2] has no
# panel below it since axes[1,2] was removed, so it gets the tick marks instead)
for ax, step in ((axes[1, 0], 1), (axes[1, 1], 1)):
    lo, hi = ax.get_extent(crs=proj)[:2]
    ax.set_xticks(np.arange(np.ceil(lo / step) * step, hi, step), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.set_xlabel('Longitude')

for ax,step in ((axes[0,0],4), (axes[0,1],4),(axes[0, 2], 4)):
    lo, hi = ax.get_extent(crs=proj)[:2]
    ax.set_xticks(np.arange(np.ceil(lo / step) * step, hi, step), crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    #ax.set_xlabel('Longitude')

# Y-axis ticks/labels — left-most panel in each row
for ax, step in ((axes[0, 0], 4), (axes[1, 0], 1)):
    lo, hi = ax.get_extent(crs=proj)[2:]
    ax.set_yticks(np.arange(np.ceil(lo / step) * step, hi, step), crs=proj)
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_ylabel('Latitude')


fig.savefig(os.path.join(OUTPUT_DIR, 'prior_comparison.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'prior_comparison.pdf'), bbox_inches='tight')
plt.show()

# %%
