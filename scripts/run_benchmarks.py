#%%
# =============================================================================
# run_benchmarks.py  —  bEPIC prior benchmark
# =============================================================================
from priors import SeismicPrior
from benchmark.background import load_background_seismicity, add_background_seismicity
from benchmark.plots import plot_prior_histograms
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pathlib import Path
from benchmark import runner as benchmark_runner
from benchmark import priors as utils
from benchmark import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir     = os.path.dirname(PROJECT_ROOT)   # 2024_NEHRP/

data_dir    = SeismicPrior.data_dir  # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

SEIS_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')
RUN_DIR     = os.path.join(PROJECT_ROOT, 'data', 'run_files')

MAX_TRIGS   = config.BENCHMARK_PARAMS['max_trigs']
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'output',  f'max_trigs_{MAX_TRIGS}')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', f'max_trigs_{MAX_TRIGS}')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

# --- Control flags ---
CONSTRUCT      = False   # rebuild all prior .tt3 files from source data
REFERENCE      = False   # run high-resolution reference locations
RUN_ALL_PRIORS = False   # run all six priors in parallel

# ── 1. Build and cache all priors ─────────────────────────────────────────
if CONSTRUCT:
    utils.build_and_cache_priors(cache_paths, data_dir, config.PRIOR_CONSTRUCTION_PARAMS)

# ── 2. Create reference locations ─────────────────────────────────────────
ref_dir = os.path.join(PROJECT_ROOT, 'data', 'reference')

if REFERENCE:
    benchmark_runner.create_reference_locations(RUN_DIR, ref_dir, cache_paths, config.REFERENCE_PARAMS)

# ── 3. Run bEPIC across priors ────────────────────────────────────────────
job_args = [
    {
        'prior_name': name,
        'cache_path': path,
        'nshm_path':  cache_paths['NSHM'],  # geometry fallback for Uniform
        'run_dir':    RUN_DIR,
        'output_dir': OUTPUT_DIR,
        'grid_size':  config.BENCHMARK_PARAMS['grid_size'],
        'grid_km':    config.BENCHMARK_PARAMS['grid_km'],
        'max_trigs':  MAX_TRIGS,
    }
    for name, path in cache_paths.items()
]

if RUN_ALL_PRIORS:
    benchmark_runner.run_all_priors_parallel(benchmark_runner.run_prior, job_args)
else:
    benchmark_runner.run_prior({
        'prior_name': 'Uniform',
        'cache_path': None,
        'nshm_path':  cache_paths['NSHM'],
        'run_dir':    RUN_DIR,
        'output_dir': OUTPUT_DIR,
        'grid_size':  config.BENCHMARK_PARAMS['grid_size'],
        'grid_km':    config.BENCHMARK_PARAMS['grid_km'],
        'max_trigs':  MAX_TRIGS,
    })

#%%
# ---------------------------------------------------------------------------
# Reference catalog and station list
# ---------------------------------------------------------------------------
catalog_path = os.path.join(PROJECT_ROOT, 'data','reference', 'bEPIC_testing_catalog.txt')
catalog_df = benchmark_runner.load_reference_catalog(catalog_path) if os.path.exists(catalog_path) else None

# --- Load reference locations (static; smooth_seismicity run to completion) ---
# REF_FILE_NAME = 'REFERENCE_100.csv'
# ref_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'reference_locations', REF_FILE_NAME))
# ref_final = ref_df.groupby('event_id').last().reset_index()

def get_unique_stations(run_dir):
    """Return a DataFrame of unique stations (by station+network) across all run files."""
    frames = [pd.read_csv(f, usecols=['station', 'network', 'longitude', 'latitude'])
              for f in Path(run_dir).glob('*.run')]
    return pd.concat(frames).drop_duplicates(subset=['station', 'network']).reset_index(drop=True)

stations_df = get_unique_stations(RUN_DIR)

bg = load_background_seismicity(
    cache_path  = SEIS_CACHE,
    bounds      = (-129, -112, 30, 45),
    start_year  = 2000,
    end_year    = 2025,
    min_mag     = 3.5,
)

#%%
"""
FIGURE GENERATION

4 figures:
1) Map view, all seismicity
2) Map view, MTJ
3) Location error hist, all
4) Location error hist, MTJ
"""
# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# ── Comparison plot: all priors ────────────────────────────────────────────
csv_files = sorted(Path(OUTPUT_DIR).glob('*_benchmark_results.csv'))

colors = plt.cm.tab10.colors

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})

ax.set_extent([-128.5, -113, 31, 44], crs=proj)
ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor='black')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='lightyellow', zorder=0)

# ax.scatter(ref_final['posterior_lon'], ref_final['posterior_lat'],
#            s=10, color='black', alpha=0.4, transform=proj,
#            label='Reference (smooth_seismicity)', zorder=1)
if catalog_df is not None:
    ax.scatter(catalog_df['usgs_lon'], catalog_df['usgs_lat'],
               s=10, color='black', alpha=0.4, transform=proj,
               label='ANSS catalog', zorder=1)
ax.scatter(stations_df['longitude'], stations_df['latitude'],
           s=20, color='orange', edgecolor='k', alpha=0.7, marker='v', transform=proj,
           label='Stations', zorder=0)
if bg is not None:
    ax.scatter(
        bg["longitude"], bg["latitude"],
        s=10, c='gray', alpha=0.1,
        transform=proj, zorder=0,
        linewidths=0,
    )

for i, csv_path in enumerate(csv_files):
    prior_name = csv_path.stem.replace('_benchmark_results', '')
    df = pd.read_csv(csv_path)
    final = df.groupby('event_id').last().reset_index()
    ax.scatter(final['posterior_lon'], final['posterior_lat'],
               s=8, color=colors[i % len(colors)], alpha=0.2,
               transform=proj, label=prior_name, zorder=2)

ax.set_title('bEPIC final locations — prior comparison')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'comparison_benchmark_locations.png'), dpi=150)
plt.show()

# %%

# ── MTJ grid: one prior per panel ─────────────────────────────────────────
lat_min = 38.5
lat_max = 42.5
lon_min = -128.5
lon_max = -122.5

def in_extent(df):
    return df[
        df['posterior_lat'].between(lat_min, lat_max) &
        df['posterior_lon'].between(lon_min, lon_max)
    ]

# ref_mtj = in_extent(ref_final)
catalog_mtj = catalog_df[
    catalog_df['usgs_lat'].between(lat_min, lat_max) &
    catalog_df['usgs_lon'].between(lon_min, lon_max)
] if catalog_df is not None else None
stations_mtj = stations_df[
    stations_df['latitude'].between(lat_min, lat_max) &
    stations_df['longitude'].between(lon_min, lon_max)
]

PRIOR_ORDER = ['Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'ETAS', 'Uniform']

proj = ccrs.PlateCarree()
fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw={'projection': proj})

for idx, (ax, prior_name) in enumerate(zip(axes.flatten(), PRIOR_ORDER)):
    row, col = divmod(idx, 3)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='lightyellow', zorder=0)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = (col == 0)
    gl.bottom_labels = (row == 1)

    # ax.scatter(ref_mtj['posterior_lon'], ref_mtj['posterior_lat'],
    #            s=18, color='black', alpha=0.5, transform=proj,
    #            label='Reference', zorder=1)
    if catalog_mtj is not None:
        ax.scatter(catalog_mtj['usgs_lon'], catalog_mtj['usgs_lat'],
                   s=18, color='black', alpha=0.5, transform=proj,
                   label='ANSS catalog', zorder=1)
    if bg is not None:
        ax.scatter(
            bg["longitude"], bg["latitude"],
            s=10, c='gray', alpha=0.1,
            transform=proj, zorder=0,
            linewidths=0,
        )
    ax.scatter(stations_mtj['longitude'], stations_mtj['latitude'],
               s=40, color='orange', edgecolor='k', alpha=0.85, marker='v',
               transform=proj, label='Stations', zorder=3)

    csv_path = os.path.join(OUTPUT_DIR, f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        final = in_extent(df.groupby('event_id').last().reset_index())

        # Draw error lines: ANSS → posterior, one plot call using NaN separators
        if catalog_df is not None and not final.empty:
            matched = final.merge(
                catalog_df[['event_id', 'usgs_lon', 'usgs_lat']],
                on='event_id', how='inner',
            )
            n = len(matched)
            seg_lons = np.empty(n * 3)
            seg_lats = np.empty(n * 3)
            seg_lons[0::3] = matched['usgs_lon'].values
            seg_lons[1::3] = matched['posterior_lon'].values
            seg_lons[2::3] = np.nan
            seg_lats[0::3] = matched['usgs_lat'].values
            seg_lats[1::3] = matched['posterior_lat'].values
            seg_lats[2::3] = np.nan
            ax.plot(seg_lons, seg_lats, color='black', linewidth=0.5,
                    alpha=0.35, transform=proj, zorder=2)

        ax.scatter(final['posterior_lon'], final['posterior_lat'],
                   s=18, color='crimson', alpha=0.5, transform=proj,
                   label=prior_name, zorder=3)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.set_title(prior_name, fontsize=11)
    ax.legend(loc='upper right', fontsize=7)

# Scale bar on bottom-left panel (axes[1, 0])
_ax_sb = axes[1, 0]
_lat_mid = (lat_min + lat_max) / 2
_scale_km = 100
_scale_deg = _scale_km / (111.32 * np.cos(np.radians(_lat_mid)))
_x0 = lon_min + 0.25
_y0 = lat_min + 0.25
_x1 = _x0 + _scale_deg
_tick_h = 0.06  # half-height of end ticks in degrees
_ax_sb.plot([_x0, _x1], [_y0, _y0], color='black', linewidth=2,
            transform=proj, zorder=10, solid_capstyle='butt')
_ax_sb.plot([_x0, _x0], [_y0 - _tick_h, _y0 + _tick_h], color='black',
            linewidth=2, transform=proj, zorder=10)
_ax_sb.plot([_x1, _x1], [_y0 - _tick_h, _y0 + _tick_h], color='black',
            linewidth=2, transform=proj, zorder=10)
_ax_sb.text((_x0 + _x1) / 2, _y0 + _tick_h + 0.04, f'{_scale_km} km',
            ha='center', va='bottom', fontsize=8, fontweight='bold',
            transform=proj, zorder=10)

fig.suptitle('bEPIC MTJ locations — prior comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'MTJ_grid_benchmark_locations.png'), dpi=150)
plt.show()

# %%
bins_frac = np.linspace(0, 0.5, 51)
bins_km   = np.linspace(0, 100, 51)

# ── MTJ fractional misfit histograms ──────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'frac_misfit',
    bins        = bins_frac,
    title       = 'bEPIC MTJ fractional misfit distributions — prior comparison',
    xlabel      = 'frac_misfit (fractional TT error)',
    save_path   = os.path.join(FIGURES_DIR, 'MTJ_grid_misfit_histograms.png'),
    filter_fn   = in_extent,
)
plt.show()

# ── Total fractional misfit histograms ────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'frac_misfit',
    bins        = bins_frac,
    title       = 'bEPIC fractional misfit distributions — prior comparison',
    xlabel      = 'frac_misfit (fractional TT error)',
    save_path   = os.path.join(FIGURES_DIR, 'Grid_misfit_histograms.png'),
)
plt.show()

# ── Location error histograms ─────────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'location_error_km',
    bins        = bins_km,
    title       = 'bEPIC location error distributions — prior comparison',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'Grid_location_error_histograms.png'),
    catalog_df  = catalog_df,
)
plt.show()

# ── MTJ location error histograms ─────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'location_error_km',
    bins        = bins_km,
    title       = 'bEPIC MTJ location error distributions — prior comparison',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'MTJ_location_error_histograms.png'),
    filter_fn   = in_extent,
    catalog_df  = catalog_df,
)
plt.show()

# %%
