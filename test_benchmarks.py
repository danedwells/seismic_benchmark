#%%
# Get priors - separate repo
from priors import SeismicPrior
from seismicity_background import load_background_seismicity, add_background_seismicity

# Get bEPIC - separate repo
from bEPIC import EPIC_locate_prelim

# Standard imports
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from concurrent.futures import ProcessPoolExecutor, as_completed

# Get custom class for this comparison
import benchmark_runner
import utils
import config

# Get root. Priors are in a different repo (priors/)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get directory of THIS project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEIS_CACHE = os.path.join(PROJECT_ROOT, 'reference_locations', 'background_seismicity.parquet')

data_dir = SeismicPrior.data_dir  # priors/data/

# Cached .tt3 paths for each prior — filenames come from config.py
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

#%%
"""
# --- Step 1: Build and cache all priors ---
# Set construct = True to rebuild; False to skip if .tt3 files already exist.
"""
construct = False

source_paths = {}

if construct:
    utils.build_and_cache_priors(cache_paths, data_dir, config.PRIOR_CONSTRUCTION_PARAMS)


"""
# --- Step 2: Build reference locations ---
# Run a version of the model in more detail. Uses smooth_seismicity.
"""

# Point to data/run files
run_dir = os.path.join(PROJECT_ROOT, 'run_files')

# --- Create reference locations ---
REFERENCE = False
ref_dir = os.path.join(PROJECT_ROOT, 'reference_locations')

if REFERENCE:
    benchmark_runner.create_reference_locations(run_dir, ref_dir, cache_paths, config.REFERENCE_PARAMS)


#%%

"""
# --- Step 2: Select and load prior for bEPIC ---
# Options: 'Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'ETAS', 'Uniform'

Single event - but with the imported benchmark_runner
classes and modules
"""
# Load prior and params from config
_bp = config.BENCHMARK_PARAMS
_cache_key = next(k for k in cache_paths if k.lower() == _bp['prior'].lower())
_tt3_path  = cache_paths[_cache_key]
if _tt3_path is None:
    p = SeismicPrior.from_tt3(next(v for v in cache_paths.values() if v is not None))
    params_use_prior = False
else:
    p = SeismicPrior.from_tt3(_tt3_path)
    params_use_prior = True

MAX_TRIGS = _bp['max_trigs']  # drives output/figures subfolders

params = EPIC_locate_prelim.EPIC_PARAMS()
params.prior           = p
params.use_prior       = params_use_prior
params.GridSize        = _bp['grid_size']
params.GridKm          = _bp['grid_km']
params.method          = 'EPIC C'
params.MAX_EVENT_TRIGS = MAX_TRIGS

# Create benchmark runner
runner = benchmark_runner.BenchmarkRunner(prior=p, params=params, run_dir=run_dir)

# --- Run all priors in parallel ---
output_subdir = os.path.join(PROJECT_ROOT, 'output', f'max_trigs_{MAX_TRIGS}')
figures_subdir = os.path.join(PROJECT_ROOT, 'figures', f'max_trigs_{MAX_TRIGS}')
os.makedirs(output_subdir,  exist_ok=True)
os.makedirs(figures_subdir, exist_ok=True)

RUN_ALL_PRIORS = False

if RUN_ALL_PRIORS:
    job_args = [
        {
            'prior_name': name,
            'cache_path': path,
            'nshm_path':  cache_paths['NSHM'],  # geometry fallback for Uniform
            'run_dir':    run_dir,
            'output_dir': output_subdir,
            'grid_size':  params.GridSize,
            'grid_km':    params.GridKm,
            'max_trigs':  MAX_TRIGS,
        }
        for name, path in cache_paths.items()
    ]

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(benchmark_runner.run_prior, args): args['prior_name']
                   for args in job_args}
        for f in as_completed(futures):
            name = futures[f]
            exc = f.exception()
            if exc:
                print(f"{name} FAILED: {exc}")
            else:
                print(f"{name} done")

benchmark_runner.run_prior({
      'prior_name': 'Uniform',
      'cache_path': None,
      'nshm_path':  cache_paths['NSHM'],
      'run_dir':    run_dir,
      'output_dir': output_subdir,
      'grid_size':  config.BENCHMARK_PARAMS['grid_size'],
      'grid_km':    config.BENCHMARK_PARAMS['grid_km'],
      'max_trigs':  MAX_TRIGS,
  })

#%%
# --- Load ANSS catalog as reference ---
catalog_path = os.path.join(root_dir, 'bEPIC', 'Zextra', 'bEPIC_testing_catalog.txt')
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

stations_df = get_unique_stations(run_dir)

bg = None
# ------ Get background seismicity
bg = load_background_seismicity(
    cache_path  = SEIS_CACHE,
    bounds      = (-129, -112, 30, 45),
    start_year  = 2000,
    end_year    = 2025,
    min_mag     = 3.5,
)

#%%
"""
FIGURE TIME
"""
# --- Comparison plot: all priors from output/ CSVs ---

csv_files = sorted(Path(output_subdir).glob('*_benchmark_results.csv'))

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
plt.savefig(os.path.join(figures_subdir, 'comparison_benchmark_locations.png'), dpi=150)
plt.show()

# %%

"""
Zoom in on Mendocino Triple Junction (MTJ) — 2x3 grid, one prior per panel
"""

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

    csv_path = os.path.join(output_subdir, f'{prior_name.lower()}_benchmark_results.csv')
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
plt.savefig(os.path.join(figures_subdir, 'MTJ_grid_benchmark_locations.png'), dpi=150)
plt.show()

# %%
"""
MTJ fractional misfit histograms — 2x3 grid, one prior per panel
frac_misfit from final version of each event, filtered to MTJ extent
"""

# ref_frac_misfits_mtj = in_extent(ref_final)['frac_misfit']
bins_frac = np.linspace(0, 0.5, 51)
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    csv_path = os.path.join(output_subdir, f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        misfits = in_extent(df.groupby('event_id').last().reset_index())['frac_misfit']
        ax.hist(misfits, bins=bins_frac, color='crimson', alpha=0.6, label=prior_name)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    # ax.hist(ref_frac_misfits_mtj, bins=bins_frac, color='black', alpha=0.4, label='Reference')
    ax.set_title(prior_name, fontsize=11)
    ax.set_xlabel('frac_misfit (fractional TT error)', fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('count', fontsize=9)
axes[1, 0].set_ylabel('count', fontsize=9)

fig.suptitle('bEPIC MTJ fractional misfit distributions — prior comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_subdir, 'MTJ_grid_misfit_histograms.png'), dpi=150)
plt.show()

"""
TOTAL fractional misfit histograms — 2x3 grid, one prior per panel
frac_misfit from final version of each event, all events
"""

# ref_frac_misfits_all = ref_final['frac_misfit']

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    csv_path = os.path.join(output_subdir, f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        misfits = df.groupby('event_id').last().reset_index()['frac_misfit']
        ax.hist(misfits, bins=bins_frac, color='crimson', alpha=0.6, label=prior_name)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    # ax.hist(ref_frac_misfits_all, bins=bins_frac, color='black', alpha=0.4, label='Reference')
    ax.set_title(prior_name, fontsize=11)
    ax.set_xlabel('frac_misfit (fractional TT error)', fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('count', fontsize=9)
axes[1, 0].set_ylabel('count', fontsize=9)

fig.suptitle('bEPIC fractional misfit distributions — prior comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_subdir, 'Grid_misfit_histograms.png'), dpi=150)
plt.show()

"""
LOCATION ERROR histograms — 2x3 grid, one prior per panel
location_error_km from final version of each event (requires catalog).
Skipped silently per prior if catalog_df is None or column is absent.
"""

# catalog_path and catalog_df loaded above in the reference section

bins_km = np.linspace(0, 100, 51)
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

# ref_loc_errors = None
# if catalog_df is not None and 'location_error_km' not in ref_final.columns:
#     ref_with_err = benchmark_runner.compute_location_error(ref_final, catalog_df)
#     ref_loc_errors = ref_with_err['location_error_km'].dropna()
# elif 'location_error_km' in ref_final.columns:
#     ref_loc_errors = ref_final['location_error_km'].dropna()

for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    csv_path = os.path.join(output_subdir, f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path) and catalog_df is not None:
        df = pd.read_csv(csv_path)
        final = df.groupby('event_id').last().reset_index()
        final = benchmark_runner.compute_location_error(final, catalog_df)
        loc_errors = final['location_error_km'].dropna()
        ax.hist(loc_errors, bins=bins_km, color='crimson', alpha=0.6, label=prior_name)
    else:
        ax.text(0.5, 0.5, 'no catalog' if catalog_df is None else 'no data',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')

    # ax.hist(ref_loc_errors, bins=bins_km, color='black', alpha=0.4, label='Reference')
    ax.set_title(prior_name, fontsize=11)
    ax.set_xlabel('location error (km)', fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('count', fontsize=9)
axes[1, 0].set_ylabel('count', fontsize=9)

fig.suptitle('bEPIC location error distributions — prior comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_subdir, 'Grid_location_error_histograms.png'), dpi=150)
plt.show()

"""
MTJ LOCATION ERROR histograms — 2x3 grid, one prior per panel
location_error_km from final version of each event, filtered to MTJ extent.
"""

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    csv_path = os.path.join(output_subdir, f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path) and catalog_df is not None:
        df = pd.read_csv(csv_path)
        final = in_extent(df.groupby('event_id').last().reset_index())
        final = benchmark_runner.compute_location_error(final, catalog_df)
        loc_errors = final['location_error_km'].dropna()
        ax.hist(loc_errors, bins=bins_km, color='crimson', alpha=0.6, label=prior_name)
    else:
        ax.text(0.5, 0.5, 'no catalog' if catalog_df is None else 'no data',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')

    ax.set_title(prior_name, fontsize=11)
    ax.set_xlabel('location error (km)', fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('count', fontsize=9)
axes[1, 0].set_ylabel('count', fontsize=9)

fig.suptitle('bEPIC MTJ location error distributions — prior comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_subdir, 'MTJ_location_error_histograms.png'), dpi=150)
plt.show()

# %%
