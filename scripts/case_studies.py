#%%
# =============================================================================
# case_studies.py  —  bEPIC case-study runner
# =============================================================================
# Downloads a USGS catalog for a predefined case study (aftershock sequence
# or mainshock region), builds .run trigger files from USGS phase data, then
# runs bEPIC across all six spatial priors — mirroring run_benchmarks.py.
#
# Usage:
#   Set ACTIVE_CASE_STUDY to one of the keys in CASE_STUDIES, flip the
#   control flags, then run cells in order (or execute the whole script).
# =============================================================================

from priors import SeismicPrior
from benchmark.background import load_background_seismicity
from benchmark.usgs import *

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from pathlib import Path
from benchmark import runner as benchmark_runner
from benchmark import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir     = os.path.dirname(PROJECT_ROOT)   # 2024_NEHRP/

data_dir    = SeismicPrior.data_dir            # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

SEIS_CACHE = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')

# ---------------------------------------------------------------------------
# Case study definitions
# ---------------------------------------------------------------------------
# bounds = (min_lon, max_lon, min_lat, max_lat)
CASE_STUDIES = {
    'Ridgecrest': {
        'name':      'Ridgecrest 2019',
        'starttime': '2019-07-04T17:00:00',
        'endtime':   '2019-08-07T00:00:00',
        'bounds':    (-118.5, -116.5, 35.0, 36.5),
        'min_mag':   3.0,
    },
    'Ferndale': {
        'name':      'Ferndale 2022',
        'starttime': '2022-12-20T10:00:00',
        'endtime':   '2023-01-20T00:00:00',
        'bounds':    (-127.0, -122.5, 39, 41.0),
        'min_mag':   3.0,
    },
    'ElMayor': {
        'name':      'El Mayor-Cucapah 2010',
        'starttime': '2010-04-04T22:00:00',
        'endtime':   '2010-05-04T00:00:00',
        'bounds':    (-117.0, -114.5, 31.5, 33.5),
        'min_mag':   3.0,
    },
}

# --- Select active case study ---
ACTIVE_CASE_STUDY = 'Ridgecrest'

cs = CASE_STUDIES[ACTIVE_CASE_STUDY]

# Per-case-study directories
CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')
CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output')
CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures')

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR, CS_FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)



#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

# --- Control flags ---
DOWNLOAD_CATALOG = True   # re-download even if cached
BUILD_RUN_FILES  = True  # build / rebuild .run files from USGS phases
RUN_ALL_PRIORS   = False   # run all six priors in parallel

MAX_TRIGS = config.BENCHMARK_PARAMS['max_trigs']

# ── 1. Download (or load cached) catalog ──────────────────────────────────
catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR)
print(catalog_df.head())

#%%

# ── 2. Build .run files ───────────────────────────────────────────────────
if BUILD_RUN_FILES:
    build_run_files_for_case_study(
        catalog_df   = catalog_df,
        run_dir      = CS_RUN_DIR,
        max_dist_deg = 5.0,
        skip_existing = not DOWNLOAD_CATALOG,
    )
#%%

# ── 3. Run bEPIC across priors ────────────────────────────────────────────
job_args = [
    {
        'prior_name': name,
        'cache_path': path,
        'nshm_path':  cache_paths['NSHM'],
        'run_dir':    CS_RUN_DIR,
        'output_dir': CS_OUTPUT_DIR,
        'grid_size':  config.BENCHMARK_PARAMS['grid_size'],
        'grid_km':    config.BENCHMARK_PARAMS['grid_km'],
        'max_trigs':  MAX_TRIGS,
    }
    for name, path in cache_paths.items()
]

if RUN_ALL_PRIORS:
    benchmark_runner.run_all_priors_parallel(benchmark_runner.run_prior, job_args)
else:
    # Run the single prior from config
    benchmark_runner.run_prior({
        'prior_name': config.BENCHMARK_PARAMS['prior'],
        'cache_path': cache_paths[config.BENCHMARK_PARAMS['prior']],
        'nshm_path':  cache_paths['NSHM'],
        'run_dir':    CS_RUN_DIR,
        'output_dir': CS_OUTPUT_DIR,
        'grid_size':  config.BENCHMARK_PARAMS['grid_size'],
        'grid_km':    config.BENCHMARK_PARAMS['grid_km'],
        'max_trigs':  MAX_TRIGS,
    })

#%%
# ---------------------------------------------------------------------------
# Reference catalog: derived from the downloaded USGS catalog
# ---------------------------------------------------------------------------
# Rename columns to match compute_location_error() expectations.
ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
    'depth':     'usgs_depth',
    'mag':       'usgs_mag',
})

#%%
# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

bg = load_background_seismicity(
    cache_path = SEIS_CACHE,
    bounds     = (-129, -112, 30, 45),
    start_year = 2000,
    end_year   = 2025,
    min_mag    = 3.5,
)

PRIOR_ORDER = ['Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'ETAS', 'Uniform']
csv_files   = sorted(Path(CS_OUTPUT_DIR).glob('*_benchmark_results.csv'))
colors      = plt.cm.tab10.colors


#%%
"""
FIGURE GENERATION

4 figures:
1) Map view, all seismicity
2) Map view, MTJ
3) Location error hist, all
4) Location error hist, MTJ
"""

# ── Map: all priors compared ──────────────────────────────────────────────
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': proj})

min_lon, max_lon, min_lat, max_lat = cs['bounds']
ax.set_extent([min_lon - 0.5, max_lon + 0.5, min_lat - 0.5, max_lat + 0.5],
              crs=proj)
ax.add_feature(cfeature.STATES,    linewidth=0.6, edgecolor='black')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.LAND,  facecolor='lightgray',   zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='lightyellow',  zorder=0)

ax.scatter(catalog_df['longitude'], catalog_df['latitude'],
           s=10, color='black', alpha=0.4, transform=proj,
           label='USGS catalog', zorder=1)

if bg is not None:
    bg_region = bg[
        bg['longitude'].between(min_lon - 1, max_lon + 1) &
        bg['latitude'].between(min_lat - 1, max_lat + 1)
    ]
    ax.scatter(bg_region['longitude'], bg_region['latitude'],
               s=6, c='gray', alpha=0.12, transform=proj, zorder=0, linewidths=0)

for i, csv_path in enumerate(csv_files):
    prior_name = csv_path.stem.replace('_benchmark_results', '')
    df    = pd.read_csv(csv_path)
    final = df.groupby('event_id').last().reset_index()
    ax.scatter(final['posterior_lon'], final['posterior_lat'],
               s=8, color=colors[i % len(colors)], alpha=0.3,
               transform=proj, label=prior_name, zorder=2)

ax.set_title(f'bEPIC locations — {cs["name"]}')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(CS_FIGURES_DIR, 'comparison_locations.png'), dpi=150)
plt.show()

# ── 2×3 grid: one panel per prior ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw={'projection': proj})

for idx, (ax, prior_name) in enumerate(zip(axes.flatten(), PRIOR_ORDER)):
    row_idx, col_idx = divmod(idx, 3)
    ax.set_extent([min_lon - 0.5, max_lon + 0.5, min_lat - 0.5, max_lat + 0.5],
                  crs=proj)
    ax.add_feature(cfeature.STATES,    linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.LAND,  facecolor='lightgray',  zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='lightyellow', zorder=0)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = (col_idx == 0)
    gl.bottom_labels = (row_idx == 1)

    ax.scatter(catalog_df['longitude'], catalog_df['latitude'],
               s=12, color='black', alpha=0.4, transform=proj,
               label='USGS catalog', zorder=1)

    if bg is not None:
        ax.scatter(bg_region['longitude'], bg_region['latitude'],
                   s=6, c='gray', alpha=0.1, transform=proj, zorder=0, linewidths=0)

    csv_path = os.path.join(CS_OUTPUT_DIR,
                            f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df    = pd.read_csv(csv_path)
        final = df.groupby('event_id').last().reset_index()

        # Error lines: USGS → posterior
        matched = final.merge(
            ref_df[['event_id', 'usgs_lon', 'usgs_lat']],
            on='event_id', how='inner',
        )
        if not matched.empty:
            n   = len(matched)
            lns = np.empty(n * 3); lts = np.empty(n * 3)
            lns[0::3] = matched['usgs_lon'].values
            lns[1::3] = matched['posterior_lon'].values
            lns[2::3] = np.nan
            lts[0::3] = matched['usgs_lat'].values
            lts[1::3] = matched['posterior_lat'].values
            lts[2::3] = np.nan
            ax.plot(lns, lts, color='black', linewidth=0.5,
                    alpha=0.35, transform=proj, zorder=2)

        ax.scatter(final['posterior_lon'], final['posterior_lat'],
                   s=14, color='crimson', alpha=0.5, transform=proj,
                   label=prior_name, zorder=3)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.set_title(prior_name, fontsize=11)
    ax.legend(loc='upper right', fontsize=7)

fig.suptitle(f'bEPIC locations — {cs["name"]} — prior comparison', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CS_FIGURES_DIR, 'grid_locations.png'), dpi=150)
plt.show()

# ── Location error histograms ─────────────────────────────────────────────
bins_km = np.linspace(0, 100, 51)
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    csv_path = os.path.join(CS_OUTPUT_DIR,
                            f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df    = pd.read_csv(csv_path)
        final = df.groupby('event_id').last().reset_index()
        final = benchmark_runner.compute_location_error(final, ref_df)
        errs  = final['location_error_km'].dropna()
        ax.hist(errs, bins=bins_km, color='crimson', alpha=0.6, label=prior_name)
        ax.axvline(errs.median(), color='k', linestyle='--', linewidth=1,
                   label=f'median {errs.median():.1f} km')
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.set_title(prior_name, fontsize=11)
    ax.set_xlabel('location error (km)', fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('count', fontsize=9)
axes[1, 0].set_ylabel('count', fontsize=9)
fig.suptitle(f'bEPIC location errors — {cs["name"]}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CS_FIGURES_DIR, 'location_error_histograms.png'), dpi=150)
plt.show()

# ── Fractional misfit histograms ──────────────────────────────────────────
bins_frac = np.linspace(0, 0.5, 51)
fig, axes  = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

for ax, prior_name in zip(axes.flatten(), PRIOR_ORDER):
    csv_path = os.path.join(CS_OUTPUT_DIR,
                            f'{prior_name.lower()}_benchmark_results.csv')
    if os.path.exists(csv_path):
        df      = pd.read_csv(csv_path)
        misfits = df.groupby('event_id').last().reset_index()['frac_misfit']
        ax.hist(misfits, bins=bins_frac, color='crimson', alpha=0.6, label=prior_name)
    else:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.set_title(prior_name, fontsize=11)
    ax.set_xlabel('frac_misfit', fontsize=9)
    ax.legend(fontsize=7)

axes[0, 0].set_ylabel('count', fontsize=9)
axes[1, 0].set_ylabel('count', fontsize=9)
fig.suptitle(f'bEPIC fractional misfit — {cs["name"]}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CS_FIGURES_DIR, 'misfit_histograms.png'), dpi=150)
plt.show()
