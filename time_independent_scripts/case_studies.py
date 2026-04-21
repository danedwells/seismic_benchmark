#%%
# =============================================================================
# case_studies.py  —  bEPIC case-study runner
# Prerequisite: run scripts/build_priors.py first to build the .tt3 cache files.
# =============================================================================
# Downloads a USGS catalog for a predefined case study (aftershock sequence
# or mainshock region), builds .run trigger files from USGS phase data, then
# runs bEPIC across all five static spatial priors — mirroring run_benchmarks.py.
#
# Usage:
#   Set ACTIVE_CASE_STUDY to one of the keys in CASE_STUDIES, flip the
#   control flags, then run cells in order (or execute the whole script).
# =============================================================================

from priors import SeismicPrior
from benchmark.background import load_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_posterior_grid, plot_location_trajectory,
                             plot_overview_map, plot_location_grid)
from benchmark.usgs import *

import os
import numpy as np
import matplotlib.pyplot as plt

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
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')
CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output',  f'max_trigs_{MAX_TRIGS}')
CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures', f'max_trigs_{MAX_TRIGS}')

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR, CS_FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)


#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

# --- Control flags ---
DOWNLOAD_CATALOG = False  # re-download even if cached
BUILD_RUN_FILES  = False  # build / rebuild .run files from USGS phases
RUN_ALL_PRIORS   = False # run all six priors in parallel
SKIP_RUN         = True # Skip all runs - will toss an error if nothing has been run before AND run_all_priors == False
# Setting skip_run to true and run_all_priors to false will skip bEPIC calls entirely and go 
# straight to plot/figure generation

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
        'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
        'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
    }
    for name, path in cache_paths.items()
]

if RUN_ALL_PRIORS:
    benchmark_runner.run_all_priors_parallel(benchmark_runner.run_prior, job_args)
else:
    if SKIP_RUN == False:
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
            'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
            'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
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

PRIOR_ORDER = list(config.PRIOR_FILENAMES.keys())

# Compute case-study map extent and filter background seismicity to the region
min_lon, max_lon, min_lat, max_lat = cs['bounds']
min_lon -= 1; max_lon += 1; min_lat -= 1; max_lat += 1
cs_extent = [min_lon - 0.5, max_lon + 0.5, min_lat - 0.5, max_lat + 0.5]

bg_region = (bg[
    bg['longitude'].between(min_lon - 1, max_lon + 1) &
    bg['latitude'].between(min_lat - 1, max_lat + 1)
] if bg is not None else None)

# ── Map: all priors compared ───────────────────────────────────────────────
fig = plot_overview_map(
    output_dir  = CS_OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = cs_extent,
    events_df   = catalog_df[['longitude', 'latitude']],
    bg          = bg_region,
    title       = f'bEPIC locations — {cs["name"]}',
    save_path   = os.path.join(CS_FIGURES_DIR, 'comparison_locations.png'),
)
plt.show()

# ── 2×3 grid: one panel per prior ─────────────────────────────────────────
fig = plot_location_grid(
    output_dir  = CS_OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = cs_extent,
    ref_catalog = ref_df,
    events_df   = catalog_df[['longitude', 'latitude']],
    bg          = bg_region,
    cache_paths = cache_paths,
    title       = f'bEPIC locations — {cs["name"]} — prior comparison',
    save_path   = os.path.join(CS_FIGURES_DIR, 'grid_locations.png'),
)
plt.show()

#%%
# ── Location error histograms ─────────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'location_error_km',
    bins        = np.linspace(0, 100, 51),
    title       = f'bEPIC location errors — {cs["name"]}',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'location_error_histograms.png'),
    catalog_df  = ref_df,
)
plt.show()

# ── Fractional misfit histograms ──────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'frac_misfit',
    bins        = np.linspace(0, 0.5, 51),
    title       = f'bEPIC fractional misfit — {cs["name"]}',
    xlabel      = 'frac_misfit',
    save_path   = os.path.join(CS_FIGURES_DIR, 'misfit_histograms.png'),
)
plt.show()

# %%
# =============================================================================
# Single-event posterior grid figure (2×3 panel, one panel per prior)
# =============================================================================
# FOCUS_EVENT_ID : str   — ANSS event ID whose .run file exists in CS_RUN_DIR.
#                          Automatically selected from FOCUS_EVENTS below based
#                          on ACTIVE_CASE_STUDY.  Override by setting
#                          FOCUS_EVENT_ID manually after this cell.
# FOCUS_VERSION  : int or None — trigger version to plot; None = last available.
#
# To add or change representative events, edit FOCUS_EVENTS below.
# Use examine_catalog.py (case-study section) to browse the catalog and pick IDs.
# =============================================================================

FOCUS_EVENTS = {
    'Ridgecrest': 'ci38548295',  # M 4.9 aftershock 
    'Ferndale':   'nc73831091',  # M 4.05 aftershock
    'ElMayor':    'ci10148002',  # M 5.2 aftershock
}

# Mainshocks
_MS_ = False
if _MS_ == True:
    FOCUS_EVENTS = {
        'Ridgecrest': 'ci38457511',   # M7.1 mainshock  2019-07-06
        'Ferndale':   'nc73821036',   # M6.4 mainshock  2022-12-20
        'ElMayor':    'ci14607652',   # M7.2 mainshock
    }

FOCUS_EVENT_ID = FOCUS_EVENTS[ACTIVE_CASE_STUDY]
FOCUS_VERSION  = None

focus_run_path = os.path.join(CS_RUN_DIR, f'{FOCUS_EVENT_ID}.run')

if not os.path.exists(focus_run_path):
    print(f"[single-event figure] .run file not found: {focus_run_path}")
    print("  → set FOCUS_EVENT_ID to a built event, or run BUILD_RUN_FILES first.")
else:
    _focus_ref = ref_df[ref_df['event_id'] == FOCUS_EVENT_ID]
    _ref_lat   = float(_focus_ref['usgs_lat'].iloc[0]) if not _focus_ref.empty else None
    _ref_lon   = float(_focus_ref['usgs_lon'].iloc[0]) if not _focus_ref.empty else None

    fig = plot_posterior_grid(
        focus_run_path = focus_run_path,
        cache_paths    = cache_paths,
        prior_order    = PRIOR_ORDER,
        params_kw      = {
            'grid_size': config.BENCHMARK_PARAMS['grid_size'],
            'grid_km':   config.BENCHMARK_PARAMS['grid_km'],
            'max_trigs': config.BENCHMARK_PARAMS['max_trigs'],
            'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
            'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
        },
        ref_lat        = _ref_lat,
        ref_lon        = _ref_lon,
        focus_version  = FOCUS_VERSION,
        title          = f'bEPIC posterior grid — {cs["name"]} — event {FOCUS_EVENT_ID}',
        save_path      = os.path.join(CS_FIGURES_DIR, f'posterior_grid_{FOCUS_EVENT_ID}.png'),
    )
    plt.show()

    fig = plot_location_trajectory(
        event_id     = FOCUS_EVENT_ID,
        output_dir   = CS_OUTPUT_DIR,
        prior_order  = PRIOR_ORDER,
        run_dir      = CS_RUN_DIR,
        min_triggers = 4,
        ref_lat      = _ref_lat,
        ref_lon      = _ref_lon,
        cache_paths  = cache_paths,
        title        = f'bEPIC location trajectory — {cs["name"]} — event {FOCUS_EVENT_ID}',
        save_path    = os.path.join(CS_FIGURES_DIR, f'trajectory_{FOCUS_EVENT_ID}.png'),
    )
    plt.show()

# %%
