#%%
# =============================================================================
# run_benchmarks.py  —  bEPIC prior benchmark
# Prerequisite: run scripts/build_priors.py first to build the .tt3 cache files.
# =============================================================================
from priors import SeismicPrior
from benchmark.background import load_background_seismicity, add_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_overview_map,
                             plot_location_grid, plot_posterior_grid,
                             plot_location_trajectory)
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
REFERENCE      = False   # run high-resolution reference locations
RUN_ALL_PRIORS = True   # run all six priors in parallel

# ── 1. Create reference locations ─────────────────────────────────────────
ref_dir = os.path.join(PROJECT_ROOT, 'data', 'reference')

if REFERENCE:
    benchmark_runner.create_reference_locations(RUN_DIR, ref_dir, cache_paths, config.REFERENCE_PARAMS)

# ── 3. Run bEPIC across priors ────────────────────────────────────────────
job_args = [
    {
        'prior_name':                name,
        'cache_path':                path,
        'nshm_path':                 cache_paths['NSHM'],  # geometry fallback for Uniform
        'run_dir':                   RUN_DIR,
        'output_dir':                OUTPUT_DIR,
        'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
        'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
        'max_trigs':                 MAX_TRIGS,
        'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
        'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
    }
    for name, path in cache_paths.items()
]

if RUN_ALL_PRIORS:
    benchmark_runner.run_all_priors_parallel(benchmark_runner.run_prior, job_args)
else:
    benchmark_runner.run_prior({
        'prior_name':                'Uniform',
        'cache_path':                None,
        'nshm_path':                 cache_paths['NSHM'],
        'run_dir':                   RUN_DIR,
        'output_dir':                OUTPUT_DIR,
        'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
        'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
        'max_trigs':                 MAX_TRIGS,
        'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
        'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
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
# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

PRIOR_ORDER = ['Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'ETAS', 'Uniform']

MTJ_EXTENT = [-128.5, -122.5, 38.5, 42.5]
mtj_lon_min, mtj_lon_max, mtj_lat_min, mtj_lat_max = MTJ_EXTENT

def in_extent(df):
    return df[
        df['posterior_lat'].between(mtj_lat_min, mtj_lat_max) &
        df['posterior_lon'].between(mtj_lon_min, mtj_lon_max)
    ]

catalog_events = (catalog_df[['usgs_lon', 'usgs_lat']]
                  .rename(columns={'usgs_lon': 'longitude', 'usgs_lat': 'latitude'})
                  if catalog_df is not None else None)
catalog_mtj = (catalog_df[
    catalog_df['usgs_lat'].between(mtj_lat_min, mtj_lat_max) &
    catalog_df['usgs_lon'].between(mtj_lon_min, mtj_lon_max)
][['usgs_lon', 'usgs_lat']].rename(columns={'usgs_lon': 'longitude', 'usgs_lat': 'latitude'})
if catalog_df is not None else None)
stations_mtj = stations_df[
    stations_df['latitude'].between(mtj_lat_min, mtj_lat_max) &
    stations_df['longitude'].between(mtj_lon_min, mtj_lon_max)
]

# ── Overview: all priors, full region ─────────────────────────────────────
fig = plot_overview_map(
    output_dir  = OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = [-128.5, -113, 31, 44],
    events_df   = catalog_events,
    stations_df = stations_df,
    bg          = bg,
    title       = 'bEPIC final locations — prior comparison',
    save_path   = os.path.join(FIGURES_DIR, 'comparison_benchmark_locations.png'),
)
plt.show()

# %%

# ── MTJ grid: one prior per panel ─────────────────────────────────────────
fig = plot_location_grid(
    output_dir     = OUTPUT_DIR,
    prior_order    = PRIOR_ORDER,
    extent         = MTJ_EXTENT,
    ref_catalog    = catalog_df,
    events_df      = catalog_mtj,
    stations_df    = stations_mtj,
    bg             = bg,
    filter_fn      = in_extent,
    show_scale_bar = True,
    title          = 'bEPIC MTJ locations — prior comparison',
    save_path      = os.path.join(FIGURES_DIR, 'MTJ_grid_benchmark_locations.png'),
)
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
# ── MTJ single-event posterior grid (prior background + posterior contours) ──
# Auto-selects the first MTJ event from the reference catalog that has a .run file.
# Override MTJ_EVENT_ID with a specific event_id (int) to pin a particular event.

MTJ_EVENT_ID   = None   # None = auto-select from MTJ region
MTJ_VERSION    = None   # None = last available trigger version

if catalog_df is not None:
    mtj_catalog = catalog_df[
        catalog_df['usgs_lat'].between(mtj_lat_min, mtj_lat_max) &
        catalog_df['usgs_lon'].between(mtj_lon_min, mtj_lon_max)
    ]

    if MTJ_EVENT_ID is None:
        # Pick first MTJ event that has a matching .run file
        for eid in mtj_catalog['event_id']:
            candidate = os.path.join(RUN_DIR, f'{eid}.run')
            if os.path.exists(candidate):
                MTJ_EVENT_ID = eid
                break

    if MTJ_EVENT_ID is not None:
        focus_run_path = os.path.join(RUN_DIR, f'{MTJ_EVENT_ID}.run')
        ref_row = catalog_df[catalog_df['event_id'] == MTJ_EVENT_ID]
        ref_lat = float(ref_row['usgs_lat'].iloc[0]) if not ref_row.empty else None
        ref_lon = float(ref_row['usgs_lon'].iloc[0]) if not ref_row.empty else None

        fig = plot_posterior_grid(
            focus_run_path = focus_run_path,
            cache_paths    = cache_paths,
            prior_order    = PRIOR_ORDER,
            params_kw      = {
                'grid_size': config.BENCHMARK_PARAMS['grid_size'],
                'grid_km':   config.BENCHMARK_PARAMS['grid_km'],
                'max_trigs': MAX_TRIGS,
                'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
                'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
            },
            ref_lat        = ref_lat,
            ref_lon        = ref_lon,
            focus_version  = MTJ_VERSION,
            title          = f'bEPIC posterior grid — MTJ event {MTJ_EVENT_ID}',
            save_path      = os.path.join(FIGURES_DIR, f'MTJ_posterior_grid_{MTJ_EVENT_ID}.png'),
        )
        plt.show()

        fig = plot_location_trajectory(
            event_id     = MTJ_EVENT_ID,
            output_dir   = OUTPUT_DIR,
            prior_order  = PRIOR_ORDER,
            run_dir      = RUN_DIR,
            min_triggers = 4,
            ref_lat      = ref_lat,
            ref_lon      = ref_lon,
            title        = f'bEPIC location trajectory — MTJ event {MTJ_EVENT_ID}',
            save_path    = os.path.join(FIGURES_DIR, f'MTJ_trajectory_{MTJ_EVENT_ID}.png'),
        )
        plt.show()
    else:
        print('[posterior grid] No MTJ event with a matching .run file found — skipping.')
else:
    print('[posterior grid] No reference catalog loaded — skipping.')

# %%
