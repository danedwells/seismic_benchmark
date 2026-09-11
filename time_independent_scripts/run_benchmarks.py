#%%
# =============================================================================
# run_benchmarks.py  —  bEPIC prior benchmark
# Prerequisite: run scripts/build_priors.py first to build the .tt3 cache files.
# =============================================================================
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Custom repository imports
from priors import SeismicPrior
from benchmark.background import load_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_coverage_panel,
                             plot_overview_map,
                             plot_location_grid, plot_posterior_grid,
                             plot_location_trajectory,
                             plot_qq_calibration, plot_qq_calibration_prior,
                             plot_qq_prior_comparison)
from benchmark.metrics import posterior_confidence_level, posterior_coverage
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import  load_station_availability_cache, get_unique_stations


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = SeismicPrior.data_dir  # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEIS_CACHE          = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
STATION_AVAIL_CACHE = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'station_availability_cache.parquet')
RUN_DIR             = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
EDT_SIGMA_S    = config.BENCHMARK_PARAMS['edt_sigma_s']
SIGMA_S        = config.BENCHMARK_PARAMS['sigma_s']
DTT_WEIGHT     = config.BENCHMARK_PARAMS['dtt_weight']
EDT_TAG        = f'edt_{EDT_SIGMA_S}'
S_TAG          = f'sig_{SIGMA_S}'
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']

_VARY_EDT      = os.environ.get('VARY_EDT', '0') == '1'
_VARY_SIG      = os.environ.get('VARY_SIG', '1') == '1'

if _VARY_EDT == True & _VARY_SIG == True:
    raise Exception("Cannot vary both EDT and Sigma at the same time")
elif _VARY_EDT == True:
    OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'output',  'time_independent', EDT_TAG, f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'figures', 'time_independent', EDT_TAG, f'max_trigs_{MAX_TRIGS}')
elif _VARY_SIG == True:
    OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'output',  'time_independent', S_TAG, f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'figures', 'time_independent', S_TAG, f'max_trigs_{MAX_TRIGS}')
else:
    OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'output',  'time_independent', f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'figures', 'time_independent', f'max_trigs_{MAX_TRIGS}')

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Reference catalog and station list
# ---------------------------------------------------------------------------
# Run bEPIC on this catalog, updating ETAS and prior as it goes.
catalog_path = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
catalog_df = benchmark_runner.load_reference_catalog(catalog_path) if os.path.exists(catalog_path) else None

cache_paths['KDE_Seismicity'] = os.path.join(data_dir, 'kde_seismicity_benchmark.tt3')

station_availability = (
    load_station_availability_cache(STATION_AVAIL_CACHE)
    if os.path.exists(STATION_AVAIL_CACHE) else None
)

#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

RUN_ALL_PRIORS  = True  # run all six priors in parallel

# ── 1. Create reference locations ─────────────────────────────────────────
ref_dir = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference')

job_args = [
    {
        'prior_name':                name,
        'cache_path':                path,
        'nshm_path':                 cache_paths['NSHM'],  # geometry fallback for Uniform
        'run_dir':                   RUN_DIR,
        'output_dir':                OUTPUT_DIR,
        'catalog_path':              catalog_path,
        'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
        'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
        'max_trigs':                 config.BENCHMARK_PARAMS['max_trigs'],
        'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
        'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
        'station_availability':      station_availability,
        'dtt_weight': DTT_WEIGHT,
        'edt_sigma_s': EDT_SIGMA_S,
        'sigma_s': SIGMA_S,
    }
    for name, path in cache_paths.items()
]

if RUN_ALL_PRIORS:
    benchmark_runner.run_all_priors_parallel(benchmark_runner.run_prior, job_args)

stations_df = get_unique_stations(RUN_DIR)

bg = load_background_seismicity(
    cache_path  = SEIS_CACHE,
    bounds      = (-129, -112, 30, 45),
    start_year  = 2000,
    end_year    = 2025,
    min_mag     = 2.0,
)

#%%
# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

PRIOR_ORDER = ['Gear1', 'NSHM', 'Helmstetter', 'KDE_Seismicity', 'Uniform']

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
    cache_paths    = cache_paths,
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
    column      = 'map_err_km',
    bins        = bins_km,
    title       = 'bEPIC location error distributions — prior comparison',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'Grid_location_error_histograms.png'),
)
plt.show()

# ── MTJ location error histograms ─────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'map_err_km',
    bins        = bins_km,
    title       = 'bEPIC MTJ location error distributions — prior comparison',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'MTJ_location_error_histograms.png'),
    filter_fn   = in_extent,
)
plt.show()

# ── posterior_confidence_level histograms ────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'posterior_confidence_level',
    bins        = np.linspace(0, 1, 41),
    title       = 'bEPIC posterior calibration — posterior_confidence_level distributions',
    xlabel      = 'posterior_confidence_level',
    save_path   = os.path.join(FIGURES_DIR, 'posterior_confidence_level_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── posterior coverage at fixed radii (2×2 panel) ─────────────────────────
fig = plot_coverage_panel(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = 'bEPIC posterior coverage at fixed radii — prior comparison',
    save_path   = os.path.join(FIGURES_DIR, 'posterior_coverage_histograms.png'),
)
plt.show()

# ── Calibration Q-Q: posterior_confidence_level vs U(0,1) ────────────────────────
fig = plot_qq_calibration(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = 'bEPIC posterior calibration — posterior_confidence_level vs U(0,1)',
    save_path   = os.path.join(FIGURES_DIR, 'qq_calibration.png'),
)
plt.show()

# ── Prior calibration Q-Q: prior_confidence_level vs U(0,1) ────────────
fig = plot_qq_calibration_prior(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = 'bEPIC prior calibration — prior_confidence_level vs U(0,1)',
    save_path   = os.path.join(FIGURES_DIR, 'qq_calibration_prior.png'),
)
plt.show()

# ── Prior-vs-prior Q-Q comparison: map_err_km ─────────────────────────────
fig = plot_qq_prior_comparison(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'map_err_km',
    title       = 'Q-Q prior comparison — map location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'qq_prior_comparison.png'),
)
plt.show()
# %%
