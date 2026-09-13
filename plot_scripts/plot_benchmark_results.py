#%%
# =============================================================================
# plot_benchmark_results.py — figures for ONE already-run main benchmark
# =============================================================================
# Reads the benchmark CSVs already written by a run_benchmarks.py /
# run_cascadia.py run and makes the standard figure set. Does not run bEPIC —
# run one of these first:
#   time_independent_scripts/run_benchmarks.py   (REGION='california')
#   time_independent_scripts/run_cascadia.py      (REGION='cascadia')
#   time_dependent_scripts/run_benchmarks.py      (REGION='california')
#   time_dependent_scripts/run_cascadia.py        (REGION='cascadia')
#   mixed_prior_scripts/run_benchmarks.py         (REGION='california' only)
#
# Works for any region / workflow combination — set REGION and WORKFLOW below
# (or override with the REGION / WORKFLOW env vars).
# =============================================================================

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'
import numpy as np
import matplotlib.pyplot as plt
from priors import SeismicPrior

# Custom repository imports
from benchmark.background import load_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_coverage_panel,
                             plot_overview_map, plot_location_grid,
                             plot_qq_calibration, plot_qq_calibration_prior,
                             plot_qq_prior_comparison)
from benchmark import runner as benchmark_runner
from benchmark.runner import get_unique_stations
from benchmark import config
from benchmark import config_cascadia

#%%
# ---------------------------------------------------------------------------
# Select which run to plot
# ---------------------------------------------------------------------------
REGION   = os.environ.get('REGION', 'california')          # 'california' | 'cascadia'
WORKFLOW = os.environ.get('WORKFLOW', 'time_independent')   # 'time_independent' | 'time_dependent' | 'mixed'

# Only used when WORKFLOW == 'mixed' — must match the ALPHA the run used
# (see mixed_prior_scripts/run_benchmarks.py). No cascadia+mixed combo exists yet.
ALPHA = 0.5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir     = SeismicPrior.data_dir            # priors/data/
MAX_TRIGS    = config.BENCHMARK_PARAMS['max_trigs']

# Per-region: run directory, background seismicity, reference catalog loader,
# KDE_Seismicity cache filename, and the overview-map extent.
if REGION == 'california':
    RUN_DIR         = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
    SEIS_CACHE       = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
    CATALOG_PATH     = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
    catalog_df       = benchmark_runner.load_reference_catalog(CATALOG_PATH) if os.path.exists(CATALOG_PATH) else None
    KDE_FILE         = 'kde_seismicity_benchmark.tt3'
    BG_BOUNDS        = (-129, -112, 30, 45)
    BG_MIN_MAG       = 2.0
    OVERVIEW_EXTENT  = [-128.5, -113, 31, 44]

elif REGION == 'cascadia':
    RUN_DIR          = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'run_files')
    SEIS_CACHE       = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference', 'background_seismicity.parquet')
    CATALOG_PATH     = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference', 'cascadia_test_catalog.csv')
    catalog_df       = benchmark_runner.load_reference_catalog_usgs(CATALOG_PATH) if os.path.exists(CATALOG_PATH) else None
    KDE_FILE         = 'kde_seismicity_cascadia.tt3'
    BG_BOUNDS        = (-132, -115, 40, 50)
    BG_MIN_MAG       = 3.5
    OVERVIEW_EXTENT  = list(config_cascadia.REFERENCE_CATALOG_CONFIG['bounds'])

else:
    raise ValueError(f"Unknown REGION '{REGION}' — expected 'california' or 'cascadia'")

cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}
cache_paths['KDE_Seismicity'] = os.path.join(data_dir, KDE_FILE)

# Per-workflow: which priors to plot, their .tt3 paths (for the location-grid
# background density — None means "don't draw one"), the output/figures
# directories to read/write, and a title suffix.
if WORKFLOW == 'time_independent':
    PRIOR_ORDER      = list(config.PRIOR_FILENAMES.keys())
    GRID_CACHE_PATHS = cache_paths
    OUTPUT_DIR       = os.path.join(PROJECT_ROOT, 'results', REGION, 'output',  'time_independent', f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR      = os.path.join(PROJECT_ROOT, 'results', REGION, 'figures', 'time_independent', f'max_trigs_{MAX_TRIGS}')
    TITLE_SUFFIX     = ''

elif WORKFLOW == 'time_dependent':
    PRIOR_ORDER      = ['ETAS_dynamic']
    GRID_CACHE_PATHS = {'ETAS_dynamic': None}   # no fixed .tt3 — prior evolves per event
    OUTPUT_DIR       = os.path.join(PROJECT_ROOT, 'results', REGION, 'output',  'time_dependent', f'max_trigs_{MAX_TRIGS}')
    FIGURES_DIR      = os.path.join(PROJECT_ROOT, 'results', REGION, 'figures', 'time_dependent', f'max_trigs_{MAX_TRIGS}')
    TITLE_SUFFIX     = ''

elif WORKFLOW == 'mixed':
    if REGION != 'california':
        raise ValueError("WORKFLOW='mixed' is only implemented for REGION='california' "
                          "(there's no mixed_prior_scripts/run_cascadia.py yet)")
    ALPHA_TAG        = f'alpha_{ALPHA:.2f}'
    PRIOR_ORDER      = [f'{name}_etas_mixed' for name in config.PRIOR_FILENAMES]
    GRID_CACHE_PATHS = {name: None for name in PRIOR_ORDER}   # blended per event — no fixed .tt3
    OUTPUT_DIR       = os.path.join(PROJECT_ROOT, 'results', REGION, 'output',  'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
    FIGURES_DIR      = os.path.join(PROJECT_ROOT, 'results', REGION, 'figures', 'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
    TITLE_SUFFIX     = f' — mixed priors (alpha={ALPHA})'

else:
    raise ValueError(f"Unknown WORKFLOW '{WORKFLOW}' — expected 'time_independent', 'time_dependent', or 'mixed'")

os.makedirs(FIGURES_DIR, exist_ok=True)

#%%
# ---------------------------------------------------------------------------
# Background seismicity and station list
# ---------------------------------------------------------------------------
stations_df = get_unique_stations(RUN_DIR)
bg = load_background_seismicity(
    cache_path  = SEIS_CACHE,
    bounds      = BG_BOUNDS,
    start_year  = 2000,
    end_year    = 2025,
    min_mag     = BG_MIN_MAG,
)

catalog_events = (catalog_df[['usgs_lon', 'usgs_lat']]
                  .rename(columns={'usgs_lon': 'longitude', 'usgs_lat': 'latitude'})
                  if catalog_df is not None else None)

#%%
# ---------------------------------------------------------------------------
# Figures — common to every region / workflow
# ---------------------------------------------------------------------------

# ── Overview: all priors, full region ─────────────────────────────────────
fig = plot_overview_map(
    output_dir  = OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = OVERVIEW_EXTENT,
    events_df   = catalog_events,
    stations_df = stations_df,
    bg          = bg,
    title       = f'bEPIC final locations — prior comparison{TITLE_SUFFIX}',
    save_path   = os.path.join(FIGURES_DIR, 'comparison_benchmark_locations.png'),
)
plt.show()

# %%
bins_frac = np.linspace(0, 0.5, 51)
bins_km   = np.linspace(0, 100, 51)

# ── Total fractional misfit histograms ────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'frac_misfit',
    bins        = bins_frac,
    title       = f'bEPIC fractional misfit distributions — prior comparison{TITLE_SUFFIX}',
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
    title       = f'bEPIC location error distributions — prior comparison{TITLE_SUFFIX}',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'Grid_location_error_histograms.png'),
)
plt.show()

# ── posterior_confidence_level histograms ──────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'posterior_confidence_level',
    bins        = np.linspace(0, 1, 41),
    title       = f'bEPIC posterior calibration — posterior_confidence_level distributions{TITLE_SUFFIX}',
    xlabel      = 'posterior_confidence_level',
    save_path   = os.path.join(FIGURES_DIR, 'posterior_confidence_level_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── posterior coverage at fixed radii (2×2 panel) ──────────────────────────
fig = plot_coverage_panel(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = f'bEPIC posterior coverage at fixed radii — prior comparison{TITLE_SUFFIX}',
    save_path   = os.path.join(FIGURES_DIR, 'posterior_coverage_histograms.png'),
)
plt.show()

# ── Calibration Q-Q: posterior_confidence_level vs U(0,1) ──────────────────
fig = plot_qq_calibration(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = f'bEPIC posterior calibration — posterior_confidence_level vs U(0,1){TITLE_SUFFIX}',
    save_path   = os.path.join(FIGURES_DIR, 'qq_calibration.png'),
)
plt.show()

# ── Prior calibration Q-Q: prior_confidence_level vs U(0,1) ────────────────
fig = plot_qq_calibration_prior(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = f'bEPIC prior calibration — prior_confidence_level vs U(0,1){TITLE_SUFFIX}',
    save_path   = os.path.join(FIGURES_DIR, 'qq_calibration_prior.png'),
)
plt.show()

# ── Prior-vs-prior Q-Q comparison: map_err_km ───────────────────────────────
fig = plot_qq_prior_comparison(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'map_err_km',
    title       = f'Q-Q prior comparison — map location error (km){TITLE_SUFFIX}',
    save_path   = os.path.join(FIGURES_DIR, 'qq_prior_comparison.png'),
)
plt.show()

#%%
# ---------------------------------------------------------------------------
# Mendocino Triple Junction (MTJ) zoom — California only, no Cascadia analog
# ---------------------------------------------------------------------------
if REGION == 'california':
    MTJ_EXTENT = [-128.5, -122.5, 38.5, 42.5]
    mtj_lon_min, mtj_lon_max, mtj_lat_min, mtj_lat_max = MTJ_EXTENT

    def in_extent(df):
        return df[
            df['posterior_lat'].between(mtj_lat_min, mtj_lat_max) &
            df['posterior_lon'].between(mtj_lon_min, mtj_lon_max)
        ]

    catalog_mtj = (catalog_df[
        catalog_df['usgs_lat'].between(mtj_lat_min, mtj_lat_max) &
        catalog_df['usgs_lon'].between(mtj_lon_min, mtj_lon_max)
    ][['usgs_lon', 'usgs_lat']].rename(columns={'usgs_lon': 'longitude', 'usgs_lat': 'latitude'})
    if catalog_df is not None else None)
    stations_mtj = stations_df[
        stations_df['latitude'].between(mtj_lat_min, mtj_lat_max) &
        stations_df['longitude'].between(mtj_lon_min, mtj_lon_max)
    ]

    # ── MTJ grid: one panel per prior ──────────────────────────────────────
    fig = plot_location_grid(
        output_dir     = OUTPUT_DIR,
        prior_order    = PRIOR_ORDER,
        extent         = MTJ_EXTENT,
        ref_catalog    = catalog_df,
        events_df      = catalog_mtj,
        stations_df    = stations_mtj,
        bg             = bg,
        cache_paths    = GRID_CACHE_PATHS,
        filter_fn      = in_extent,
        show_scale_bar = True,
        title          = f'bEPIC MTJ locations — prior comparison{TITLE_SUFFIX}',
        save_path      = os.path.join(FIGURES_DIR, 'MTJ_grid_benchmark_locations.png'),
    )
    plt.show()

    # ── MTJ fractional misfit histograms ───────────────────────────────────
    fig = plot_prior_histograms(
        prior_names = PRIOR_ORDER,
        output_dir  = OUTPUT_DIR,
        column      = 'frac_misfit',
        bins        = bins_frac,
        title       = f'bEPIC MTJ fractional misfit distributions — prior comparison{TITLE_SUFFIX}',
        xlabel      = 'frac_misfit (fractional TT error)',
        save_path   = os.path.join(FIGURES_DIR, 'MTJ_grid_misfit_histograms.png'),
        filter_fn   = in_extent,
    )
    plt.show()

    # ── MTJ location error histograms ──────────────────────────────────────
    fig = plot_prior_histograms(
        prior_names = PRIOR_ORDER,
        output_dir  = OUTPUT_DIR,
        column      = 'map_err_km',
        bins        = bins_km,
        title       = f'bEPIC MTJ location error distributions — prior comparison{TITLE_SUFFIX}',
        xlabel      = 'location error (km)',
        save_path   = os.path.join(FIGURES_DIR, 'MTJ_location_error_histograms.png'),
        filter_fn   = in_extent,
    )
    plt.show()

# %%
