#%%
# =============================================================================
# run_benchmarks.py  —  bEPIC prior benchmark
# Prerequisite: run scripts/build_priors.py first to build the .tt3 cache files.
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Custom repository imports
from priors import SeismicPrior
from benchmark.background import load_background_seismicity, add_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_overview_map,
                             plot_location_grid, plot_posterior_grid,
                             plot_location_trajectory,
                             plot_qq_calibration, plot_qq_prior_comparison)
from benchmark.metrics import usgs_credible_level, posterior_coverage
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import (BenchmarkRunner, runner_results_to_df, get_unique_stations,
                              run_single_event_get_grid, make_epic_params)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = SeismicPrior.data_dir  # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEIS_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')
RUN_DIR     = os.path.join(PROJECT_ROOT, 'data', 'run_files')

MAX_TRIGS   = config.BENCHMARK_PARAMS['max_trigs']
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'output',  f'max_trigs_{MAX_TRIGS}')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', f'max_trigs_{MAX_TRIGS}')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Reference catalog and station list
# ---------------------------------------------------------------------------
# Run bEPIC on this catalog, updating ETAS and prior as it goes.
catalog_path = os.path.join(PROJECT_ROOT, 'data', 'reference', 'bEPIC_testing_catalog.txt')
catalog_df = benchmark_runner.load_reference_catalog(catalog_path) if os.path.exists(catalog_path) else None

# Lookup: event_id (int) → USGS time, lat, lon, magnitude
_usgs_ref_lookup = (
    catalog_df[['event_id', 'usgs_time', 'usgs_lat', 'usgs_lon', 'usgs_mag']]
    .rename(columns={'usgs_time': 'time', 'usgs_lat': 'latitude',
                     'usgs_lon': 'longitude', 'usgs_mag': 'magnitude'})
    .set_index('event_id')
    if catalog_df is not None else pd.DataFrame()
)

#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

# --- Control flags ---
REFERENCE       = False   # run high-resolution reference locations
RUN_ALL_PRIORS  = True  # run all six priors in parallel
SKIP_RUN        = False

# ── 1. Create reference locations ─────────────────────────────────────────
ref_dir = os.path.join(PROJECT_ROOT, 'data', 'reference')

if REFERENCE:
    benchmark_runner.create_reference_locations(RUN_DIR, ref_dir, cache_paths, config.REFERENCE_PARAMS)


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
        'max_trigs':                 MAX_TRIGS,
        'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
        'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
    }
    for name, path in cache_paths.items()
]

if RUN_ALL_PRIORS:
    benchmark_runner.run_all_priors_parallel(benchmark_runner.run_prior, job_args)
else:
    if SKIP_RUN == False:
        benchmark_runner.run_prior({
            'prior_name':                'Uniform',
            'cache_path':                None,
            'nshm_path':                 cache_paths['NSHM'],
            'run_dir':                   RUN_DIR,
            'output_dir':                OUTPUT_DIR,
            'catalog_path':              catalog_path,
            'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
            'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
            'max_trigs':                 MAX_TRIGS,
            'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
            'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
        })



stations_df = get_unique_stations(RUN_DIR)

bg = load_background_seismicity(
    cache_path  = SEIS_CACHE,
    bounds      = (-129, -112, 30, 45),
    start_year  = 2000,
    end_year    = 2025,
    min_mag     = 3.0,
)

#%%
# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

PRIOR_ORDER = ['Gear1', 'NSHM', 'Helmstetter', 'Smooth_seismicity', 'Uniform']

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

# ── usgs_credible_level histograms ────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'usgs_credible_level',
    bins        = np.linspace(0, 1, 41),
    title       = 'bEPIC posterior calibration — usgs_credible_level distributions',
    xlabel      = 'usgs_credible_level',
    save_path   = os.path.join(FIGURES_DIR, 'usgs_credible_level_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── posterior-mass ────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'coverage',
    bins        = np.linspace(0, 1, 41),
    title       = 'bEPIC posterior calibration — posterior coverage within location error',
    xlabel      = 'Posterior Coverage',
    save_path   = os.path.join(FIGURES_DIR, 'posterior_coverage_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── Calibration Q-Q: usgs_credible_level vs U(0,1) ────────────────────────
fig = plot_qq_calibration(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = 'bEPIC posterior calibration — usgs_credible_level vs U(0,1)',
    save_path   = os.path.join(FIGURES_DIR, 'qq_calibration.png'),
)
plt.show()

# ── Prior-vs-prior Q-Q comparison: map_err_km ─────────────────────────────
fig = plot_qq_prior_comparison(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'map_err_km',
    title       = 'Q-Q prior comparison — map location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'qq_prior_comparison.png'),
    catalog_df  = catalog_df,
)
plt.show()

# %%
# ── MTJ single-event posterior grid (prior background + posterior contours) ──
# Auto-selects the first MTJ event from the reference catalog that has a .run file.
# Override MTJ_EVENT_ID with a specific event_id (int) to pin a particular event.

MTJ_EVENT_ID   = 130646   # None = auto-select from MTJ region
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

        params_kw = {
            'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
            'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
            'max_trigs':                 MAX_TRIGS,
            'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
            'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
        }

        # Run bEPIC once per prior and cache — reused by plot_posterior_grid
        # and the posterior statistics block below so bEPIC is not called twice.
        prior_results = {}
        for prior_name in PRIOR_ORDER:
            tt3_path = cache_paths[prior_name]
            if tt3_path is None:
                p         = SeismicPrior.from_tt3(cache_paths['NSHM'])
                use_prior = False
            else:
                p         = SeismicPrior.from_tt3(tt3_path)
                use_prior = True
            t_out, out_df, actual_v = benchmark_runner.run_single_event_get_grid(
                focus_run_path, p, use_prior, params_kw, focus_version=MTJ_VERSION,
            )
            prior_results[prior_name] = (t_out, out_df, actual_v, p, tt3_path)

        fig = plot_posterior_grid(
            focus_run_path = focus_run_path,
            cache_paths    = cache_paths,
            prior_order    = PRIOR_ORDER,
            params_kw      = params_kw,
            prior_results  = prior_results,
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
            cache_paths  = cache_paths,
            title        = f'bEPIC location trajectory — MTJ event {MTJ_EVENT_ID}',
            save_path    = os.path.join(FIGURES_DIR, f'MTJ_trajectory_{MTJ_EVENT_ID}.png'),
        )
        plt.show()

        # ── Posterior statistics ───────────────────────────────────────────────
        if ref_lat is not None and ref_lon is not None:
            from obspy.geodetics import gps2dist_azimuth as _gps2dist
            print(f'\n── Posterior statistics — event {MTJ_EVENT_ID} ──')
            for prior_name, (t_out, out_df, actual_v, sp, pcache) in prior_results.items():
                if t_out is None or out_df is None:
                    continue
                bEPIC_lat  = t_out.posterior_lat
                bEPIC_lon  = t_out.posterior_lon
                map_err_km = _gps2dist(ref_lat, ref_lon, bEPIC_lat, bEPIC_lon)[0] / 1000.0
                frac       = posterior_coverage(out_df, ref_lat, ref_lon, radii_km=map_err_km)
                usgs_contf = 100 * usgs_credible_level(out_df, ref_lat, ref_lon)
                print(f'  {prior_name}:')
                print(f'    MAP error         : {map_err_km:.1f} km')
                print(f'    Coverage (dist)   : {frac * 100:.1f}%  '
                      f'(mass within {map_err_km:.1f} km of USGS)')
                print(f'    USGS credible lvl : {usgs_contf:.1f}%')

    else:
        print('[posterior grid] No MTJ event with a matching .run file found — skipping.')
else:
    print('[posterior grid] No reference catalog loaded — skipping.')

# %%
