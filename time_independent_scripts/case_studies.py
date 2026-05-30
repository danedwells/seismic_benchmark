#%%
# =============================================================================
# case_studies.py  —  bEPIC case-study runner (static priors)
# =============================================================================
# Runs bEPIC across all static spatial priors for a predefined aftershock
# sequence, mirroring run_benchmarks.py.
#
# Prerequisites
# -------------
#   preparation_scripts/case_study_preparation.py  — download catalog + .run files
#   preparation_scripts/build_priors.py            — build .tt3 prior cache
#
# Usage:
#   Set ACTIVE_CASE_STUDY to one of the keys in CASE_STUDIES, flip the
#   control flags, then run cells in order (or execute the whole script).
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Custom repository imports
from priors import SeismicPrior
from benchmark.background import load_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_coverage_panel,
                             plot_posterior_grid, plot_location_trajectory,
                             plot_overview_map, plot_location_grid,
                             plot_qq_calibration, plot_qq_calibration_prior,
                             plot_qq_prior_comparison)

from benchmark.usgs import *
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import (BenchmarkRunner, runner_results_to_df, get_unique_stations,
                              run_single_event_get_grid, make_epic_params,
                              load_station_availability_cache)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = SeismicPrior.data_dir            # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)
SEIS_CACHE   = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')
#AVAIL_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'case_studies',f'{ACTIVE_CASE_STUDY}', 'station_availability_cache.parquet')

# ---------------------------------------------------------------------------
# Case study definitions — loaded from benchmark/config.py
# ---------------------------------------------------------------------------
CASE_STUDIES = config.CASE_STUDIES

# --- Select active case study ---
ACTIVE_CASE_STUDY = 'Ferndale'
AVAIL_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'case_studies',f'{ACTIVE_CASE_STUDY}', 'station_availability_cache.parquet')
cs = CASE_STUDIES[ACTIVE_CASE_STUDY]

# Per-case-study directories
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')
CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output',  'time_independent', f'max_trigs_{MAX_TRIGS}')
CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures', 'time_independent', f'max_trigs_{MAX_TRIGS}')

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR, CS_FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

cache_paths['KDE_Seismicity'] = os.path.join(data_dir, f'kde_seismicity_{ACTIVE_CASE_STUDY}.tt3')

#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

# --- Control flags ---
RUN_ALL_PRIORS = True   # run all static priors in parallel

# ---------------------------------------------------------------------------
# Load catalog from cache (preparation_scripts/case_study_preparation.py must
# have been run first)
# ---------------------------------------------------------------------------

catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR, REDOWNLOAD=False)

print(f"{len(catalog_df)} events in {cs['name']} catalog.")
print(catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']].head())

#%%
# ------------------------------------------------------------------------------
# ── 1. Run bEPIC across priors ────────────────────────────────────────────
# ------------------------------------------------------------------------------

# Build reference catalog before job_args so it can be passed to each worker.
# Workers run in separate processes (ProcessPoolExecutor) — the DataFrame is
# pickled with the args dict, which is fine for a small case-study catalog.
_cs_ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]

_avail = load_station_availability_cache(AVAIL_CACHE) if os.path.exists(AVAIL_CACHE) else None
if _avail:
    print("Station availability cache loaded")
#_avail = None
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
        'catalog_df': _cs_ref_df,
        'station_availability': _avail,
    }
    for name, path in cache_paths.items()
]

# run_stems = {f.stem for f in Path(CS_RUN_DIR).glob('*.run')}
# if _avail:
#     covered = sum(1 for s in run_stems if str(s) in _avail)
#     print(f"Inventory coverage: {covered}/{len(run_stems)} events")
#     sample = next(iter(_avail.values()))
#     print(f"Sample inventory size: {len(sample)} stations")

# for eid in list(run_stems)[:5]:
#     inv = _avail.get(str(eid))
#     run_df = pd.read_csv(Path(CS_RUN_DIR) / f'{eid}.run')
#     run_df.columns = [c.replace(' ', '_') for c in run_df.columns]
#     n_trig = run_df['version'].eq(run_df['version'].max()).sum()
#     print(f"{eid}: inventory={len(inv)}, max_triggered={n_trig}, "
#         f"extra_stations={len(inv) - n_trig}")
#   If coverage is low, the cache was built from a different event set. If the sample size equals trigger_number, the
#   inventory only contains the triggered stations (see #3 below).
#%%
if RUN_ALL_PRIORS:
    benchmark_runner.run_all_priors_parallel(benchmark_runner.run_prior, job_args)


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


# ── posterior_confidence_level histograms ────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'posterior_confidence_level',
    bins        = np.linspace(0, 1, 41),
    title       = 'bEPIC posterior calibration — posterior_confidence_level distributions',
    xlabel      = 'posterior_confidence_level',
    save_path   = os.path.join(CS_FIGURES_DIR, 'posterior_confidence_level_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── posterior coverage at fixed radii (2×2 panel) ─────────────────────────
fig = plot_coverage_panel(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = f'bEPIC posterior coverage at fixed radii — {cs["name"]}',
    save_path   = os.path.join(CS_FIGURES_DIR, 'posterior_coverage_histograms.png'),
)

# ── Calibration Q-Q: posterior_confidence_level vs U(0,1) ────────────────────────
fig = plot_qq_calibration(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = 'bEPIC posterior calibration — posterior_confidence_level vs U(0,1)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_calibration.png'),
)
plt.show()

# ── Prior calibration Q-Q: prior_confidence_level vs U(0,1) ────────────
fig = plot_qq_calibration_prior(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = 'bEPIC prior calibration — prior_confidence_level vs U(0,1)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_calibration_prior.png'),
)
plt.show()

# ── Prior-vs-prior Q-Q comparison: map_err_km ─────────────────────────────
fig = plot_qq_prior_comparison(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'map_err_km',
    title       = 'Q-Q prior comparison — map location error (km)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_prior_comparison.png'),
    catalog_df  = ref_df,
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
# To add or change representative events, edit config.FOCUS_EVENTS in benchmark/config.py.
# Use examine_catalog.py (case-study section) to browse the catalog and pick IDs.
# =============================================================================

_MS_ = False  # set True to use mainshock events instead of representative aftershocks
FOCUS_EVENT_ID = config.FOCUS_EVENTS_MAINSHOCK[ACTIVE_CASE_STUDY] if _MS_ else config.FOCUS_EVENTS[ACTIVE_CASE_STUDY]
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
