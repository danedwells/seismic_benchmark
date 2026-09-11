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
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Custom repository imports
from priors import SeismicPrior
from benchmark.background import load_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_coverage_panel,
                             plot_overview_map, plot_location_grid,
                             plot_qq_calibration, plot_qq_calibration_prior,
                             plot_qq_prior_comparison)

from benchmark.usgs import *
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import load_station_availability_cache, make_epic_params


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = SeismicPrior.data_dir            # priors/data/
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}
CASE_STUDIES = config.CASE_STUDIES
# --- Select active case study --- (override with CASE_STUDY env var)

DEFAULT_CASE_STUDY = "Ferndale"
ACTIVE_CASE_STUDY = os.environ.get('CASE_STUDY', DEFAULT_CASE_STUDY)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEIS_CACHE   = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
AVAIL_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'case_studies',f'{ACTIVE_CASE_STUDY}', 'station_availability_cache.parquet')
cs = CASE_STUDIES[ACTIVE_CASE_STUDY]

# Params 
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
SIGMA_S        = config.BENCHMARK_PARAMS['sigma_s']
SIGMA_S = 0.22
config.BENCHMARK_PARAMS['sigma_s'] = SIGMA_S

# Directories
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

catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR, REDOWNLOAD=False)
print(f"{len(catalog_df)} events in {cs['name']} catalog.")
print(catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']].head())

# ------------------------------------------------------------------------------
# ── 1. Run bEPIC across priors ────────────────────────────────────────────
# ------------------------------------------------------------------------------

# Build reference catalog before job_args so it can be passed to each worker.
cs_ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]

# Get the station availability inventory from (preparation_scripts/build_station_availability.py)
_avail = (load_station_availability_cache(AVAIL_CACHE)
          if os.path.exists(AVAIL_CACHE) else None)
if _avail:
    print("Station availability cache loaded")

#%%
# ---------------------------------------------------------
# Construct parameters and run
#----------------------------------------------------------
priors_to_run = ['GEAR1']#, 'NSHM', 'KDE_Seismicity', 'Helmstetter', 'Uniform']
for name,path in cache_paths.items():
    if path is not None and name in priors_to_run:
        prior = SeismicPrior.from_tt3(path)
        use_prior = True
    elif path is None and name == 'Uniform':
        path = cache_paths['NSHM']
    params = make_epic_params(prior, use_prior, config.BENCHMARK_PARAMS, station_inventory=_avail)
 
    benchmark_runner.run_prior(params, name, cs_ref_df, CS_RUN_DIR,CS_OUTPUT_DIR)

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
    ref_catalog = cs_ref_df,
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
    column      = 'map_err_km',
    bins        = np.linspace(0, 100, 51),
    title       = f'bEPIC location errors — {cs["name"]}',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'location_error_histograms.png'),
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
)
plt.show()


# %%
