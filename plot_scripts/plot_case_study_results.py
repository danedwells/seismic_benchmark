#%%
# =============================================================================
# plot_case_study_results.py — figures for ONE already-run case study
# =============================================================================
# Reads the benchmark CSVs already written by a case_studies.py run and makes
# the standard figure set. Does not run bEPIC — run one of these first:
#   time_independent_scripts/case_studies.py
#   time_dependent_scripts/case_studies.py
#   mixed_prior_scripts/case_studies.py
#
# Works for any case study / workflow combination — just set ACTIVE_CASE_STUDY
# and WORKFLOW below (or override with the CASE_STUDY / WORKFLOW env vars).
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
from benchmark.usgs import download_case_study_catalog
from benchmark import config

#%%
# ---------------------------------------------------------------------------
# Select which run to plot
# ---------------------------------------------------------------------------
DEFAULT_CASE_STUDY = "Ferndale"
ACTIVE_CASE_STUDY  = os.environ.get('CASE_STUDY', DEFAULT_CASE_STUDY)

# 'time_independent' | 'time_dependent' | 'mixed'
WORKFLOW = os.environ.get('WORKFLOW', 'time_independent')

# Only used when WORKFLOW == 'mixed' — must match the ALPHA / SCHED_TAG the
# run was produced with (see mixed_prior_scripts/case_studies.py).
ALPHA     = 0.5
SCHED_TAG = 'off'   # 'off', 'tempering_only', or 'full_blend'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir     = SeismicPrior.data_dir            # priors/data/
CASE_STUDIES = config.CASE_STUDIES
cs           = CASE_STUDIES[ACTIVE_CASE_STUDY]
MAX_TRIGS    = config.BENCHMARK_PARAMS['max_trigs']

CS_DATA_DIR  = os.path.join(PROJECT_ROOT, 'data', 'case_studies', ACTIVE_CASE_STUDY)
SEIS_CACHE   = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')

cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}
cache_paths['KDE_Seismicity'] = os.path.join(data_dir, f'kde_seismicity_{ACTIVE_CASE_STUDY}.tt3')

# Per-workflow: which priors to plot, their .tt3 paths (for the location-grid
# background density — None means "don't draw one"), the output/figures
# directories to read/write, and a title suffix.
if WORKFLOW == 'time_independent':
    PRIOR_ORDER      = list(config.PRIOR_FILENAMES.keys())
    GRID_CACHE_PATHS = cache_paths
    CS_OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output',  'time_independent', f'max_trigs_{MAX_TRIGS}')
    CS_FIGURES_DIR   = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures', 'time_independent', f'max_trigs_{MAX_TRIGS}')
    TITLE_SUFFIX     = ''

elif WORKFLOW == 'time_dependent':
    PRIOR_ORDER      = ['ETAS_dynamic']
    GRID_CACHE_PATHS = {'ETAS_dynamic': None}   # no fixed .tt3 — prior evolves per event
    CS_OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output',  'time_dependent', f'max_trigs_{MAX_TRIGS}')
    CS_FIGURES_DIR   = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures', 'time_dependent', f'max_trigs_{MAX_TRIGS}')
    TITLE_SUFFIX     = ''

elif WORKFLOW == 'mixed':
    ALPHA_TAG        = f'alpha_{ALPHA:.2f}'
    PRIOR_ORDER      = [f'{name}_etas_mixed' for name in config.PRIOR_FILENAMES]
    GRID_CACHE_PATHS = {name: None for name in PRIOR_ORDER}   # blended per event — no fixed .tt3
    CS_OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY,
                                    'output', 'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG, f'sched_{SCHED_TAG}')
    CS_FIGURES_DIR   = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY,
                                    'figures', 'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG, f'sched_{SCHED_TAG}')
    TITLE_SUFFIX     = f' — mixed priors (alpha={ALPHA})'

else:
    raise ValueError(f"Unknown WORKFLOW '{WORKFLOW}' — expected 'time_independent', 'time_dependent', or 'mixed'")

os.makedirs(CS_FIGURES_DIR, exist_ok=True)

#%%
# ---------------------------------------------------------------------------
# Reference catalog + background seismicity (read from cache — no network
# access as long as case_study_preparation.py has already been run)
# ---------------------------------------------------------------------------
catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR, REDOWNLOAD=False)
cs_ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]

bg = load_background_seismicity(
    cache_path = SEIS_CACHE,
    bounds     = (-129, -112, 30, 45),
    start_year = 2000,
    end_year   = 2025,
    min_mag    = 3.5,
)

min_lon, max_lon, min_lat, max_lat = cs['bounds']
min_lon -= 1; max_lon += 1; min_lat -= 1; max_lat += 1
cs_extent = [min_lon - 0.5, max_lon + 0.5, min_lat - 0.5, max_lat + 0.5]

bg_region = (bg[
    bg['longitude'].between(min_lon - 1, max_lon + 1) &
    bg['latitude'].between(min_lat - 1, max_lat + 1)
] if bg is not None else None)

#%%
# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# ── Map: all priors compared ───────────────────────────────────────────────
fig = plot_overview_map(
    output_dir  = CS_OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = cs_extent,
    events_df   = catalog_df[['longitude', 'latitude']],
    bg          = bg_region,
    title       = f'bEPIC locations — {cs["name"]}{TITLE_SUFFIX}',
    save_path   = os.path.join(CS_FIGURES_DIR, 'comparison_locations.png'),
)
plt.show()

# ── Grid: one panel per prior ──────────────────────────────────────────────
fig = plot_location_grid(
    output_dir  = CS_OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = cs_extent,
    ref_catalog = cs_ref_df,
    events_df   = catalog_df[['longitude', 'latitude']],
    bg          = bg_region,
    cache_paths = GRID_CACHE_PATHS,
    title       = f'bEPIC locations — {cs["name"]}{TITLE_SUFFIX} — prior comparison',
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
    title       = f'bEPIC location errors — {cs["name"]}{TITLE_SUFFIX}',
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
    title       = f'bEPIC fractional misfit — {cs["name"]}{TITLE_SUFFIX}',
    xlabel      = 'frac_misfit',
    save_path   = os.path.join(CS_FIGURES_DIR, 'misfit_histograms.png'),
)
plt.show()

# ── posterior_confidence_level histograms ──────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'posterior_confidence_level',
    bins        = np.linspace(0, 1, 41),
    title       = f'bEPIC posterior calibration — {cs["name"]}{TITLE_SUFFIX}',
    xlabel      = 'posterior_confidence_level',
    save_path   = os.path.join(CS_FIGURES_DIR, 'posterior_confidence_level_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── posterior coverage at fixed radii (2×2 panel) ──────────────────────────
fig = plot_coverage_panel(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = f'bEPIC posterior coverage — {cs["name"]}{TITLE_SUFFIX}',
    save_path   = os.path.join(CS_FIGURES_DIR, 'posterior_coverage_histograms.png'),
)
plt.show()

# ── Calibration Q-Q: posterior_confidence_level vs U(0,1) ──────────────────
fig = plot_qq_calibration(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = f'bEPIC posterior calibration Q-Q — {cs["name"]}{TITLE_SUFFIX}',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_calibration.png'),
)
plt.show()

# ── Prior calibration Q-Q: prior_confidence_level vs U(0,1) ────────────────
fig = plot_qq_calibration_prior(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = f'bEPIC prior calibration Q-Q — {cs["name"]}{TITLE_SUFFIX}',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_calibration_prior.png'),
)
plt.show()

# ── Prior-vs-prior Q-Q comparison: map_err_km ───────────────────────────────
fig = plot_qq_prior_comparison(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'map_err_km',
    title       = f'Q-Q prior comparison — {cs["name"]}{TITLE_SUFFIX}',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_prior_comparison.png'),
)
plt.show()

# %%
