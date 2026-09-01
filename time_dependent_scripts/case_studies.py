#%%
# =============================================================================
# case_studies.py  —  bEPIC case-study runner with dynamic ETAS prior
# =============================================================================
# Runs bEPIC with a time-evolving ETAS prior over a predefined aftershock
# sequence.  The ETAS prior updates after every located event so the spatial
# prior reflects the current aftershock distribution.
#
# How the dynamic ETAS prior works
# ---------------------------------
# Before each event is located, EtasPriorUpdater.update() evaluates the ETAS
# conditional intensity using all events in the rolling catalog up to that
# point.  After the event is located, it is appended to the rolling catalog
# so that the next event's prior sees it.
#
# Causal order per event:
#   update_prior(updater.update(t))  →  run_event()  →  updater.append_events()
#
# Prerequisites
# -------------
#   preparation_scripts/case_study_preparation.py  — download catalog + .run files
#   time_dependent_scripts/build_initial_prior.py  — ETAS parameter inversion
#     (produces data/etas_inversion/parameters_benchmark.json)
#
# Usage
# -----
#   Set ACTIVE_CASE_STUDY, flip control flags, run cells in order.
# =============================================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Custom repository imports
from priors import SeismicPrior, EtasPriorUpdater
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

from bEPIC import EPIC_locate_prelim

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = SeismicPrior.data_dir
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Case study definitions — loaded from benchmark/config.py
# ---------------------------------------------------------------------------
CASE_STUDIES = config.CASE_STUDIES

SIGMA_S = float(os.environ.get('BENCHMARK_SIGMA_S', 0.22))
print(SIGMA_S)
config.BENCHMARK_PARAMS['sigma_s'] = SIGMA_S

BW_SQ = 4
config.ETAS_INVERSION_CONFIG['bw_sq'] = BW_SQ

# ── CONFIGURE ─────────────────────────────────────────────────────────────────

#DEFAULT_CASE_STUDY = "Ferndale"
DEFAULT_CASE_STUDY = "MTJ_2024_M7"
ACTIVE_CASE_STUDY = os.environ.get('CASE_STUDY', DEFAULT_CASE_STUDY)

SEIS_CACHE       = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
INVERSION_JSON   = os.path.join(PROJECT_ROOT, 'data', 'case_studies',ACTIVE_CASE_STUDY, 'etas_inversion',
                                f'parameters_{config.etas_output_id(ACTIVE_CASE_STUDY)}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'case_studies', ACTIVE_CASE_STUDY, 'etas_inversion', 'input',
                                  f'catalog_{config.etas_catalog_tag(ACTIVE_CASE_STUDY)}.csv')

cs = CASE_STUDIES[ACTIVE_CASE_STUDY]
AVAIL_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'case_studies',f'{ACTIVE_CASE_STUDY}', 'station_availability_cache.parquet')


# How often to re-evaluate the ETAS prior (in seconds of event time).
# 0  → update before every event  (most accurate, slowest)
# 3600 → update at most once per hour of event time
ETAS_UPDATE_INTERVAL_S = int(os.environ.get('ETAS_UPDATE_INTERVAL_S', 0))

# Focus event for single-event posterior grid / trajectory figures.
# The prior used for this event is saved to disk during the run so it can be
# visualised even though every event has a different prior.
_MS_ = False  # set True to use mainshock events instead of representative aftershocks
FOCUS_EVENT_ID = config.FOCUS_EVENTS_MAINSHOCK[ACTIVE_CASE_STUDY] if _MS_ else config.FOCUS_EVENTS[ACTIVE_CASE_STUDY]
FOCUS_VERSION  = None   # None = last available trigger version

# Per-case-study directories
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
EDT_SIGMA_S    = config.BENCHMARK_PARAMS['edt_sigma_s']
SIGMA_S        = config.BENCHMARK_PARAMS['sigma_s']
SIGMA_S = 0.22
config.BENCHMARK_PARAMS['sigma_s'] = SIGMA_S
DTT_WEIGHT     = config.BENCHMARK_PARAMS['dtt_weight']
EDT_TAG        = f'edt_{EDT_SIGMA_S}'
S_TAG          = f'sig_{SIGMA_S}'

# Tags the ETAS inversion flags (free_background/free_productivity/mc/m_ref)
# these results were run against, so different inversion configs land in
# their own subfolder instead of overwriting each other's benchmark results
# — same idea as etas_output_id() for the inversion outputs themselves.
ETAS_TAG       = config.etas_run_tag()

_VARY_EDT      = os.environ.get('VARY_EDT', '0') == '1'
_VARY_SIG      = os.environ.get('VARY_SIG', '1') == '1'

# Quick-test toggle: DISABLE_ACTIVITY_MASK=1 skips loading the per-event
# station availability cache, so params.station_inventory stays None and
# bEPIC's activity-fraction mask (EPIC_locate_prelim.py's
# "Per-grid-point activity mask" block) never engages. Results land in a
# '_nomask'-suffixed subfolder so they don't clobber the normal run.
_DISABLE_ACTIVITY_MASK = os.environ.get('DISABLE_ACTIVITY_MASK', '0') == '1'
_MASK_SUFFIX = '_nomask' if _DISABLE_ACTIVITY_MASK else ''

CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')

if _VARY_EDT == True & _VARY_SIG == True:
    raise Exception("Cannot vary both EDT and Sigma at the same time")
elif _VARY_EDT == True:
    CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output',  'time_dependent', EDT_TAG, f'max_trigs_{MAX_TRIGS}{_MASK_SUFFIX}', ETAS_TAG)
    CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures', 'time_dependent', EDT_TAG, f'max_trigs_{MAX_TRIGS}{_MASK_SUFFIX}', ETAS_TAG)
elif _VARY_SIG == True:
    CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output',  'time_dependent', S_TAG, f'max_trigs_{MAX_TRIGS}{_MASK_SUFFIX}', ETAS_TAG)
    CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures', 'time_dependent', S_TAG, f'max_trigs_{MAX_TRIGS}{_MASK_SUFFIX}', ETAS_TAG)
else:
    CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output',  'time_dependent', f'max_trigs_{MAX_TRIGS}{_MASK_SUFFIX}', ETAS_TAG)
    CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures', 'time_dependent', f'max_trigs_{MAX_TRIGS}{_MASK_SUFFIX}', ETAS_TAG)

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR, CS_FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

focus_run_path = os.path.join(CS_RUN_DIR, f'{FOCUS_EVENT_ID}.run')

#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

# --- Control flags ---
RUN_DYNAMIC_PRIORS = True   # run all time-dependent priors (serial, event-by-event)
DEBUG_PLOT_PRIOR   = False  # plot ETAS lambda grid before each event

# Prior tempering exponent.  1.0 = full ETAS weight; <1.0 compresses the
# dynamic range, reducing overconfidence.  0.5 is a reasonable starting point.
PRIOR_ALPHA = float(os.environ.get('PRIOR_ALPHA', 1))  # UNCHANGED behavior if this == 1

_avail = (load_station_availability_cache(AVAIL_CACHE)
          if os.path.exists(AVAIL_CACHE) and not _DISABLE_ACTIVITY_MASK else None)
if _avail:
    print("Station availability cache loaded")
elif _DISABLE_ACTIVITY_MASK:
    print("DISABLE_ACTIVITY_MASK=1 — station_inventory left None, activity mask disabled")

#%%
# ---------------------------------------------------------------------------
# Load catalog from cache (preparation_scripts/case_study_preparation.py must
# have been run first)
# ---------------------------------------------------------------------------
catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR, REDOWNLOAD=False)

print(f"{len(catalog_df)} events in {cs['name']} catalog.")
print(catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']].head())

#%%

# ---------------------------------------------------------------------------
# 1. Dynamic prior runs (serial — prior state evolves event-by-event)
# ---------------------------------------------------------------------------
# Time-dependent priors cannot use ProcessPoolExecutor because their updaters
# hold mutable state (rolling catalog) that evolves through the sequence.
# Add new time-dependent priors here as they are implemented.
#
# Output: {prior}_benchmark_results.csv — same format as static priors.

if RUN_DYNAMIC_PRIORS:

    # ── ETAS dynamic ──────────────────────────────────────────────────────────

    # -- Verify inversion output exists --------------------------------------
    if not os.path.exists(INVERSION_JSON):
        raise FileNotFoundError(
            f"ETAS inversion output not found:\n  {INVERSION_JSON}\n"
            "Run time_dependent_scripts/build_initial_prior.py first."
        )

    # -- Load historical catalog (background seismicity for ETAS) ------------
    # This is the same catalog used for inversion.  Case-study events will be
    # appended to it incrementally as they are located.
    print(f"Loading historical catalog from:\n  {os.path.abspath(HISTORICAL_CATALOG)}")
    hist_catalog = pd.read_csv(
        HISTORICAL_CATALOG,
        index_col=0,
        parse_dates=['time'],
        dtype={'url': str, 'alert': str},
    )
    print(f"  {len(hist_catalog)} events loaded.")

    # -- Build EtasPriorUpdater from the pre-inverted parameters -------------
    print(f"\nBuilding EtasPriorUpdater from:\n  {INVERSION_JSON}")
    benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)
    updater = EtasPriorUpdater.from_inversion_json(
        json_path  = INVERSION_JSON,
        catalog_df = hist_catalog,
        **config.ETAS_UPDATER_CONFIG,
    )
    print(updater)

    # -- Prepare the case-study catalog in ETAS column format ----------------
    # Events are sorted chronologically so append_events() stays causal.
    # Only events above m_ref are meaningful for the ETAS intensity sum.
    # ETAS_INVERSION_CONFIG['mc'] is 'positive' (a rolling per-event
    # completeness computed inside ETASParameterCalculation, not a fixed
    # number) so it can't be used as a magnitude filter here — m_ref is the
    # fixed floor that mode still requires.
    m_ref = config.ETAS_INVERSION_CONFIG['m_ref']
    cs_etas_catalog = (
        catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']]
        .rename(columns={'mag': 'magnitude'})
        .assign(time=lambda df: pd.to_datetime(df['time']).dt.tz_localize(None))
        .query(f'magnitude >= {m_ref}')
        .sort_values('time')
        .reset_index(drop=True)
    )
    # Build a lookup: ANSS event_id → catalog row (for after_event_fn)
    cs_event_lookup = cs_etas_catalog.set_index('id')
    print(f"  {len(cs_etas_catalog)} case-study events above m_ref={m_ref} "
          f"will be fed to ETAS incrementally.")

    # -- Define callbacks ----------------------------------------------------

    def etas_update_fn(event_time_unix: float) -> SeismicPrior:
        t     = pd.Timestamp(event_time_unix, unit='s')
        prior = updater.update(t)
        if PRIOR_ALPHA != 1.0:
            prior.grid  = prior.grid ** PRIOR_ALPHA
            prior.grid /= prior.grid.sum()
        print(f"  [ETAS] prior updated at {t.strftime('%Y-%m-%d %H:%M:%S')} "
              f"— catalog size: {updater.n_catalog_events}")
        if DEBUG_PLOT_PRIOR:
            _fig, _ax = plt.subplots(1, 1, figsize=(7, 5))
            _pcm = _ax.pcolormesh(prior.lons, prior.lats,
                                  np.log10(prior.grid + 1e-12),
                                  cmap='viridis', shading='auto')
            plt.colorbar(_pcm, ax=_ax, label='log₁₀ λ')
            _ax.set_title(f'ETAS prior  {t.strftime("%Y-%m-%d %H:%M:%S")}  '
                          f'(n_cat={updater.n_catalog_events})', fontsize=9)
            _ax.set_xlabel('longitude'); _ax.set_ylabel('latitude')
            plt.tight_layout(); plt.pause(0.01); plt.close(_fig)
        return prior

    # def after_event_fn(event_id, event_time_unix: float) -> None:
    #     """
    #     Called by BenchmarkRunner immediately after each event is located.
    #     Appends the just-located event to the rolling ETAS catalog so the
    #     next prior update reflects it.
    #     """
    #     if event_id in cs_event_lookup.index:
    #         row = cs_event_lookup.loc[[event_id],
    #                                   ['time', 'latitude', 'longitude', 'magnitude']]
    #         updater.append_events(row)

    def after_event_fn(event_id):
        """
        Called by BenchmarkRunner immediately after each event is located.
        Appends the just-located event to updater.catalog so the next
        prior update sees it.
        """
        if event_id in cs_event_lookup.index:
            row = cs_event_lookup.loc[[event_id],
                                      ['time', 'latitude', 'longitude', 'magnitude']]
            updater.append_events(row)

    # -- Set up BenchmarkRunner with the initial (pre-sequence) prior --------
    t0 = pd.Timestamp(cs['starttime'])
    initial_prior = updater.update(t0)

    params = make_epic_params(initial_prior, True, config.BENCHMARK_PARAMS)

    cs_ref_df = catalog_df.rename(columns={
        'id':        'event_id',
        'latitude':  'usgs_lat',
        'longitude': 'usgs_lon',
    })[['event_id', 'usgs_lat', 'usgs_lon']]
    runner = BenchmarkRunner(prior=initial_prior, params=params, run_dir=CS_RUN_DIR,
                             catalog_df=cs_ref_df,station_availability=_avail)

    # Collect event IDs from available .run files, sorted by first trigger time
    # (chronological order is critical so ETAS updates are causal).
    run_files   = sorted(Path(CS_RUN_DIR).glob('*.run'))
    event_ids   = [f.stem for f in run_files]

    def _trigger_time(stem):
        try:
            df = pd.read_csv(CS_RUN_DIR + f'/{stem}.run', nrows=1)
            col = 'trigger time' if 'trigger time' in df.columns else 'trigger_time'
            return float(df[col].iloc[0])
        except Exception:
            return 0.0

    event_ids = sorted(event_ids, key=_trigger_time)
    print(f"\nRunning dynamic ETAS prior over {len(event_ids)} events "
          f"(update interval: {'per-event' if ETAS_UPDATE_INTERVAL_S == 0 else f'{ETAS_UPDATE_INTERVAL_S}s'})…\n")

    # -- Run ------------------------------------------------------------------
    runner.run_all(
        event_ids          = event_ids,
        etas_update_fn     = etas_update_fn,
        update_interval_s  = ETAS_UPDATE_INTERVAL_S,
        after_event_fn     = after_event_fn,
    )

    # -- Save results ---------------------------------------------------------
    out_path = os.path.join(CS_OUTPUT_DIR, 'etas_dynamic_benchmark_results.csv')
    runner_results_to_df(runner).to_csv(out_path, index=False)
    print(f"\nDynamic ETAS results saved to:\n  {out_path}")


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

PRIOR_ORDER    = ['ETAS_dynamic']
td_cache_paths = {'ETAS_dynamic': None}

min_lon, max_lon, min_lat, max_lat = cs['bounds']
min_lon -= 1; max_lon += 1; min_lat -= 1; max_lat += 1
cs_extent = [min_lon - 0.5, max_lon + 0.5, min_lat - 0.5, max_lat + 0.5]

bg_region = (bg[
    bg['longitude'].between(min_lon - 1, max_lon + 1) &
    bg['latitude'].between(min_lat - 1, max_lat + 1)
] if bg is not None else None)

# ── Overview map ──────────────────────────────────────────────────────────────
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

# ── Prior comparison grid ─────────────────────────────────────────────────────
fig = plot_location_grid(
    output_dir  = CS_OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = cs_extent,
    ref_catalog = cs_ref_df,
    events_df   = catalog_df[['longitude', 'latitude']],
    bg          = bg_region,
    cache_paths = td_cache_paths,
    title       = f'bEPIC locations — {cs["name"]} — prior comparison',
    save_path   = os.path.join(CS_FIGURES_DIR, 'grid_locations.png'),
)
plt.show()

#%%
# ── Location error histograms ─────────────────────────────────────────────────
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

# ── Fractional misfit histograms ──────────────────────────────────────────────
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
# =============================================================================
# Single-event posterior grid and location trajectory
# =============================================================================
# Builds a fresh ETAS prior for FOCUS_EVENT_ID from the historical catalog plus
# all case-study events that occurred before it.  No dependency on a saved .tt3
# or on having run the full benchmark loop first.
#
# TIME_PRIOR_BUFFER_DAYS : int or None
#     Lookback window for appending pre-event case-study catalog entries.
#     None = include all pre-event entries.
# =============================================================================

TIME_PRIOR_BUFFER_DAYS = 1   # lookback window for pre-event case-study events

_params_kw = {
    'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
    'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
    'max_trigs':                 MAX_TRIGS,
    'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
    'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
    'station_availability': _avail,
    'dtt_weight': DTT_WEIGHT,
    'edt_sigma_s': EDT_SIGMA_S,
    'sigma_s': SIGMA_S,

}

# The curated FOCUS_EVENT_ID (config.FOCUS_EVENTS[ACTIVE_CASE_STUDY]) doesn't
# always have a .run file for every case study / prep state — fall back to a
# random event from CS_RUN_DIR rather than just skipping this section.
if not os.path.exists(focus_run_path):
    print(f'[single-event] configured FOCUS_EVENT_ID {FOCUS_EVENT_ID} has no .run file: {focus_run_path}')
    _fallback_run_files = sorted(Path(CS_RUN_DIR).glob('*.run'))
    if _fallback_run_files:
        FOCUS_EVENT_ID = random.choice(_fallback_run_files).stem
        focus_run_path = os.path.join(CS_RUN_DIR, f'{FOCUS_EVENT_ID}.run')
        print(f'  → picked a random event instead: {FOCUS_EVENT_ID}')
    else:
        print(f'  → no .run files found in {CS_RUN_DIR} either; run BUILD_RUN_FILES first.')

if not os.path.exists(focus_run_path):
    print(f'[single-event] .run file not found: {focus_run_path}')
    print('  → set FOCUS_EVENT_ID to a built event, or run BUILD_RUN_FILES first.')
elif not os.path.exists(INVERSION_JSON):
    print(f'[single-event] inversion JSON not found: {INVERSION_JSON}')
else:
    _focus_ref = cs_ref_df[cs_ref_df['event_id'] == FOCUS_EVENT_ID]
    _ref_lat   = float(_focus_ref['usgs_lat'].iloc[0]) if not _focus_ref.empty else None
    _ref_lon   = float(_focus_ref['usgs_lon'].iloc[0]) if not _focus_ref.empty else None

    _focus_cat = catalog_df[catalog_df['id'] == FOCUS_EVENT_ID]
    if _focus_cat.empty:
        print(f'[single-event] event {FOCUS_EVENT_ID} not found in catalog.')
    else:
        _focus_t = pd.Timestamp(_focus_cat['time'].iloc[0]).replace(tzinfo=None)
        print(f'[single-event] focus event {FOCUS_EVENT_ID}  t = {_focus_t}')

        # -- Build fresh updater from historical catalog ----------------------
        try:
            _hist = hist_catalog
        except NameError:
            _hist = pd.read_csv(
                HISTORICAL_CATALOG,
                index_col=0,
                parse_dates=['time'],
                dtype={'url': str, 'alert': str},
            )

        benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)
        _updater = EtasPriorUpdater.from_inversion_json(
            json_path  = INVERSION_JSON,
            catalog_df = _hist,
            **config.ETAS_UPDATER_CONFIG,
        )

        # -- Append pre-event case-study catalog entries ----------------------
        m_ref = config.ETAS_INVERSION_CONFIG['m_ref']
        _cs_cat = (
            catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']]
            .rename(columns={'mag': 'magnitude'})
            .assign(time=lambda df: pd.to_datetime(df['time']).dt.tz_localize(None))
            .query(f'magnitude >= {m_ref}')
            .sort_values('time')
            .reset_index(drop=True)
        )
        _window_start = (
            _focus_t - pd.Timedelta(days=TIME_PRIOR_BUFFER_DAYS)
            if TIME_PRIOR_BUFFER_DAYS is not None else pd.Timestamp.min
        )
        _pre = _cs_cat[
            (_cs_cat['time'] < _focus_t) &
            (_cs_cat['time'] >= _window_start) &
            (_cs_cat['id'] != FOCUS_EVENT_ID)
        ][['time', 'latitude', 'longitude', 'magnitude']]
        if not _pre.empty:
            _updater.append_events(_pre)
            print(f'[single-event] appended {len(_pre)} pre-event case-study events.')

        # -- Compute ETAS prior at focus event time ---------------------------
        _standalone_prior = _updater.update(_focus_t)
        if PRIOR_ALPHA != 1.0:
            _standalone_prior.grid  = _standalone_prior.grid ** PRIOR_ALPHA
            _standalone_prior.grid /= _standalone_prior.grid.sum()
        print(f'[single-event] prior computed  (catalog size: {_updater.n_catalog_events})')

        _standalone_prior_path = os.path.join(CS_OUTPUT_DIR, f'standalone_prior_{FOCUS_EVENT_ID}.tt3')
        _standalone_prior.to_tt3(_standalone_prior_path)

        # -- Run bEPIC on just this event (for trajectory CSV) ---------------
        _s_params = make_epic_params(_standalone_prior, True, config.BENCHMARK_PARAMS)
        _single_runner = BenchmarkRunner(
            prior   = _standalone_prior,
            params  = _s_params,
            run_dir = CS_RUN_DIR,
        )
        _single_runner.run_event(FOCUS_EVENT_ID)

        _standalone_out_dir = os.path.join(CS_OUTPUT_DIR, f'standalone_{FOCUS_EVENT_ID}')
        os.makedirs(_standalone_out_dir, exist_ok=True)
        _standalone_csv = os.path.join(_standalone_out_dir, 'etas_dynamic_benchmark_results.csv')
        runner_results_to_df(_single_runner).to_csv(_standalone_csv, index=False)
        print(f'[single-event] results written → {_standalone_csv}')

        # -- Run bEPIC again to get the posterior grid object ----------------
        _t_cov, _odf_cov, _actual_v = run_single_event_get_grid(
            focus_run_path, _standalone_prior, True, _params_kw,
            focus_version=FOCUS_VERSION,
        )

        _standalone_cache = {'ETAS_dynamic': _standalone_prior_path}
        _precomputed = {
            'ETAS_dynamic': (_t_cov, _odf_cov, _actual_v,
                             _standalone_prior, _standalone_prior_path),
        }
        _buffer_label = (f'{TIME_PRIOR_BUFFER_DAYS}d lookback'
                         if TIME_PRIOR_BUFFER_DAYS else 'full history')
        deg_buf = 0.5
        _extent = [_ref_lon - deg_buf, _ref_lon + deg_buf,
                   _ref_lat - deg_buf, _ref_lat + deg_buf]

        # -- Posterior grid (prior background + posterior contours) ----------
        fig = plot_posterior_grid(
            focus_run_path = focus_run_path,
            cache_paths    = _standalone_cache,
            prior_order    = ['ETAS_dynamic'],
            params_kw      = _params_kw,
            prior_results  = _precomputed,
            ref_lat        = _ref_lat,
            ref_lon        = _ref_lon,
            extent         = _extent,
            focus_version  = FOCUS_VERSION,
            title          = f'ETAS prior/posterior — {cs["name"]} — event {FOCUS_EVENT_ID} ({_buffer_label})',
            save_path      = os.path.join(CS_FIGURES_DIR, f'standalone_posterior_{FOCUS_EVENT_ID}.png'),
        )
        plt.show()

        # -- Location trajectory ---------------------------------------------
        fig = plot_location_trajectory(
            event_id       = FOCUS_EVENT_ID,
            output_dir     = _standalone_out_dir,
            prior_order    = ['ETAS_dynamic'],
            run_dir        = CS_RUN_DIR,
            min_triggers   = 4,
            ref_lat        = _ref_lat,
            ref_lon        = _ref_lon,
            cache_paths    = _standalone_cache,
            extent         = _extent,
            extent_pad_deg = 0.1,
            title          = f'bEPIC location trajectory — {cs["name"]} — event {FOCUS_EVENT_ID} ({_buffer_label})',
            save_path      = os.path.join(CS_FIGURES_DIR, f'standalone_trajectory_{FOCUS_EVENT_ID}.png'),
        )
        plt.show()

# %%
