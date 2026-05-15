#%%
# =============================================================================
# case_studies.py  —  bEPIC case-study runner with mixed (TI + ETAS) priors
# =============================================================================
# Runs bEPIC with each of the five time-independent spatial priors linearly
# blended with a time-evolving ETAS prior:
#
#   combined = ALPHA * etas_prior + (1 - ALPHA) * ti_prior
#
# The ETAS prior is evaluated once per event from a rolling catalog that grows
# as each event is located.  Events run in chronological order so the ETAS
# state remains causal.
#
# Prerequisites
# -------------
#   preparation_scripts/case_study_preparation.py  — download catalog + .run files
#   preparation_scripts/build_priors.py            — build .tt3 prior cache
#   time_dependent_scripts/build_initial_prior.py  — ETAS parameter inversion
#
# Usage
# -----
#   Set ACTIVE_CASE_STUDY, flip control flags, run cells in order.
# =============================================================================

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from priors import SeismicPrior, EtasPriorUpdater
from benchmark.background import load_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_coverage_panel,
                             plot_location_trajectory, plot_overview_map,
                             plot_location_grid, plot_qq_calibration,
                             plot_qq_calibration_prior, plot_qq_prior_comparison)
from benchmark.usgs import *
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import (BenchmarkRunner, runner_results_to_df, get_unique_stations,
                              make_epic_params)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Prior data directory - lives in priors/ repository folder
data_dir    = SeismicPrior.data_dir
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

# Root folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Case study definitions — loaded from benchmark/config.py
# ---------------------------------------------------------------------------
CASE_STUDIES = config.CASE_STUDIES

# ── CONFIGURE ────────────────────────────────────────────────────────────────

ACTIVE_CASE_STUDY = 'ElMayor'

# Background seismicity catalog (plotting only)
SEIS_CACHE         = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')
# ETAS inversion parameters and catalog — context-specific
INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion',
                                   f'parameters_{ACTIVE_CASE_STUDY}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion', 'input',
                                   f'catalog_{ACTIVE_CASE_STUDY}.csv')
cs = CASE_STUDIES[ACTIVE_CASE_STUDY]

# Blending weights: ALPHA on the ETAS component, (1-ALPHA) on the static prior.
ALPHA     = 0.5
ALPHA_TAG = f'alpha_{ALPHA:.2f}'

# How often to re-evaluate the ETAS prior (seconds of event time).
# 0 = update before every event (most accurate, slowest).
ETAS_UPDATE_INTERVAL_S = 0

# Focus event for the standalone posterior / trajectory figures.
_MS_ = False  # set True to use mainshock events instead of representative aftershocks
FOCUS_EVENT_ID = config.FOCUS_EVENTS_MAINSHOCK[ACTIVE_CASE_STUDY] if _MS_ else config.FOCUS_EVENTS[ACTIVE_CASE_STUDY]
FOCUS_VERSION  = None

# Per-case-study directories
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')
CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY,
                               'output', 'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY,
                               'figures', 'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR, CS_FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

focus_run_path = os.path.join(CS_RUN_DIR, f'{FOCUS_EVENT_ID}.run')

#%%
# ---------------------------------------------------------------------------
# Control flags
# ---------------------------------------------------------------------------

RUN_MIXED        = True   # run the blended prior benchmark
DEBUG_PLOT_PRIOR = False  # plot ETAS lambda grid before each event

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
# Blending utility
# ---------------------------------------------------------------------------

def blend_priors(ti_prior, etas_prior, alpha=0.5):
    """
    Blend ti_prior onto the ETAS grid and return a new SeismicPrior:

        combined = alpha * etas_prior.grid + (1 - alpha) * ti_resampled

    ti_prior  — SeismicPrior (static) or None for a Uniform base prior.
    etas_prior — SeismicPrior (time-dependent); defines the output grid.
    alpha      — weight on the ETAS component in [0, 1].
    """
    if ti_prior is None:
        ti_grid = np.ones_like(etas_prior.grid)
    else:
        interp = RegularGridInterpolator(
            (ti_prior.lats, ti_prior.lons),
            ti_prior.grid,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        lat_mesh, lon_mesh = np.meshgrid(etas_prior.lats, etas_prior.lons, indexing='ij')
        ti_grid = interp(
            np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
        ).reshape(len(etas_prior.lats), len(etas_prior.lons))
        ti_grid = np.clip(ti_grid, 0.0, None)
        ti_grid = np.nan_to_num(ti_grid, nan=0.0)

    ti_sum = ti_grid.sum()
    if ti_sum > 0:
        ti_grid /= ti_sum
    else:
        ti_grid = np.ones_like(etas_prior.grid) / etas_prior.grid.size

    combined      = alpha * etas_prior.grid + (1.0 - alpha) * ti_grid
    combined     /= combined.sum()

    mixed      = copy.deepcopy(etas_prior)
    mixed.grid = combined
    return mixed

cache_paths['KDE_Seismicity'] = os.path.join(data_dir, f'kde_seismicity_{ACTIVE_CASE_STUDY}.tt3')

#%%
# ---------------------------------------------------------------------------
# 3. Load time-independent priors
# ---------------------------------------------------------------------------

print("Loading time-independent priors:")
ti_priors = {}
for name, path in cache_paths.items():
    if path is None:
        ti_priors[name] = None
        print(f'  {name}: Uniform (flat base on ETAS grid)')
    elif not os.path.exists(path):
        raise FileNotFoundError(
            f"Static prior cache not found for '{name}':\n  {path}\n"
            "Run time_independent_scripts/build_priors.py first."
        )
    else:
        ti_priors[name] = SeismicPrior.from_tt3(path)
        print(f'  {name}: loaded from {os.path.basename(path)}')

#%%
# ---------------------------------------------------------------------------
# 4. Build EtasPriorUpdater
# ---------------------------------------------------------------------------

if not os.path.exists(INVERSION_JSON):
    raise FileNotFoundError(
        f"ETAS inversion output not found:\n  {INVERSION_JSON}\n"
        "Run time_dependent_scripts/build_initial_prior.py first."
    )

print(f"\nLoading historical catalog from:\n  {os.path.abspath(HISTORICAL_CATALOG)}")
hist_catalog = pd.read_csv(
    HISTORICAL_CATALOG,
    index_col=0,
    dtype={'url': str, 'alert': str},
)
hist_catalog['time'] = pd.to_datetime(
    hist_catalog['time'], format='ISO8601', utc=True
).dt.tz_convert(None)
print(f"  {len(hist_catalog)} events loaded.")

print(f"\nBuilding EtasPriorUpdater from:\n  {INVERSION_JSON}")
updater = EtasPriorUpdater.from_inversion_json(
    json_path  = INVERSION_JSON,
    catalog_df = hist_catalog,
    **config.ETAS_UPDATER_CONFIG,
)
print(updater)

# Prepare case-study events for incremental ETAS feeding.
# Only events at or above mc are meaningful for the ETAS intensity sum.
mc = config.ETAS_INVERSION_CONFIG['mc']
cs_etas_catalog = (
    catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']]
    .rename(columns={'mag': 'magnitude'})
    .assign(time=lambda df: pd.to_datetime(df['time']).dt.tz_localize(None))
    .query(f'magnitude >= {mc}')
    .sort_values('time')
    .reset_index(drop=True)
)
cs_event_lookup = cs_etas_catalog.set_index('id')
print(f"  {len(cs_etas_catalog)} case-study events at or above mc={mc} "
      f"will be fed to ETAS incrementally.")

#%%
# ---------------------------------------------------------------------------
# 5. Sort events chronologically (causal ETAS ordering)
# ---------------------------------------------------------------------------

def _trigger_time(stem):
    try:
        df  = pd.read_csv(os.path.join(CS_RUN_DIR, f'{stem}.run'), nrows=1)
        col = 'trigger time' if 'trigger time' in df.columns else 'trigger_time'
        return float(df[col].iloc[0])
    except Exception:
        return 0.0

run_files = sorted(Path(CS_RUN_DIR).glob('*.run'))
event_ids = sorted([f.stem for f in run_files], key=_trigger_time)

# Reference catalog (for location error computation)
cs_ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]

#%%
# ---------------------------------------------------------------------------
# 6. Mixed-prior benchmark loop
# ---------------------------------------------------------------------------
# Events run serially in chronological order.  The shared ETAS updater is
# advanced once per event; each of the 5 TI priors is blended with the
# current ETAS prior before running bEPIC.
station_inventory = None

if RUN_MIXED:

    # Initialise runners with blended priors at the start of the sequence
    _t0           = pd.Timestamp(cs['starttime'])
    _current_etas = updater.update(_t0)
    print(f"\nInitial ETAS prior evaluated at {_t0.strftime('%Y-%m-%d %H:%M:%S')}")

    runners = {}
    for name, ti_prior in ti_priors.items():
        initial_mixed = blend_priors(ti_prior, _current_etas, ALPHA)
        params        = make_epic_params(initial_mixed, True, config.BENCHMARK_PARAMS,
                                         station_inventory=station_inventory)
        runners[name] = BenchmarkRunner(
            prior      = initial_mixed,
            params     = params,
            run_dir    = CS_RUN_DIR,
            catalog_df = cs_ref_df,
        )

    _last_etas_update_unix = _t0.timestamp()

    print(f"\nRunning mixed-prior benchmark over {len(event_ids)} events "
          f"({len(runners)} TI priors × ETAS, alpha={ALPHA})…\n")

    for i, event_id in enumerate(event_ids):
        event_time_unix = _trigger_time(event_id)
        t = pd.Timestamp(event_time_unix, unit='s')

        # Re-evaluate ETAS if the update interval has elapsed
        if (ETAS_UPDATE_INTERVAL_S == 0 or
                event_time_unix - _last_etas_update_unix >= ETAS_UPDATE_INTERVAL_S):
            _current_etas          = updater.update(t)
            _last_etas_update_unix = event_time_unix
            print(f"  [ETAS] updated at {t.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"— catalog: {updater.n_catalog_events} events")

            if DEBUG_PLOT_PRIOR:
                _fig, _ax = plt.subplots(1, 1, figsize=(7, 5))
                _pcm = _ax.pcolormesh(
                    _current_etas.lons, _current_etas.lats,
                    np.log10(_current_etas.grid + 1e-12),
                    cmap='viridis', shading='auto',
                )
                plt.colorbar(_pcm, ax=_ax, label='log₁₀ λ (ETAS only)')
                _ax.set_title(f'ETAS prior  {t.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=9)
                _ax.set_xlabel('longitude'); _ax.set_ylabel('latitude')
                plt.tight_layout(); plt.pause(0.01); plt.close(_fig)

        # Run bEPIC for each blended prior
        for name, ti_prior in ti_priors.items():
            mixed = blend_priors(ti_prior, _current_etas, ALPHA)
            runners[name].update_prior(mixed)
            runners[name].run_event(event_id)

        # Feed case-study event location back to ETAS (once per event, causal)
        if event_id in cs_event_lookup.index:
            row = cs_event_lookup.loc[[event_id], ['time', 'latitude', 'longitude', 'magnitude']]
            updater.append_events(row)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(event_ids)} events complete.")

    print(f"\nSaving results to:\n  {CS_OUTPUT_DIR}")
    for name, runner in runners.items():
        out_path = os.path.join(CS_OUTPUT_DIR, f'{name.lower()}_etas_mixed_benchmark_results.csv')
        runner_results_to_df(runner).to_csv(out_path, index=False)
        print(f'  {name} → {os.path.basename(out_path)}')

#%%
# ---------------------------------------------------------------------------
# Reference catalog (for figures)
# ---------------------------------------------------------------------------

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

PRIOR_ORDER       = [f'{name}_etas_mixed' for name in ti_priors]
mixed_cache_paths = {pname: None for pname in PRIOR_ORDER}

min_lon, max_lon, min_lat, max_lat = cs['bounds']
min_lon -= 1; max_lon += 1; min_lat -= 1; max_lat += 1
cs_extent = [min_lon - 0.5, max_lon + 0.5, min_lat - 0.5, max_lat + 0.5]

bg_region = (bg[
    bg['longitude'].between(min_lon - 1, max_lon + 1) &
    bg['latitude'].between(min_lat - 1, max_lat + 1)
] if bg is not None else None)

# ── Overview map ──────────────────────────────────────────────────────────
fig = plot_overview_map(
    output_dir  = CS_OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = cs_extent,
    events_df   = catalog_df[['longitude', 'latitude']],
    bg          = bg_region,
    title       = f'bEPIC locations — {cs["name"]} — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(CS_FIGURES_DIR, 'comparison_locations.png'),
)
plt.show()

# ── 2×3 location grid ────────────────────────────────────────────────────
fig = plot_location_grid(
    output_dir  = CS_OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = cs_extent,
    ref_catalog = ref_df,
    events_df   = catalog_df[['longitude', 'latitude']],
    bg          = bg_region,
    cache_paths = mixed_cache_paths,
    title       = f'bEPIC locations — {cs["name"]} — mixed priors (alpha={ALPHA})',
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
    title       = f'bEPIC location errors — {cs["name"]} — mixed priors (alpha={ALPHA})',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'location_error_histograms.png'),
    catalog_df  = ref_df,
)
plt.show()

# ── Fractional misfit histograms ─────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'frac_misfit',
    bins        = np.linspace(0, 0.5, 51),
    title       = f'bEPIC fractional misfit — {cs["name"]} — mixed priors (alpha={ALPHA})',
    xlabel      = 'frac_misfit',
    save_path   = os.path.join(CS_FIGURES_DIR, 'misfit_histograms.png'),
)
plt.show()

# ── usgs_credible_level histograms ────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'usgs_credible_level',
    bins        = np.linspace(0, 1, 41),
    title       = f'bEPIC posterior calibration — {cs["name"]} — mixed priors (alpha={ALPHA})',
    xlabel      = 'usgs_credible_level',
    save_path   = os.path.join(CS_FIGURES_DIR, 'usgs_credible_level_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── Posterior coverage at fixed radii ────────────────────────────────────
fig = plot_coverage_panel(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = f'bEPIC posterior coverage — {cs["name"]} — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(CS_FIGURES_DIR, 'posterior_coverage_histograms.png'),
)
plt.show()

# ── Calibration Q-Q: usgs_credible_level vs U(0,1) ───────────────────────
fig = plot_qq_calibration(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = f'bEPIC posterior calibration Q-Q — {cs["name"]} — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_calibration.png'),
)
plt.show()

# ── Prior calibration Q-Q: usgs_prior_credible_level vs U(0,1) ──────────
fig = plot_qq_calibration_prior(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = f'bEPIC prior calibration Q-Q — {cs["name"]} — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_calibration_prior.png'),
)
plt.show()

# ── Prior-vs-prior Q-Q comparison: map_err_km ────────────────────────────
fig = plot_qq_prior_comparison(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'map_err_km',
    title       = f'Q-Q prior comparison — {cs["name"]} — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_prior_comparison.png'),
    catalog_df  = ref_df,
)
plt.show()

# %%
# =============================================================================
# Standalone single-event location trajectory
# =============================================================================
# Builds a fresh blended prior for FOCUS_EVENT_ID from the historical catalog
# plus all case-study events that preceded it.  No dependency on having run
# the full benchmark loop first.
#
# TIME_PRIOR_BUFFER_DAYS : int or None
#     Lookback window for appending pre-event case-study events to ETAS.
#     None = include all pre-event entries.
#
# Produces one CSV and one trajectory figure per TI prior, saved to a
# per-event subdirectory under CS_OUTPUT_DIR.
# =============================================================================

TIME_PRIOR_BUFFER_DAYS = 1

_params_kw = {
    'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
    'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
    'max_trigs':                 MAX_TRIGS,
    'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
    'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
}

if not os.path.exists(focus_run_path):
    print(f'[single-event] .run file not found: {focus_run_path}')
    print('  → set FOCUS_EVENT_ID to a built event, or run BUILD_RUN_FILES first.')
elif not os.path.exists(INVERSION_JSON):
    print(f'[single-event] inversion JSON not found: {INVERSION_JSON}')
else:
    _focus_ref = ref_df[ref_df['event_id'] == FOCUS_EVENT_ID]
    _ref_lat   = float(_focus_ref['usgs_lat'].iloc[0]) if not _focus_ref.empty else None
    _ref_lon   = float(_focus_ref['usgs_lon'].iloc[0]) if not _focus_ref.empty else None

    _focus_cat = catalog_df[catalog_df['id'] == FOCUS_EVENT_ID]
    if _focus_cat.empty:
        print(f'[single-event] event {FOCUS_EVENT_ID} not found in catalog.')
    else:
        _focus_t = pd.Timestamp(_focus_cat['time'].iloc[0]).replace(tzinfo=None)
        print(f'[single-event] focus event {FOCUS_EVENT_ID}  t = {_focus_t}')

        # Build a fresh updater from the historical catalog
        try:
            _hist = hist_catalog
        except NameError:
            _hist = pd.read_csv(
                HISTORICAL_CATALOG,
                index_col=0,
                dtype={'url': str, 'alert': str},
            )
            _hist['time'] = pd.to_datetime(
                _hist['time'], format='ISO8601', utc=True
            ).dt.tz_convert(None)

        _updater = EtasPriorUpdater.from_inversion_json(
            json_path  = INVERSION_JSON,
            catalog_df = _hist,
            **config.ETAS_UPDATER_CONFIG,
        )

        # Append pre-event case-study events within the lookback window
        _window_start = (
            _focus_t - pd.Timedelta(days=TIME_PRIOR_BUFFER_DAYS)
            if TIME_PRIOR_BUFFER_DAYS is not None else pd.Timestamp.min
        )
        _pre = cs_etas_catalog[
            (cs_etas_catalog['time'] < _focus_t) &
            (cs_etas_catalog['time'] >= _window_start) &
            (cs_etas_catalog['id'] != FOCUS_EVENT_ID)
        ][['time', 'latitude', 'longitude', 'magnitude']]
        if not _pre.empty:
            _updater.append_events(_pre)
            print(f'[single-event] appended {len(_pre)} pre-event case-study events.')

        # Evaluate ETAS prior at focus event time
        _standalone_etas = _updater.update(_focus_t)
        print(f'[single-event] ETAS prior computed (catalog size: {_updater.n_catalog_events})')

        _buffer_label = (f'{TIME_PRIOR_BUFFER_DAYS}d lookback'
                         if TIME_PRIOR_BUFFER_DAYS else 'full history')
        _standalone_out_dir = os.path.join(CS_OUTPUT_DIR, f'standalone_{FOCUS_EVENT_ID}')
        os.makedirs(_standalone_out_dir, exist_ok=True)

        deg_buf = 0.5
        _extent = [_ref_lon - deg_buf, _ref_lon + deg_buf,
                   _ref_lat - deg_buf, _ref_lat + deg_buf]

        # Run bEPIC once per TI prior and collect results
        _standalone_prior_order = PRIOR_ORDER
        for name, ti_prior in ti_priors.items():
            mixed_name   = f'{name}_etas_mixed'
            mixed_prior  = blend_priors(ti_prior, _standalone_etas, ALPHA)
            _s_params    = make_epic_params(mixed_prior, True, config.BENCHMARK_PARAMS)
            _s_runner    = BenchmarkRunner(
                prior   = mixed_prior,
                params  = _s_params,
                run_dir = CS_RUN_DIR,
            )
            _s_runner.run_event(FOCUS_EVENT_ID)
            _standalone_csv = os.path.join(
                _standalone_out_dir, f'{mixed_name.lower()}_benchmark_results.csv'
            )
            runner_results_to_df(_s_runner).to_csv(_standalone_csv, index=False)

        print(f'[single-event] results written → {_standalone_out_dir}/')

        # Location trajectory for all mixed priors
        fig = plot_location_trajectory(
            event_id       = FOCUS_EVENT_ID,
            output_dir     = _standalone_out_dir,
            prior_order    = _standalone_prior_order,
            run_dir        = CS_RUN_DIR,
            min_triggers   = 4,
            ref_lat        = _ref_lat,
            ref_lon        = _ref_lon,
            cache_paths    = {pname: None for pname in _standalone_prior_order},
            extent         = _extent,
            extent_pad_deg = 0.1,
            title          = (f'bEPIC location trajectory — {cs["name"]} — '
                              f'event {FOCUS_EVENT_ID} ({_buffer_label}, alpha={ALPHA})'),
            save_path      = os.path.join(CS_FIGURES_DIR,
                                          f'standalone_trajectory_{FOCUS_EVENT_ID}.png'),
        )
        plt.show()

# %%
