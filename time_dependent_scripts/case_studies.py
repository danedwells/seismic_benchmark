#%%
# =============================================================================
# case_studies.py  —  bEPIC case-study runner with dynamic ETAS prior
# =============================================================================
# Downloads a USGS catalog for a predefined aftershock sequence, builds .run
# trigger files from USGS phase data, then runs bEPIC with a time-evolving
# ETAS prior that updates after every located event.
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
# Prerequisite
# ------------
# Run time_dependent_scripts/build_initial_prior.py first to produce
#   data/etas_inversion/parameters_benchmark.json
# That file holds the pre-inverted ETAS parameters consumed here.
#
# Usage
# -----
#   Set ACTIVE_CASE_STUDY, flip control flags, run cells in order.
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from priors import SeismicPrior, EtasPriorUpdater
from benchmark.background import load_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_posterior_grid,
                              plot_location_trajectory, plot_overview_map,
                              plot_location_grid,
                              plot_qq_calibration, plot_qq_prior_comparison)
from benchmark.metrics import usgs_credible_level, posterior_coverage

from benchmark.usgs import *
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import BenchmarkRunner, compute_location_error
from bEPIC import EPIC_locate_prelim

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)

data_dir    = SeismicPrior.data_dir
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

SEIS_CACHE       = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')
INVERSION_JSON   = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion',
                                f'parameters_{config.ETAS_INVERSION_CONFIG["id"]}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion', 'input','example_catalog.csv')

# ---------------------------------------------------------------------------
# Case study definitions
# ---------------------------------------------------------------------------
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

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
ACTIVE_CASE_STUDY = 'Ferndale'

# How often to re-evaluate the ETAS prior (in seconds of event time).
# 0  → update before every event  (most accurate, slowest)
# 3600 → update at most once per hour of event time
ETAS_UPDATE_INTERVAL_S = 0

# Focus event for single-event posterior grid / trajectory figures.
# The prior used for this event is saved to disk during the run so it can be
# visualised even though every event has a different prior.
_MS_ = False
FOCUS_EVENTS = {
    'Ridgecrest': 'ci38548295',   # M 4.9 aftershock
    'Ferndale':   'nc73831091',   # M 4.05 aftershock
    'ElMayor':    'ci10148002',   # M 5.2 aftershock
}
if _MS_:
    FOCUS_EVENTS = {
        'Ridgecrest': 'ci38457511',
        'Ferndale':   'nc73821036',
        'ElMayor':    'ci14607652',
    }
FOCUS_EVENT_ID = FOCUS_EVENTS[ACTIVE_CASE_STUDY]
FOCUS_VERSION  = None

cs = CASE_STUDIES[ACTIVE_CASE_STUDY]

MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')
CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY,
                               'output', 'time_dependent',  f'max_trigs_{MAX_TRIGS}')
CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY,
                               'figures', 'time_dependent', f'max_trigs_{MAX_TRIGS}')

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR, CS_FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

focus_run_path   = os.path.join(CS_RUN_DIR,    f'{FOCUS_EVENT_ID}.run')
FOCUS_PRIOR_PATH = os.path.join(CS_OUTPUT_DIR, f'focus_prior_{FOCUS_EVENT_ID}.tt3')

#%%
# ---------------------------------------------------------------------------
# Control flags
# ---------------------------------------------------------------------------
DOWNLOAD_CATALOG   = False   # re-download even if cached
BUILD_RUN_FILES    = False   # build / rebuild .run files from USGS phases
RUN_DYNAMIC_PRIORS = True    # run all time-dependent priors (serial, event-by-event)
SKIP_RUN           = False   # skip all bEPIC calls; go straight to figures
DEBUG_PLOT_PRIOR   = True    # plot ETAS lambda grid before each event (comment out to disable)

#%%
# ---------------------------------------------------------------------------
# 1. Download (or load cached) catalog
# ---------------------------------------------------------------------------
catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR)
print(f"{len(catalog_df)} events in {cs['name']} catalog.")
print(catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']].head())

#%%
# ---------------------------------------------------------------------------
# 2. Build .run files
# ---------------------------------------------------------------------------
if BUILD_RUN_FILES:
    build_run_files_for_case_study(
        catalog_df    = catalog_df,
        run_dir       = CS_RUN_DIR,
        max_dist_deg  = 5.0,
        skip_existing = not DOWNLOAD_CATALOG,
    )

#%%


# ---------------------------------------------------------------------------
# 3. Dynamic prior runs (serial — prior state evolves event-by-event)
# ---------------------------------------------------------------------------
# Time-dependent priors cannot use ProcessPoolExecutor because their updaters
# hold mutable state (rolling catalog) that evolves through the sequence.
# Add new time-dependent priors here as they are implemented.
#
# Output: {prior}_benchmark_results.csv — same format as static priors.

if RUN_DYNAMIC_PRIORS and not SKIP_RUN:

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
    updater = EtasPriorUpdater.from_inversion_json(
        json_path  = INVERSION_JSON,
        catalog_df = hist_catalog,
        **config.ETAS_UPDATER_CONFIG,
    )
    print(updater)

    # -- Prepare the case-study catalog in ETAS column format ----------------
    # Events are sorted chronologically so append_events() stays causal.
    # Only events above mc are meaningful for the ETAS intensity sum.
    mc = config.ETAS_INVERSION_CONFIG['mc']
    cs_etas_catalog = (
        catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']]
        .rename(columns={'mag': 'magnitude'})
        .assign(time=lambda df: pd.to_datetime(df['time']).dt.tz_localize(None))
        .query(f'magnitude >= {mc}')
        .sort_values('time')
        .reset_index(drop=True)
    )
    # Build a lookup: ANSS event_id → catalog row (for after_event_fn)
    cs_event_lookup = cs_etas_catalog.set_index('id')
    print(f"  {len(cs_etas_catalog)} case-study events above mc={mc} "
          f"will be fed to ETAS incrementally.")

    # -- Define callbacks ----------------------------------------------------

    # Pre-compute the focus event's trigger time so etas_update_fn can detect
    # the first update at or past that point and save the prior to disk.
    def _read_trigger_time(run_path):
        try:
            df  = pd.read_csv(run_path, nrows=1)
            col = 'trigger time' if 'trigger time' in df.columns else 'trigger_time'
            return float(df[col].iloc[0])
        except Exception:
            return None

    _focus_trigger_time = _read_trigger_time(focus_run_path)
    _focus_prior_saved  = [False]   # mutable flag for closure

    def etas_update_fn(event_time_unix: float) -> SeismicPrior:
        """
        Called by BenchmarkRunner before each event (or on schedule).
        Evaluates the ETAS conditional intensity and returns a fresh prior.
        Saves the prior to disk the first time event_time reaches the focus event.
        """
        t = pd.Timestamp(event_time_unix, unit='s')
        save_path = None
        if (not _focus_prior_saved[0]
                and _focus_trigger_time is not None
                and event_time_unix >= _focus_trigger_time):
            save_path = FOCUS_PRIOR_PATH
            _focus_prior_saved[0] = True
            print(f"  [ETAS] saving focus prior for {FOCUS_EVENT_ID} → {FOCUS_PRIOR_PATH}")
        prior = updater.update(t, cache_path=save_path)
        print(f"  [ETAS] prior updated at {t.strftime('%Y-%m-%d %H:%M:%S')} "
              f"— catalog size: {updater.n_catalog_events}")

        # ── DEBUG: plot lambda grid ──────────────────────────────────────────
        if DEBUG_PLOT_PRIOR:
            _fig, _ax = plt.subplots(1, 1, figsize=(7, 5))
            _pcm = _ax.pcolormesh(prior.lons, prior.lats,
                                  np.log10(prior.grid + 1e-12),
                                  cmap='viridis', shading='auto')
            plt.colorbar(_pcm, ax=_ax, label='log₁₀ λ')
            _ax.set_title(f'ETAS prior  {t.strftime("%Y-%m-%d %H:%M:%S")}  '
                          f'(n_cat={updater.n_catalog_events})', fontsize=9)
            _ax.set_xlabel('longitude'); _ax.set_ylabel('latitude')
            plt.tight_layout()
            plt.pause(0.01)
            plt.close(_fig)
        # ── END DEBUG ────────────────────────────────────────────────────────

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

    params = EPIC_locate_prelim.EPIC_PARAMS()
    params.prior                     = initial_prior
    params.use_prior                 = True
    params.GridSize                  = config.BENCHMARK_PARAMS['grid_size']
    params.GridKm                    = config.BENCHMARK_PARAMS['grid_km']
    params.method                    = 'EPIC C'
    params.MAX_EVENT_TRIGS           = MAX_TRIGS
    params.migrate_grid              = config.BENCHMARK_PARAMS['migrate_grid']
    params.migrate_grid_min_triggers = config.BENCHMARK_PARAMS['migrate_grid_min_triggers']

    runner = BenchmarkRunner(prior=initial_prior, params=params, run_dir=CS_RUN_DIR)

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
    rows = [
        {
            'event_id':      eid,
            'version':       ver,
            'posterior_lat': t.posterior_lat,
            'posterior_lon': t.posterior_lon,
            'best_misfit':   t.best_misfit,
            'best_like':     t.best_like,
            'best_prior':    t.best_prior,
            'frac_misfit':   t.frac_misfit,
        }
        for (eid, ver), t in runner.results.items()
    ]
    out_path = os.path.join(CS_OUTPUT_DIR, 'etas_dynamic_benchmark_results.csv')
    (pd.DataFrame(rows)
       .sort_values(['event_id', 'version'])
       .to_csv(out_path, index=False))
    print(f"\nDynamic ETAS results saved to:\n  {out_path}")

#%%
# ---------------------------------------------------------------------------
# Reference catalog
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

PRIOR_ORDER    = ['ETAS_dynamic']
td_cache_paths = {'ETAS_dynamic': None}

# For the single-event posterior grid, swap in the saved focus prior (.tt3)
# so the plot function can render the actual prior used for that event.
focus_prior_cache_paths = {
    'ETAS_dynamic': FOCUS_PRIOR_PATH if os.path.exists(FOCUS_PRIOR_PATH) else None,
}

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
    ref_catalog = ref_df,
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
    column      = 'location_error_km',
    bins        = np.linspace(0, 100, 51),
    title       = f'bEPIC location errors — {cs["name"]}',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'location_error_histograms.png'),
    catalog_df  = ref_df,
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

# ── usgs_credible_level histograms ────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'usgs_credible_level',
    bins        = np.linspace(0, 1, 41),
    title       = 'bEPIC posterior calibration — usgs_credible_level distributions',
    xlabel      = 'usgs_credible_level',
    save_path   = os.path.join(CS_FIGURES_DIR, 'usgs_credible_level_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── posterior-mass ────────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'coverage',
    bins        = np.linspace(0, 1, 41),
    title       = 'bEPIC posterior calibration — posterior coverage within location error',
    xlabel      = 'Posterior Coverage',
    save_path   = os.path.join(CS_FIGURES_DIR, 'posterior_coverage_histograms.png'),
    color       = 'steelblue',
)

# ── Calibration Q-Q: usgs_credible_level vs U(0,1) ────────────────────────
fig = plot_qq_calibration(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    title       = 'bEPIC posterior calibration — usgs_credible_level vs U(0,1)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_calibration.png'),
)
plt.show()

# ── Prior-vs-prior Q-Q comparison: map_err_km ─────────────────────────────
fig = plot_qq_prior_comparison(
    prior_names = PRIOR_ORDER,
    output_dir  = CS_OUTPUT_DIR,
    column      = 'map_err_km',
    title       = 'Q-Q prior comparison — map location error (km)',
    save_path   = os.path.join(CS_FIGURES_DIR, 'qq_prior_comparison.png'),
    catalog_df  = catalog_df,
)
plt.show()

# %%
# =============================================================================
# Single-event posterior grid and location trajectory
# =============================================================================
if not os.path.exists(focus_run_path):
    print(f"[single-event figure] .run file not found: {focus_run_path}")
    print("  → set FOCUS_EVENT_ID to a built event, or run BUILD_RUN_FILES first.")
else:
    _focus_ref = ref_df[ref_df['event_id'] == FOCUS_EVENT_ID]
    _ref_lat   = float(_focus_ref['usgs_lat'].iloc[0]) if not _focus_ref.empty else None
    _ref_lon   = float(_focus_ref['usgs_lon'].iloc[0]) if not _focus_ref.empty else None

    if focus_prior_cache_paths['ETAS_dynamic'] is None:
        print(f"[single-event figure] focus prior not found: {FOCUS_PRIOR_PATH}")
        print("  → run with RUN_DYNAMIC_PRIORS=True first to generate it.")
    else:
        fig = plot_posterior_grid(
            focus_run_path = focus_run_path,
            cache_paths    = focus_prior_cache_paths,
            prior_order    = PRIOR_ORDER,
            params_kw      = {
                'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
                'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
                'max_trigs':                 MAX_TRIGS,
                'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
                'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
            },
            ref_lat       = _ref_lat,
            ref_lon       = _ref_lon,
            focus_version = FOCUS_VERSION,
            title         = f'bEPIC posterior grid — {cs["name"]} — event {FOCUS_EVENT_ID}',
            save_path     = os.path.join(CS_FIGURES_DIR, f'posterior_grid_{FOCUS_EVENT_ID}.png'),
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
        cache_paths  = td_cache_paths,
        title        = f'bEPIC location trajectory — {cs["name"]} — event {FOCUS_EVENT_ID}',
        save_path    = os.path.join(CS_FIGURES_DIR, f'trajectory_{FOCUS_EVENT_ID}.png'),
    )
    plt.show()

# %%
