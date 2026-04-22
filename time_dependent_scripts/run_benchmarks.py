#%%
# =============================================================================
# run_benchmarks.py  —  bEPIC prior benchmark
# Prerequisite: run scripts/build_priors.py first to build the .tt3 cache files.
# =============================================================================
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from priors import SeismicPrior, EtasPriorUpdater
from benchmark.background import load_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_overview_map,
                             plot_location_grid, plot_posterior_grid,
                             plot_location_trajectory)
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import BenchmarkRunner, compute_location_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This should be /home/a01738353/2024_NEHRP/seismic_benchmark/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEIS_CACHE         = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')
RUN_DIR            = os.path.join(PROJECT_ROOT, 'data', 'run_files')
INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion',
                                   f'parameters_{config.ETAS_INVERSION_CONFIG["id"]}.json')
# This catalog is what was used for the ORIGINAL ETAS inersion
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion', 'input', 'downloaded_catalog.csv')

MAX_TRIGS   = config.BENCHMARK_PARAMS['max_trigs']
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'output', 'time_dependent',  f'max_trigs_{MAX_TRIGS}')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', 'time_dependent',  f'max_trigs_{MAX_TRIGS}')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Pick an event of interest to plot posterior and trajectory
MTJ_EVENT_ID     = 130646   # None = auto-select from MTJ region
MTJ_VERSION      = None     # None = last available trigger version
FOCUS_PRIOR_PATH = os.path.join(OUTPUT_DIR, f'focus_prior_{MTJ_EVENT_ID}.tt3')

# ---------------------------------------------------------------------------
# Reference catalog and station list
# ---------------------------------------------------------------------------
# Run bEPIC on this catalog, updating ETAS and prior as it goes.
catalog_path = os.path.join(PROJECT_ROOT, 'data', 'reference', 'bEPIC_testing_catalog.txt')
catalog_df = benchmark_runner.load_reference_catalog(catalog_path) if os.path.exists(catalog_path) else None

# Lookup used by after_event_fn: event_id (int) → USGS lat, lon, magnitude
_usgs_ref_lookup = (
    catalog_df[['event_id', 'usgs_lat', 'usgs_lon', 'usgs_mag']]
    .rename(columns={'usgs_lat': 'latitude', 'usgs_lon': 'longitude', 'usgs_mag': 'magnitude'})
    .set_index('event_id')
    if catalog_df is not None else pd.DataFrame()
)

# ---------------------------------------------------------------------------
# Control flags
# ---------------------------------------------------------------------------
RUN_DYNAMIC_PRIORS = True   # run time-dependent ETAS prior (serial, event-by-event)
SKIP_RUN           =  False
DEBUG_PLOT_PRIOR   = False  # plot ETAS lambda grid before each event

# How often to re-evaluate the ETAS prior (in seconds of event time).
# 0  → update before every event  (most accurate, slowest)
# 3600 → update at most once per hour of event time
ETAS_UPDATE_INTERVAL_S = 0

#%%
# ---------------------------------------------------------------------------
# Dynamic ETAS prior (serial — prior state evolves event-by-event)
# ---------------------------------------------------------------------------
# Time-dependent priors cannot use ProcessPoolExecutor because their updaters
# hold mutable state (rolling catalog) that evolves through the sequence.
# Events are sorted chronologically so ETAS updates remain causal.

if RUN_DYNAMIC_PRIORS and not SKIP_RUN:

    if not os.path.exists(INVERSION_JSON):
        raise FileNotFoundError(
            f"ETAS inversion output not found:\n  {INVERSION_JSON}\n"
            "Run time_dependent_scripts/build_initial_prior.py first."
        )

    # 
    # -- Load historical catalog (background seismicity for ETAS) ------------
    # Convert to correct datetime
    print(f"Loading historical catalog from:\n  {os.path.abspath(HISTORICAL_CATALOG)}")
    hist_catalog = pd.read_csv(
        HISTORICAL_CATALOG,
        index_col=0,
        dtype={'url': str, 'alert': str},
    )
    hist_catalog['time'] = pd.to_datetime(
        hist_catalog['time'], format='ISO8601', utc=True
    ).dt.tz_convert(None)
    print(f"  {len(hist_catalog)} events loaded.")


    # -- Build EtasPriorUpdater from the pre-inverted parameters -------------
    # Give it the historical catalog as context
    # TODO - decide if that is prudent, or give it a shorter window, or don't give it 
    # the historical catalog as context at all
    print(f"\nBuilding EtasPriorUpdater from:\n  {INVERSION_JSON}")
    updater = EtasPriorUpdater.from_inversion_json(
        json_path  = INVERSION_JSON,
        catalog_df = hist_catalog,
        **config.ETAS_UPDATER_CONFIG,
    )
    print(updater)

    # -- Helper: read first trigger time from a .run file --------------------
    def _run_trigger_time(event_id):
        path = os.path.join(RUN_DIR, f'{event_id}.run')
        try:
            df  = pd.read_csv(path, nrows=1)
            col = 'trigger time' if 'trigger time' in df.columns else 'trigger_time'
            return float(df[col].iloc[0])
        except Exception:
            return 0.0

    # -- Collect events and sort chronologically -----------------------------
    run_files  = sorted(Path(RUN_DIR).glob('*.run'))
    event_ids  = sorted([f.stem for f in run_files], key=_run_trigger_time)
    print(f"\nRunning dynamic ETAS prior over {len(event_ids)} events "
          f"(update interval: "
          f"{'per-event' if ETAS_UPDATE_INTERVAL_S == 0 else f'{ETAS_UPDATE_INTERVAL_S}s'})…\n")

    # NOTE - Prior is updated periodically, so plotting the prior/posterior
    # Isn't meaninful without saving the prior that existed at a given event
    # This saves the prior for an event of interest (MTJ_EVENT_ID)
    # -- Define callbacks ----------------------------------------------------
    _focus_trigger_time = _run_trigger_time(str(MTJ_EVENT_ID))
    _focus_prior_saved  = [False]

    def etas_update_fn(event_time_unix: float) -> SeismicPrior:
        t = pd.Timestamp(event_time_unix, unit='s')
        save_path = None
        if (not _focus_prior_saved[0]
                and _focus_trigger_time
                and event_time_unix >= _focus_trigger_time):
            save_path = FOCUS_PRIOR_PATH
            _focus_prior_saved[0] = True
            print(f"  [ETAS] saving focus prior for event {MTJ_EVENT_ID} → {FOCUS_PRIOR_PATH}")
        
        # Update the prior:
        prior = updater.update(t, cache_path=save_path)
        print(f"  [ETAS] prior updated at {t.strftime('%Y-%m-%d %H:%M:%S')} "
              f"— catalog size: {updater.n_catalog_events}")

        # Track prior changing for debugging (optional)
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

    def after_event_fn(event_id):
        """
        Append the just-located benchmark event to the rolling ETAS catalog
        so the next prior update reflects it.  Uses the first trigger time
        as a proxy for origin time (a few seconds late; negligible for ETAS).
        """
        eid_int = int(event_id)
        if catalog_df is None or eid_int not in _usgs_ref_lookup.index:
            return
        row = _usgs_ref_lookup.loc[eid_int]
        t   = _run_trigger_time(event_id)
        if t == 0.0:
            return
        updater.append_events(pd.DataFrame([{
            'time':      pd.Timestamp(t, unit='s'),
            'latitude':  row['latitude'],
            'longitude': row['longitude'],
            'magnitude': row['magnitude'],
        }]))

    # -- Set up BenchmarkRunner with the initial prior -----------------------
    _t0 = pd.Timestamp(_run_trigger_time(event_ids[0]), unit='s')
    initial_prior = updater.update(_t0)

    from bEPIC import EPIC_locate_prelim
    params = EPIC_locate_prelim.EPIC_PARAMS()
    params.prior                     = initial_prior
    params.use_prior                 = True
    params.GridSize                  = config.BENCHMARK_PARAMS['grid_size']
    params.GridKm                    = config.BENCHMARK_PARAMS['grid_km']
    params.method                    = 'EPIC C'
    params.MAX_EVENT_TRIGS           = MAX_TRIGS
    params.migrate_grid              = config.BENCHMARK_PARAMS['migrate_grid']
    params.migrate_grid_min_triggers = config.BENCHMARK_PARAMS['migrate_grid_min_triggers']

    dyn_runner = BenchmarkRunner(prior=initial_prior, params=params, run_dir=RUN_DIR)

    dyn_runner.run_all(
        event_ids         = event_ids,
        etas_update_fn    = etas_update_fn,
        update_interval_s = ETAS_UPDATE_INTERVAL_S,
        after_event_fn    = after_event_fn,
    )

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
        for (eid, ver), t in dyn_runner.results.items()
    ]
    out_path = os.path.join(OUTPUT_DIR, 'etas_dynamic_benchmark_results.csv')
    (pd.DataFrame(rows)
       .sort_values(['event_id', 'version'])
       .to_csv(out_path, index=False))
    print(f"\nDynamic ETAS results saved to:\n  {out_path}")

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

# Overall background seismicity for plotting (this is a static file)
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

PRIOR_ORDER    = ['ETAS_dynamic']
td_cache_paths = {'ETAS_dynamic': None}
catalog_events = (catalog_df[['usgs_lon', 'usgs_lat']]
                  .rename(columns={'usgs_lon': 'longitude', 'usgs_lat': 'latitude'})
                  if catalog_df is not None else None)


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


# Get Mendecino Triple Junction (MTJ) specific events
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
    cache_paths    = td_cache_paths,
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



# %%
# ── MTJ single-event posterior grid (prior background + posterior contours) ──
# Auto-selects the first MTJ event from the reference catalog that has a .run file.
# Override MTJ_EVENT_ID with a specific event_id (int) to pin a particular event.



# include the saved focus prior so plot_posterior_grid can render it
focus_cache_paths = {'ETAS_dynamic': FOCUS_PRIOR_PATH if os.path.exists(FOCUS_PRIOR_PATH) else None}

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
            cache_paths    = focus_cache_paths,
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
            cache_paths  = td_cache_paths,
            title        = f'bEPIC location trajectory — MTJ event {MTJ_EVENT_ID}',
            save_path    = os.path.join(FIGURES_DIR, f'MTJ_trajectory_{MTJ_EVENT_ID}.png'),
        )
        plt.show()
    else:
        print('[posterior grid] No MTJ event with a matching .run file found — skipping.')
else:
    print('[posterior grid] No reference catalog loaded — skipping.')

# %%
