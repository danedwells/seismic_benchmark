#%%
# =============================================================================
# run_benchmarks.py  —  bEPIC prior benchmark
# Prerequisite: run scripts/build_priors.py first to build the .tt3 cache files.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Custom repository imports
from priors import SeismicPrior, EtasPriorUpdater
from benchmark.background import load_background_seismicity, add_background_seismicity
from benchmark.plots import (plot_prior_histograms, plot_coverage_panel,
                             plot_location_grid, plot_posterior_grid,
                             plot_overview_map, plot_location_trajectory,
                             plot_qq_calibration, plot_qq_calibration_prior,
                             plot_qq_prior_comparison)
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import (BenchmarkRunner, runner_results_to_df, get_unique_stations,
                              run_single_event_get_grid, make_epic_params,
                              load_station_availability_cache)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEIS_CACHE          = os.path.join(PROJECT_ROOT, 'data', 'reference', 'background_seismicity.parquet')
STATION_AVAIL_CACHE = os.path.join(PROJECT_ROOT, 'data', 'reference', 'station_availability_cache.parquet')
RUN_DIR             = os.path.join(PROJECT_ROOT, 'data', 'run_files')
INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion',
                                   f'parameters_{config.ETAS_INVERSION_CONFIG["id"]}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'etas_inversion', 'input',
                                   f'catalog_{config.ETAS_INVERSION_CONFIG["id"]}.csv')

MAX_TRIGS   = config.BENCHMARK_PARAMS['max_trigs']
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'output', 'time_dependent',  f'max_trigs_{MAX_TRIGS}')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', 'time_dependent',  f'max_trigs_{MAX_TRIGS}')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MTJ_EVENT_ID = 130646  # event used in standalone prior/posterior test below
MTJ_VERSION  = None    # None = last available trigger version

#%%
# ---------------------------------------------------------------------------
# Reference catalog and station list
# ---------------------------------------------------------------------------
# Run bEPIC on this catalog, updating ETAS and prior as it goes.
catalog_path = os.path.join(PROJECT_ROOT, 'data', 'reference', 'bEPIC_testing_catalog.txt')
catalog_df = benchmark_runner.load_reference_catalog(catalog_path) if os.path.exists(catalog_path) else None

station_availability = (
    load_station_availability_cache(STATION_AVAIL_CACHE)
    if os.path.exists(STATION_AVAIL_CACHE) else None
)

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
RUN_DYNAMIC_PRIORS = True   # run time-dependent ETAS prior (serial, event-by-event)
DEBUG_PLOT_PRIOR   = False  # plot ETAS lambda grid before each event

# How often to re-evaluate the ETAS prior (in seconds of event time).
# 0  → update before every event  (most accurate, slowest)
# 3600 → update at most once per hour of event time
ETAS_UPDATE_INTERVAL_S = 0

# Prior tempering exponent.  1.0 = full ETAS weight; <1.0 compresses the
# dynamic range, reducing overconfidence.  0.5 is a reasonable starting point.
PRIOR_ALPHA = 1 # UNCHANGED behavior if this == 1

#%%
# ---------------------------------------------------------------------------
# Dynamic ETAS prior (serial — prior state evolves event-by-event)
# ---------------------------------------------------------------------------
# Time-dependent priors cannot use ProcessPoolExecutor because their updaters
# hold mutable state (rolling catalog) that evolves through the sequence.
# Events are sorted chronologically so ETAS updates remain causal.

if RUN_DYNAMIC_PRIORS:

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

    def etas_update_fn(event_time_unix: float) -> SeismicPrior:
        t     = pd.Timestamp(event_time_unix, unit='s')
        prior = updater.update(t)
        if PRIOR_ALPHA != 1.0:
            prior.grid  = prior.grid ** PRIOR_ALPHA
            prior.grid /= prior.grid.sum()
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
        # Feeds USGS final location into ETAS — deliberately NOT the bEPIC estimate.
        eid_int = int(event_id)
        if eid_int not in _usgs_ref_lookup.index:
            return
        row = _usgs_ref_lookup.loc[eid_int]
        updater.append_events(pd.DataFrame([{
            'time':      row['time'],
            'latitude':  row['latitude'],
            'longitude': row['longitude'],
            'magnitude': row['magnitude'],
        }]))

    # -- Set up BenchmarkRunner with the initial prior -----------------------
    _t0 = pd.Timestamp(_run_trigger_time(event_ids[0]), unit='s')
    initial_prior = updater.update(_t0)

    params = make_epic_params(initial_prior, True, config.BENCHMARK_PARAMS)

    dyn_runner = BenchmarkRunner(prior=initial_prior, params=params, run_dir=RUN_DIR,
                                 catalog_df=catalog_df,
                                 station_availability=station_availability)

    dyn_runner.run_all(
        event_ids         = event_ids,
        etas_update_fn    = etas_update_fn,
        update_interval_s = ETAS_UPDATE_INTERVAL_S,
        after_event_fn    = after_event_fn,
    )

    out_path = os.path.join(OUTPUT_DIR, 'etas_dynamic_benchmark_results.csv')
    runner_results_to_df(dyn_runner).to_csv(out_path, index=False)
    print(f"\nDynamic ETAS results saved to:\n  {out_path}")

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

###########################################################################
# Get Mendecino Triple Junction (MTJ) specific events - filter catalog
###########################################################################
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
    title       = 'bEPIC posterior coverage at fixed radii — dynamic ETAS',
    save_path   = os.path.join(FIGURES_DIR, 'posterior_coverage_histograms.png'),
)

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
    catalog_df  = catalog_df,
)
plt.show()

#%%
# ---------------------------------------------------------------------------
# Standalone single-event prior/posterior test
# ---------------------------------------------------------------------------
# Builds a fresh ETAS prior for MTJ_EVENT_ID from scratch — no dependency on
# the run_all loop or any saved .tt3 file.  Useful for interactive testing.
#
# Prior catalog:
#   - hist_catalog (2000–2018) — always fully included via from_inversion_json
#   - bEPIC testing catalog events preceding MTJ_EVENT_ID within
#     TIME_PRIOR_BUFFER_DAYS (USGS reference locations, not bEPIC estimates)
#     Set TIME_PRIOR_BUFFER_DAYS = None to include all pre-event entries.

MTJ_VERSION            = None    # None = last available trigger version
TIME_PRIOR_BUFFER_DAYS = 1     # lookback window for bEPIC catalog events

_focus_run_path = os.path.join(RUN_DIR, f'{MTJ_EVENT_ID}.run')

if not os.path.exists(_focus_run_path):
    print(f'[standalone] .run file not found: {_focus_run_path}')
elif not os.path.exists(INVERSION_JSON):
    print(f'[standalone] inversion JSON not found: {INVERSION_JSON}')
else:
    # -- Get focus event time and reference location from USGS lookup ------
    if MTJ_EVENT_ID not in _usgs_ref_lookup.index:
        print(f'[standalone] event {MTJ_EVENT_ID} not found in reference catalog.')
    else:
        # Get the event time
        _focus_t = pd.Timestamp(_usgs_ref_lookup.loc[MTJ_EVENT_ID, 'time'])

        # Get USGS lat/lon
        _ref_lat = float(_usgs_ref_lookup.loc[MTJ_EVENT_ID, 'latitude'])
        _ref_lon = float(_usgs_ref_lookup.loc[MTJ_EVENT_ID, 'longitude'])
        print(f'[standalone] focus event {MTJ_EVENT_ID}  t = {_focus_t}')

        # -- Load historical catalog and build a fresh updater ---------------
        # Reuse hist_catalog if available (set in the main loop block above);
        # fall back to re-reading from disk when running the standalone section alone.
        try:
            _hist = hist_catalog
        except NameError:
            _hist = pd.read_csv(HISTORICAL_CATALOG, index_col=0, dtype={'url': str, 'alert': str})
            _hist['time'] = pd.to_datetime(_hist['time'], format='ISO8601', utc=True).dt.tz_convert(None)

        _updater = EtasPriorUpdater.from_inversion_json(
            json_path  = INVERSION_JSON,
            catalog_df = _hist,
            **config.ETAS_UPDATER_CONFIG,
        )

        # -- Append bEPIC catalog events preceding the focus event -----------
        _window_start = (
            _focus_t - pd.Timedelta(days=TIME_PRIOR_BUFFER_DAYS)
            if TIME_PRIOR_BUFFER_DAYS is not None else pd.Timestamp.min
        )
        _mask = (
            (_usgs_ref_lookup['time'] < _focus_t) &
            (_usgs_ref_lookup['time'] >= _window_start) &
            (_usgs_ref_lookup.index != MTJ_EVENT_ID)
        )
        _pre = _usgs_ref_lookup.loc[_mask, ['time', 'latitude', 'longitude', 'magnitude']]
        if not _pre.empty:
            _updater.append_events(_pre)
            print(f'[standalone] appended {len(_pre)} pre-event bEPIC catalog events.')

        # -- Compute ETAS conditional intensity prior at focus event time ----
        _standalone_prior = _updater.update(_focus_t)
        if PRIOR_ALPHA != 1.0:
            _standalone_prior.grid  = _standalone_prior.grid ** PRIOR_ALPHA
            _standalone_prior.grid /= _standalone_prior.grid.sum()
        print(f'[standalone] prior computed  (catalog size: {_updater.n_catalog_events}, '
              f'alpha={PRIOR_ALPHA})')

        _standalone_prior_path = os.path.join(OUTPUT_DIR, f'standalone_prior_{MTJ_EVENT_ID}.tt3')
        _standalone_prior.to_tt3(_standalone_prior_path)

        # -- Run bEPIC on just this event ------------------------------------
        _params = make_epic_params(_standalone_prior, True, config.BENCHMARK_PARAMS)

        _single_runner = BenchmarkRunner(
            prior                = _standalone_prior,
            params               = _params,
            run_dir              = RUN_DIR,
            station_availability = station_availability,
        )
        _single_runner.run_event(str(MTJ_EVENT_ID))
        print(f'[standalone] bEPIC location complete.')

        # -- Write standalone results CSV to a per-event dir so the full-loop
        #    etas_dynamic_benchmark_results.csv in OUTPUT_DIR is not overwritten.
        _standalone_out_dir = os.path.join(OUTPUT_DIR, f'standalone_{MTJ_EVENT_ID}')
        os.makedirs(_standalone_out_dir, exist_ok=True)
        _standalone_rows = [
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
            for (eid, ver), t in _single_runner.results.items()
        ]
        _standalone_csv = os.path.join(_standalone_out_dir, 'etas_dynamic_benchmark_results.csv')
        (pd.DataFrame(_standalone_rows)
           .sort_values(['event_id', 'version'])
           .to_csv(_standalone_csv, index=False))
        print(f'[standalone] results written → {_standalone_csv}')

        # -- Grid + coverage (single bEPIC run, reused for both) ---------------
        from benchmark.runner import run_single_event_get_grid
        _params_kw = {
            'grid_size':                 config.BENCHMARK_PARAMS['grid_size'],
            'grid_km':                   config.BENCHMARK_PARAMS['grid_km'],
            'max_trigs':                 MAX_TRIGS,
            'migrate_grid':              config.BENCHMARK_PARAMS['migrate_grid'],
            'migrate_grid_min_triggers': config.BENCHMARK_PARAMS['migrate_grid_min_triggers'],
        }

        # Get the posterior (_t_cov), out_df (_odf_cov), and version (# triggers) (_actual_v)
        _t_cov, _odf_cov, _actual_v = run_single_event_get_grid(
            _focus_run_path, _standalone_prior, True, _params_kw,
            focus_version=MTJ_VERSION,
        )

        _standalone_cache = {'ETAS_dynamic': _standalone_prior_path}
        _precomputed = {
            'ETAS_dynamic': (_t_cov, _odf_cov, _actual_v,
                             _standalone_prior, _standalone_prior_path),
        }
        _buffer_label = f'{TIME_PRIOR_BUFFER_DAYS}d lookback' if TIME_PRIOR_BUFFER_DAYS else 'full history'
        deg_buf = 0.5
        # extent = [-125, -124,40,40.6]
        extent = [_ref_lon - deg_buf, _ref_lon + deg_buf, _ref_lat - deg_buf, _ref_lat + deg_buf]
        # -- Posterior grid (prior background + posterior contours) ----------
        fig = plot_posterior_grid(
            focus_run_path = _focus_run_path,
            cache_paths    = _standalone_cache,
            prior_order    = ['ETAS_dynamic'],
            params_kw      = _params_kw,
            prior_results  = _precomputed,
            ref_lat        = _ref_lat,
            ref_lon        = _ref_lon,
            extent         = extent,
            focus_version  = MTJ_VERSION,
            title          = f'ETAS prior/posterior — event {MTJ_EVENT_ID} ({_buffer_label})',
            save_path      = os.path.join(FIGURES_DIR, f'standalone_posterior_{MTJ_EVENT_ID}.png'),
        )
        plt.show()

        # -- Location trajectory ---------------------------------------------
        fig = plot_location_trajectory(
            event_id     = MTJ_EVENT_ID,
            output_dir   = _standalone_out_dir,
            prior_order  = ['ETAS_dynamic'],
            run_dir      = RUN_DIR,
            min_triggers = 4,
            ref_lat      = _ref_lat,
            ref_lon      = _ref_lon,
            cache_paths  = _standalone_cache,
            extent_pad_deg = 0.1,
            title        = f'bEPIC location trajectory — event {MTJ_EVENT_ID} ({_buffer_label})',
            save_path    = os.path.join(FIGURES_DIR, f'standalone_trajectory_{MTJ_EVENT_ID}.png'),
        )
        plt.show()

#%%

# ######################################
# # Compute posterior statistics
# ######################################
# bEPIC_lat = _t_cov.posterior_lat
# bEPIC_lon = _t_cov.posterior_lon


# # Posterior coverage # 1 - Geometry first
# # how much of probability mass is contained within
# # a circle around the USGS location with a radius of the distance to the bEPIC location?
# if _odf_cov is not None:
#     from obspy.geodetics import gps2dist_azimuth as _gps2dist
#     _map_err_km = _gps2dist(
#         _ref_lat, _ref_lon,
#         bEPIC_lat, bEPIC_lon,
#     )[0] / 1000.0
#     _frac = posterior_coverage(
#         _odf_cov, _ref_lat, _ref_lon,
#         radii_km=(_map_err_km),
#     )
#     print(f'MAP location error : {_map_err_km:.1f} km')
#     print(f'Posterior Coverage # 1:')
#     print(f'  Probability mass within {_map_err_km:6.1f} km : {_frac * 100:5.1f}%')


# # Posterior coverage #2 - Probability first
# # What confidence contour is the USGS locatoin on w/respect to the bEPIC location?
# # USGS credible_level computes the contour level around 
# # the bEPIC location that the USGS lies on. I.e., it returns 0.5
# # if 50% of the probability mass of hte posterior is contained within the contour 
# # the the USGS location lies on. 
# if _odf_cov is not None:
#     usgs_contf = 100*posterior_confidence_level(_odf_cov,_ref_lat,_ref_lon)
#     print("Posterior Coverage # 2")
#     print(f'   Confidence contour of USGS location: {usgs_contf:5.1f}%')

# # # 3 metrics to compare. We probably want to implement this on all of 
# # # the priors, including time independent
# # post_covs = []
# # distance_err = []
# # post_conts = []


# %%
