#%%
# =============================================================================
# run_benchmarks.py  —  bEPIC mixed-prior benchmark
# Blends each of the five time-independent spatial priors with a time-evolving
# ETAS prior using a weighted linear combination:
#
#   combined = ALPHA * etas_prior + (1 - ALPHA) * ti_prior
#
# Prerequisites:
#   - time_independent_scripts/build_priors.py  (builds .tt3 cache files)
#   - time_dependent_scripts/build_initial_prior.py  (ETAS parameter inversion)
#
# Output:
#   results/output/mixed/max_trigs_{N}/{prior}_etas_mixed_benchmark_results.csv
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
                             plot_location_grid, plot_overview_map,
                             plot_qq_calibration, plot_qq_calibration_prior,
                             plot_qq_prior_comparison)
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import (BenchmarkRunner, runner_results_to_df, get_unique_stations,
                              make_epic_params, load_station_availability_cache)

# ---------------------------------------------------------------------------
# Control flags
# ---------------------------------------------------------------------------

# Blending weights: ALPHA on the ETAS component, (1-ALPHA) on the static prior.
# 0.0 = pure time-independent; 1.0 = pure ETAS; 0.5 = equal weight.
ALPHA     = 0.5
ALPHA_TAG = f'alpha_{ALPHA:.2f}'

# How often to re-evaluate the ETAS prior (seconds of event time).
# 0 = update before every event (most accurate, slowest).
ETAS_UPDATE_INTERVAL_S = 0

# Power-law tempering applied to the raw ETAS grid before blending.
# Values < 1 compress dynamic range (reduce aftershock cluster dominance).
# 1 = no change.
PRIOR_ALPHA = 1

# Set True to plot the raw ETAS grid before each update (diagnostic).
DEBUG_PLOT_PRIOR = False

RUN_MIXED = True
SKIP_RUN  = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir    = SeismicPrior.data_dir
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

SEIS_CACHE          = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
STATION_AVAIL_CACHE = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'station_availability_cache.parquet')
RUN_DIR             = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion',
                                   f'parameters_{config.etas_output_id(config.ETAS_INVERSION_CONFIG["id"])}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion', 'input',
                                   f'catalog_{config.etas_catalog_tag(config.ETAS_INVERSION_CONFIG["id"])}.csv')

MAX_TRIGS   = config.BENCHMARK_PARAMS['max_trigs']
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'output',  'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'figures', 'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG)
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference catalog
# ---------------------------------------------------------------------------

catalog_path = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
catalog_df   = benchmark_runner.load_reference_catalog(catalog_path) if os.path.exists(catalog_path) else None

station_availability = (
    load_station_availability_cache(STATION_AVAIL_CACHE)
    if os.path.exists(STATION_AVAIL_CACHE) else None
)

_usgs_ref_lookup = (
    catalog_df[['event_id', 'usgs_time', 'usgs_lat', 'usgs_lon', 'usgs_mag']]
    .rename(columns={'usgs_time': 'time', 'usgs_lat': 'latitude',
                     'usgs_lon': 'longitude', 'usgs_mag': 'magnitude'})
    .set_index('event_id')
    if catalog_df is not None else pd.DataFrame()
)



#%%
# ---------------------------------------------------------------------------
# Blending utility
# ---------------------------------------------------------------------------

def blend_priors(ti_prior, etas_prior, alpha=0.5, prior_alpha=1):
    """
    Blend ti_prior onto the ETAS grid and return a new SeismicPrior:

        combined = alpha * etas_tempered + (1 - alpha) * ti_resampled

    ti_prior    — SeismicPrior (static) or None for a Uniform base prior.
    etas_prior  — SeismicPrior (time-dependent); defines the output grid.
    alpha       — weight on the ETAS component in [0, 1].
    prior_alpha — power-law exponent applied to the ETAS grid before blending.
                  Values < 1 compress dynamic range (reduce cluster dominance).
                  1 = no change.

    The TI prior is bilinearly interpolated onto the ETAS lon/lat grid before
    mixing so both components are on the same support.  When ti_prior is None
    a flat (uniform) grid is used as the base, making the result equivalent
    to a linearly tempered ETAS prior.
    """
    etas_grid = etas_prior.grid.copy()
    if prior_alpha != 1.0:
        etas_grid  = etas_grid ** prior_alpha
        etas_grid /= etas_grid.sum()

    if ti_prior is None:
        # Uniform base: equal probability on every ETAS grid cell
        ti_grid = np.ones_like(etas_grid)
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
        ti_grid = np.ones_like(etas_grid) / etas_grid.size

    combined      = alpha * etas_grid + (1.0 - alpha) * ti_grid
    combined     /= combined.sum()

    mixed      = copy.deepcopy(etas_prior)
    mixed.grid = combined
    return mixed

cache_paths['KDE_Seismicity'] = os.path.join(data_dir, 'kde_seismicity_benchmark.tt3')

#%%
# ---------------------------------------------------------------------------
# Load time-independent priors
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
# Build EtasPriorUpdater
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

#%%
# ---------------------------------------------------------------------------
# Sort events chronologically (causal ETAS ordering)
# ---------------------------------------------------------------------------

def _run_trigger_time(event_id):
    path = os.path.join(RUN_DIR, f'{event_id}.run')
    try:
        df  = pd.read_csv(path, nrows=1)
        col = 'trigger time' if 'trigger time' in df.columns else 'trigger_time'
        return float(df[col].iloc[0])
    except Exception:
        return 0.0

run_files = sorted(Path(RUN_DIR).glob('*.run'))
event_ids = sorted([f.stem for f in run_files], key=_run_trigger_time)

#%%
# ---------------------------------------------------------------------------
# Mixed-prior benchmark loop
# ---------------------------------------------------------------------------
# Events run serially in chronological order so the shared ETAS updater
# stays causal.  For each event:
#   1. Evaluate ETAS prior once (if update interval has elapsed).
#   2. Blend the fresh ETAS prior with each of the 5 TI priors.
#   3. Run bEPIC for each blended prior.
#   4. Append the USGS reference location to the ETAS catalog (once).

if RUN_MIXED and not SKIP_RUN:

    # Initialise runners with blended priors at t0
    _t0_unix      = _run_trigger_time(event_ids[0])
    _t0           = pd.Timestamp(_t0_unix, unit='s')
    _current_etas = updater.update(_t0)
    print(f"\nInitial ETAS prior evaluated at {_t0.strftime('%Y-%m-%d %H:%M:%S')}")

    runners = {}
    for name, ti_prior in ti_priors.items():
        initial_mixed   = blend_priors(ti_prior, _current_etas, ALPHA, PRIOR_ALPHA)
        params          = make_epic_params(initial_mixed, True, config.BENCHMARK_PARAMS)
        runners[name]   = BenchmarkRunner(
            prior                = initial_mixed,
            params               = params,
            run_dir              = RUN_DIR,
            catalog_df           = catalog_df,
            station_availability = station_availability,
        )

    _last_etas_update_unix = _t0_unix

    print(f"\nRunning mixed-prior benchmark over {len(event_ids)} events "
          f"({len(runners)} TI priors × ETAS, alpha={ALPHA})…\n")

    for i, event_id in enumerate(event_ids):
        event_time_unix = _run_trigger_time(event_id)
        t = pd.Timestamp(event_time_unix, unit='s')

        # Re-evaluate ETAS if the update interval has elapsed
        if (ETAS_UPDATE_INTERVAL_S == 0 or
                event_time_unix - _last_etas_update_unix >= ETAS_UPDATE_INTERVAL_S):
            _current_etas              = updater.update(t)
            _last_etas_update_unix     = event_time_unix
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
            mixed = blend_priors(ti_prior, _current_etas, ALPHA, PRIOR_ALPHA)
            runners[name].update_prior(mixed)
            runners[name].run_event(event_id)

        # Feed USGS reference location back to ETAS — once per event, using ground
        # truth (not bEPIC estimates) to keep the updater causally consistent.
        eid_int = int(event_id)
        if eid_int in _usgs_ref_lookup.index:
            row = _usgs_ref_lookup.loc[eid_int]
            updater.append_events(pd.DataFrame([{
                'time':      row['time'],
                'latitude':  row['latitude'],
                'longitude': row['longitude'],
                'magnitude': row['magnitude'],
            }]))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(event_ids)} events complete.")

    print(f"\nSaving results to:\n  {OUTPUT_DIR}")
    for name, runner in runners.items():
        out_path = os.path.join(OUTPUT_DIR, f'{name.lower()}_etas_mixed_benchmark_results.csv')
        runner_results_to_df(runner).to_csv(out_path, index=False)
        print(f'  {name} → {os.path.basename(out_path)}')

#%%
# ---------------------------------------------------------------------------
# Background seismicity and station list (for plotting)
# ---------------------------------------------------------------------------

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
# CSV filenames are {name}_etas_mixed_benchmark_results.csv, so plot functions
# receive prior_names like 'Gear1_etas_mixed' and construct the path themselves.

PRIOR_ORDER    = [f'{name}_etas_mixed' for name in ti_priors]
mixed_cache_paths = {pname: None for pname in PRIOR_ORDER}  # no single .tt3 for mixed priors

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

# ── Overview: all mixed priors, full region ───────────────────────────────
fig = plot_overview_map(
    output_dir  = OUTPUT_DIR,
    prior_order = PRIOR_ORDER,
    extent      = [-128.5, -113, 31, 44],
    events_df   = catalog_events,
    stations_df = stations_df,
    bg          = bg,
    title       = f'bEPIC final locations — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(FIGURES_DIR, 'comparison_benchmark_locations.png'),
)
plt.show()

# %%

# ── MTJ grid: one panel per mixed prior ──────────────────────────────────
fig = plot_location_grid(
    output_dir     = OUTPUT_DIR,
    prior_order    = PRIOR_ORDER,
    extent         = MTJ_EXTENT,
    ref_catalog    = catalog_df,
    events_df      = catalog_mtj,
    stations_df    = stations_mtj,
    bg             = bg,
    cache_paths    = mixed_cache_paths,
    filter_fn      = in_extent,
    show_scale_bar = True,
    title          = f'bEPIC MTJ locations — mixed priors (alpha={ALPHA})',
    save_path      = os.path.join(FIGURES_DIR, 'MTJ_grid_benchmark_locations.png'),
)
plt.show()

# %%
bins_frac = np.linspace(0, 0.5, 51)
bins_km   = np.linspace(0, 100, 51)

# ── MTJ fractional misfit histograms ────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'frac_misfit',
    bins        = bins_frac,
    title       = f'bEPIC MTJ fractional misfit — mixed priors (alpha={ALPHA})',
    xlabel      = 'frac_misfit (fractional TT error)',
    save_path   = os.path.join(FIGURES_DIR, 'MTJ_grid_misfit_histograms.png'),
    filter_fn   = in_extent,
)
plt.show()

# ── Total fractional misfit histograms ──────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'frac_misfit',
    bins        = bins_frac,
    title       = f'bEPIC fractional misfit — mixed priors (alpha={ALPHA})',
    xlabel      = 'frac_misfit (fractional TT error)',
    save_path   = os.path.join(FIGURES_DIR, 'Grid_misfit_histograms.png'),
)
plt.show()

# ── MTJ location error histograms ───────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'map_err_km',
    bins        = bins_km,
    title       = f'bEPIC MTJ location error — mixed priors (alpha={ALPHA})',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'MTJ_location_error_histograms.png'),
    filter_fn   = in_extent,
)
plt.show()

# ── Total location error histograms ─────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'map_err_km',
    bins        = bins_km,
    title       = f'bEPIC location error — mixed priors (alpha={ALPHA})',
    xlabel      = 'location error (km)',
    save_path   = os.path.join(FIGURES_DIR, 'Grid_location_error_histograms.png'),
)
plt.show()

# ── posterior_confidence_level histograms ──────────────────────────────────────
fig = plot_prior_histograms(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'posterior_confidence_level',
    bins        = np.linspace(0, 1, 41),
    title       = f'bEPIC posterior calibration — mixed priors (alpha={ALPHA})',
    xlabel      = 'posterior_confidence_level',
    save_path   = os.path.join(FIGURES_DIR, 'posterior_confidence_level_histograms.png'),
    color       = 'steelblue',
)
plt.show()

# ── Posterior coverage at fixed radii (2×2 panel) ───────────────────────
fig = plot_coverage_panel(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = f'bEPIC posterior coverage at fixed radii — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(FIGURES_DIR, 'posterior_coverage_histograms.png'),
)
plt.show()

# ── Calibration Q-Q: posterior_confidence_level vs U(0,1) ──────────────────────
fig = plot_qq_calibration(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = f'bEPIC posterior calibration Q-Q — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(FIGURES_DIR, 'qq_calibration.png'),
)
plt.show()

# ── Prior calibration Q-Q: prior_confidence_level vs U(0,1) ──────────
fig = plot_qq_calibration_prior(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    title       = f'bEPIC prior calibration Q-Q — mixed priors (alpha={ALPHA})',
    save_path   = os.path.join(FIGURES_DIR, 'qq_calibration_prior.png'),
)
plt.show()

# ── Prior-vs-prior Q-Q comparison: map_err_km ───────────────────────────
fig = plot_qq_prior_comparison(
    prior_names = PRIOR_ORDER,
    output_dir  = OUTPUT_DIR,
    column      = 'map_err_km',
    title       = f'Q-Q prior comparison — map location error (km) — mixed (alpha={ALPHA})',
    save_path   = os.path.join(FIGURES_DIR, 'qq_prior_comparison.png'),
)
plt.show()

# %%
