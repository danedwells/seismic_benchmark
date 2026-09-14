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
import pandas as pd
from pathlib import Path

from priors import SeismicPrior, EtasPriorUpdater
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.priors import blend_priors
from benchmark.runner import (BenchmarkRunner, runner_results_to_df,
                              make_epic_params, load_station_availability_cache)
from benchmark.time_dependent_helpers import run_trigger_time

# ---------------------------------------------------------------------------
# Control flags
# ---------------------------------------------------------------------------

# Blending weights: ALPHA on the ETAS component, (1-ALPHA) on the static prior.
# 0.0 = pure time-independent; 1.0 = pure ETAS; 0.5 = equal weight.
ALPHA     = 0.5
ALPHA_TAG = f'alpha_{ALPHA:.2f}'

# Set True to plot the raw ETAS grid before each update (diagnostic).
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir    = SeismicPrior.data_dir
cache_paths = {
    name: os.path.join(data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}

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
# Blending utility (imported from benchmark.priors — see benchmark/priors.py)
# ---------------------------------------------------------------------------

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
# Main workflow
# ---------------------------------------------------------------------------
# How often to re-evaluate the ETAS prior (in seconds of event time).
ETAS_UPDATE_INTERVAL_S = 0

# Prior tempering exponent.  1.0 = full ETAS weight; <1.0 compresses the
PRIOR_ALPHA = 1 # UNCHANGED behavior if this == 1

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
benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)
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

run_files = sorted(Path(RUN_DIR).glob('*.run'))
event_ids = sorted([f.stem for f in run_files], key=lambda eid: run_trigger_time(eid, RUN_DIR))

#%%
# ---------------------------------------------------------------------------
# Mixed-prior benchmark loop


# Initialise runners with blended priors at t0
_t0_unix      = run_trigger_time(event_ids[0], RUN_DIR)
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
    event_time_unix = run_trigger_time(event_id, RUN_DIR)
    t = pd.Timestamp(event_time_unix, unit='s')

    # Re-evaluate ETAS if the update interval has elapsed
    if (ETAS_UPDATE_INTERVAL_S == 0 or
            event_time_unix - _last_etas_update_unix >= ETAS_UPDATE_INTERVAL_S):
        _current_etas              = updater.update(t)
        _last_etas_update_unix     = event_time_unix
        print(f"  [ETAS] updated at {t.strftime('%Y-%m-%d %H:%M:%S')} "
                f"— catalog: {updater.n_catalog_events} events")

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

