#%%
# =============================================================================
# california_large_events.py  —  bEPIC prior benchmark restricted to large events
#
# One-off script: copies the logic of time_independent_scripts/run_benchmarks.py
# (KDE_Seismicity, Uniform) and time_dependent_scripts/run_benchmarks.py (ETAS),
# but filters the California event list down to events with usgs_mag >= MIN_MAG
# and writes results under results/california/{MAG_FILTER}/ instead of the main
# benchmark output tree.
#
# Prerequisites: preparation_scripts/build_priors.py and
# time_dependent_scripts/build_initial_prior.py must already have been run.
# =============================================================================
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'

import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Custom repository imports
from priors import SeismicPrior, EtasPriorUpdater

from benchmark import runner as benchmark_runner
import benchmark.config_california as config
from benchmark.runner import (BenchmarkRunner, runner_results_to_df,
                               make_epic_params,
                               load_station_availability_cache)
from benchmark.time_dependent_helpers import *

# ---------------------------------------------------------------------------
# Magnitude filter
# ---------------------------------------------------------------------------
MIN_MAG    = 5.0   # edit for different thresholds (e.g. 6.0, 6.5)
MAG_FILTER = f"M{MIN_MAG:g}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATION_AVAIL_CACHE = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'station_availability_cache.parquet')
RUN_DIR             = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']

# Make a min mag subfolder directory structure
MAG_DIR            = os.path.join(PROJECT_ROOT, 'results', 'california', MAG_FILTER)
STATIC_OUTPUT_DIR  = os.path.join(MAG_DIR, 'output', 'time_independent', f'max_trigs_{MAX_TRIGS}')
DYNAMIC_OUTPUT_DIR = os.path.join(MAG_DIR, 'output', 'time_dependent',   f'max_trigs_{MAX_TRIGS}')

os.makedirs(STATIC_OUTPUT_DIR,  exist_ok=True)
os.makedirs(DYNAMIC_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference catalog, station list, and large-event filter
# ---------------------------------------------------------------------------
catalog_path = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')
catalog_df = benchmark_runner.load_reference_catalog(catalog_path) if os.path.exists(catalog_path) else None

ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]

run_ids = {int(f.stem) for f in Path(RUN_DIR).glob('*.run')}
large_event_ids = sorted(
    int(eid) for eid in catalog_df.loc[catalog_df['usgs_mag'] >= MIN_MAG, 'event_id']
    if int(eid) in run_ids
)
print(f"{len(large_event_ids)} events with usgs_mag >= {MIN_MAG} have run files: {large_event_ids}")

cache_paths = {
    name: os.path.join(SeismicPrior.data_dir, fname) if fname is not None else None
    for name, fname in config.PRIOR_FILENAMES.items()
}
cache_paths['KDE_Seismicity'] = os.path.join(SeismicPrior.data_dir, 'kde_seismicity_benchmark.tt3')

_avail = (
    load_station_availability_cache(STATION_AVAIL_CACHE)
    if os.path.exists(STATION_AVAIL_CACHE) else None
)

#%%
# ---------------------------------------------------------------------------
# Part 1 — Static priors (KDE_Seismicity, Uniform), large events only
# ---------------------------------------------------------------------------
priors_to_run = ['KDE_Seismicity', 'Uniform']

def _run_one(name, path):
    if path is not None:
        prior = SeismicPrior.from_tt3(path)
        use_prior = True
    else:  # Uniform
        path = cache_paths['NSHM']
        prior = SeismicPrior.from_tt3(path)
        use_prior = False
    params = make_epic_params(prior, use_prior, config.BENCHMARK_PARAMS, station_inventory=_avail)
    runner = BenchmarkRunner(prior=prior, params=params, run_dir=RUN_DIR,
                             catalog_df=ref_df,
                             station_availability=params.station_inventory)
    runner.run_all(large_event_ids)
    out_path = os.path.join(STATIC_OUTPUT_DIR, f"{name.lower()}_benchmark_results.csv")
    runner_results_to_df(runner).to_csv(out_path, index=False)
    return name

with ProcessPoolExecutor(max_workers=len(priors_to_run)) as ex:
    futures = [ex.submit(_run_one, name, cache_paths[name]) for name in priors_to_run]
    for f in futures:
        f.result()  # re-raise any worker exception

#%%
# ---------------------------------------------------------------------------
# Part 2 — Dynamic ETAS prior, large events only
# ---------------------------------------------------------------------------
BW_SQ = 4
config.ETAS_INVERSION_CONFIG['bw_sq'] = BW_SQ
spatial_factor = None  # multiply inverted d (spatial decay size) by this factor

INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion',
                                   f'parameters_{config.etas_output_id(config.ETAS_INVERSION_CONFIG["id"])}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion', 'input',
                                   f'catalog_{config.etas_catalog_tag(config.ETAS_INVERSION_CONFIG["id"])}.csv')

SIGMA_S = 0.35
config.BENCHMARK_PARAMS['sigma_s'] = SIGMA_S

station_availability = _avail

# Full-catalog context feed — even though only large_event_ids are actually
# located, the ETAS updater should still see every smaller event (down to
# the catalog's Mc, M3 here) that occurred in between, so its context
# matches what really happened rather than just the located large events.
_full_context = (
    catalog_df[['usgs_time', 'usgs_lat', 'usgs_lon', 'usgs_mag']]
    .rename(columns={'usgs_time': 'time', 'usgs_lat': 'latitude',
                     'usgs_lon': 'longitude', 'usgs_mag': 'magnitude'})
    .sort_values('time')
)

ETAS_UPDATE_INTERVAL_S = 0
PRIOR_ALPHA = 1  # UNCHANGED behavior if this == 1

if not os.path.exists(INVERSION_JSON):
    raise FileNotFoundError(
        f"ETAS inversion output not found:\n  {INVERSION_JSON}\n"
        "Run time_dependent_scripts/build_initial_prior.py first."
    )

# Get historical catalog for the prior updater (need a context before the test sequence
# to simulate that there have been recent events)
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

print(f"\nBuilding EtasPriorUpdater from:\n  {INVERSION_JSON}")
benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)

updater = EtasPriorUpdater.from_inversion_json(
    json_path  = INVERSION_JSON,
    catalog_df = hist_catalog,
    **config.ETAS_UPDATER_CONFIG,
    spatial_factor = spatial_factor
)

_context_fed_upto = pd.Timestamp.min

def etas_update_fn_with_context(event_time_unix):
    # Feeds every not-yet-fed full-catalog event (any magnitude down to Mc)
    # with a true origin time before this update into the ETAS updater,
    # then updates the prior as usual. Causal: events at/after event_time_unix
    # are left for a later update.
    global _context_fed_upto
    t = pd.Timestamp(event_time_unix, unit='s')
    new = _full_context[(_full_context['time'] > _context_fed_upto) & (_full_context['time'] < t)]
    if not new.empty:
        updater.append_events(new)
        _context_fed_upto = new['time'].max()
    return etas_update_fn(event_time_unix, updater, PRIOR_ALPHA)

event_ids = sorted(
    (str(eid) for eid in large_event_ids),
    key=lambda eid: run_trigger_time(eid, RUN_DIR)
)


#%%
#--------------------------------------------------
#-- Run ETAS (dynamic prior)
#--------------------------------------------------
print(f"\nRunning dynamic ETAS prior over {len(event_ids)} large events "
        f"(update interval: "
        f"{'per-event' if ETAS_UPDATE_INTERVAL_S == 0 else f'{ETAS_UPDATE_INTERVAL_S}s'})…\n")

_t0 = pd.Timestamp(run_trigger_time(event_ids[0], RUN_DIR), unit='s')
initial_prior = updater.update(_t0)

params = make_epic_params(initial_prior, True, config.BENCHMARK_PARAMS)

runner = BenchmarkRunner(prior=initial_prior,
                        params=params,
                        run_dir=RUN_DIR,
                        catalog_df=catalog_df,
                        station_availability=station_availability)

runner.run_all(
    event_ids         = event_ids,
    etas_update_fn    = etas_update_fn_with_context,
    update_interval_s = ETAS_UPDATE_INTERVAL_S,
)

out_path = os.path.join(DYNAMIC_OUTPUT_DIR, 'etas_dynamic_benchmark_results.csv')
runner_results_to_df(runner).to_csv(out_path, index=False)
print(f"\nDynamic ETAS results saved to:\n  {out_path}")
