#%%
# =============================================================================
# run_cascadia.py  —  bEPIC prior benchmark for Cascadia specifically
# Prerequisite: run preparation_scripts/build_initial_prior_cascadia.py first to build the .tt3 cache files.
import os
import pandas as pd
from pathlib import Path

# Custom repository imports
from priors import EtasPriorUpdater
from benchmark import runner as benchmark_runner
from benchmark import config_cascadia as config
from benchmark.runner import (BenchmarkRunner, runner_results_to_df,
                              make_epic_params, load_station_availability_cache)
from benchmark.time_dependent_helpers import *

# ---------------------------------------------------------------------------
# ETAS inversion variant selection
# ---------------------------------------------------------------------------

BW_SQ = 16 # 4 is the default value. Unchanged behavior
config.ETAS_INVERSION_CONFIG['bw_sq'] = BW_SQ

#manual override of spatial kernel size
# put an integeor or None
spatial_factor = None # multiply inverted d (spatial decay size) by this factor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATION_AVAIL_CACHE = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'reference', 'station_availability_cache.parquet')
RUN_DIR             = os.path.join(PROJECT_ROOT, 'data', 'cascadia', 'run_files')
INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'cascadia','etas_inversion',
                                   f'parameters_{config.etas_output_id(config.ETAS_INVERSION_CONFIG["id"], config.ETAS_INVERSION_CONFIG)}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'cascadia','etas_inversion', 'input',
                                   f'catalog_{config.etas_catalog_tag(config.ETAS_INVERSION_CONFIG["id"], config.ETAS_INVERSION_CONFIG)}.csv')

MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
EDT_SIGMA_S    = config.BENCHMARK_PARAMS['edt_sigma_s']
#SIGMA_S        = config.BENCHMARK_PARAMS['sigma_s']
SIGMA_S = 0.60
config.BENCHMARK_PARAMS['sigma_s']      = SIGMA_S
DTT_WEIGHT     = config.BENCHMARK_PARAMS['dtt_weight']

OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'cascadia', 'output',  'time_dependent', f'max_trigs_{MAX_TRIGS}')

os.makedirs(OUTPUT_DIR,  exist_ok=True)

#%%
# ---------------------------------------------------------------------------
# Reference catalog and station list
# ---------------------------------------------------------------------------
# Run bEPIC on this catalog, updating ETAS and prior as it goes.
catalog_path = os.path.join(PROJECT_ROOT, 'data', 'cascadia','reference', 'cascadia_test_catalog_west.csv')
catalog_df = benchmark_runner.load_reference_catalog_usgs(catalog_path) if os.path.exists(catalog_path) else None

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
# How often to re-evaluate the ETAS prior (in seconds of event time).
ETAS_UPDATE_INTERVAL_S = 0

# Prior tempering exponent.  1.0 = full ETAS weight; <1.0 compresses the
PRIOR_ALPHA = 1 # UNCHANGED behavior if this == 1

#%%
# ---------------------------------------------------------------------------
# Dynamic ETAS prior (serial — prior state evolves event-by-event)
# ---------------------------------------------------------------------------

# Check directory
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
print(f"\nBuilding EtasPriorUpdater from:\n  {INVERSION_JSON}")
benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)

# Make ETAS updater
updater = EtasPriorUpdater.from_inversion_json(
    json_path  = INVERSION_JSON,
    catalog_df = hist_catalog,
    **config.ETAS_UPDATER_CONFIG,
    spatial_factor = spatial_factor
)

def after_event_fn(event_id):
    # Feeds USGS final location into ETAS — deliberately NOT the bEPIC estimate.
    # event_id is the ANSS string id (.run files are named by it directly),
    # not an int like the CA benchmark's postgres ids.
    if event_id not in _usgs_ref_lookup.index:
        return
    row = _usgs_ref_lookup.loc[event_id]
    updater.append_events(pd.DataFrame([{
        'time':      row['time'],
        'latitude':  row['latitude'],
        'longitude': row['longitude'],
        'magnitude': row['magnitude'],
    }]))

#%%
# Collect event IDs from available .run files, sorted by first trigger time
# (chronological order is critical so ETAS updates are causal).
run_files  = sorted(Path(RUN_DIR).glob('*.run'))
event_ids  = sorted([f.stem for f in run_files], key=lambda eid: run_trigger_time(eid, RUN_DIR))
print(f"\nRunning dynamic ETAS prior over {len(event_ids)} events "
        f"(update interval: "
        f"{'per-event' if ETAS_UPDATE_INTERVAL_S == 0 else f'{ETAS_UPDATE_INTERVAL_S}s'})…\n")

# -----------------------------------------------------------------------
# -- Set up BenchmarkRunner with the initial prior
# -- Run the dynamic prior ----------------------
# -----------------------------------------------------------------------
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
    etas_update_fn    = lambda t: etas_update_fn(t, updater, PRIOR_ALPHA),
    update_interval_s = ETAS_UPDATE_INTERVAL_S,
    after_event_fn    = after_event_fn,
)

# -- Save results ---------------------------------------------------------
out_path = os.path.join(OUTPUT_DIR, 'etas_dynamic_benchmark_results.csv')
runner_results_to_df(runner).to_csv(out_path, index=False)
print(f"\nDynamic ETAS results saved to:\n  {out_path}")
