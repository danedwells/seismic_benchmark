#%%
# =============================================================================
# bw_sq_calibration.py — sweep ETAS's bw_sq, one full inversion + dynamic
# benchmark run per value (California 'benchmark' context only)
# =============================================================================
# Unlike sigma_calibration.py (a runtime-only knob), bw_sq is baked into the
# ETAS inversion itself, so every value needs its own ETASParameterCalculation
# fit *and* its own full causal-order bEPIC pass with that prior. This script
# does both per value, tagging outputs the same way build_initial_prior.py /
# run_benchmarks.py already do so nothing collides:
#   - parameters_{tagged_id}.json      (config.etas_output_id, includes bw_sq)
#   - etas_dynamic_benchmark_results_bw{bw_sq}.csv
#
# Inputs shared across every bw_sq value (reference catalog, station
# availability, the filtered ETAS catalog, event order) are built once in the
# parent process and handed to each worker — only bw_sq differs between runs.
#
# Prerequisite: preparation_scripts/build_priors.py (static prior .tt3 cache)
# is NOT needed here (ETAS-only sweep, no static prior). The shared seismicity
# catalog is downloaded once if not already cached.
#
# Usage: edit BW_SQ_VALUES below, then run top-to-bottom or cell-by-cell.
# =============================================================================

import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from etas import set_up_logger
from etas.inversion import ETASParameterCalculation

from priors import EtasPriorUpdater
from benchmark import runner as benchmark_runner
from benchmark import config_california as config
from benchmark.background import load_background_seismicity
from benchmark.runner import (BenchmarkRunner, runner_results_to_df,
                               make_epic_params, load_station_availability_cache,
                               repair_inversion_json_paths)
from benchmark.time_dependent_helpers import run_trigger_time, etas_update_fn

#%%
# ---------------------------------------------------------------------------
# Sweep values — edit as needed
# ---------------------------------------------------------------------------
BW_SQ_VALUES = [1,64]

# Skip re-inverting a bw_sq value whose tagged parameters_*.json already
# exists (e.g. re-running this script after a crash partway through).
SKIP_EXISTING_INVERSION = True

# Held fixed for the whole sweep — matches time_dependent_scripts/run_benchmarks.py's
# current calibrated value. Not swept here; sigma_s and bw_sq are calibrated
# separately so their effects aren't conflated in one sweep.
SIGMA_S = 0.35

ETAS_UPDATE_INTERVAL_S = 0   # per-event ETAS updates (matches run_benchmarks.py)
PRIOR_ALPHA            = 1   # no tempering (matches run_benchmarks.py)

# 'benchmark' context cutoff — matches CONTEXTS['benchmark'] in
# preparation_scripts/build_initial_prior.py.
CONTEXT_NAME = 'benchmark'
CUTOFF_STR   = '2018-09-30T00:00:00'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_TRIGS    = config.BENCHMARK_PARAMS['max_trigs']

RUN_DIR              = os.path.join(PROJECT_ROOT, 'data', 'california', 'run_files')
STATION_AVAIL_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'station_availability_cache.parquet')
CATALOG_PATH         = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'bEPIC_testing_catalog.txt')

ETAS_DIR             = os.path.join(PROJECT_ROOT, 'data', 'california', 'etas_inversion')
ETAS_INPUT_DIR       = os.path.join(ETAS_DIR, 'input')
SHARED_CATALOG_CACHE = os.path.join(ETAS_INPUT_DIR, 'etas_base_seismicity.parquet')
os.makedirs(ETAS_INPUT_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'output', 'time_dependent', f'max_trigs_{MAX_TRIGS}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%%
# ---------------------------------------------------------------------------
# BEPIC SETUP
# Shared inputs — built once, passed to every worker. More of this in the run_one() function.
# ---------------------------------------------------------------------------
catalog_df = benchmark_runner.load_reference_catalog(CATALOG_PATH) if os.path.exists(CATALOG_PATH) else None

station_availability = (
    load_station_availability_cache(STATION_AVAIL_CACHE)
    if os.path.exists(STATION_AVAIL_CACHE) else None
)


# ---------------------------------------------------------------------------
# ETAS SETUP 
# ---------------------------------------------------------------------------
# -- Shared ETAS seismicity catalog (independent of bw_sq — mc/m_ref/region
#    fixed across the sweep) — download once, filter to the context cutoff.
_m_ref = config.ETAS_INVERSION_CONFIG['m_ref']
_shape_lats = [pt[0] for pt in config.ETAS_INVERSION_CONFIG['shape_coords']]
_shape_lons = [pt[1] for pt in config.ETAS_INVERSION_CONFIG['shape_coords']]
_query_bounds = (
    min(_shape_lons) - 0.5, 
    max(_shape_lons) + 0.5,
    min(_shape_lats) - 0.5, 
    max(_shape_lats) + 0.5,
)

_aux_start  = config.ETAS_INVERSION_CONFIG['auxiliary_start']
_start_year = pd.Timestamp(_aux_start).year
_end_year   = pd.Timestamp(CUTOFF_STR).year

raw_catalog = load_background_seismicity(
    cache_path    = SHARED_CATALOG_CACHE,
    bounds        = _query_bounds,
    start_year    = _start_year,
    end_year      = _end_year,
    min_mag       = _m_ref,
    force_refresh = False,
)

# Normalise to ETAS column format: id, latitude, longitude, time, magnitude.
# background.py returns: time (UTC-aware), latitude, longitude, depth, mag.
raw_catalog = raw_catalog.copy()
raw_catalog.insert(0, 'id', range(len(raw_catalog)))
raw_catalog = raw_catalog.rename(columns={'mag': 'magnitude'})
raw_catalog['time'] = pd.to_datetime(raw_catalog['time'], utc=True).dt.tz_convert(None)
raw_catalog = raw_catalog[['id', 'latitude', 'longitude', 'time', 'magnitude']]
raw_catalog = raw_catalog[raw_catalog['magnitude'] >= _m_ref].reset_index(drop=True)

# -- Filter shared catalog to strictly before the cutoff ------------------
_cutoff_ts = pd.Timestamp(CUTOFF_STR)
etas_catalog_df = (
    raw_catalog[raw_catalog['time'] < _cutoff_ts]
    .reset_index(drop=True)
    .copy()
)
etas_catalog_df['id'] = range(len(etas_catalog_df))

#print(f"ETAS catalog: {len(etas_catalog_df):,} events before {CUTOFF_STR}")


#%%
# ---------------------------------------------------------------------------
# Worker: invert ETAS for one bw_sq, then run the full dynamic benchmark
# ---------------------------------------------------------------------------
def _run_one(bw_sq):

    # This is needed to correctly route the etas output inversion name per
    # the full set of parameters we care about (such as bw_sq)
    local_cfg = dict(config.ETAS_INVERSION_CONFIG,
                      bw_sq=bw_sq, timewindow_end=CUTOFF_STR, id=CONTEXT_NAME)
    
    tagged_id   = config.etas_output_id(CONTEXT_NAME, cfg=local_cfg)

    output_json = os.path.join(ETAS_DIR, f'parameters_{tagged_id}.json')

    if not (SKIP_EXISTING_INVERSION and os.path.exists(output_json)):

        # Save per-context catalog so results are reproducible without re-filtering.
        _catalog_tag = config.etas_catalog_tag(CONTEXT_NAME)
        _catalog_csv = os.path.join(ETAS_INPUT_DIR, f'catalog_{_catalog_tag}.csv')
        if not os.path.exists(_catalog_csv):
            etas_catalog_df.to_csv(_catalog_csv, index=False)
            print(f"  Catalog cached: {_catalog_csv}")

        # -- Build inversion metadata ---------------------------------------------
        #print(f"[bw_sq={bw_sq}] running ETAS inversion (id='{tagged_id}')…")
        inversion_metadata = config.ETAS_INVERSION_CONFIG
        inversion_metadata['timewindow_end']= CUTOFF_STR
        inversion_metadata['id'] =        CONTEXT_NAME
        inversion_metadata['bw_sq']     = bw_sq
        inversion_metadata['catalog']   = etas_catalog_df.copy()
        inversion_metadata['data_path'] = ETAS_DIR + os.sep
        inversion_metadata.pop('fn_catalog', None)

        # -- Run inversion --------------------------------------------------------
        set_up_logger(level=logging.WARNING)  # keep parallel worker output readable
        calculation = ETASParameterCalculation(inversion_metadata)
        calculation.prepare()
        calculation.invert()
        calculation.store_results(ETAS_DIR + os.sep, store_pij=False, store_spatial_fields=True)
        print(f"[bw_sq={bw_sq}] inversion complete -> {output_json}")

    else:
        print(f"[bw_sq={bw_sq}] reusing existing inversion -> {output_json}")

    # STart bEPIC location
    _usgs_ref_lookup = (
        catalog_df[['event_id', 'usgs_time', 'usgs_lat', 'usgs_lon', 'usgs_mag']]
        .rename(columns={'usgs_time': 'time', 'usgs_lat': 'latitude',
                          'usgs_lon': 'longitude', 'usgs_mag': 'magnitude'})
        .set_index('event_id')
    )

    repair_inversion_json_paths(output_json)
    updater = EtasPriorUpdater.from_inversion_json(
        json_path  = output_json,
        catalog_df = etas_catalog_df,
        **config.ETAS_UPDATER_CONFIG,
        spatial_factor = None,
    )


    def after_event_fn(event_id):
        eid_int = int(event_id)
        if eid_int not in _usgs_ref_lookup.index:
            return
        row = _usgs_ref_lookup.loc[eid_int]
        updater.append_events(pd.DataFrame([{
            'time': row['time'], 
            'latitude': row['latitude'],
            'longitude': row['longitude'], 
            'magnitude': row['magnitude'],
        }]))

    # Collect event IDs from available .run files, sorted by first trigger time
    run_files = sorted(Path(RUN_DIR).glob('*.run'))
    event_ids = sorted([f.stem for f in run_files], key=lambda eid: run_trigger_time(eid, RUN_DIR))
    print(f"{len(event_ids)} events in {RUN_DIR}")

    # -----------------------------------------------------------------------
    # -- Set up BenchmarkRunner with the initial prior
    # -- Run the dynamic prior ----------------------
    # -----------------------------------------------------------------------
    t0            = pd.Timestamp(run_trigger_time(event_ids[0], RUN_DIR), unit='s')
    initial_prior = updater.update(t0)

    params_dict = {**config.BENCHMARK_PARAMS, 'sigma_s': SIGMA_S}   # local copy — no shared mutable state
    params      = make_epic_params(initial_prior, True, params_dict)

    runner = BenchmarkRunner(prior=initial_prior, 
                             params=params,
                            run_dir=RUN_DIR,
                            catalog_df=catalog_df, 
                            station_availability=station_availability)

    print(f"[bw_sq={bw_sq}] running dynamic benchmark over {len(event_ids)} events…")
    runner.run_all(
        event_ids         = event_ids,
        etas_update_fn    = lambda t: etas_update_fn(t, updater, PRIOR_ALPHA),
        update_interval_s = ETAS_UPDATE_INTERVAL_S,
        after_event_fn    = after_event_fn,
    )

    out_path = os.path.join(OUTPUT_DIR, f'etas_dynamic_benchmark_results_bw{bw_sq:g}.csv')
    runner_results_to_df(runner).to_csv(out_path, index=False)
    print(f"[bw_sq={bw_sq}] saved -> {out_path}")
    return out_path

#%%
# ---------------------------------------------------------------------------
# Sweep bw_sq — one process per value
# ---------------------------------------------------------------------------
with ProcessPoolExecutor(max_workers=len(BW_SQ_VALUES)) as ex:
    futures = [ex.submit(_run_one, bw_sq) for bw_sq in BW_SQ_VALUES]
    for f in futures:
        f.result()  # re-raise any worker exception

# %%
