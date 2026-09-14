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
import pandas as pd

from pathlib import Path

# Custom repository imports
from priors import SeismicPrior, EtasPriorUpdater

from benchmark.usgs import *
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.runner import (BenchmarkRunner, runner_results_to_df, 
                              load_station_availability_cache,
                              make_epic_params)
from benchmark.time_dependent_helpers import *

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

BW_SQ = 4
config.ETAS_INVERSION_CONFIG['bw_sq'] = BW_SQ

# ── CONFIGURE ─────────────────────────────────────────────────────────────────

DEFAULT_CASE_STUDY = "Ridgecrest"
#DEFAULT_CASE_STUDY = "ElMayor"
#DEFAULT_CASE_STUDY = "Ferndale"
#DEFAULT_CASE_STUDY = "MTJ_2024_M7"
ACTIVE_CASE_STUDY = os.environ.get('CASE_STUDY', DEFAULT_CASE_STUDY)

SEIS_CACHE       = os.path.join(PROJECT_ROOT, 'data', 'california', 'reference', 'background_seismicity.parquet')
INVERSION_JSON   = os.path.join(PROJECT_ROOT, 'data', 'case_studies',ACTIVE_CASE_STUDY, 'etas_inversion',
                                f'parameters_{config.etas_output_id(ACTIVE_CASE_STUDY)}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'case_studies', ACTIVE_CASE_STUDY, 'etas_inversion', 'input',
                                  f'catalog_{config.etas_catalog_tag(ACTIVE_CASE_STUDY)}.csv')

cs = CASE_STUDIES[ACTIVE_CASE_STUDY]
AVAIL_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'case_studies',f'{ACTIVE_CASE_STUDY}', 'station_availability_cache.parquet')

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

CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')

CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'output',  'time_dependent', f'max_trigs_{MAX_TRIGS}')
CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'case_studies', ACTIVE_CASE_STUDY, 'figures', 'time_dependent', f'max_trigs_{MAX_TRIGS}')

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

focus_run_path = os.path.join(CS_RUN_DIR, f'{FOCUS_EVENT_ID}.run')



#%%
# ---------------------------------------------------------------------------
# Load catalog from cache (preparation_scripts/case_study_preparation.py must
# have been run first)
# ---------------------------------------------------------------------------
catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR, REDOWNLOAD=False)

_avail = (load_station_availability_cache(AVAIL_CACHE)
          if os.path.exists(AVAIL_CACHE)  else None)
if _avail:
    print("Station availability cache loaded")

# -- Prepare the case-study catalog in ETAS column format ----------------
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


cs_ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]


#%%
# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
# How often to re-evaluate the ETAS prior (in seconds of event time).
ETAS_UPDATE_INTERVAL_S = 0

# Prior tempering exponent.  1.0 = full ETAS weight; <1.0 compresses the
PRIOR_ALPHA = 1 # UNCHANGED behavior if this == 1

#%%
# ── ETAS dynamic ──────────────────────────────────────────────────────────
# -- Verify inversion output exists --------------------------------------

if not os.path.exists(INVERSION_JSON):
    raise FileNotFoundError(
        f"ETAS inversion output not found:\n  {INVERSION_JSON}\n"
        "Run time_dependent_scripts/build_initial_prior.py first."
    )

# -- Load historical catalog (background seismicity for ETAS) ------------
print(f"Loading historical catalog from:\n  {os.path.abspath(HISTORICAL_CATALOG)}")
hist_catalog = pd.read_csv(
    HISTORICAL_CATALOG,
    index_col=0,
    parse_dates=['time'],
    dtype={'url': str, 'alert': str},
)
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
    spatial_factor = None,
)

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


# Collect event IDs from available .run files, sorted by first trigger time
# (chronological order is critical so ETAS updates are causal).
run_files   = sorted(Path(CS_RUN_DIR).glob('*.run'))
event_ids  = sorted([f.stem for f in run_files], key=lambda eid: run_trigger_time(eid, CS_RUN_DIR))
print(f"\nRunning dynamic ETAS prior over {len(event_ids)} events "
        f"(update interval: "
        f"{'per-event' if ETAS_UPDATE_INTERVAL_S == 0 else f'{ETAS_UPDATE_INTERVAL_S}s'})…\n")


#%%
# -----------------------------------------------------------------------
# -- Set up BenchmarkRunner with the initial prior
# -- Run the dynamic prior ----------------------
# -----------------------------------------------------------------------
t0 = pd.Timestamp(cs['starttime'])
initial_prior = updater.update(t0)

params = make_epic_params(initial_prior, True, config.BENCHMARK_PARAMS)

runner = BenchmarkRunner(prior=initial_prior, 
                        params=params, 
                        run_dir=CS_RUN_DIR,
                        catalog_df=cs_ref_df,
                        station_availability=_avail)


runner.run_all(
    event_ids          = event_ids,
    etas_update_fn     = lambda t: etas_update_fn(t, updater, PRIOR_ALPHA),
    update_interval_s  = ETAS_UPDATE_INTERVAL_S,
    after_event_fn     = after_event_fn,
)

# -- Save results ---------------------------------------------------------
out_path = os.path.join(CS_OUTPUT_DIR, 'etas_dynamic_benchmark_results.csv')
runner_results_to_df(runner).to_csv(out_path, index=False)
print(f"\nDynamic ETAS results saved to:\n  {out_path}")


