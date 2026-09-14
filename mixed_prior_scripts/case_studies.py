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
import pandas as pd
from pathlib import Path

from priors import SeismicPrior, EtasPriorUpdater
from benchmark.usgs import *
from benchmark import runner as benchmark_runner
from benchmark import config
from benchmark.priors import blend_priors
from benchmark.runner import (BenchmarkRunner, runner_results_to_df,
                              make_epic_params, load_station_availability_cache)
from benchmark.time_dependent_helpers import run_trigger_time

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
# Case study definitions — loaded from benchmark/config_california.py
# ---------------------------------------------------------------------------
CASE_STUDIES = config.CASE_STUDIES

# ── CONFIGURE ────────────────────────────────────────────────────────────────

ACTIVE_CASE_STUDY = 'MTJ_2024_M7'

# ETAS inversion parameters and catalog — context-specific
INVERSION_JSON     = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', ACTIVE_CASE_STUDY, 'etas_inversion',
                                   f'parameters_{config.etas_output_id(ACTIVE_CASE_STUDY)}.json')
HISTORICAL_CATALOG = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies', ACTIVE_CASE_STUDY, 'etas_inversion', 'input',
                                   f'catalog_{config.etas_catalog_tag(ACTIVE_CASE_STUDY)}.csv')
cs = CASE_STUDIES[ACTIVE_CASE_STUDY]
AVAIL_CACHE  = os.path.join(PROJECT_ROOT, 'data', 'california', 'case_studies',f'{ACTIVE_CASE_STUDY}', 'station_availability_cache.parquet')

# Blending weights: ALPHA on the ETAS component, (1-ALPHA) on the static prior.
# Higher alpha = ETAS prior more important
# Lower alpha = STATIC prior more important
ALPHA     = 0.5
ALPHA_TAG = f'alpha_{ALPHA:.2f}'

# ---------------------------------------------------------------------------
# N-dependent prior schedule (per-trigger-count ALPHA/PRIOR_ALPHA taper)
# ---------------------------------------------------------------------------
# Master switch. False (default) = ALPHA/PRIOR_ALPHA above are used unchanged
# for every trigger count, exactly as before this feature existed.
USE_N_TRIGS_SCHEDULE = True

# 'tempering_only' — alpha stays pinned at 1.0 (no TI-prior contribution,
#                     ever); only prior_alpha (ETAS tempering) tapers with N.
#                     All 5 TI-prior runners converge to the identical
#                     tempered-ETAS-alone prior — a clean baseline.
# 'full_blend'      — alpha *also* tapers with N, so each runner's TI 
#                     (time-independent/static) prior
#                     gains influence as N grows, on top of the tempering.
#SCHED_MODE = 'full_blend'
SCHED_MODE = 'tempering_only'

# Taper shape: full ETAS sharpness (alpha=1, prior_alpha=1) through
# SCHED_FLAT_THROUGH_N triggers, then a linear ramp down to the configured
# minimums by SCHED_TAPER_END_N triggers. Starting points from this
# session's constant-PRIOR_ALPHA sweep — meant to be re-swept.
SCHED_FLAT_THROUGH_N  = 4
SCHED_TAPER_END_N     = 10
SCHED_MIN_PRIOR_ALPHA = 0.5 # Minimum etas alpha
SCHED_MIN_ALPHA       = 0.5 # minimum weight assigned to ETAS (corresponds to 0.5 weight for static)

SCHED_TAG = ('off' if not USE_N_TRIGS_SCHEDULE else SCHED_MODE)


def _linear_taper(n_trigs, min_value):
    """1.0 for n_trigs <= SCHED_FLAT_THROUGH_N, ramping linearly to
    min_value by SCHED_TAPER_END_N, then flat at min_value beyond that."""
    if n_trigs <= SCHED_FLAT_THROUGH_N:
        return 1.0
    if n_trigs >= SCHED_TAPER_END_N:
        return min_value
    frac = (n_trigs - SCHED_FLAT_THROUGH_N) / (SCHED_TAPER_END_N - SCHED_FLAT_THROUGH_N)
    return 1.0 + frac * (min_value - 1.0)


def prior_alpha_schedule(n_trigs):
    """ETAS tempering exponent as a function of trigger count. Active in
    both SCHED_MODE variants whenever USE_N_TRIGS_SCHEDULE is True."""
    return _linear_taper(n_trigs, SCHED_MIN_PRIOR_ALPHA)


def alpha_schedule(n_trigs):
    """ETAS blend weight as a function of trigger count. Pinned at 1.0
    (no TI-prior contribution) in 'tempering_only' mode; tapers like
    prior_alpha_schedule in 'full_blend' mode."""
    if SCHED_MODE == 'tempering_only':
        return 1.0
    return _linear_taper(n_trigs, SCHED_MIN_ALPHA)


def build_n_trigs_prior_fn(ti_prior, base_etas_prior):
    """Factory passed as BenchmarkRunner.run_all's n_trigs_schedule_fn (or
    called directly per-event here). Returns callable(n_trigs) -> SeismicPrior
    that blends ti_prior with base_etas_prior at schedule-derived alpha/
    prior_alpha for that trigger count."""
    def _prior_for_n_trigs(n_trigs):
        return blend_priors(
            ti_prior, base_etas_prior,
            alpha       = alpha_schedule(n_trigs),
            prior_alpha = prior_alpha_schedule(n_trigs),
        )
    return _prior_for_n_trigs

# Focus event for the standalone posterior / trajectory figures.
_MS_ = False  # set True to use mainshock events instead of representative aftershocks
FOCUS_EVENT_ID = config.FOCUS_EVENTS_MAINSHOCK[ACTIVE_CASE_STUDY] if _MS_ else config.FOCUS_EVENTS[ACTIVE_CASE_STUDY]
FOCUS_VERSION  = None

# Per-case-study directories
MAX_TRIGS      = config.BENCHMARK_PARAMS['max_trigs']
CS_DATA_DIR    = os.path.join(PROJECT_ROOT, 'data',    'california', 'case_studies', ACTIVE_CASE_STUDY)
CS_RUN_DIR     = os.path.join(CS_DATA_DIR, 'run_files')
CS_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', ACTIVE_CASE_STUDY,
                               'output', 'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG, f'sched_{SCHED_TAG}')
CS_FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'california', 'case_studies', ACTIVE_CASE_STUDY,
                               'figures', 'mixed', f'max_trigs_{MAX_TRIGS}', ALPHA_TAG, f'sched_{SCHED_TAG}')

for _d in (CS_DATA_DIR, CS_RUN_DIR, CS_OUTPUT_DIR, CS_FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

focus_run_path = os.path.join(CS_RUN_DIR, f'{FOCUS_EVENT_ID}.run')

#%%
# ---------------------------------------------------------------------------
# Load catalog from cache (preparation_scripts/case_study_preparation.py must
# have been run first)
# ---------------------------------------------------------------------------

catalog_df = download_case_study_catalog(cs, cache_dir=CS_DATA_DIR, REDOWNLOAD=False)

_avail = (load_station_availability_cache(AVAIL_CACHE)
          if os.path.exists(AVAIL_CACHE) else None)
if _avail:
    print("Station availability cache loaded")

# Reference catalog (for location error computation)
cs_ref_df = catalog_df.rename(columns={
    'id':        'event_id',
    'latitude':  'usgs_lat',
    'longitude': 'usgs_lon',
})[['event_id', 'usgs_lat', 'usgs_lon']]

# Prepare case-study events for incremental ETAS feeding.
m_ref = config.ETAS_INVERSION_CONFIG['m_ref']
cs_etas_catalog = (
    catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']]
    .rename(columns={'mag': 'magnitude'})
    .assign(time=lambda df: pd.to_datetime(df['time']).dt.tz_localize(None))
    .query(f'magnitude >= {m_ref}')
    .sort_values('time')
    .reset_index(drop=True)
)
cs_event_lookup = cs_etas_catalog.set_index('id')
print(f"  {len(cs_etas_catalog)} case-study events at or above m_ref={m_ref} "
      f"will be fed to ETAS incrementally.")

# ---------------------------------------------------------------------------
# Blending utility (imported from benchmark.priors — see benchmark/priors.py)
# ---------------------------------------------------------------------------

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
# Main workflow
# ---------------------------------------------------------------------------
# How often to re-evaluate the ETAS prior (in seconds of event time).
ETAS_UPDATE_INTERVAL_S = 0

# Power-law tempering applied to the raw ETAS grid before blending.
PRIOR_ALPHA = 1.2

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
benchmark_runner.repair_inversion_json_paths(INVERSION_JSON)

# Make updater
updater = EtasPriorUpdater.from_inversion_json(
    json_path  = INVERSION_JSON,
    catalog_df = hist_catalog,
    **config.ETAS_UPDATER_CONFIG,
)

#%%
# ---------------------------------------------------------------------------
# 5. Sort events chronologically (causal ETAS ordering)
# ---------------------------------------------------------------------------

run_files = sorted(Path(CS_RUN_DIR).glob('*.run'))
event_ids = sorted([f.stem for f in run_files], key=lambda eid: run_trigger_time(eid, CS_RUN_DIR))
print(f"\nRunning dynamic ETAS prior over {len(event_ids)} events "
        f"(update interval: "
        f"{'per-event' if ETAS_UPDATE_INTERVAL_S == 0 else f'{ETAS_UPDATE_INTERVAL_S}s'})…\n")


#%%
# ---------------------------------------------------------------------------
# 6. Mixed-prior benchmark loop
# ---------------------------------------------------------------------------

# Initialise runners with blended priors at the start of the sequence
_t0           = pd.Timestamp(cs['starttime'])
_current_etas = updater.update(_t0)
print(f"\nInitial ETAS prior evaluated at {_t0.strftime('%Y-%m-%d %H:%M:%S')}")

runners = {}
for name, ti_prior in ti_priors.items():
    # Initial-state placeholder prior: for the N-trigs schedule this is
    # immediately superseded per-version by prior_for_n_trigs below.
    initial_mixed = (
        blend_priors(ti_prior, _current_etas, alpha_schedule(1), prior_alpha_schedule(1))
        if USE_N_TRIGS_SCHEDULE else
        blend_priors(ti_prior, _current_etas, ALPHA, PRIOR_ALPHA)
    )
    params        = make_epic_params(initial_mixed, True, config.BENCHMARK_PARAMS)
    runners[name] = BenchmarkRunner(
        prior                = initial_mixed,
        params               = params,
        run_dir              = CS_RUN_DIR,
        catalog_df           = cs_ref_df,
        station_availability = _avail,
    )

_last_etas_update_unix = _t0.timestamp()

if USE_N_TRIGS_SCHEDULE:
    print(f"\nRunning mixed-prior benchmark over {len(event_ids)} events "
            f"({len(runners)} TI priors × ETAS, N-trigs schedule mode={SCHED_MODE})…\n")
else:
    print(f"\nRunning mixed-prior benchmark over {len(event_ids)} events "
            f"({len(runners)} TI priors × ETAS, alpha={ALPHA})…\n")

for i, event_id in enumerate(event_ids):
    event_time_unix = run_trigger_time(event_id, CS_RUN_DIR)
    t = pd.Timestamp(event_time_unix, unit='s')

    # Re-evaluate ETAS if the update interval has elapsed
    if (ETAS_UPDATE_INTERVAL_S == 0 or
            event_time_unix - _last_etas_update_unix >= ETAS_UPDATE_INTERVAL_S):
        _current_etas          = updater.update(t)
        _last_etas_update_unix = event_time_unix
        print(f"  [ETAS] updated at {t.strftime('%Y-%m-%d %H:%M:%S')} "
                f"— catalog: {updater.n_catalog_events} events")

    # Run bEPIC for each blended prior
    for name, ti_prior in ti_priors.items():
        if USE_N_TRIGS_SCHEDULE:
            runners[name].run_event(
                event_id,
                prior_for_n_trigs=build_n_trigs_prior_fn(ti_prior, _current_etas),
            )
        else:
            mixed = blend_priors(ti_prior, _current_etas, ALPHA, PRIOR_ALPHA)
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

