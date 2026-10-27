#%%
# =============================================================================
# run_benchmarks.py — bEPIC on the Magna, UT (March 2020) sequence
# =============================================================================
# Runs bEPIC over every .run file in Magna/data/run_files with three priors:
#   Uniform         — no prior (likelihood only)
#   KDE_Seismicity  — static KDE of M>=3 Utah seismicity before the sequence
#   ETAS            — dynamic ETAS prior, updated before each event and fed the
#                     USGS locations of already-located events (causal order)
#
# The KDE .tt3 and the ETAS parameter inversion are built on first use (they are
# skipped if their outputs already exist). Everything is written under Magna/.
#
# Prerequisite: Magna/download_triggers.py
# Usage:        python Magna/run_benchmarks.py
# =============================================================================
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'

import copy
import json
import logging
from datetime import datetime, timezone
from math import radians, sin, cos, pi
from pathlib import Path

import io

import numpy as np
import pandas as pd
import requests
import scipy.linalg
from scipy.stats import gaussian_kde
from scipy.stats._stats import gaussian_kernel_estimate

from priors import SeismicPrior, EtasPriorUpdater
from benchmark import runner as benchmark_runner
from benchmark import config_california as config
from benchmark.runner import BenchmarkRunner, runner_results_to_df, make_epic_params
from benchmark.time_dependent_helpers import etas_update_fn, run_trigger_time
from variant import BW_SQ, NOCUT, MAX_LON, CONTEXT, OUT_SUFFIX, ETAS_SUFFIX   # env switches, see variant.py

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAGNA_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(MAGNA_DIR, 'data')
RUN_DIR    = os.path.join(DATA_DIR, 'run_files')
PRIOR_DIR  = os.path.join(DATA_DIR, 'priors')
ETAS_DIR   = os.path.join(DATA_DIR, 'etas_inversion')
OUTPUT_DIR = os.path.join(MAGNA_DIR, 'results', f'output{OUT_SUFFIX}')
CATALOG_PATH = os.path.join(DATA_DIR, 'Magna_2020_catalog.parquet')   # from download_triggers.py
for d in (PRIOR_DIR, ETAS_DIR, os.path.join(ETAS_DIR, 'input'), OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

SEQUENCE_START = '2020-03-18T13:00:00'          # prior builds only see seismicity before this
BOUNDS         = (-114.5, -109.0, 37.0, 42.5)   # (lon_min, lon_max, lat_min, lat_max): Utah + margin
KDE_START_YEAR = 1980                          # KDE training catalog and ETAS timewindow_start
ETAS_START_YEAR = 1980                          # ETAS auxiliary_start (= timewindow_start: no earlier source events)
M_MIN          = 2.5                            # magnitude of completeness for the KDE and ETAS background
BACKGROUND_PATH = os.path.join(DATA_DIR, f'background_earthquakes_M{M_MIN:g}.parquet')

# ETAS inversion polygon, [lat, lon] pairs (etas_2 convention), inside BOUNDS.
ETAS_POLYGON = [[42.0, -114.0], [42.0, -109.5], [37.5, -109.5], [37.5, -114.0], [42.0, -114.0]]

KDE_TT3   = os.path.join(PRIOR_DIR, f'kde_seismicity_{CONTEXT}.tt3')
ETAS_CFG  = copy.deepcopy(config.ETAS_INVERSION_CONFIG)
ETAS_CFG.update({'mc': M_MIN, 'm_ref': M_MIN})   # fixed completeness, also used in the output file tags
if BW_SQ is not None:
    ETAS_CFG['bw_sq'] = BW_SQ
ETAS_ID   = config.etas_output_id(CONTEXT, ETAS_CFG)
ETAS_JSON = os.path.join(ETAS_DIR, f'parameters_{ETAS_ID}.json')
ETAS_HIST = os.path.join(ETAS_DIR, 'input', f'catalog_{config.etas_catalog_tag(CONTEXT, ETAS_CFG)}.csv')

PARAMS = copy.deepcopy(config.BENCHMARK_PARAMS)
PARAMS['sigma_s'] = 0.22       # as in the case-study scripts

FORCE_REBUILD_KDE   = False
FORCE_RERUN_ETAS    = False   # the inversion is slow
RUN_STATIC          = BW_SQ is None or NOCUT     # Uniform + KDE_Seismicity (unaffected by bw_sq, but by the cut)
RUN_ETAS            = True
ETAS_UPDATE_INTERVAL_S = 0     # 0 = update the ETAS prior before every event
PRIOR_ALPHA         = 0.5      # tempering exponent: prior**alpha, renormalized (1 = none; <1 flattens contrast)
FLOOR_FRAC          = 0.05     # then raise every prior's minimum to this fraction of its own maximum (None = off)


def adjust_prior(prior, alpha=PRIOR_ALPHA, frac=FLOOR_FRAC):
    """Temper (grid**alpha), then raise cells below frac * max up to frac * max; renormalize. In place.
    Tempering first makes the final minimum exactly frac of the final maximum."""
    grid = prior.grid ** alpha if alpha != 1 else prior.grid
    if frac is not None:
        grid = np.maximum(grid, frac * grid.max())
    prior.grid = grid / grid.sum()
    return prior

#%%
# ---------------------------------------------------------------------------
# Background seismicity (shared by the KDE and ETAS builds)
# ---------------------------------------------------------------------------
# Only events labeled as earthquakes: mining/other explosions, mine collapses and sonic booms
# are excluded (eventtype=earthquake). Mining-induced seismicity that ComCat labels as an
# earthquake is NOT removed by this.
if os.path.exists(BACKGROUND_PATH):
    background = pd.read_parquet(BACKGROUND_PATH)
else:
    resp = requests.get('https://earthquake.usgs.gov/fdsnws/event/1/query', timeout=120, params={
        'format': 'csv', 'eventtype': 'earthquake', 'orderby': 'time-asc',
        'starttime': f'{ETAS_START_YEAR}-01-01', 'endtime': SEQUENCE_START,
        'minmagnitude': M_MIN, 'minlongitude': BOUNDS[0], 'maxlongitude': BOUNDS[1],
        'minlatitude': BOUNDS[2], 'maxlatitude': BOUNDS[3]})
    resp.raise_for_status()
    background = pd.read_csv(io.StringIO(resp.text))[['time', 'latitude', 'longitude', 'depth', 'mag']]
    background['time'] = pd.to_datetime(background['time'], utc=True)
    background.to_parquet(BACKGROUND_PATH, index=False)
background = background.copy()
background['time'] = pd.to_datetime(background['time'], utc=True).dt.tz_convert(None)
background = background[(background['time'] < pd.Timestamp(SEQUENCE_START))
                        & (background['mag'] >= M_MIN)].reset_index(drop=True)
if MAX_LON is not None:
    background = background[background['longitude'] <= MAX_LON].reset_index(drop=True)
print(f"{len(background):,} background earthquakes M>={M_MIN} before {SEQUENCE_START}"
      + (f", lon <= {MAX_LON}" if MAX_LON is not None else ""))

#%%
# ---------------------------------------------------------------------------
# KDE prior (Scott bandwidth, factor 0.05, diagonal covariance — as in
# preparation_scripts/build_priors.py)
# ---------------------------------------------------------------------------
if FORCE_REBUILD_KDE or not os.path.exists(KDE_TT3):
    kde_cat  = background[background['time'] >= pd.Timestamp(f'{KDE_START_YEAR}-01-01')]
    lons_pts = kde_cat['longitude'].values.astype(float)
    lats_pts = kde_cat['latitude'].values.astype(float)

    lon_min, lon_max, lat_min, lat_max = BOUNDS
    ref_lat = radians(lat_min)
    mpd = 6378137.0 * (1 - sin(ref_lat) ** 2 / 298.257) * pi / 180    # m per degree

    def to_km(lon, lat):
        return (lon - lon_min) * mpd * cos(ref_lat) / 1000, (lat - lat_min) * mpd / 1000

    nx = ny = 200
    lons = np.linspace(lon_min, lon_max, nx)
    lats = np.linspace(lat_min, lat_max, ny)
    xx, yy = np.meshgrid(lons, lats)

    points   = np.column_stack(to_km(lons_pts, lats_pts))
    grid_pts = np.column_stack(to_km(xx.ravel(), yy.ravel()))

    k = gaussian_kde(points.T, bw_method='scott')
    cho_cov = (scipy.linalg.cholesky(k._data_covariance, lower=True) * 0.05).astype(np.float64)
    cho_cov[0, 1] = cho_cov[1, 0] = 0
    density = gaussian_kernel_estimate['double'](
        points, np.ones(len(points))[:, None], grid_pts, cho_cov,
        np.common_type(k.covariance, grid_pts))
    grid = np.clip(density.reshape(xx.shape), 0.0, None)
    grid = grid / np.nansum(grid)

    metadata = {'bw_method': 'scott', 'grid_size': [nx, ny], 'n_events': int(len(points)),
                'bounds': list(BOUNDS), 'generated_at': datetime.now(timezone.utc).isoformat()}
    SeismicPrior(name=f'kde_seismicity_{CONTEXT}', lons=lons, lats=lats, grid=grid,
                 metadata=metadata).to_tt3(KDE_TT3)
    print(f"KDE prior: {len(points):,} events -> {KDE_TT3}")
else:
    print(f"KDE prior exists: {KDE_TT3}")

#%%
# ---------------------------------------------------------------------------
# ETAS parameter inversion (as in preparation_scripts/build_initial_prior.py)
# ---------------------------------------------------------------------------
if FORCE_RERUN_ETAS or not os.path.exists(ETAS_JSON):
    from etas import set_up_logger
    from etas.inversion import ETASParameterCalculation

    etas_catalog = (background[['longitude', 'latitude', 'time', 'mag']]
                    .rename(columns={'mag': 'magnitude'})
                    .reset_index(drop=True))
    etas_catalog.insert(0, 'id', range(len(etas_catalog)))
    etas_catalog = etas_catalog[['id', 'latitude', 'longitude', 'time', 'magnitude']]
    etas_catalog.to_csv(ETAS_HIST, index=False)

    meta = copy.deepcopy(ETAS_CFG)
    meta.update({
        'auxiliary_start':  f'{ETAS_START_YEAR}-01-01 00:00:00',
        'timewindow_start': f'{KDE_START_YEAR}-01-01 00:00:00',
        'timewindow_end':   SEQUENCE_START,
        'shape_coords':     ETAS_POLYGON,
        'id':               ETAS_ID,
        'catalog':          etas_catalog,
        'data_path':        ETAS_DIR + os.sep,
    })
    meta.pop('fn_catalog', None)

    set_up_logger(level=logging.INFO)
    calculation = ETASParameterCalculation(meta)
    calculation.prepare()
    calculation.invert()
    calculation.store_results(ETAS_DIR + os.sep, store_pij=False, store_spatial_fields=True)

    with open(ETAS_JSON) as fh:
        for name, val in json.load(fh).get('final_parameters', {}).items():
            print(f"  {name:12s}: {val}")
else:
    print(f"ETAS inversion exists: {ETAS_JSON}")

#%%
# ---------------------------------------------------------------------------
# Reference catalog + event list
# ---------------------------------------------------------------------------
catalog_df = pd.read_parquet(CATALOG_PATH)
cs_ref_df = catalog_df.rename(columns={'id': 'event_id', 'latitude': 'usgs_lat',
                                       'longitude': 'usgs_lon'})[['event_id', 'usgs_lat', 'usgs_lon']]
print(f"{len(catalog_df)} events in catalog; {len(list(Path(RUN_DIR).glob('*.run')))} .run files")

#%%
# ---------------------------------------------------------------------------
# Static priors: Uniform + KDE_Seismicity
# ---------------------------------------------------------------------------
if RUN_STATIC:
    kde_prior = adjust_prior(SeismicPrior.from_tt3(KDE_TT3))
    for name, use_prior in (('Uniform', False), ('KDE_Seismicity', True)):
        print(f"\n=== {name} ===")
        params = make_epic_params(kde_prior, use_prior, PARAMS)   # Uniform: prior object unused
        benchmark_runner.run_prior(params, name, cs_ref_df, RUN_DIR, OUTPUT_DIR)

#%%
# ---------------------------------------------------------------------------
# Dynamic ETAS prior (causal: update prior -> locate event -> append USGS location)
# ---------------------------------------------------------------------------
if RUN_ETAS:
    m_ref = ETAS_CFG['m_ref']
    cs_etas_catalog = (
        catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']]
        .rename(columns={'mag': 'magnitude'})
        .assign(time=lambda df: pd.to_datetime(df['time']).dt.tz_localize(None))
        .query(f'magnitude >= {m_ref}')
        .sort_values('time')
        .reset_index(drop=True)
    )
    cs_event_lookup = cs_etas_catalog.set_index('id')

    hist_catalog = pd.read_csv(ETAS_HIST, index_col=0, parse_dates=['time'])
    benchmark_runner.repair_inversion_json_paths(ETAS_JSON)
    updater = EtasPriorUpdater.from_inversion_json(
        json_path      = ETAS_JSON,
        catalog_df     = hist_catalog,
        **{**config.ETAS_UPDATER_CONFIG, 'bounds': BOUNDS},
        spatial_factor = None,
    )

    def after_event_fn(event_id):
        if event_id in cs_event_lookup.index:
            updater.append_events(cs_event_lookup.loc[[event_id],
                                                      ['time', 'latitude', 'longitude', 'magnitude']])

    event_ids = sorted((f.stem for f in Path(RUN_DIR).glob('*.run')),
                       key=lambda eid: run_trigger_time(eid, RUN_DIR))
    print(f"\n=== ETAS dynamic: {len(event_ids)} events ===")

    initial_prior = adjust_prior(updater.update(pd.Timestamp(SEQUENCE_START)))
    params = make_epic_params(initial_prior, True, PARAMS)
    runner = BenchmarkRunner(prior=initial_prior, params=params, run_dir=RUN_DIR,
                             catalog_df=cs_ref_df)
    runner.run_all(
        event_ids         = event_ids,
        etas_update_fn    = lambda t: adjust_prior(etas_update_fn(t, updater, 1)),   # tempering + floor in adjust_prior
        update_interval_s = ETAS_UPDATE_INTERVAL_S,
        after_event_fn    = after_event_fn,
    )

    out_path = os.path.join(OUTPUT_DIR, f'etas_dynamic{ETAS_SUFFIX}_benchmark_results.csv')
    runner_results_to_df(runner).to_csv(out_path, index=False)
    print(f"ETAS results -> {out_path}")
