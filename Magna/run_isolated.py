#%%
# =============================================================================
# run_isolated.py — priors on isolated M>=3 Wasatch Front events (where the ETAS background matters)
# =============================================================================
# For each event of data/wasatch_isolated (from download_isolated.py), in time order, with every prior built from
# seismicity BEFORE that event only (no leakage from the event itself or later ones):
#   Uniform       — no prior
#   KDE           — KDE of the M>=2.5 catalog before the event (all longitudes; Magna's -111.5 cut is not used)
#   ETAS          — dynamic ETAS prior whose background grid is the inverted one, recomputed from events before the event
#   ETAS_NSHMbg   — the same ETAS prior with the background replaced by the NSHM shape (same total background rate)
#   NSHM          — static NSHM prior
#   ETAS_bgonly   — only the (causal, inverted) background grid of the ETAS prior, no triggered terms
#   ETAS_trigonly — only the triggered terms (background scaled by 1e-6)
# CAUSALITY: every prior is evaluated 1 s BEFORE the event's origin time. (An earlier version used the first pick time,
# a few seconds after origin; the test event is itself in the ETAS history / background catalogs, so it then sat in the
# prior as a freshly triggering source at its own location and flattered ETAS. Those results are in *_LEAKED folders.)
# With MAGNA_BG_FLOOR set (variant.py), the ETAS-family priors instead use a fixed floor on the background RATE field
# (lambda = max(mu, floor) + triggered, no tempering / peak-relative floor); KDE and NSHM static keep adjust_prior.
# All priors are tempered + floored like the Magna runs (adjust_prior). The ETAS parameters come from the full inversion
# (a mild leak). Uses the ETAS inversion selected by the env switches in variant.py:
#   MAGNA_NOCUT=1 MAGNA_BW_SQ=16 python Magna/run_isolated.py
# Results: Magna/results/output_isolated/{prior}_benchmark_results.csv
# =============================================================================
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'

import copy
import json
from math import radians, sin, cos, pi

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from scipy.stats._stats import gaussian_kernel_estimate
from etas.intensity import _compute_background_field

from priors import SeismicPrior, EtasPriorUpdater
from benchmark import runner as benchmark_runner
from benchmark import config_california as config
from benchmark.runner import BenchmarkRunner, runner_results_to_df, make_epic_params
from benchmark.time_dependent_helpers import run_trigger_time
from variant import BW_SQ, CONTEXT, NSHM_TAG, ISO_TAG, PRIOR_ALPHA, FLOOR_FRAC, PRIOR_TAG, BG_FLOOR

MAGNA_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(MAGNA_DIR, 'data')
ISO_DIR    = os.path.join(DATA_DIR, f'wasatch_isolated{ISO_TAG}')
RUN_DIR    = os.path.join(ISO_DIR, 'run_files')
ETAS_DIR   = os.path.join(DATA_DIR, 'etas_inversion')
OUTPUT_DIR = os.path.join(MAGNA_DIR, 'results', f'output_isolated{ISO_TAG}{PRIOR_TAG}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDS  = (-114.5, -109.0, 37.0, 42.5)
M_MIN   = 2.5
ETAS_CFG = copy.deepcopy(config.ETAS_INVERSION_CONFIG)
ETAS_CFG.update({'mc': M_MIN, 'm_ref': M_MIN})
if BW_SQ is not None:
    ETAS_CFG['bw_sq'] = BW_SQ
ETAS_JSON = os.path.join(ETAS_DIR, f'parameters_{config.etas_output_id(CONTEXT, ETAS_CFG)}.json')
ETAS_HIST = os.path.join(ETAS_DIR, 'input', f'catalog_{config.etas_catalog_tag(CONTEXT, ETAS_CFG)}.csv')
NSHM_TT3  = os.path.join(DATA_DIR, 'priors', f'nshm_Magna{NSHM_TAG}.tt3')
PARAMS = copy.deepcopy(config.BENCHMARK_PARAMS); PARAMS['sigma_s'] = 0.22


def adjust_prior(prior, alpha=PRIOR_ALPHA, frac=FLOOR_FRAC):
    """Temper (grid**alpha), raise cells below frac * max to frac * max, renormalize. In place."""
    grid = prior.grid ** alpha if alpha != 1 else prior.grid
    grid = np.maximum(grid, frac * grid.max())
    prior.grid = grid / grid.sum()
    return prior


def build_kde(cat):
    """KDE prior of a catalog (Scott bandwidth, factor 0.05, diagonal covariance — as in run_benchmarks.py)."""
    lon_min, lon_max, lat_min, lat_max = BOUNDS
    ref_lat = radians(lat_min); mpd = 6378137.0 * (1 - sin(ref_lat) ** 2 / 298.257) * pi / 180
    to_km = lambda lo, la: ((lo - lon_min) * mpd * cos(ref_lat) / 1000, (la - lat_min) * mpd / 1000)
    lons = np.linspace(lon_min, lon_max, 200); lats = np.linspace(lat_min, lat_max, 200); xx, yy = np.meshgrid(lons, lats)
    pts = np.column_stack(to_km(cat['longitude'].values.astype(float), cat['latitude'].values.astype(float)))
    gp = np.column_stack(to_km(xx.ravel(), yy.ravel()))
    k = gaussian_kde(pts.T, bw_method='scott')
    cho = (scipy.linalg.cholesky(k._data_covariance, lower=True) * 0.05).astype(np.float64); cho[0, 1] = cho[1, 0] = 0
    dens = gaussian_kernel_estimate['double'](pts, np.ones(len(pts))[:, None], gp, cho, np.common_type(k.covariance, gp))
    grid = np.clip(dens.reshape(xx.shape), 0, None)
    return SeismicPrior(name='kde_seismicity', lons=lons, lats=lats, grid=grid / np.nansum(grid), metadata={'n_events': len(pts)})

#%%
# ── Events, background catalog, ETAS updater, NSHM background ───────────────────────────
iso = pd.read_parquet(os.path.join(ISO_DIR, 'catalog.parquet')).query('has_run').reset_index(drop=True)
ref_df = iso.rename(columns={'id': 'event_id', 'latitude': 'usgs_lat', 'longitude': 'usgs_lon'})[['event_id', 'usgs_lat', 'usgs_lon']]
event_ids = sorted(iso['id'], key=lambda e: run_trigger_time(e, RUN_DIR))
origin = dict(zip(iso['id'], pd.to_datetime(iso['time']).dt.tz_localize(None) if pd.to_datetime(iso['time']).dt.tz is not None else pd.to_datetime(iso['time'])))
print(f'{len(event_ids)} isolated events')

bg_cat = pd.read_parquet(os.path.join(DATA_DIR, 'background_earthquakes_M2.5.parquet'))
bg_cat['time'] = pd.to_datetime(bg_cat['time'], utc=True).dt.tz_convert(None)

hist_catalog = pd.read_csv(ETAS_HIST, index_col=0, parse_dates=['time'])
benchmark_runner.repair_inversion_json_paths(ETAS_JSON)
updater = EtasPriorUpdater.from_inversion_json(json_path=ETAS_JSON, catalog_df=hist_catalog,
                                               **{**config.ETAS_UPDATER_CONFIG, 'bounds': BOUNDS}, spatial_factor=None)
cfg_json = json.load(open(ETAS_JSON))
bg_events = pd.read_csv(cfg_json['fn_ip'], index_col=0, parse_dates=['time'])      # per-event P_background from the inversion
orig_bg = updater._bg_field.copy()

nshm = SeismicPrior.from_tt3(NSHM_TT3)
f = RegularGridInterpolator((nshm.lats, nshm.lons), nshm.grid, bounds_error=False, fill_value=np.nan)(
    np.column_stack([updater.grid_lats_masked, updater.grid_lons_masked]))
nshm_bg = np.where(np.isnan(f), np.nanmin(f), f); nshm_bg = nshm_bg / nshm_bg.sum() * orig_bg.sum()
nshm_static = adjust_prior(SeismicPrior.from_tt3(NSHM_TT3))


def causal_bg(t):
    """Inverted spatial background recomputed from events before t, rescaled to the full-sample total background rate."""
    e = bg_events[bg_events['time'] < t]
    f = _compute_background_field(updater.grid_lats_masked, updater.grid_lons_masked, e['latitude'].values, e['longitude'].values,
                                  e['P_background'].values, cfg_json['bw_sq'], cfg_json['timewindow_length'])
    return f / f.sum() * orig_bg.sum()

#%%
# ── Run ──────────────────────────────────────────────────────────────────────────────────
runners = {}
for name in ('Uniform', 'KDE', 'ETAS', 'ETAS_NSHMbg', 'NSHM', 'ETAS_bgonly', 'ETAS_trigonly'):
    p = nshm_static
    runners[name] = BenchmarkRunner(prior=p, params=make_epic_params(p, name != 'Uniform', PARAMS), run_dir=RUN_DIR, catalog_df=ref_df)

for i, eid in enumerate(event_ids, 1):
    t = origin[eid] - pd.Timedelta(seconds=1)      # strictly before the event: it must not be in any training catalog
    priors = {'Uniform': None,
              'KDE': adjust_prior(build_kde(bg_cat[bg_cat['time'] < t])),
              'NSHM': nshm_static}
    bg_t = causal_bg(t)
    if BG_FLOOR is None:
        for name, bg in (('ETAS', bg_t), ('ETAS_NSHMbg', nshm_bg), ('ETAS_trigonly', bg_t * 1e-6)):
            updater._bg_field = bg
            priors[name] = adjust_prior(updater.update(t))
        priors['ETAS_bgonly'] = adjust_prior(SeismicPrior.from_etas(
            lats=updater.grid_lats_masked, lons=updater.grid_lons_masked, lambda_grid=bg_t, forecast_time=t, bounds=BOUNDS,
            out_of_bounds_fill=config.ETAS_UPDATER_CONFIG['out_of_bounds_fill']))
    else:                                              # fixed rate floor on mu(x); the prior is not otherwise adjusted
        for name, bg in (('ETAS', np.maximum(bg_t, BG_FLOOR)), ('ETAS_NSHMbg', np.maximum(nshm_bg, BG_FLOOR)),
                         ('ETAS_trigonly', np.zeros_like(bg_t))):
            updater._bg_field = bg
            priors[name] = updater.update(t)
        priors['ETAS_bgonly'] = SeismicPrior.from_etas(
            lats=updater.grid_lats_masked, lons=updater.grid_lons_masked, lambda_grid=np.maximum(bg_t, BG_FLOOR), forecast_time=t,
            bounds=BOUNDS, out_of_bounds_fill=config.ETAS_UPDATER_CONFIG['out_of_bounds_fill'])
    for name, runner in runners.items():
        if priors[name] is not None:
            runner.update_prior(priors[name])
        runner.run_event(eid)
    print(f'  [{i}/{len(event_ids)}] {eid} done')

for name, runner in runners.items():
    runner_results_to_df(runner).to_csv(os.path.join(OUTPUT_DIR, f'{name.lower()}_benchmark_results.csv'), index=False)
print('results ->', OUTPUT_DIR)
