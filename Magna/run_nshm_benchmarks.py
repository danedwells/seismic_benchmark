#%%
# =============================================================================
# run_nshm_benchmarks.py — NSHM and ETAS+NSHM priors on the Magna, UT sequence
# =============================================================================
# Utah is a sparse-seismicity, normal-faulting region, so the USGS NSHM gridded +
# fault moment-rate model (Wasatch fault etc.) is tried in place of the KDE prior:
#   NSHM             — static NSHM prior, tempered + floored like the other Magna priors
#   NSHM_a1          — same, with no tempering (PRIOR_ALPHA = 1)
#   ETAS_NSHM_a{w}   — weight w on the dynamic ETAS prior + (1 - w) on NSHM, run serially
#                      in causal order (as mixed_prior_scripts/run_benchmarks.py)
#   ETAS_NSHMbg_x{s} — the ETAS prior with its inverted background-rate grid replaced by the NSHM shape
#                      (same total background rate x s); the triggering kernel/parameters are left as inverted
# Uses the same ETAS inversion, run files and bEPIC settings as Magna/run_benchmarks.py,
# which must have been run first. Results go to Magna/results/output/.
#
# Env switches (Magna/variant.py): MAGNA_NOCUT=1, MAGNA_BW_SQ=16, MAGNA_NSHM_RES=0.02 select the ETAS inversion / NSHM grid.
# Usage: python Magna/run_nshm_benchmarks.py
# =============================================================================
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']  = '1'

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from priors import SeismicPrior, EtasPriorUpdater
from benchmark import runner as benchmark_runner
from benchmark import config_california as config
from benchmark.priors import blend_priors
from benchmark.runner import BenchmarkRunner, runner_results_to_df, make_epic_params
from benchmark.time_dependent_helpers import run_trigger_time
from variant import BW_SQ, CONTEXT, OUT_SUFFIX, ETAS_SUFFIX, NSHM_RES, NSHM_TAG   # env switches, see variant.py

# ---------------------------------------------------------------------------
# Config — must match Magna/run_benchmarks.py
# ---------------------------------------------------------------------------
MAGNA_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(MAGNA_DIR, 'data')
RUN_DIR    = os.path.join(DATA_DIR, 'run_files')
PRIOR_DIR  = os.path.join(DATA_DIR, 'priors')
ETAS_DIR   = os.path.join(DATA_DIR, 'etas_inversion')
OUTPUT_DIR = os.path.join(MAGNA_DIR, 'results', f'output{OUT_SUFFIX}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
CATALOG_PATH = os.path.join(DATA_DIR, 'Magna_2020_catalog.parquet')

SEQUENCE_START = '2020-03-18T13:00:00'
BOUNDS         = (-114.5, -109.0, 37.0, 42.5)
M_MIN          = 2.5
ETAS_CFG  = copy.deepcopy(config.ETAS_INVERSION_CONFIG)
ETAS_CFG.update({'mc': M_MIN, 'm_ref': M_MIN})
if BW_SQ is not None:
    ETAS_CFG['bw_sq'] = BW_SQ
ETAS_ID   = config.etas_output_id(CONTEXT, ETAS_CFG)
ETAS_JSON = os.path.join(ETAS_DIR, f'parameters_{ETAS_ID}.json')
ETAS_HIST = os.path.join(ETAS_DIR, 'input', f'catalog_{config.etas_catalog_tag(CONTEXT, ETAS_CFG)}.csv')

PARAMS = copy.deepcopy(config.BENCHMARK_PARAMS)
PARAMS['sigma_s'] = 0.22

NSHM_DIR  = os.path.join(SeismicPrior.data_dir, 'USGS_NSHM_data')
NSHM_TT3  = os.path.join(PRIOR_DIR, f'nshm_Magna{NSHM_TAG}.tt3')   # NSHM_RES: bilinear-resampled finer grid (native is 0.1 deg)
PRIOR_ALPHA, FLOOR_FRAC = 0.5, 0.05      # tempering + floor, as for the KDE/ETAS priors in run_benchmarks.py
STATIC_VARIANTS = {f'NSHM{NSHM_TAG}': PRIOR_ALPHA, f'NSHM{NSHM_TAG}_a1': 1.0}    # name -> tempering exponent
BLEND_WEIGHTS   = (0.25, 0.5, 0.75)                        # weight on ETAS in ETAS+NSHM
BG_SCALES       = (0.25, 1.0, 4.0)                         # total background rate (x the inverted one) for the NSHM-background ETAS


def adjust_prior(prior, alpha=PRIOR_ALPHA, frac=FLOOR_FRAC):
    """Temper (grid**alpha), raise cells below frac * max to frac * max, renormalize. In place."""
    grid = prior.grid ** alpha if alpha != 1 else prior.grid
    if frac is not None:
        grid = np.maximum(grid, frac * grid.max())
    prior.grid = grid / grid.sum()
    return prior

#%%
# ── NSHM prior for Utah (the cached California NSHM prior stops at -112 deg) ──
if not os.path.exists(NSHM_TT3):
    nshm = SeismicPrior.from_nshm(os.path.join(NSHM_DIR, 'gridded_moment_rates.xyz'),
                                  fault_data_path=os.path.join(NSHM_DIR, 'fault_moment_rates.xyz'),
                                  bounds=BOUNDS)
    if NSHM_RES is not None:
        nshm = nshm.resample(NSHM_RES)       # bilinear: smoother/less blocky, adds no new information
        # SeismicPrior.resample can land its last row/column a float epsilon outside the source grid -> NaN; copy the neighbor
        g = pd.DataFrame(nshm.grid).ffill(axis=0).ffill(axis=1).bfill(axis=0).bfill(axis=1).values
        nshm.grid = g / g.sum()
    nshm.to_tt3(NSHM_TT3)
    print(f'NSHM prior built -> {NSHM_TT3}  grid {nshm.grid.shape}')
else:
    print(f'NSHM prior exists: {NSHM_TT3}')

catalog_df = pd.read_parquet(CATALOG_PATH)
cs_ref_df = catalog_df.rename(columns={'id': 'event_id', 'latitude': 'usgs_lat',
                                       'longitude': 'usgs_lon'})[['event_id', 'usgs_lat', 'usgs_lon']]

#%%
# ── Static NSHM ─────────────────────────────────────────────────────────────
for name, alpha in STATIC_VARIANTS.items():
    print(f'\n=== {name} (tempering {alpha:g}) ===')
    prior = adjust_prior(SeismicPrior.from_tt3(NSHM_TT3), alpha=alpha)
    benchmark_runner.run_prior(make_epic_params(prior, True, PARAMS), name, cs_ref_df, RUN_DIR, OUTPUT_DIR)

#%%
# ── ETAS + NSHM blends (causal: update ETAS -> blend -> locate -> append USGS location) ──
m_ref = ETAS_CFG['m_ref']
cs_etas_catalog = (catalog_df[['id', 'time', 'latitude', 'longitude', 'mag']]
                   .rename(columns={'mag': 'magnitude'})
                   .assign(time=lambda df: pd.to_datetime(df['time']).dt.tz_localize(None))
                   .query(f'magnitude >= {m_ref}').sort_values('time').reset_index(drop=True))
cs_event_lookup = cs_etas_catalog.set_index('id')

hist_catalog = pd.read_csv(ETAS_HIST, index_col=0, parse_dates=['time'])
benchmark_runner.repair_inversion_json_paths(ETAS_JSON)
updater = EtasPriorUpdater.from_inversion_json(
    json_path=ETAS_JSON, catalog_df=hist_catalog,
    **{**config.ETAS_UPDATER_CONFIG, 'bounds': BOUNDS}, spatial_factor=None)

nshm_adj = adjust_prior(SeismicPrior.from_tt3(NSHM_TT3))
event_ids = sorted((f.stem for f in Path(RUN_DIR).glob('*.run')), key=lambda eid: run_trigger_time(eid, RUN_DIR))

def blended(etas_prior, w):
    return blend_priors(nshm_adj, adjust_prior(etas_prior), w)     # both components tempered + floored

t0 = pd.Timestamp(SEQUENCE_START)
runners = {}
for w in BLEND_WEIGHTS:
    p0 = blended(updater.update(t0), w)
    runners[w] = BenchmarkRunner(prior=p0, params=make_epic_params(p0, True, PARAMS), run_dir=RUN_DIR, catalog_df=cs_ref_df)

print(f'\n=== ETAS+NSHM blends {BLEND_WEIGHTS}: {len(event_ids)} events ===')
for event_id in event_ids:
    etas = updater.update(pd.Timestamp(run_trigger_time(event_id, RUN_DIR), unit='s'))
    for w, runner in runners.items():
        runner.update_prior(blended(copy.deepcopy(etas), w))
        runner.run_event(event_id)
    if event_id in cs_event_lookup.index:
        updater.append_events(cs_event_lookup.loc[[event_id], ['time', 'latitude', 'longitude', 'magnitude']])

for w, runner in runners.items():
    out = os.path.join(OUTPUT_DIR, f'etas_nshm{NSHM_TAG}{ETAS_SUFFIX}_a{w:g}_benchmark_results.csv')
    runner_results_to_df(runner).to_csv(out, index=False)
    print(f'ETAS+NSHM (w={w:g}) -> {out}')

#%%
# ── ETAS with the background-rate grid replaced by NSHM ───────────────────────────
# lambda(x, t) = mu(x) + sum of triggered terms. mu(x) is updater._bg_field (the inverted, spatially smoothed
# background); here it becomes the NSHM shape rescaled to the same total, times BG_SCALES. Parameters are not re-fit.
from scipy.interpolate import RegularGridInterpolator

updater = EtasPriorUpdater.from_inversion_json(
    json_path=ETAS_JSON, catalog_df=hist_catalog,
    **{**config.ETAS_UPDATER_CONFIG, 'bounds': BOUNDS}, spatial_factor=None)
orig_bg = updater._bg_field.copy()
nshm_raw = SeismicPrior.from_tt3(NSHM_TT3)
f = RegularGridInterpolator((nshm_raw.lats, nshm_raw.lons), nshm_raw.grid, bounds_error=False, fill_value=np.nan)(
    np.column_stack([updater.grid_lats_masked, updater.grid_lons_masked]))
f = np.where(np.isnan(f), np.nanmin(f), f)
nshm_bg = f / f.sum() * orig_bg.sum()                    # same total background rate as the inverted field
i_m = np.argmin(np.hypot(updater.grid_lats_masked - 40.75, (updater.grid_lons_masked + 112.08) * 0.76))
print(f'\nbackground at Magna / grid mean: inverted {orig_bg[i_m] / orig_bg.mean():.2f}, NSHM {nshm_bg[i_m] / nshm_bg.mean():.2f}')

def bg_prior(t, scale):
    updater._bg_field = nshm_bg * scale
    return adjust_prior(updater.update(t))               # tempered + floored like the other ETAS runs

t0 = pd.Timestamp(SEQUENCE_START)
runners = {sc: BenchmarkRunner(prior=(p0 := bg_prior(t0, sc)), params=make_epic_params(p0, True, PARAMS),
                               run_dir=RUN_DIR, catalog_df=cs_ref_df) for sc in BG_SCALES}
print(f'=== ETAS with NSHM background, scales {BG_SCALES}: {len(event_ids)} events ===')
for event_id in event_ids:
    t = pd.Timestamp(run_trigger_time(event_id, RUN_DIR), unit='s')
    for sc, runner in runners.items():
        runner.update_prior(bg_prior(t, sc))
        runner.run_event(event_id)
    if event_id in cs_event_lookup.index:               # once per event, after all scales have located it
        updater.append_events(cs_event_lookup.loc[[event_id], ['time', 'latitude', 'longitude', 'magnitude']])

for sc, runner in runners.items():
    out = os.path.join(OUTPUT_DIR, f'etas_nshmbg{NSHM_TAG}{ETAS_SUFFIX}_x{sc:g}_benchmark_results.csv')
    runner_results_to_df(runner).to_csv(out, index=False)
    print(f'ETAS + NSHM background (x{sc:g}) -> {out}')
