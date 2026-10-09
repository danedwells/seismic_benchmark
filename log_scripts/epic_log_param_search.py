"""
Hyperparameter search for the 2024 M7.0 MTJ mainshock (nc75095651) using the stations/triggers
from the EPIC replay log (log_files/MTJ-selected/MTJ_sigma_and_prior), located with the
benchmark's bEPIC locator and the dynamic ETAS prior.

For each distinct station count (up to N_MAX) the first EPIC version that reaches it is turned
into a .run file (one trigger per station, in the log's own order; written to
data/.../MTJ_2024_M7/epic_log_run_files/).  Each parameter combination is then scored by the
MAP error (km) against the USGS epicenter over those station counts.

Cheap trick: the bEPIC posterior is exactly likelihood * prior (checked below), so PRIOR_ALPHA
(prior ** alpha) is swept analytically from one locator run per (sigma_s, dtt_weight, Vp, depths).
One event only, so treat the winner as a diagnostic, not a tuned setting.
"""
#%%
import io
import contextlib
import copy
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from priors import EtasPriorUpdater
import benchmark.config_california as config
from benchmark import runner as benchmark_runner
from benchmark.runner import make_epic_params, run_single_event_get_grid
from benchmark.usgs import download_case_study_catalog
from benchmark.metrics import _haversine_km
from bEPIC import EPIC_locate_prelim

ROOT  = Path(__file__).resolve().parents[1]
HERE  = Path(__file__).resolve().parent
CASE, EVENT_ID = 'MTJ_2024_M7', 'nc75095651'
LOG   = ROOT / 'log_files' / 'MTJ-selected' / 'MTJ_sigma_and_prior' / 'epic' / f'{EVENT_ID}.log'
CS_DATA = ROOT / 'data' / 'california' / 'case_studies' / CASE
RUN_DIR = CS_DATA / 'epic_log_run_files'
USGS  = (40.3740, -125.0217)

N_MIN, N_MAX = 4, 10            # station counts scored (3 stations is too ambiguous to score)
BASELINE = dict(sigma_s=0.22, dtt_weight=0.0, vp=6.0, depths=(8.0,), alpha=1.0)   # the case-study settings
GRID = dict(
    sigma_s    = [0.1, 0.22, 0.35, 0.5, 1.0],      # travel-time uncertainty per pick (s)
    dtt_weight = [0.0, 0.5],                        # weight of differential travel times
    vp         = [5.5, 6.0, 6.5, 7.0],              # bEPIC P velocity (km/s)
    depths     = [(8.0,), (4.0, 8.0, 12.0, 16.0)],  # candidate source depths (km)
)
ALPHAS = [0.5, 1.0, 1.3, 2.0, 3.0]                  # ETAS prior exponent (PRIOR_ALPHA)
EDT_SIGMA_S = 0.02

#%% Triggers per EPIC version (E:I:T:), one trigger per station in the log's order
hdr, rows = None, []
for line in LOG.open(errors='replace'):
    _, _, msg = line.partition('| INFO | E:I:T:')
    f = msg.split()
    if not f:
        continue
    if f[0] == 'H:':
        hdr = hdr or f[1:]
    elif hdr and len(f) >= len(hdr) and f[0].lstrip('-').isdigit():
        rows.append(f[:len(hdr)])
trig = pd.DataFrame(rows, columns=hdr)
for c in ('eventid', 'ver', 'order', 'lat', 'lon', 'tterr'):
    trig[c] = pd.to_numeric(trig[c])
trig = trig[trig.eventid > 0].copy()
trig['t'] = pd.to_datetime(trig['trigger_time']).dt.tz_localize('UTC').map(lambda x: x.timestamp())

sets = {}                                           # n stations -> trigger DataFrame (first version reaching n)
for ver, g in trig.groupby('ver'):
    g = g.sort_values('t').drop_duplicates('sta')
    if len(g) <= N_MAX and len(g) not in sets:
        sets[len(g)] = g.sort_values('order').assign(epic_ver=ver)

RUN_DIR.mkdir(exist_ok=True)
for n, g in sets.items():
    pd.DataFrame({'version': 0, 'order': range(1, n + 1), 'station': g.sta.values, 'channel': g.chan.values,
                  'network': g.net.values, 'location': g['loc'].values, 'longitude': g.lon.values,
                  'latitude': g.lat.values, 'trigger time': g.t.values, 'tterr': g.tterr.values,
                  'logPd': 0.0}).to_csv(RUN_DIR / f'{EVENT_ID}_nS{n}.run', index=False)
print('station counts:', {n: f'EPIC ver {g.epic_ver.iloc[0]}' for n, g in sets.items()}, '-> .run files in', RUN_DIR)

#%% EPIC's own location error at each station count (first E:I: / E:I:F: row reaching it)
epic_err = {}
for line in LOG.open(errors='replace'):
    _, _, msg = line.partition('| INFO | E:I:')
    msg = msg[2:] if msg.startswith('F:') else msg
    f = msg.split()
    if len(f) >= 17 and f[0].lstrip('-').isdigit() and int(f[0]) > 0 and int(f[16]) not in epic_err:
        epic_err[int(f[16])] = float(_haversine_km(*USGS, np.array([float(f[2])]), np.array([float(f[3])]))[0])

#%% ETAS prior at the mainshock (as time_dependent_scripts/case_studies.py builds it)
config.ETAS_INVERSION_CONFIG['bw_sq'] = 4
inv  = CS_DATA / 'etas_inversion' / f'parameters_{config.etas_output_id(CASE)}.json'
hist = CS_DATA / 'etas_inversion' / 'input' / f'catalog_{config.etas_catalog_tag(CASE)}.csv'
benchmark_runner.repair_inversion_json_paths(str(inv))
updater = EtasPriorUpdater.from_inversion_json(
    json_path=str(inv),
    catalog_df=pd.read_csv(hist, index_col=0, parse_dates=['time'], dtype={'url': str, 'alert': str}),
    **config.ETAS_UPDATER_CONFIG, spatial_factor=None)
cat = download_case_study_catalog(config.CASE_STUDIES[CASE], cache_dir=str(CS_DATA), REDOWNLOAD=False)
cs = (cat[['id', 'time', 'latitude', 'longitude', 'mag']].rename(columns={'mag': 'magnitude'})
      .assign(time=lambda d: pd.to_datetime(d['time']).dt.tz_localize(None))
      .query(f"magnitude >= {config.ETAS_INVERSION_CONFIG['m_ref']}").sort_values('time'))
t_event = pd.Timestamp(sets[N_MAX if N_MAX in sets else max(sets)].t.min(), unit='s')
# Only events strictly before this one; the event itself joins the context AFTER it is located.
before = cs[(cs.time < t_event) & (cs.id != EVENT_ID)]
if len(before):
    updater.append_events(before[['time', 'latitude', 'longitude', 'magnitude']])
etas = updater.update(t_event)
print(f'ETAS prior at {t_event:%H:%M:%S}; {len(before)} case-study events in context')

#%% Locator
def locate(g, prior, sigma_s, dtt_weight, vp, depths):
    pk = dict(config.BENCHMARK_PARAMS)
    pk.update(sigma_s=sigma_s, edt_sigma_s=EDT_SIGMA_S, dtt_weight=dtt_weight, search_depths=list(depths),
              max_trigs=N_MAX)
    params = make_epic_params(prior, True, pk)
    params.LocationPVelocity = vp
    first = g.iloc[0]
    event = EPIC_locate_prelim.Event(lat=first.lat, lon=first.lon, time=first.t, misfit_rms=0, misfit_ave=0,
                                     eventid=EVENT_ID, version=0)
    event.trigs = [EPIC_locate_prelim.TriggerManager(lon=r.lon, lat=r.lat, trigger_time=r.t, sta=r.sta,
                                                     net=r.net, chan=r.chan) for r in g.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        _, odf = EPIC_locate_prelim.E2Location_locate(params, event)
    return odf

def err_km(lat, lon):
    return float(_haversine_km(*USGS, np.array([lat]), np.array([lon]))[0])

#%% Sanity checks: (1) same answer as the benchmark runner on the .run file, (2) posterior == like * prior^alpha
g = sets[4]
odf = locate(g, etas, **{k: BASELINE[k] for k in ('sigma_s', 'dtt_weight', 'vp', 'depths')})
pk = dict(config.BENCHMARK_PARAMS); pk.update(sigma_s=BASELINE['sigma_s'], max_trigs=N_MAX)
with contextlib.redirect_stdout(io.StringIO()):
    _, ref, _ = run_single_event_get_grid(str(RUN_DIR / f'{EVENT_ID}_nS4.run'), etas, True, pk, focus_version=0)
assert np.allclose(odf['post'], ref['post']), 'direct locate() differs from the benchmark runner'
for a in (1.3, 2.0):
    with contextlib.redirect_stdout(io.StringIO()):
        t_ = copy.deepcopy(etas); t_.grid = t_.grid ** a; t_.grid /= t_.grid.sum()
        direct = locate(g, t_, **{k: BASELINE[k] for k in ('sigma_s', 'dtt_weight', 'vp', 'depths')})
    ana = odf['like'].values * odf['prior'].values ** a
    assert direct['post'].values.argmax() == ana.argmax(), f'analytic alpha={a} sweep disagrees with the locator'
print('sanity checks passed (matches benchmark runner; alpha sweep is exact)')

#%% Search
lat, lon = odf['lat'].values, odf['lon'].values
results = []
for sigma_s, dtt_weight, vp, depths in itertools.product(*GRID.values()):
    for n, g in sets.items():
        if n < N_MIN:
            continue
        o = locate(g, etas, sigma_s, dtt_weight, vp, depths)
        like, prior = np.log(np.maximum(o['like'].values, 1e-300)), np.log(np.maximum(o['prior'].values, 1e-300))
        for a in ALPHAS:
            i = int(np.argmax(like + a * prior))
            results.append(dict(sigma_s=sigma_s, dtt_weight=dtt_weight, vp=vp, depths=str(list(depths)),
                                alpha=a, n=n, err_km=err_km(o['lat'].values[i], o['lon'].values[i])))
res = pd.DataFrame(results)
res.to_csv(HERE / 'epic_log_param_search.csv', index=False)

keys = ['sigma_s', 'dtt_weight', 'vp', 'depths', 'alpha']
score = (res.groupby(keys).err_km.agg(mean='mean', worst='max').reset_index())
b = res[(res.sigma_s == BASELINE['sigma_s']) & (res.dtt_weight == BASELINE['dtt_weight']) & (res.vp == BASELINE['vp'])
        & (res.depths == str(list(BASELINE['depths']))) & (res.alpha == BASELINE['alpha'])]
pd.set_option('display.width', 200)
print(f'\nBaseline {BASELINE}: mean {b.err_km.mean():.1f} km, worst {b.err_km.max():.1f} km over n={N_MIN}..{N_MAX}')
print(f'EPIC log:  mean {np.mean([epic_err[n] for n in sets if n >= N_MIN]):.1f} km')
print('\nBest 15 by mean error:'); print(score.sort_values('mean').head(15).round(1).to_string(index=False))
print('\nMarginal (best mean error with each value fixed):')
for k in keys:
    print(f'  {k}:', {v: round(float(s['mean'].min()), 1) for v, s in score.groupby(k)})

#%% Plot: error vs station count, EPIC log vs baseline vs best
best = score.sort_values('mean').iloc[0]
sel = lambda r: res[(res[keys] == r[keys].values).all(axis=1)].sort_values('n')
fig, ax = plt.subplots(figsize=(7, 4.5))
ns = [n for n in sets if n >= N_MIN]
ax.plot(ns, [epic_err[n] for n in ns], 'o-', c='gray', label='EPIC log')
bs = b.sort_values('n')
ax.plot(bs['n'], bs.err_km, 's-', c='darkblue', label='benchmark, case-study settings')
ax.plot(sel(best)['n'], sel(best).err_km, '^-', c='red',
        label=f"best: σ={best.sigma_s}, α={best.alpha}, Vp={best.vp}, dtt={best.dtt_weight}")
ax.set_xlabel('Stations reporting'); ax.set_ylabel('Distance to USGS (km)'); ax.set_ylim(bottom=0)
ax.set_title('M7.0 MTJ mainshock, EPIC-log triggers + ETAS prior'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / 'epic_log_param_search.png', dpi=150)
print('Saved', HERE / 'epic_log_param_search.png')
