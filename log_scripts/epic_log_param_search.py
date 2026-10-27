"""
Hyperparameter search for the 2024 M7.0 MTJ mainshock (nc75095651) using the stations/triggers
from an EPIC replay log (log_files/MTJ-selected/), located with the benchmark's bEPIC locator and
the matching prior.  PRIOR='ETAS' uses MTJ_sigma_and_prior + the dynamic ETAS prior; PRIOR='KDE' uses
MTJ_control + the KDE_Seismicity prior (set with the PRIOR environment variable).  Outputs are tagged
with the prior name.

For each distinct station count (up to N_MAX) the first EPIC version that reaches it is turned
into a .run file (one trigger per station, in the log's own order; written to
data/.../MTJ_2024_M7/epic_log_run_files/).  Each parameter combination is then scored by the
MAP error (km) against the USGS epicenter over those station counts.

Cheap trick: the bEPIC posterior is exactly likelihood * prior (checked below), so PRIOR_ALPHA
(prior ** alpha) is swept analytically from one locator run per (sigma_s, dtt_weight, depths).
One event only, so treat the winner as a diagnostic, not a tuned setting.

Last cells: drop-one-station test (does removing the worst-fitting station at each station count help?)
and its plot.  The drop test also gives a magnitude for the new location from EPIC's own scaling
relation (Kuyuk & Allen, 2013; Williamson et al., 2023): M = 1.23*log10(Pd) + 1.39*log10(R_km) + 5.39
per station, with the logged log_pd and R = distance from the (new) location to the station; the event
magnitude is the weighted average over the triggers EPIC used (updm == 1); stations with < 4 s of waveform
(the log's `buftime`) are down-weighted linearly, weight = clip(buftime / 4, 0.3, 1) (Williamson et al. 2023;
A. Williamson, pers. comm.; the 0.3 floor is fitted to the logged event magnitudes, rms 0.04).  Note the log's R is the
epicentral distance (that reproduces its pdmag to rms 0.02; a hypocentral R does not).
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

from priors import EtasPriorUpdater, SeismicPrior
import benchmark.config_california as config
from benchmark import runner as benchmark_runner
from benchmark.runner import make_epic_params, run_single_event_get_grid
from benchmark.usgs import download_case_study_catalog
from benchmark.metrics import _haversine_km
from bEPIC import EPIC_locate_prelim

ROOT  = Path(__file__).resolve().parents[1]
HERE  = Path(__file__).resolve().parent
CASE, EVENT_ID = 'MTJ_2024_M7', 'nc75095651'
PRIOR = os.environ.get('PRIOR', 'ETAS')                       # 'ETAS' or 'KDE'
RUN   = {'ETAS': 'MTJ_sigma_and_prior', 'KDE': 'MTJ_control'}[PRIOR]
LOG   = ROOT / 'log_files' / 'MTJ-selected' / RUN / 'epic' / f'{EVENT_ID}.log'
CS_DATA = ROOT / 'data' / 'california' / 'case_studies' / CASE
RUN_DIR = CS_DATA / 'epic_log_run_files'
USGS  = (40.3740, -125.0217)

N_MIN, N_MAX = 4, 10            # station counts scored (3 stations is too ambiguous to score)
BASELINE = dict(sigma_s=0.22, dtt_weight=0.0, depths=(8.0,), alpha=1.0)   # the case-study settings
GRID = dict(
    sigma_s    = [0.1, 0.22, 0.35, 0.5, 1.0],      # travel-time uncertainty per pick (s)
    dtt_weight = [0.0, 0.5],                        # weight of differential travel times
    # Source depth is held at 8 km. Searching it is future work (travel-time tables
    # bEPIC/data/h2p+ak135.* exist only for 1, 8, 20, 50, 100 km). params.LocationPVelocity is not
    # used by the locator, which reads those tables, so Vp is not searched either.
    depths     = [(8.0,)],
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
for c in ('eventid', 'ver', 'order', 'lat', 'lon', 'tterr', 'log_pd', 'pdmag', 'updm', 'distkm', 'buftime'):
    trig[c] = pd.to_numeric(trig[c])
trig = trig[trig.eventid > 0].copy()
trig['t'] = pd.to_datetime(trig['trigger_time']).dt.tz_localize('UTC').map(lambda x: x.timestamp())

# EPIC's Pd magnitude scaling (Kuyuk & Allen 2013; Williamson et al. 2023); check it against the log
PD_A, PD_B, PD_C = 1.23, 1.39, 5.39
PD_WINDOW_S, W_FLOOR = 4.0, 0.3
pdfit = trig.drop_duplicates(['sta', 'chan'])
pdfit = pdfit[(pdfit.pdmag > 0) & (pdfit.distkm > 0) & np.isfinite(pdfit.log_pd)]
print('Pd magnitude formula vs logged pdmag: rms',
      round(float(np.sqrt(((PD_A * pdfit.log_pd + PD_B * np.log10(pdfit.distkm) + PD_C - pdfit.pdmag) ** 2).mean())), 3))

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
                  'logPd': 0.0}).to_csv(RUN_DIR / f'{EVENT_ID}_{PRIOR}_nS{n}.run', index=False)
print('station counts:', {n: f'EPIC ver {g.epic_ver.iloc[0]}' for n, g in sets.items()}, '-> .run files in', RUN_DIR)

#%% EPIC's own location error at each station count (first E:I: / E:I:F: row reaching it)
epic_err, epic_loc, epic_mag = {}, {}, {}
for line in LOG.open(errors='replace'):
    _, _, msg = line.partition('| INFO | E:I:')
    msg = msg[2:] if msg.startswith('F:') else msg
    f = msg.split()
    if len(f) >= 17 and f[0].lstrip('-').isdigit() and int(f[0]) > 0 and int(f[16]) not in epic_err:
        epic_err[int(f[16])] = float(_haversine_km(*USGS, np.array([float(f[2])]), np.array([float(f[3])]))[0])
        epic_loc[int(f[16])] = (float(f[2]), float(f[3]))
        epic_mag[int(f[16])] = float(f[5])

#%% Prior: ETAS at the mainshock (as time_dependent_scripts/case_studies.py builds it), or KDE_Seismicity
if PRIOR == 'ETAS':
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
else:
    etas = SeismicPrior.from_tt3(os.path.join(SeismicPrior.data_dir, f'kde_seismicity_{CASE}.tt3'))   # named `etas` below for either prior

#%% Locator
def locate(g, prior, sigma_s, dtt_weight, depths):
    pk = dict(config.BENCHMARK_PARAMS)
    pk.update(sigma_s=sigma_s, edt_sigma_s=EDT_SIGMA_S, dtt_weight=dtt_weight, search_depths=list(depths),
              max_trigs=N_MAX)
    params = make_epic_params(prior, True, pk)
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
odf = locate(g, etas, **{k: BASELINE[k] for k in ('sigma_s', 'dtt_weight', 'depths')})
pk = dict(config.BENCHMARK_PARAMS); pk.update(sigma_s=BASELINE['sigma_s'], max_trigs=N_MAX)
with contextlib.redirect_stdout(io.StringIO()):
    _, ref, _ = run_single_event_get_grid(str(RUN_DIR / f'{EVENT_ID}_{PRIOR}_nS4.run'), etas, True, pk, focus_version=0)
assert np.allclose(odf['post'], ref['post']), 'direct locate() differs from the benchmark runner'
for a in (1.3, 2.0):
    with contextlib.redirect_stdout(io.StringIO()):
        t_ = copy.deepcopy(etas); t_.grid = t_.grid ** a; t_.grid /= t_.grid.sum()
        direct = locate(g, t_, **{k: BASELINE[k] for k in ('sigma_s', 'dtt_weight', 'depths')})
    ana = odf['like'].values * odf['prior'].values ** a
    assert direct['post'].values.argmax() == ana.argmax(), f'analytic alpha={a} sweep disagrees with the locator'
print('sanity checks passed (matches benchmark runner; alpha sweep is exact)')

#%% Search
lat, lon = odf['lat'].values, odf['lon'].values
results = []
for sigma_s, dtt_weight, depths in itertools.product(*GRID.values()):
    for n, g in sets.items():
        if n < N_MIN:
            continue
        o = locate(g, etas, sigma_s, dtt_weight, depths)
        like, prior = np.log(np.maximum(o['like'].values, 1e-300)), np.log(np.maximum(o['prior'].values, 1e-300))
        for a in ALPHAS:
            i = int(np.argmax(like + a * prior))
            results.append(dict(sigma_s=sigma_s, dtt_weight=dtt_weight, depths=str(list(depths)),
                                alpha=a, n=n, err_km=err_km(o['lat'].values[i], o['lon'].values[i])))
res = pd.DataFrame(results)
res.to_csv(HERE / f'epic_log_param_search_{PRIOR}.csv', index=False)

keys = ['sigma_s', 'dtt_weight', 'depths', 'alpha']
score = (res.groupby(keys).err_km.agg(mean='mean', worst='max').reset_index())
b = res[(res.sigma_s == BASELINE['sigma_s']) & (res.dtt_weight == BASELINE['dtt_weight'])
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
        label=f"best: σ={best.sigma_s}, α={best.alpha}, dtt={best.dtt_weight}, depths={best.depths}")
ax.set_xlabel('Stations reporting'); ax.set_ylabel('Distance to USGS (km)'); ax.set_ylim(bottom=0)
ax.set_title(f'M7.0 MTJ mainshock, EPIC-log triggers + {PRIOR} prior'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / f'epic_log_param_search_{PRIOR}.png', dpi=150)
print('Saved', HERE / f'epic_log_param_search_{PRIOR}.png')

#%% Drop one station at each station count
# 'worst residual': locate with all stations, drop the station with the largest travel-time residual
#                   at that solution, relocate.
# 'best-fit drop':  try dropping each station in turn; keep the drop whose solution has the lowest rms residual.
# 'oracle':         the single drop that happens to give the smallest location error (not usable in practice;
#                   upper bound on what dropping one station can do).
from bEPIC.EPIC_locate_prelim import _get_travel_time_function
TT8 = _get_travel_time_function(8.0)
DROP_SETTINGS = [(0.22, 1.0), (0.22, 1.3), (0.5, 1.0)]       # (sigma_s, PRIOR_ALPHA)

def resid(g, lat, lon):
    """Per-station travel-time residuals (s) at a candidate location, origin time = mean (as the locator)."""
    d = _haversine_km(lat, lon, g.lat.values, g.lon.values)
    ot = g.t.values - TT8(np.sqrt(d ** 2 + 8.0 ** 2))
    return ot - ot.mean()

def map_loc(g, sigma_s, alpha):
    o = locate(g, etas, sigma_s, 0.0, (8.0,))
    i = int(np.argmax(np.log(np.maximum(o['like'].values, 1e-300)) + alpha * np.log(np.maximum(o['prior'].values, 1e-300))))
    return o['lat'].values[i], o['lon'].values[i]

rows, resid_all = [], {}
for sigma_s, alpha in DROP_SETTINGS:
    for n, g in sets.items():
        if n < N_MIN:
            continue
        g = g.reset_index(drop=True)
        lat0, lon0 = map_loc(g, sigma_s, alpha)
        r0 = resid(g, lat0, lon0)
        w = int(np.argmax(np.abs(r0)))
        resid_all[(sigma_s, alpha, n)] = (list(g.sta), r0)
        row = dict(sigma_s=sigma_s, alpha=alpha, n=n, all=err_km(lat0, lon0), worst_station=g.sta[w], worst_resid_s=r0[w],
                   all_lat=lat0, all_lon=lon0)
        row['drop_lat'], row['drop_lon'] = map_loc(g.drop(index=w), sigma_s, alpha)
        row['drop_worst_resid'] = err_km(row['drop_lat'], row['drop_lon'])
        # Magnitude at the new location from the Pd scaling relation (weighted average over the triggers EPIC used)
        tv = trig[(trig.ver == g.epic_ver.iloc[0]) & (trig.updm == 1) & (trig.sta != g.sta[w])]
        r_new = _haversine_km(row['drop_lat'], row['drop_lon'], tv.lat.values, tv.lon.values)
        row['mag_log'] = epic_mag[n]
        row['mag_drop'] = float(np.average(PD_A * tv.log_pd.values + PD_B * np.log10(r_new) + PD_C,
                                           weights=np.clip(tv.buftime.values / PD_WINDOW_S, W_FLOOR, 1)))
        loo = []
        for k in range(n):
            gk = g.drop(index=k)
            la, lo = map_loc(gk, sigma_s, alpha)
            loo.append((float(np.sqrt(np.mean(resid(gk, la, lo) ** 2))), err_km(la, lo), g.sta[k]))
        best_fit = min(loo)                                   # lowest rms residual
        oracle = min(loo, key=lambda x: x[1])                 # lowest location error
        row.update(drop_best_fit=best_fit[1], best_fit_station=best_fit[2],
                   oracle=oracle[1], oracle_station=oracle[2], rms_all=float(np.sqrt(np.mean(r0 ** 2))))
        rows.append(row)
drop = pd.DataFrame(rows)
drop.to_csv(HERE / f'epic_log_drop_station_{PRIOR}.csv', index=False)
print('\nMAP error (km) vs USGS after dropping one station:')
for (sigma_s, alpha), d in drop.groupby(['sigma_s', 'alpha']):
    print(f'\n sigma_s={sigma_s}, alpha={alpha}')
    print(d[['n', 'rms_all', 'all', 'worst_station', 'drop_worst_resid', 'mag_log', 'mag_drop', 'best_fit_station',
             'drop_best_fit', 'oracle_station', 'oracle']].round(1).to_string(index=False))

#%% Plot: effect of dropping the worst-fitting station
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

MAIN = DROP_SETTINGS[0]
dm = drop[(drop.sigma_s == MAIN[0]) & (drop.alpha == MAIN[1])].sort_values('n')
n_show = max(dm.n)
stas, r_show = resid_all[(*MAIN, n_show)]
worst = dm[dm.n == n_show].worst_station.iloc[0]
PC = ccrs.PlateCarree()

fig = plt.figure(figsize=(18, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.7, 1.2])
# (a) per-station residuals at the all-station solution
ax = fig.add_subplot(gs[0])
order = np.argsort(r_show)
cols = ['red' if stas[i] == worst else 'gray' for i in order]
ax.barh(range(len(order)), r_show[order], color=cols)
ax.set_yticks(range(len(order))); ax.set_yticklabels([stas[i] for i in order])
ax.axvline(0, c='k', lw=.8); ax.axvspan(-2 * MAIN[0], 2 * MAIN[0], color='darkblue', alpha=.15, label=f'±2σ (σ={MAIN[0]} s)')
ax.set_xlabel('Travel-time residual at the located solution (s)')
ax.set_title(f'(a)  Residuals, {n_show} stations, all used', fontweight='bold'); ax.legend(loc='lower right', fontsize=8)

# (b) map of the located points
ax = fig.add_subplot(gs[1], projection=PC)
ax.set_extent([-125.8, -123.9, 40.0, 40.75], crs=PC)
ax.add_feature(cfeature.LAND, facecolor='#f2f2f2'); ax.add_feature(cfeature.OCEAN, facecolor='lightcyan')
ax.add_feature(cfeature.COASTLINE, lw=.7)
g_show = sets[n_show]
bad = g_show[g_show.sta == worst]
ax.scatter(g_show.lon, g_show.lat, marker='^', s=45, c='w', edgecolor='k', zorder=4, transform=PC)
ax.scatter(bad.lon, bad.lat, marker='^', s=110, c='red', edgecolor='k', zorder=5, transform=PC)
ax.annotate(worst, (bad.lon.iloc[0], bad.lat.iloc[0]), xytext=(6, 8), textcoords='offset points', color='red', fontweight='bold', transform=PC)
ax.plot(USGS[1], USGS[0], '*', c='gold', mec='k', ms=20, zorder=6, transform=PC)
ax.plot(dm.all_lon, dm.all_lat, 'o', c='darkblue', mec='k', ms=7, zorder=5, transform=PC)
ax.plot(dm.drop_lon[dm.n >= 5], dm.drop_lat[dm.n >= 5], 's', c='red', mec='k', ms=7, zorder=5, transform=PC)
ax.plot([epic_loc[n][1] for n in dm.n], [epic_loc[n][0] for n in dm.n], 'x', c='gray', ms=8, mew=2, zorder=5, transform=PC)
gl = ax.gridlines(draw_labels=True, lw=.3, alpha=.5, ls='--'); gl.top_labels = gl.right_labels = False
ax.set_title('(b)  Locations, 4–10 stations', fontweight='bold')
ax.legend(handles=[Line2D([], [], marker='*', ls='', c='gold', mec='k', ms=14, label='USGS epicenter'),
                   Line2D([], [], marker='x', ls='', c='gray', ms=8, mew=2, label='EPIC log'),
                   Line2D([], [], marker='o', ls='', c='darkblue', mec='k', label='all stations'),
                   Line2D([], [], marker='s', ls='', c='red', mec='k', label=f'worst station ({worst}) dropped'),
                   Line2D([], [], marker='^', ls='', c='w', mec='k', label='stations')], loc='upper left', fontsize=8, framealpha=.95)

# (c) error vs station count
ax = fig.add_subplot(gs[2])
ax.plot(dm.n, [epic_err[n] for n in dm.n], 'x-', c='gray', mew=2, label='EPIC log')
ax.plot(dm.n, dm['all'], 'o-', c='darkblue', label=f'all stations (σ={MAIN[0]}, α={MAIN[1]})')
d5 = dm[dm.n >= 5]
ax.plot(d5.n, d5.drop_worst_resid, 's-', c='red', label='worst-residual station dropped')
# One distinct style per extra setting (colour, linestyle and marker all differ)
EXTRA_STYLES = [dict(c='darkorange', ls='--', marker='^'), dict(c='purple', ls='-.', marker='v'),
                dict(c='teal', ls=':', marker='D')]
extra = [(k, d) for k, d in drop.groupby(['sigma_s', 'alpha']) if k != MAIN]
for (sg, al), d in extra:
    st = EXTRA_STYLES[[k for k, _ in extra].index((sg, al)) % len(EXTRA_STYLES)]
    d = d[d.n >= 5].sort_values('n')
    ax.plot(d.n, d.drop_worst_resid, lw=1.5, ms=5, label=f'  dropped, σ={sg}, α={al}', **st)
ax.set_xlabel('Stations reporting'); ax.set_ylabel('Distance to USGS (km)'); ax.set_ylim(bottom=0)
ax.set_title('(c)  Location error', fontweight='bold'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / f'epic_log_drop_station_{PRIOR}.png', dpi=150)
print('Saved', HERE / f'epic_log_drop_station_{PRIOR}.png')
