"""
Re-run EPIC replay logs with the bEPIC locator, the dynamic ETAS prior and the station activity mask.  RUN ONLY: writes
CSVs to output/, no figures (figures come from a separate script).

Scenarios (SEQUENCE):
  California  the 237+ event benchmark catalog (bEPIC_testing_catalog.txt)      PLACEHOLDER, no logs yet
  Ridgecrest  Ridgecrest 2019 case study                                         sigma_s 0.22
  MTJ         Mendocino Triple Junction 2024 M7 case study                       sigma_s 0.22
  Testsuite   individual events                                                  PLACEHOLDER, no logs yet

For every ComCat event with an EPIC log (<anss_id>.log in the sigma_and_prior run's folder) the script
  1. finds the EPIC event that matches the USGS/ANSS event (parsing_tools.match_event),
  2. starts the search grid at lat0/lon0 of the FIRST L:E: line of that event in the log
     (EPIC's own starting point; see AMY_SCRIPTS_REVIEW.md, "Reproducing the EPIC replays"),
  3. steps through the logged versions in order.  Each version's E:I:T: triggers (every trigger, up to two per station,
     all counting equally) go to E2Location_locate, which re-centres the grid on its location after every station,
  4. recomputes the magnitude at the new location from the logged Pd values (log_pd column), each trigger weighted by its seconds of waveform
     (linear taper, 0 s -> 0, >= 4 s -> 1; every trigger counts, whatever the number per station).

Alert rule: the first version with >= ALERT_MIN_STATIONS distinct stations (nS, not triggers) is flagged `alert`.

Output, one row per (event, version): output/<SEQUENCE>_station_masking.csv.
"""
#%%
import io
import os
import sys
import contextlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from priors import EtasPriorUpdater
import benchmark.config_california as config
from benchmark import runner as benchmark_runner
from benchmark.runner import make_epic_params
from benchmark.usgs import get_usgs_event, download_case_study_catalog
from benchmark.metrics import _haversine_km
from bEPIC import EPIC_locate_prelim
import parsing_tools

SEQUENCE = 'MTJ'      # 'California' | 'Ridgecrest' | 'MTJ' | 'Testsuite'

SCENARIOS = {
    'California': dict(ready=False, sigma_s=0.35),    # logs not available yet
    'Ridgecrest': dict(ready=True, case='Ridgecrest',  sigma_s=0.22, channels='Ridgecrest_channels.dat'),
    'MTJ':        dict(ready=True, case='MTJ_2024_M7', sigma_s=0.22, channels='MTJ_channels.dat'),
    'Testsuite':  dict(ready=False, sigma_s=None),     # logs not available yet; sigma_s to be decided
}
SC = SCENARIOS[SEQUENCE]
if not SC['ready']:
    raise NotImplementedError(f'{SEQUENCE}: no EPIC logs yet (placeholder)')

# -- bEPIC settings --
GRID_SIZE, GRID_KM = 25, 50          # 2 km spacing, (2*13+1)^2 nodes, 52 x 52 km box
SIGMA_S, EDT_SIGMA_S = SC['sigma_s'], 0.02
MIN_STATIONS_GRID_SEARCH = 3         # EPIC: below 3 stations no grid search (station / 2:1 point)
MIGRATE_MIN_STATIONS = 1             # re-centre the grid after every station
MAX_TRIGS = 10                       # locator trigger cap; high so late versions with 2 triggers/station are not truncated
# -- Knobs to turn (defaults = EPIC-like) --
ACTIVITY_THRESHOLD = 0.40            # station mask: minimum availability
SEARCH_DEPTHS = [8.0]                # km
# -- Alert, matching, magnitude --
ALERT_MIN_STATIONS = 4               # alert criterion: distinct stations (not triggers)
MAX_ORIGIN_DIFF_S = 30
MIN_USGS_MAG = 3.5                   # only events with USGS magnitude >= this (None = all)
PD_A, PD_B, PD_C = 1.23, 1.39, 5.39  # EPIC Pd magnitude scaling
PD_WINDOW_S = 4.0                    # magnitude weight of a trigger: buftime / PD_WINDOW_S, linear taper, 1 from 4 s of waveform on

CASE = SC['case']
CS_DATA = ROOT / 'data' / 'california' / 'case_studies' / CASE
OUT = Path(__file__).resolve().parent / 'output'
LOG_DIR = ROOT / 'log_files' / f'{SEQUENCE}-selected' / f'{SEQUENCE}_sigma_and_prior' / 'epic'   # always the sigma_and_prior (ETAS) logs

#%% Station inventory for the activity mask: one row per net.sta from the collaborators' channels file
COLS = 'network station location channel latitude longitude elevation samplerate gain units'.split()
INVENTORY = (pd.read_csv(ROOT / SC['channels'], sep=r'\s+', comment='#', names=COLS, dtype=str, keep_default_na=False)
             .drop_duplicates(['network', 'station'])[['station', 'network', 'longitude', 'latitude']]
             .astype({'longitude': float, 'latitude': float}).reset_index(drop=True))
print(f'{len(INVENTORY)} stations in the mask inventory')

#%% Logs and USGS solutions (cached)
logs = {p.stem: parsing_tools.read_ei(p) for p in sorted(LOG_DIR.glob('*.log'))}
print(f'{len(logs)} logs; {sum(not d.empty for d in logs.values())} contain an EPIC event')

def fetch_usgs(anss_id):
    try:
        g = get_usgs_event(anss_id)
        lon, lat, dep = g['geometry']['coordinates']
        return dict(anss_id=anss_id, lat=lat, lon=lon, mag=g['properties']['mag'],
                    time=pd.to_datetime(g['properties']['time'], unit='ms'))
    except Exception as e:
        print(f'USGS lookup failed for {anss_id}: {e}')

USGS_CACHE = OUT / f'usgs_cache_{SEQUENCE}.csv'
usgs = pd.read_csv(USGS_CACHE, parse_dates=['time']) if USGS_CACHE.exists() else pd.DataFrame(
    columns=['anss_id', 'lat', 'lon', 'mag', 'time'])
todo = sorted({i for i, d in logs.items() if not d.empty} - set(usgs.anss_id))
if todo:
    with ThreadPoolExecutor(8) as ex:
        new = [r for r in ex.map(fetch_usgs, todo) if r]
    usgs = pd.concat([usgs, pd.DataFrame(new)], ignore_index=True)
    usgs.to_csv(USGS_CACHE, index=False)
usgs = usgs.set_index('anss_id')

#%% Log readers and the locator
def read_le(path):
    """L:E: lines (EPIC's location searches): eventid, ver, lat0, lon0 = where each search started."""
    rows = []
    for f in parsing_tools._lines(path, '| INFO | L:E:'):
        if len(f) >= 5 and f[0].lstrip('-').isdigit():
            rows.append((int(f[0]), int(f[1]), float(f[3]), float(f[4])))
    return pd.DataFrame(rows, columns=['eventid', 'ver', 'lat0', 'lon0'])

def locate(event, g, prior):
    """One bEPIC step: triggers g (a DataFrame) -> (lat, lon).  Re-centres `event` on the result."""
    event.time = g.t.min()
    event.trigs = [EPIC_locate_prelim.TriggerManager(lon=r.lon, lat=r.lat, trigger_time=r.t, sta=r.sta, net=r.net,
                                                     chan=r.chan) for r in g.itertuples()]
    params = make_epic_params(prior, True, dict(
        grid_size=GRID_SIZE, grid_km=GRID_KM, max_trigs=MAX_TRIGS, migrate_grid=True,
        migrate_grid_min_stations=MIGRATE_MIN_STATIONS, min_stations_grid_search=MIN_STATIONS_GRID_SEARCH,
        activity_threshold=ACTIVITY_THRESHOLD, resample_distant_events=False, sigma_s=SIGMA_S,
        edt_sigma_s=EDT_SIGMA_S, dtt_weight=0.0, search_depths=SEARCH_DEPTHS), station_inventory=INVENTORY)
    with contextlib.redirect_stdout(io.StringIO()):
        t, _ = EPIC_locate_prelim.E2Location_locate(params, event)
    return t.posterior_lat, t.posterior_lon

def pd_magnitude(tv, lat, lon):
    """EPIC Pd magnitude at (lat, lon) from the triggers used for magnitude (updm == 1)."""
    tw = tv[tv.updm == 1]
    if tw.empty:
        return np.nan
    r = np.maximum(_haversine_km(lat, lon, tw.lat.values, tw.lon.values), 1.0)   # 1 km floor
    m = PD_A * tw.log_pd.values + PD_B * np.log10(r) + PD_C
    w = np.clip(tw.buftime.values / PD_WINDOW_S, 0, 1)              # per trigger (a station's HH and HN each count), no floor
    return float(np.average(m, weights=w)) if w.sum() > 0 else np.nan

#%% ETAS prior: case-study inversion, fed every catalog event before the trigger time
config.ETAS_INVERSION_CONFIG['bw_sq'] = 4
inv  = CS_DATA / 'etas_inversion' / f'parameters_{config.etas_output_id(CASE)}.json'
hist = CS_DATA / 'etas_inversion' / 'input' / f'catalog_{config.etas_catalog_tag(CASE)}.csv'
benchmark_runner.repair_inversion_json_paths(str(inv))
updater = EtasPriorUpdater.from_inversion_json(
    json_path=str(inv), catalog_df=pd.read_csv(hist, index_col=0, parse_dates=['time'], dtype={'url': str, 'alert': str}),
    **config.ETAS_UPDATER_CONFIG, spatial_factor=None)
cat = download_case_study_catalog(config.CASE_STUDIES[CASE], cache_dir=str(CS_DATA), REDOWNLOAD=False)
cs = (cat[['id', 'time', 'latitude', 'longitude', 'mag']].rename(columns={'mag': 'magnitude'})
      .assign(time=lambda d: pd.to_datetime(d['time']).dt.tz_localize(None))
      .query(f"magnitude >= {config.ETAS_INVERSION_CONFIG['m_ref']}").sort_values('time'))
added = set()

def etas_prior(t, anss_id):
    t_event = pd.Timestamp(t, unit='s')
    new = cs[(cs.time < t_event) & (cs.id != anss_id) & ~cs.id.isin(added)]
    if len(new):
        updater.append_events(new[['time', 'latitude', 'longitude', 'magnitude']])
        added.update(new.id)
    return updater.update(t_event)

#%% Select the matched EPIC event of every ComCat event and collect its logged triggers
jobs = []
for anss_id, ei in logs.items():
    if ei.empty or anss_id not in usgs.index:
        continue
    u = usgs.loc[anss_id]
    if MIN_USGS_MAG is not None and not u.mag >= MIN_USGS_MAG:
        continue
    eventid = parsing_tools.match_event(ei, u.time, u.lat, u.lon, max_dt_s=MAX_ORIGIN_DIFF_S)
    if eventid is None:
        continue
    path = LOG_DIR / f'{anss_id}.log'
    trig = parsing_tools.read_eit(path)
    for c in ('eventid', 'ver', 'order', 'lat', 'lon', 'log_pd', 'updm', 'buftime'):
        trig[c] = pd.to_numeric(trig[c], errors='coerce')
    trig = trig[trig.eventid == eventid]
    le = read_le(path)
    le = le[le.eventid == eventid]                                  # the event's L:E: lines in log order
    if trig.empty or le.empty:
        continue
    jobs.append((trig.t.min(), anss_id, eventid, u, trig, le.iloc[0], ei[ei.eventid == eventid].set_index('ver')))
print(f'{len(jobs)} events to re-run')

#%% Re-run, oldest event first (the ETAS prior must only see earlier events)
rows = []
for t0, anss_id, eventid, u, trig, start, ei in sorted(jobs, key=lambda j: j[0]):
    prior = etas_prior(t0, anss_id)
    event = EPIC_locate_prelim.Event(lat=start.lat0, lon=start.lon0, time=t0, misfit_rms=0, misfit_ave=0,
                                     eventid=str(eventid), version=0)       # grid starts at EPIC's own starting point
    alerted = False
    for ver, tv in trig.groupby('ver'):                                     # versions in order; the grid migrates between them
        g = tv.sort_values('order').reset_index(drop=True)
        lat, lon = locate(event, g, prior)
        n_sta = g[['net', 'sta']].drop_duplicates().shape[0]
        alert = (not alerted) and n_sta >= ALERT_MIN_STATIONS               # first version with enough stations
        alerted |= alert
        e = ei.loc[ver] if ver in ei.index else None                        # EPIC's own solution, for comparison only; never used in the relocation
        rows.append(dict(anss_id=anss_id, eventid=eventid, ver=ver, nS=n_sta, nT=len(g), alert=alert,
                         lat=lat, lon=lon, mag=pd_magnitude(g, lat, lon),
                         usgs_lat=u.lat, usgs_lon=u.lon, usgs_mag=u.mag,
                         epic_lat=e.lat if e is not None else np.nan, epic_lon=e.lon if e is not None else np.nan,
                         epic_mag=e.mag if e is not None else np.nan, epic_alerted=bool(e.Ast == 1) if e is not None else False))

res = pd.DataFrame(rows)
res['loc_err_km'] = _haversine_km(res.usgs_lat.values, res.usgs_lon.values, res.lat.values, res.lon.values)
res['epic_err_km'] = _haversine_km(res.usgs_lat.values, res.usgs_lon.values, res.epic_lat.values, res.epic_lon.values)
res['mag_err'] = res.mag - res.usgs_mag
res.to_csv(OUT / f'{SEQUENCE}_station_masking.csv', index=False)

a = res[res.alert]
print(f'{res.anss_id.nunique()} events, {len(a)} reach nS >= {ALERT_MIN_STATIONS}')
print(f'at the alert version: median location error {a.loc_err_km.median():.2f} km (EPIC log {a.epic_err_km.median():.2f} km), '
      f'median magnitude error {a.mag_err.median():+.2f}')
print('Saved', OUT / f'{SEQUENCE}_station_masking.csv')
