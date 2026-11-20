"""
Alarm-signal map for one EPIC replay run: which USGS events got an alarm and which should have.

For every <anss_id>.log of a run folder, the EPIC event that matches the USGS/ANSS event (parsing_tools.summarize,
same matching as the other log scripts) either alerted or not (see ALARM_MIN_EST_MAG for what counts).  That is the "alarm sent" flag.
What *should* have happened comes from the USGS solution alone, following the ShakeAlert rule (paper_figures/shakealert_footprint.py):
an alarm is warranted if USGS M >= 4.5 and the BA08/Worden table has an MMI 3 contour for that magnitude.

Four outcomes, drawn at the USGS epicentre:
  alarm and warranted          (true alarm)
  alarm, not warranted         (false alarm)
  no alarm, but warranted      (missed)
  no alarm, not warranted      (correct silence)

Writes output/alarm_signal_<SEQUENCE>_<RUN>.csv (one row per event) and figures/alarm_signal_<SEQUENCE>_<RUN>.png.
USGS solutions are cached in output/usgs_cache_<SEQUENCE>.csv (shared with run_error_histograms_comp.py).
"""
#%%
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'paper_figures'))
from benchmark.usgs import get_usgs_event
from benchmark import parsing_tools
from shakealert_footprint import contour_distance_km

SEQUENCE = os.environ.get('SEQUENCE', 'MTJ')            # 'MTJ' | 'Ridgecrest' | 'ElMayor' | 'Ferndale'
RUN = os.environ.get('RUN', 'control')                  # 'control' | 'sigma_only' | 'sigma_and_prior'
MAX_ORIGIN_DIFF_S = 30                                  # match log event to USGS origin time within this
MIN_ALERT_MAG, ALERT_MMI = 4.5, 3                       # warranted if USGS M >= this and the MMI contour exists
# Ast == 1 is set for M~3 versions too, so it is not a public alarm.  An alarm here = an alerted version (Ast == 1) of the matched
# EPIC event whose EPIC magnitude is >= this (None = any Ast == 1 version counts).
ALARM_MIN_EST_MAG = 4.5
USGS_CACHE = HERE / 'output' / f'usgs_cache_{SEQUENCE}.csv'

# (alarm sent, warranted) -> label, colour, marker
OUTCOMES = {(True, True):   ('Alarm, warranted',        'tab:green',  'o'),
            (True, False):  ('Alarm, not warranted',    'tab:red',    's'),
            (False, True):  ('No alarm, warranted',     'black',      'X'),
            (False, False): ('No alarm, not warranted', 'lightgray',  '.')}

#%% USGS solutions for every log (cached)
run_dir = ROOT / 'log_files' / f'{SEQUENCE}-selected' / f'{SEQUENCE}_{RUN}' / 'epic'
if not run_dir.is_dir():
    raise FileNotFoundError(f'No logs for {SEQUENCE}/{RUN}: {run_dir} does not exist')
logs = sorted(run_dir.glob('*.log'))

def fetch_usgs(anss_id):
    try:
        g = get_usgs_event(anss_id)
        lon, lat, dep = g['geometry']['coordinates']
        return dict(anss_id=anss_id, lat=lat, lon=lon, mag=g['properties']['mag'],
                    time=pd.to_datetime(g['properties']['time'], unit='ms'))
    except Exception as e:
        print(f'USGS lookup failed for {anss_id}: {e}')

usgs = pd.read_csv(USGS_CACHE, parse_dates=['time']) if USGS_CACHE.exists() else pd.DataFrame(
    columns=['anss_id', 'lat', 'lon', 'mag', 'time'])
todo = sorted({p.stem for p in logs} - set(usgs.anss_id))
if todo:
    print(f'Fetching {len(todo)} USGS events')
    with ThreadPoolExecutor(8) as ex:
        usgs = pd.concat([usgs, pd.DataFrame([r for r in ex.map(fetch_usgs, todo) if r])], ignore_index=True)
    usgs.to_csv(USGS_CACHE, index=False)
usgs = usgs.drop_duplicates('anss_id').set_index('anss_id')

#%% Alarm sent vs. warranted
rows = []
for p in logs:
    if p.stem not in usgs.index:
        continue
    u = usgs.loc[p.stem]
    s = parsing_tools.summarize(p, u.time, u.lat, u.lon, max_dt_s=MAX_ORIGIN_DIFF_S)
    alarm = s['matched_alerted']
    if alarm and ALARM_MIN_EST_MAG is not None:
        ei = parsing_tools.read_ei(p)
        alarm = bool(((ei.eventid == s['matched_eventid']) & (ei.Ast == 1) & (ei.mag >= ALARM_MIN_EST_MAG)).any())
    warranted = bool(u.mag >= MIN_ALERT_MAG and contour_distance_km(u.mag, ALERT_MMI) > 0)
    rows.append(dict(anss_id=p.stem, lat=u.lat, lon=u.lon, mag=u.mag, status=s['status'],
                     alarm=alarm, warranted=warranted))
ev = pd.DataFrame(rows)
ev['outcome'] = [OUTCOMES[(a, w)][0] for a, w in zip(ev.alarm, ev.warranted)]
ev.to_csv(HERE / 'output' / f'alarm_signal_{SEQUENCE}_{RUN}.csv', index=False)
print(f'{SEQUENCE} / {RUN}: {len(ev)} events')
print(ev.outcome.value_counts().to_string())

#%% Map
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection=ccrs.PlateCarree())
pad = 0.5
ax.set_extent([ev.lon.min() - pad, ev.lon.max() + pad, ev.lat.min() - pad, ev.lat.max() + pad])
ax.add_feature(cfeature.LAND, facecolor='0.96')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='0.5')
for (alarm, warranted), (label, color, marker) in OUTCOMES.items():
    d = ev[(ev.alarm == alarm) & (ev.warranted == warranted)]
    ax.scatter(d.lon, d.lat, c=color, marker=marker, s=25 if marker != '.' else 20,
               edgecolors='k' if marker != '.' else 'none', linewidths=0.4, transform=ccrs.PlateCarree(),
               label=f'{label} ({len(d)})', zorder=3 + warranted + alarm)
ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6)
ax.legend(loc='upper right', fontsize=8)
ax.set_title(f'Alarm signal, {SEQUENCE} {RUN} replay (warranted: USGS M ≥ {MIN_ALERT_MAG})')
(HERE / 'figures').mkdir(exist_ok=True)
fig.savefig(HERE / 'figures' / f'alarm_signal_{SEQUENCE}_{RUN}.png', dpi=200, bbox_inches='tight')
plt.show()
